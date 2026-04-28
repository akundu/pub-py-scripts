"""Regression test for the index-mismatch crash in
`compute_range_percentiles_multi_window` when both `exclude_outliers=True`
and a momentum filter (auto-detected from a >=2-day streak) are active.

Before the fix, IQR cleaning rebuilt `return_pct` with a default RangeIndex,
losing the date alignment with `df_window`. The momentum-filter block then
indexed `return_pct.loc[momentum_mask]` with a date-indexed mask, raising
`Unalignable boolean Series provided as indexer`. The HTML endpoint
surfaced this as "Error Computing Multi-Window Range Percentiles".
"""

import asyncio
from pathlib import Path
import sys
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.range_percentiles import (
    _iqr_keep_mask,
    compute_range_percentiles_multi_window,
)


class TestIqrKeepMask:
    """Directly test the new mask helper since the bug-fix uses it."""

    def test_returns_bool_array_same_length(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 250)
        mask = _iqr_keep_mask(data)
        assert mask.dtype == bool
        assert len(mask) == len(data)

    def test_drops_extreme_tails(self):
        rng = np.random.default_rng(0)
        bulk = rng.normal(0, 1, 245)
        outliers = np.array([-12.0, -10.0, -9.0, 9.0, 12.0])
        data = np.concatenate([bulk, outliers])
        mask = _iqr_keep_mask(data, factor=3.0)
        # The 5 extreme outliers should be marked False.
        assert mask.sum() < len(data)
        assert not mask[-5:].any(), "extreme outliers should be dropped"
        assert mask[:245].all(), "bulk should be kept"

    def test_small_sample_returns_all_true(self):
        # n<20 short-circuit
        data = np.array([1.0, 2.0, 3.0])
        assert _iqr_keep_mask(data).all()

    def test_zero_iqr_returns_all_true(self):
        data = np.ones(50)
        assert _iqr_keep_mask(data).all()

    def test_safety_fallback_returns_all_true(self):
        # If cleaning would drop >20% of samples, return all-True so the
        # downstream calculation runs on the raw data.
        rng = np.random.default_rng(0)
        # Heavy-tailed mixture: ~30% of samples land far from the bulk,
        # which would naively trigger a >20% drop -> safety fallback engages.
        bulk = rng.normal(0, 1, 70)
        tails = rng.normal(0, 30, 30)
        data = np.concatenate([bulk, tails])
        mask = _iqr_keep_mask(data, factor=1.5)
        assert mask.all(), "safety fallback should keep all samples"


def _make_synthetic_close_df(n_days: int = 260, seed: int = 7) -> pd.DataFrame:
    """Build a daily-close DataFrame indexed by business days, ending today.

    Forces the last several days into a sustained UP move so
    `_compute_consecutive_days_series` reports a streak >= 2 — that's the
    trigger for `compute_range_percentiles_multi_window` to auto-set a
    momentum filter, which combined with `exclude_outliers=True` is the
    crash path the bug fix targets.
    """
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.utcnow().normalize().tz_localize(None)
    dates = pd.bdate_range(end=end, periods=n_days)
    rets = rng.normal(0.0, 0.012, n_days)
    # Plant 3 consecutive up days at the tail to force the auto-momentum trigger.
    rets[-3:] = np.array([0.008, 0.006, 0.010])
    closes = 20000.0 * (1.0 + rets).cumprod()
    return pd.DataFrame({"close": closes}, index=dates)


@pytest.mark.asyncio
async def test_multi_window_does_not_crash_on_outliers_plus_momentum():
    """Regression: ?ticker=NDX&windows=*&percentiles=80,90 used to raise
    'Unalignable boolean Series provided as indexer'. With the index-
    preserving IQR filter in place it should return a normal result dict.
    """
    df = _make_synthetic_close_df()

    fake_db = AsyncMock()
    fake_db.get_stock_data = AsyncMock(return_value=df)
    fake_db.close = AsyncMock()

    with patch(
        "common.range_percentiles.StockQuestDB",
        return_value=fake_db,
    ):
        result = await compute_range_percentiles_multi_window(
            ticker="NDX",
            windows=[1, 3, 5, 10],
            lookback=250,
            percentiles=[80, 90],
            min_days=20,
            min_direction_days=10,
            db_config="dummy",
            enable_cache=False,
            exclude_outliers=True,
        )

    assert result["ticker"] == "NDX"
    assert "windows" in result and result["windows"], "expected non-empty windows"
    # The auto-detected streak should have produced a momentum filter on the
    # earliest computed window (proving the previously-crashing code path
    # actually executed in this test).
    md = result["metadata"]
    assert md.get("current_streak", 0) >= 2, (
        "test setup failed to plant a >=2-day streak; "
        "the bug path wasn't exercised"
    )
    assert "momentum_filter" in md, (
        "auto-momentum should have engaged on this synthetic data; "
        "without it the regression isn't actually being tested"
    )


@pytest.mark.asyncio
async def test_multi_window_no_outliers_no_momentum_still_works():
    """Sanity check: the non-bug path keeps working."""
    df = _make_synthetic_close_df(seed=11)

    fake_db = AsyncMock()
    fake_db.get_stock_data = AsyncMock(return_value=df)
    fake_db.close = AsyncMock()

    with patch(
        "common.range_percentiles.StockQuestDB",
        return_value=fake_db,
    ):
        result = await compute_range_percentiles_multi_window(
            ticker="NDX",
            windows=[1, 5],
            lookback=250,
            percentiles=[90],
            min_days=20,
            min_direction_days=10,
            db_config="dummy",
            enable_cache=False,
            exclude_outliers=False,
        )

    assert result["ticker"] == "NDX"
    assert "1" in result["windows"]
