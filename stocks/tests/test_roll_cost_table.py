"""Tests for scripts/roll_cost_table.py"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.roll_cost_table import (
    _compute_spread_value,
    analyze_single_day,
    et_time_to_utc,
    find_dte_date,
    find_nearest_strike,
    get_all_available_dates,
    get_equity_price_at,
    get_option_bidask,
    get_option_mid,
    get_trading_dates,
    load_equity_day,
    load_options_day,
    pst_time_to_utc,
    snap_options,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_equity_df():
    """5-min equity bars for a single day, price rises from 2600 to 2650."""
    times = pd.date_range("2025-10-01 13:30", periods=78, freq="5min", tz="UTC")
    prices = np.linspace(2600, 2650, len(times))
    return pd.DataFrame({
        "timestamp": times, "ticker": "I:RUT",
        "open": prices - 1, "high": prices + 2, "low": prices - 2, "close": prices,
    })


def _make_options_rows(timestamps, expirations, strike_range, base_price=2625):
    rows = []
    for ts in timestamps:
        for exp in expirations:
            dte_days = max(0, (pd.Timestamp(exp, tz="UTC") - ts.normalize()).days)
            for strike in strike_range:
                for opt_type in ("call", "put"):
                    intrinsic = max(0, base_price - strike) if opt_type == "call" else max(0, strike - base_price)
                    mid = intrinsic + 3 + dte_days * 2
                    rows.append({
                        "timestamp": ts, "ticker": f"O:RUT{exp.replace('-', '')}",
                        "type": opt_type, "strike": strike, "expiration": exp,
                        "bid": max(0.01, mid - 1), "ask": mid + 1,
                    })
    return rows


@pytest.fixture
def sample_options_df():
    timestamps = pd.date_range("2025-10-01 13:30", periods=28, freq="15min", tz="UTC")
    return pd.DataFrame(_make_options_rows(
        timestamps, ["2025-10-01", "2025-10-02", "2025-10-03"], range(2580, 2680, 5)))


@pytest.fixture
def temp_data_dirs(sample_equity_df, sample_options_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        eq_dir = os.path.join(tmpdir, "equities_output", "I:RUT")
        os.makedirs(eq_dir)
        eq_df = sample_equity_df.copy()
        eq_df["timestamp"] = eq_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
        eq_df.to_csv(os.path.join(eq_dir, "I:RUT_equities_2025-10-01.csv"), index=False)

        opt_dir = os.path.join(tmpdir, "options", "RUT")
        os.makedirs(opt_dir)
        opt_df = sample_options_df.copy()
        opt_df["timestamp"] = opt_df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        opt_df.to_csv(os.path.join(opt_dir, "RUT_options_2025-10-01.csv"), index=False)
        yield {"equities_dir": os.path.join(tmpdir, "equities_output"),
               "options_dir": os.path.join(tmpdir, "options")}


@pytest.fixture
def temp_crossday_dirs(sample_equity_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        eq_dir = os.path.join(tmpdir, "equities_output", "I:RUT")
        os.makedirs(eq_dir)
        eq_df = sample_equity_df.copy()
        eq_df["timestamp"] = eq_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
        eq_df.to_csv(os.path.join(eq_dir, "I:RUT_equities_2025-10-01.csv"), index=False)

        opt_dir = os.path.join(tmpdir, "options", "RUT")
        os.makedirs(opt_dir)
        for day, exp in [("2025-10-01", "2025-10-01"), ("2025-10-02", "2025-10-02"),
                         ("2025-10-03", "2025-10-03")]:
            ts = pd.date_range(f"{day} 13:30", periods=28, freq="5min", tz="UTC")
            df = pd.DataFrame(_make_options_rows(ts, [exp], range(2580, 2680, 5)))
            df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            df.to_csv(os.path.join(opt_dir, f"RUT_options_{day}.csv"), index=False)
        yield {"equities_dir": os.path.join(tmpdir, "equities_output"),
               "options_dir": os.path.join(tmpdir, "options")}


# ── Unit Tests ───────────────────────────────────────────────────────────────

class TestTimeConversion:
    def test_pst_to_utc_standard(self):
        # Dec is PST (UTC-8)
        ts = pst_time_to_utc("2025-12-01", "12:00")
        assert ts.hour == 20

    def test_pst_to_utc_daylight(self):
        # Oct 1 is PDT (UTC-7)
        ts = pst_time_to_utc("2025-10-01", "12:00")
        assert ts.hour == 19

    def test_et_to_utc_standard(self):
        ts = et_time_to_utc("2025-12-01", "12:00")
        assert ts.hour == 17

    def test_et_to_utc_daylight(self):
        ts = et_time_to_utc("2025-10-01", "12:00")
        assert ts.hour == 16


class TestGetEquityPriceAt:
    def test_exact_match(self, sample_equity_df):
        assert get_equity_price_at(sample_equity_df, pd.Timestamp("2025-10-01 14:00", tz="UTC")) is not None

    def test_too_stale(self, sample_equity_df):
        assert get_equity_price_at(sample_equity_df, pd.Timestamp("2025-10-01 12:00", tz="UTC")) is None

    def test_empty_df(self):
        assert get_equity_price_at(pd.DataFrame(), pd.Timestamp("2025-10-01", tz="UTC")) is None


class TestSnapOptions:
    def test_finds_correct_expiration(self, sample_options_df):
        snap = snap_options(sample_options_df, pd.Timestamp("2025-10-01 14:00", tz="UTC"), "2025-10-02", "call")
        assert not snap.empty
        assert all(snap["expiration"] == "2025-10-02")

    def test_tolerance_exceeded(self, sample_options_df):
        assert snap_options(sample_options_df, pd.Timestamp("2025-10-01 10:00", tz="UTC"),
                            "2025-10-01", "call", tolerance_mins=15).empty

    def test_missing_expiration(self, sample_options_df):
        assert snap_options(sample_options_df, pd.Timestamp("2025-10-01 14:00", tz="UTC"),
                            "2025-12-01", "call").empty


class TestGetOptionMid:
    def test_returns_mid(self):
        assert get_option_mid(pd.DataFrame([{"strike": 100, "bid": 4.0, "ask": 6.0}]), 100) == 5.0

    def test_missing(self):
        assert get_option_mid(pd.DataFrame([{"strike": 100, "bid": 4.0, "ask": 6.0}]), 200) is None


class TestGetOptionBidAsk:
    def test_returns_tuple(self):
        assert get_option_bidask(pd.DataFrame([{"strike": 100, "bid": 4.0, "ask": 6.0}]), 100) == (4.0, 6.0)

    def test_missing(self):
        assert get_option_bidask(pd.DataFrame([{"strike": 100, "bid": 4.0, "ask": 6.0}]), 200) is None


class TestFindNearestStrike:
    def test_exact(self, sample_options_df):
        assert find_nearest_strike(sample_options_df, 2600) == 2600

    def test_between(self, sample_options_df):
        assert find_nearest_strike(sample_options_df, 2602) == 2600

    def test_empty(self):
        assert find_nearest_strike(pd.DataFrame(), 100) is None


class TestFindDteDate:
    def test_dte1(self):
        assert find_dte_date("2025-10-01", 1, ["2025-10-01", "2025-10-02"]) == "2025-10-02"

    def test_weekend(self):
        assert find_dte_date("2025-10-03", 2, ["2025-10-03", "2025-10-06"]) == "2025-10-06"

    def test_none(self):
        assert find_dte_date("2025-10-01", 5, ["2025-10-01"]) is None


class TestComputeSpreadValue:
    def test_mid(self):
        df = pd.DataFrame([{"strike": 100, "bid": 8, "ask": 12}, {"strike": 120, "bid": 3, "ask": 5}])
        assert _compute_spread_value(df, 100, 120, True) == pytest.approx(6.0)

    def test_bidask(self):
        df = pd.DataFrame([{"strike": 100, "bid": 8, "ask": 12}, {"strike": 120, "bid": 3, "ask": 5}])
        assert _compute_spread_value(df, 100, 120, False) == pytest.approx(3.0)

    def test_missing(self):
        assert _compute_spread_value(pd.DataFrame([{"strike": 100, "bid": 8, "ask": 12}]), 100, 120, True) is None


class TestLoadFunctions:
    def test_equity_valid(self, temp_data_dirs):
        assert not load_equity_day("RUT", "2025-10-01", temp_data_dirs["equities_dir"]).empty

    def test_equity_missing(self, temp_data_dirs):
        assert load_equity_day("RUT", "2025-12-01", temp_data_dirs["equities_dir"]).empty

    def test_options_valid(self, temp_data_dirs):
        assert not load_options_day("RUT", "2025-10-01", temp_data_dirs["options_dir"]).empty

    def test_options_missing(self, temp_data_dirs):
        assert load_options_day("RUT", "2025-12-01", temp_data_dirs["options_dir"]).empty

    def test_get_dates(self, temp_data_dirs):
        assert "2025-10-01" in get_trading_dates("2025-10-01", "2025-10-31", "RUT", temp_data_dirs["options_dir"])

    def test_get_all_dates(self, temp_crossday_dirs):
        dates = get_all_available_dates("RUT", temp_crossday_dirs["options_dir"])
        assert len(dates) == 3


class TestAnalyzeSingleDay:
    def test_multiexp_produces_results(self, temp_data_dirs):
        all_dates = get_all_available_dates("RUT", temp_data_dirs["options_dir"])
        results = analyze_single_day(
            "2025-10-01", "RUT", 20, ["09:00"], [1, 2], [100, 50], [100, 0],
            temp_data_dirs["options_dir"], temp_data_dirs["equities_dir"], all_dates)
        assert len(results) > 0
        for r in results:
            assert "entry_breach_pct" in r
            assert "target_breach_pct" in r
            assert "net_roll_cost" in r

    def test_crossday_produces_results(self, temp_crossday_dirs):
        all_dates = get_all_available_dates("RUT", temp_crossday_dirs["options_dir"])
        results = analyze_single_day(
            "2025-10-01", "RUT", 20, ["09:00"], [1], [100], [100],
            temp_crossday_dirs["options_dir"], temp_crossday_dirs["equities_dir"], all_dates)
        assert len(results) > 0

    def test_entry_breach_variation(self, temp_data_dirs):
        all_dates = get_all_available_dates("RUT", temp_data_dirs["options_dir"])
        results = analyze_single_day(
            "2025-10-01", "RUT", 20, ["09:00"], [1], [100, 75, 50, 25], [100],
            temp_data_dirs["options_dir"], temp_data_dirs["equities_dir"], all_dates)
        entry_pcts = {r["entry_breach_pct"] for r in results}
        assert len(entry_pcts) >= 2  # at least two distinct entry levels

    def test_missing_data_empty(self, temp_data_dirs):
        all_dates = get_all_available_dates("RUT", temp_data_dirs["options_dir"])
        assert len(analyze_single_day(
            "2025-12-01", "RUT", 20, ["09:00"], [1], [100], [100],
            temp_data_dirs["options_dir"], temp_data_dirs["equities_dir"], all_dates)) == 0


class TestCLI:
    def test_help(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/roll_cost_table.py", "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]))
        assert result.returncode == 0
        assert "--entry-breach-pcts" in result.stdout
        assert "--target-breach-pcts" in result.stdout
        assert "--options-dir" in result.stdout
        assert "PST" in result.stdout
