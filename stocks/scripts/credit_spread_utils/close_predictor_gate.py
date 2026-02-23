"""
Close Predictor Risk Gate for credit spread analysis.

Uses the unified close predictor to filter or annotate spreads based on
whether the predicted close band suggests the short strike is safe.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

from scripts.close_predictor.models import ET_TZ, _intraday_vol_cache
from scripts.close_predictor.prediction import train_both_models, make_unified_prediction
from scripts.close_predictor.live import _build_day_context, _find_nearest_time_label
from scripts.close_predictor.features import get_intraday_vol_factor
from scripts.csv_prediction_backtest import (
    load_csv_data,
    get_available_dates,
    get_day_high_low,
    DayContext,
)
from scripts.percentile_range_backtest import collect_all_data


@dataclass
class ClosePredictorGateConfig:
    """Configuration for the close predictor risk gate."""
    enabled: bool = False
    band_level: str = "P95"          # P95, P98, P99
    buffer_points: float = 0.0       # absolute points buffer
    buffer_pct: float = 0.0          # percentage buffer (0.005 = 0.5%)
    mode: str = "gate"               # "gate" or "annotate"
    lookback: int = 250


def parse_close_predictor_buffer(value: str) -> Tuple[float, float]:
    """Parse buffer value into (buffer_points, buffer_pct).

    Args:
        value: Buffer specification. If ends with '%', parsed as percentage.
               Otherwise parsed as absolute points.

    Returns:
        Tuple of (buffer_points, buffer_pct)
    """
    value = value.strip()
    if value.endswith('%'):
        pct = float(value[:-1]) / 100.0
        return (0.0, pct)
    else:
        return (float(value), 0.0)


class ClosePredictorGate:
    """Manages close predictor model lifecycle and per-result decisions.

    Lazily trains models on first use and evaluates whether each spread's
    short strike is safe relative to the predicted close band.
    """

    def __init__(self, config: ClosePredictorGateConfig, ticker: str, logger: logging.Logger):
        self.config = config
        self.ticker = ticker
        self.logger = logger
        self._models_ready = False
        self._pct_df = None
        self._stat_predictor = None
        self._all_dates = []
        self._pct_train_dates = set()
        self._train_dates_sorted = []
        self._day_context_cache = {}  # date_str -> DayContext
        self._day_data_cache = {}     # date_str -> DataFrame

    def ensure_models_trained(self) -> bool:
        """Lazy-train models on first call. Returns True if ready."""
        if self._models_ready:
            return True

        self.logger.info(f"Close predictor gate: training models for {self.ticker} (lookback={self.config.lookback})...")

        pct_df, stat_predictor, all_dates = train_both_models(
            self.ticker, lookback=self.config.lookback
        )

        if pct_df is None or pct_df.empty:
            self.logger.warning("Close predictor gate: no percentile data available, gate disabled.")
            return False

        self._pct_df = pct_df
        self._stat_predictor = stat_predictor
        self._all_dates = all_dates

        # Build train dates set from percentile data
        unique_dates = sorted(pct_df['date'].unique())
        test_idx = len(unique_dates) - 1
        start_idx = max(0, test_idx - self.config.lookback)
        self._pct_train_dates = set(unique_dates[start_idx:test_idx])
        self._train_dates_sorted = unique_dates[start_idx:test_idx]

        # Clear intraday vol cache
        _intraday_vol_cache.clear()

        self._models_ready = True
        self.logger.info(
            f"Close predictor gate: models trained. "
            f"{len(self._pct_train_dates)} training dates, "
            f"stat predictor {'ready' if stat_predictor else 'unavailable'}."
        )
        return True

    def _get_day_context(self, date_str: str) -> Optional[DayContext]:
        """Get or build DayContext for a given date."""
        if date_str in self._day_context_cache:
            return self._day_context_cache[date_str]

        test_df = self._get_day_data(date_str)
        if test_df is None or test_df.empty:
            return None

        day_ctx = _build_day_context(self.ticker, date_str, test_df)
        self._day_context_cache[date_str] = day_ctx
        return day_ctx

    def _get_day_data(self, date_str: str):
        """Get or load CSV data for a given date."""
        if date_str in self._day_data_cache:
            return self._day_data_cache[date_str]

        df = load_csv_data(self.ticker, date_str)
        self._day_data_cache[date_str] = df
        return df

    def evaluate_spread(self, result: dict) -> Tuple[bool, str]:
        """Evaluate whether a spread is safe based on close predictor.

        Args:
            result: A spread result dict from analyze_interval().

        Returns:
            Tuple of (is_safe, annotation_string)
        """
        if not self._models_ready:
            if not self.ensure_models_trained():
                return (True, "CLOSE PREDICTOR: models unavailable, passing through")

        # Extract info from result
        timestamp = result['timestamp']
        option_type = result.get('option_type', '').lower()
        short_strike = result['best_spread']['short_strike']
        current_price = result.get('current_close') or result['prev_close']

        # Determine the date and time in ET
        if hasattr(timestamp, 'astimezone'):
            ts_et = timestamp.astimezone(ET_TZ)
        else:
            import pandas as pd
            ts_et = pd.Timestamp(timestamp, tz='UTC').tz_convert(ET_TZ)

        date_str = ts_et.strftime('%Y-%m-%d')
        hour_et = ts_et.hour
        minute_et = ts_et.minute

        # Map to nearest time label
        time_label = _find_nearest_time_label(hour_et, minute_et)

        # Get day context
        day_ctx = self._get_day_context(date_str)
        if day_ctx is None:
            return (True, f"CLOSE PREDICTOR [{self.config.band_level}]: no day context for {date_str}, passing through")

        # Get day high/low
        test_df = self._get_day_data(date_str)
        if test_df is not None and not test_df.empty:
            day_high, day_low = get_day_high_low(test_df)
        else:
            day_high = current_price
            day_low = current_price

        prev_close = day_ctx.prev_close

        # Get intraday vol factor
        if test_df is not None and not test_df.empty:
            intraday_vol_factor = get_intraday_vol_factor(
                self.ticker, date_str, time_label, test_df, self._train_dates_sorted
            )
        else:
            intraday_vol_factor = 1.0

        # Make prediction
        prediction = make_unified_prediction(
            self._pct_df,
            self._stat_predictor,
            self.ticker,
            current_price,
            prev_close,
            ts_et,
            time_label,
            day_ctx,
            day_high,
            day_low,
            self._pct_train_dates,
            intraday_vol_factor=intraday_vol_factor,
        )

        if prediction is None:
            return (True, f"CLOSE PREDICTOR [{self.config.band_level}]: prediction unavailable, passing through")

        # Get the band at the configured level
        band = prediction.combined_bands.get(self.config.band_level)
        if band is None:
            # Fall back to percentile bands
            band = prediction.percentile_bands.get(self.config.band_level)
        if band is None:
            return (True, f"CLOSE PREDICTOR [{self.config.band_level}]: band not available, passing through")

        # Compute effective buffer (whichever is larger)
        effective_buffer = max(
            self.config.buffer_points,
            current_price * self.config.buffer_pct
        )

        # Decision logic
        if option_type == 'put':
            band_edge = band.lo_price
            actual_buffer = band_edge - short_strike
            is_safe = (band_edge - effective_buffer) > short_strike
            direction_label = "PUT"
        elif option_type == 'call':
            band_edge = band.hi_price
            actual_buffer = short_strike - band_edge
            is_safe = (band_edge + effective_buffer) < short_strike
            direction_label = "CALL"
        else:
            return (True, f"CLOSE PREDICTOR [{self.config.band_level}]: unknown option type '{option_type}', passing through")

        # Build annotation string
        status = "SAFE" if is_safe else "REJECT"
        check_mark = "\u2713" if is_safe else "\u2717"
        annotation = (
            f"CLOSE PREDICTOR [{self.config.band_level}]: "
            f"{direction_label} {short_strike:.0f} {status} — "
            f"band [{band.lo_price:,.0f} - {band.hi_price:,.0f}] "
            f"buffer={actual_buffer:.0f}pts (need {effective_buffer:.0f}) {check_mark}"
        )

        return (is_safe, annotation)

    def filter_results(self, results: list) -> list:
        """Filter or annotate results based on close predictor evaluation.

        Args:
            results: List of spread result dicts.

        Returns:
            Filtered (or annotated) list of results.
        """
        if not results:
            return results

        if not self.ensure_models_trained():
            self.logger.warning("Close predictor gate: models not ready, returning all results.")
            return results

        filtered = []
        for result in results:
            is_safe, annotation = self.evaluate_spread(result)

            if self.config.mode == 'annotate':
                result['close_predictor_annotation'] = annotation
                print(f"  {annotation}")
                filtered.append(result)
            elif self.config.mode == 'gate':
                if is_safe:
                    result['close_predictor_annotation'] = annotation
                    filtered.append(result)
                else:
                    print(f"  {annotation}")
                    self.logger.info(f"Close predictor gate filtered: {annotation}")
            else:
                filtered.append(result)

        if self.config.mode == 'gate' and len(filtered) < len(results):
            self.logger.info(
                f"Close predictor gate: {len(results)} -> {len(filtered)} results "
                f"({len(results) - len(filtered)} filtered)"
            )

        return filtered
