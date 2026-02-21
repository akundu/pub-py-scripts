#!/usr/bin/env python3
"""
Close Predictor Gate Backtest Grid

Runs a comprehensive backtest to evaluate the close predictor gate's impact
on credit spread P&L across multiple dimensions:
  - Tickers: NDX, SPX
  - Option types: put, call
  - Band levels: P95, P98, P99, P100
  - Time periods: 3 months, 1 month, 2 weeks

Simulates live 5-min-by-5-min evaluation: for each spread entry point,
the close predictor is evaluated at the exact timestamp of entry.

Usage:
    cd stocks/
    python scripts/close_predictor_gate_backtest.py [--buffer 0] [--risk-cap 500000]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Path setup
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from common.questdb_db import StockQuestDB
from common.logging_utils import get_logger

from credit_spread_utils.data_loader import find_csv_files_in_dir, load_data_cached
from credit_spread_utils.interval_analyzer import analyze_interval
from credit_spread_utils.spread_builder import parse_percent_beyond, parse_max_spread_width
from credit_spread_utils.close_predictor_gate import (
    ClosePredictorGate,
    ClosePredictorGateConfig,
)

# ============================================================================
# Configuration
# ============================================================================

TICKERS = ["NDX", "SPX"]
OPTION_TYPES = ["put", "call"]
BAND_LEVELS = ["P95", "P98", "P99", "P100"]
TODAY = date(2026, 2, 8)

PERIODS = {
    "3mo": TODAY - timedelta(days=90),
    "1mo": TODAY - timedelta(days=30),
    "2wk": TODAY - timedelta(days=14),
}

# Default credit spread parameters (matching typical daily usage)
DEFAULT_PARAMS = {
    "NDX": {
        "percent_beyond": "0.005:0.015",
        "max_spread_width": "20:30",
        "risk_cap": 500000.0,
        "min_spread_width": 5.0,
        "min_contract_price": 0.0,
        "max_credit_width_ratio": 0.60,
        "max_trading_hour": 15,
        "min_trading_hour": 6,
        "profit_target_pct": 0.80,
    },
    "SPX": {
        "percent_beyond": "0.005:0.015",
        "max_spread_width": "20:30",
        "risk_cap": 500000.0,
        "min_spread_width": 5.0,
        "min_contract_price": 0.0,
        "max_credit_width_ratio": 0.60,
        "max_trading_hour": 15,
        "min_trading_hour": 6,
        "profit_target_pct": 0.80,
    },
}


# ============================================================================
# Metrics collection
# ============================================================================

@dataclass
class GateMetrics:
    """Collected metrics for a single (ticker, opt_type, band, period) combo."""
    ticker: str = ""
    option_type: str = ""
    band_level: str = ""
    period: str = ""

    total_spreads: int = 0
    gate_passed: int = 0
    gate_rejected: int = 0
    gate_unavailable: int = 0  # prediction not available

    # P&L for ALL spreads (baseline)
    all_wins: int = 0
    all_losses: int = 0
    all_total_pnl: float = 0.0
    all_total_credit: float = 0.0
    all_max_drawdown: float = 0.0

    # P&L for GATE-PASSED spreads only
    passed_wins: int = 0
    passed_losses: int = 0
    passed_total_pnl: float = 0.0
    passed_total_credit: float = 0.0
    passed_max_drawdown: float = 0.0

    # P&L for GATE-REJECTED spreads (what we avoided)
    rejected_wins: int = 0
    rejected_losses: int = 0
    rejected_total_pnl: float = 0.0

    # Confusion matrix
    true_positive: int = 0   # Gate rejected, spread was a loss (correct filter)
    false_positive: int = 0  # Gate rejected, spread was a win (missed opportunity)
    true_negative: int = 0   # Gate passed, spread was a win (correct pass)
    false_negative: int = 0  # Gate passed, spread was a loss (missed filter)

    # Time-of-day breakdown (hour -> {passed_pnl, rejected_pnl, count})
    hourly_data: Dict = field(default_factory=dict)

    @property
    def all_win_rate(self) -> float:
        total = self.all_wins + self.all_losses
        return self.all_wins / total * 100 if total > 0 else 0.0

    @property
    def passed_win_rate(self) -> float:
        total = self.passed_wins + self.passed_losses
        return self.passed_wins / total * 100 if total > 0 else 0.0

    @property
    def rejection_rate(self) -> float:
        return self.gate_rejected / self.total_spreads * 100 if self.total_spreads > 0 else 0.0

    @property
    def all_roi(self) -> float:
        return self.all_total_pnl / self.all_total_credit * 100 if self.all_total_credit > 0 else 0.0

    @property
    def passed_roi(self) -> float:
        return self.passed_total_pnl / self.passed_total_credit * 100 if self.passed_total_credit > 0 else 0.0

    @property
    def all_avg_pnl(self) -> float:
        total = self.all_wins + self.all_losses
        return self.all_total_pnl / total if total > 0 else 0.0

    @property
    def passed_avg_pnl(self) -> float:
        total = self.passed_wins + self.passed_losses
        return self.passed_total_pnl / total if total > 0 else 0.0

    @property
    def precision(self) -> float:
        """Of the spreads the gate rejected, what % were actually losses?"""
        total = self.true_positive + self.false_positive
        return self.true_positive / total * 100 if total > 0 else 0.0

    @property
    def recall(self) -> float:
        """Of the actual losses, what % did the gate catch?"""
        total = self.true_positive + self.false_negative
        return self.true_positive / total * 100 if total > 0 else 0.0

    @property
    def pnl_saved(self) -> float:
        """Net P&L improvement from using the gate (positive = gate helped)."""
        return self.passed_total_pnl - self.all_total_pnl

    @property
    def all_profit_factor(self) -> float:
        """Gross profit / gross loss for all spreads."""
        gross_profit = sum(1 for _ in range(self.all_wins))  # placeholder
        return 0.0  # computed below in aggregate

    def compute_drawdown(self, pnl_series: List[float]) -> float:
        """Compute max drawdown from a P&L series."""
        if not pnl_series:
            return 0.0
        cumulative = np.cumsum(pnl_series)
        peak = np.maximum.accumulate(cumulative)
        drawdown = cumulative - peak
        return float(np.min(drawdown)) if len(drawdown) > 0 else 0.0


# ============================================================================
# Core backtest engine
# ============================================================================

async def run_backtest_for_period(
    ticker: str,
    start_date: str,
    end_date: str,
    logger: logging.Logger,
    risk_cap: float = 500000.0,
) -> List[Dict[str, Any]]:
    """Run credit spread backtest for a ticker/period, return all results with P&L."""

    params = DEFAULT_PARAMS[ticker]
    percent_beyond = parse_percent_beyond(params["percent_beyond"])
    max_spread_width = parse_max_spread_width(params["max_spread_width"])

    csv_dir = "options_csv_output"
    csv_paths = find_csv_files_in_dir(csv_dir, ticker, start_date, end_date, logger)

    if not csv_paths:
        logger.warning(f"No CSV files found for {ticker} from {start_date} to {end_date}")
        return []

    csv_paths = [str(p) for p in csv_paths]
    logger.info(f"Loading {len(csv_paths)} CSV files for {ticker} ({start_date} to {end_date})...")

    try:
        df = load_data_cached(csv_paths, cache_dir=".options_cache", no_cache=False, logger=logger)
    except Exception as e:
        logger.error(f"Failed to load data for {ticker}: {e}")
        return []

    # Connect to DB
    db_config = os.getenv('QUESTDB_CONNECTION_STRING', '') or os.getenv('QUESTDB_URL', '')
    db = StockQuestDB(db_config, enable_cache=True, logger=logger)

    try:
        from credit_spread_utils.timezone_utils import resolve_timezone
        output_tz = resolve_timezone("America/Los_Angeles")

        intervals_grouped = df.groupby('interval')
        logger.info(f"  {len(intervals_grouped)} intervals to analyze...")

        results = []
        for interval_time, interval_df in intervals_grouped:
            for opt_type in OPTION_TYPES:
                result = await analyze_interval(
                    db, interval_df, opt_type, percent_beyond,
                    risk_cap,
                    params["min_spread_width"],
                    max_spread_width,
                    False,  # use_mid_price
                    params["min_contract_price"],
                    ticker,
                    logger,
                    params["max_credit_width_ratio"],
                    None,  # max_strike_distance_pct
                    False,  # use_current_price
                    params["max_trading_hour"],
                    params["min_trading_hour"],
                    params["profit_target_pct"],
                    output_tz,
                    None,  # force_close_hour
                    None,  # min_premium_diff
                    None,  # dynamic_width_config
                    None,  # delta_filter_config
                )
                if result:
                    results.append(result)

        logger.info(f"  Got {len(results)} spread results for {ticker}")
        return results

    finally:
        await db.close()


def evaluate_gate_on_results(
    gate: ClosePredictorGate,
    results: List[Dict],
    band_level: str,
    logger: logging.Logger,
) -> List[Tuple[Dict, bool, str]]:
    """Evaluate the gate on each result, return list of (result, is_safe, annotation)."""
    # Temporarily override gate band level
    original_level = gate.config.band_level
    gate.config.band_level = band_level

    evaluated = []
    for result in results:
        is_safe, annotation = gate.evaluate_spread(result)
        evaluated.append((result, is_safe, annotation))

    gate.config.band_level = original_level
    return evaluated


def compute_metrics(
    ticker: str,
    option_type: str,
    band_level: str,
    period: str,
    evaluated: List[Tuple[Dict, bool, str]],
) -> GateMetrics:
    """Compute comprehensive metrics from evaluated results."""
    m = GateMetrics(
        ticker=ticker,
        option_type=option_type,
        band_level=band_level,
        period=period,
    )

    # Filter to just this option type
    type_results = [(r, safe, ann) for r, safe, ann in evaluated
                    if r.get('option_type', '').lower() == option_type]

    all_pnl_series = []
    passed_pnl_series = []

    for result, is_safe, annotation in type_results:
        pnl = result.get('actual_pnl_per_share')
        if pnl is None:
            continue

        num_contracts = result['best_spread'].get('num_contracts', 1) or 1
        total_pnl = pnl * num_contracts * 100
        total_credit = (result['best_spread'].get('total_credit') or
                       result['best_spread'].get('net_credit_per_contract', 0) * num_contracts)
        is_win = total_pnl > 0

        # Get hour for time-of-day analysis
        ts = result['timestamp']
        if hasattr(ts, 'hour'):
            hour = ts.hour
        else:
            hour = pd.Timestamp(ts).hour

        m.total_spreads += 1
        all_pnl_series.append(total_pnl)

        # All spreads baseline
        m.all_total_pnl += total_pnl
        m.all_total_credit += total_credit
        if is_win:
            m.all_wins += 1
        else:
            m.all_losses += 1

        # Gate classification
        if "passing through" in annotation or "unavailable" in annotation or "no day context" in annotation:
            m.gate_unavailable += 1
            # Treat as passed when prediction unavailable
            is_safe = True

        if is_safe:
            m.gate_passed += 1
            m.passed_total_pnl += total_pnl
            m.passed_total_credit += total_credit
            passed_pnl_series.append(total_pnl)
            if is_win:
                m.passed_wins += 1
                m.true_negative += 1  # correctly passed a winner
            else:
                m.passed_losses += 1
                m.false_negative += 1  # failed to filter a loser
        else:
            m.gate_rejected += 1
            m.rejected_total_pnl += total_pnl
            if is_win:
                m.rejected_wins += 1
                m.false_positive += 1  # incorrectly rejected a winner
            else:
                m.rejected_losses += 1
                m.true_positive += 1  # correctly rejected a loser

        # Hourly tracking
        if hour not in m.hourly_data:
            m.hourly_data[hour] = {"passed_pnl": 0, "rejected_pnl": 0, "count": 0, "rejected_count": 0}
        m.hourly_data[hour]["count"] += 1
        if is_safe:
            m.hourly_data[hour]["passed_pnl"] += total_pnl
        else:
            m.hourly_data[hour]["rejected_pnl"] += total_pnl
            m.hourly_data[hour]["rejected_count"] += 1

    # Compute drawdowns
    m.all_max_drawdown = m.compute_drawdown(all_pnl_series)
    m.passed_max_drawdown = m.compute_drawdown(passed_pnl_series)

    return m


# ============================================================================
# Display / formatting
# ============================================================================

def print_main_grid(all_metrics: List[GateMetrics]):
    """Print the main performance grid."""

    print("\n" + "=" * 140)
    print("CLOSE PREDICTOR GATE — PERFORMANCE GRID")
    print("=" * 140)

    for ticker in TICKERS:
        print(f"\n{'━' * 140}")
        print(f"  {ticker}")
        print(f"{'━' * 140}")

        for opt_type in OPTION_TYPES:
            print(f"\n  {opt_type.upper()} SPREADS")
            print(f"  {'─' * 136}")

            # Header
            print(f"  {'Period':<8} {'Band':<6} "
                  f"{'Trades':>6} {'Reject%':>8} "
                  f"{'All P&L':>12} {'All Win%':>9} {'All ROI':>8} "
                  f"{'Gate P&L':>12} {'Gate Win%':>10} {'Gate ROI':>9} "
                  f"{'P&L Saved':>10} {'Precision':>10} {'Recall':>8} "
                  f"{'Drawdown':>10}")
            print(f"  {'─' * 136}")

            for period_name in ["3mo", "1mo", "2wk"]:
                for band in BAND_LEVELS:
                    m = next((x for x in all_metrics
                             if x.ticker == ticker and x.option_type == opt_type
                             and x.band_level == band and x.period == period_name), None)
                    if m is None or m.total_spreads == 0:
                        print(f"  {period_name:<8} {band:<6} {'—no data—':>6}")
                        continue

                    # P&L saved indicator
                    pnl_saved = m.passed_total_pnl - m.all_total_pnl
                    saved_indicator = f"${pnl_saved:>+,.0f}"

                    print(
                        f"  {period_name:<8} {band:<6} "
                        f"{m.total_spreads:>6} {m.rejection_rate:>7.1f}% "
                        f"${m.all_total_pnl:>+11,.0f} {m.all_win_rate:>8.1f}% {m.all_roi:>7.1f}% "
                        f"${m.passed_total_pnl:>+11,.0f} {m.passed_win_rate:>9.1f}% {m.passed_roi:>8.1f}% "
                        f"{saved_indicator:>10} {m.precision:>9.1f}% {m.recall:>7.1f}% "
                        f"${m.passed_max_drawdown:>9,.0f}"
                    )

                if period_name != "2wk":
                    print(f"  {'─' * 136}")


def print_confusion_matrix_summary(all_metrics: List[GateMetrics]):
    """Print confusion matrix breakdown."""
    print("\n" + "=" * 100)
    print("CONFUSION MATRIX SUMMARY (Gate Filtering Accuracy)")
    print("=" * 100)
    print(f"  {'Ticker':<6} {'Type':<6} {'Period':<8} {'Band':<6} "
          f"{'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5} "
          f"{'Precision':>10} {'Recall':>8} {'F1':>7}")
    print(f"  {'─' * 90}")

    for m in all_metrics:
        if m.total_spreads == 0:
            continue
        tp, fp, tn, fn = m.true_positive, m.false_positive, m.true_negative, m.false_negative
        prec = m.precision
        rec = m.recall
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        print(f"  {m.ticker:<6} {m.option_type:<6} {m.period:<8} {m.band_level:<6} "
              f"{tp:>5} {fp:>5} {tn:>5} {fn:>5} "
              f"{prec:>9.1f}% {rec:>7.1f}% {f1:>6.1f}%")


def print_hourly_analysis(all_metrics: List[GateMetrics]):
    """Print time-of-day analysis for the most interesting combos."""
    print("\n" + "=" * 100)
    print("TIME-OF-DAY ANALYSIS (Gate rejection rate & P&L by hour, P95 band)")
    print("=" * 100)

    for ticker in TICKERS:
        for opt_type in OPTION_TYPES:
            # Use 3mo P95 for the hourly analysis
            m = next((x for x in all_metrics
                     if x.ticker == ticker and x.option_type == opt_type
                     and x.band_level == "P95" and x.period == "3mo"), None)
            if m is None or not m.hourly_data:
                continue

            print(f"\n  {ticker} {opt_type.upper()} (3mo, P95)")
            print(f"  {'Hour':>6} {'Trades':>7} {'Rejected':>9} {'Rej%':>6} "
                  f"{'Passed P&L':>12} {'Rejected P&L':>13} {'Avg Pass P&L':>13}")
            print(f"  {'─' * 80}")

            for hour in sorted(m.hourly_data.keys()):
                h = m.hourly_data[hour]
                count = h['count']
                rej = h['rejected_count']
                rej_pct = rej / count * 100 if count > 0 else 0
                pass_count = count - rej
                avg_pass = h['passed_pnl'] / pass_count if pass_count > 0 else 0

                print(f"  {hour:>4}:00 {count:>7} {rej:>9} {rej_pct:>5.1f}% "
                      f"${h['passed_pnl']:>+11,.0f} ${h['rejected_pnl']:>+12,.0f} "
                      f"${avg_pass:>+12,.0f}")


def print_insights_and_suggestions(all_metrics: List[GateMetrics]):
    """Print analysis insights and alternative experiment ideas."""
    print("\n" + "=" * 140)
    print("INSIGHTS & ALTERNATIVE EXPERIMENT IDEAS")
    print("=" * 140)

    # Find best performing combos
    best_pnl_saved = max(all_metrics, key=lambda m: m.passed_total_pnl - m.all_total_pnl if m.total_spreads > 0 else float('-inf'))
    best_precision = max(all_metrics, key=lambda m: m.precision if m.gate_rejected > 0 else 0)
    best_recall = max(all_metrics, key=lambda m: m.recall if m.total_spreads > 0 else 0)
    highest_rejection = max(all_metrics, key=lambda m: m.rejection_rate if m.total_spreads > 0 else 0)

    print(f"""
  KEY FINDINGS:
  {'─' * 80}
  Best P&L improvement: {best_pnl_saved.ticker} {best_pnl_saved.option_type} {best_pnl_saved.band_level} {best_pnl_saved.period}
    P&L saved: ${best_pnl_saved.passed_total_pnl - best_pnl_saved.all_total_pnl:+,.0f}

  Best precision:       {best_precision.ticker} {best_precision.option_type} {best_precision.band_level} {best_precision.period}
    {best_precision.precision:.1f}% of rejections were actual losses

  Best recall:          {best_recall.ticker} {best_recall.option_type} {best_recall.band_level} {best_recall.period}
    Caught {best_recall.recall:.1f}% of all losses

  Highest rejection:    {highest_rejection.ticker} {highest_rejection.option_type} {highest_rejection.band_level} {highest_rejection.period}
    {highest_rejection.rejection_rate:.1f}% of spreads rejected

  ADDITIONAL METRICS WORTH MONITORING:
  {'─' * 80}
  1. Band Width vs Spread Width Ratio
     - When predicted band is narrow relative to spread width, higher confidence
     - Track: band_width / spread_width as a "conviction score"

  2. Prediction Stability (5-min delta)
     - How much does the prediction change between consecutive 5-min evaluations?
     - Stable predictions = higher confidence; volatile = wait for clarity

  3. VIX1D Regime Correlation
     - Does the gate perform better in high-vol vs low-vol regimes?
     - Track gate accuracy by VIX1D quartile

  4. Distance-to-Band-Edge Histogram
     - Distribution of how close short strikes are to the band edge
     - Sweet spot analysis: what buffer value maximizes risk-adjusted return?

  5. Intraday Prediction Convergence
     - As market close approaches, predictions should tighten
     - Track: band_width_at_entry vs actual_close_distance

  ALTERNATIVE EXPERIMENT FRAMINGS:
  {'─' * 80}
  1. POSITION SIZING SIGNAL (instead of binary gate)
     - Use (band_edge - short_strike) / band_width as a continuous confidence score
     - Scale contract count: high confidence = full size, low = half size
     - Avoids the binary accept/reject tradeoff

  2. DYNAMIC PERCENT-BEYOND
     - Use predicted band to SET the percent-beyond parameter dynamically
     - percent_beyond = (band_edge_pct from close) + buffer
     - Adapts automatically to volatility regime

  3. ENTRY TIMING OPTIMIZER
     - Don't filter spreads — instead, time the entry
     - If prediction says "unsafe now but likely safe later", wait
     - Track: at what time label does the band narrow enough?

  4. ASYMMETRIC BUFFER BY DIRECTION
     - Use tighter buffer for the side price is moving AWAY from
     - Use wider buffer for the side price is moving TOWARD
     - E.g., if price trending up, widen call buffer, tighten put buffer

  5. ENSEMBLE WITH DELTA
     - Combine close predictor confidence with option delta
     - Gate requires: (a) prediction says safe AND (b) delta < threshold
     - Two independent signals = higher accuracy than either alone

  6. TRAILING GATE RE-EVALUATION
     - After entry, re-evaluate every 5 min
     - If prediction flips to "unsafe", close position early
     - Combines entry gate with active risk management

  7. BACKTEST THE PREDICTOR ITSELF (separate from spread P&L)
     - For each day, compare predicted band vs actual close
     - Track band hit rates at each time slot
     - Identify: at what time of day is the predictor most/least accurate?
     - This isolates predictor quality from spread construction noise
""")


# ============================================================================
# Main
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Close Predictor Gate Backtest Grid")
    parser.add_argument("--buffer", type=str, default="0", help="Buffer value (points or pct, e.g. '50' or '0.3%%')")
    parser.add_argument("--risk-cap", type=float, default=500000.0, help="Risk cap per spread")
    parser.add_argument("--log-level", default="WARNING", help="Logging level")
    parser.add_argument("--lookback", type=int, default=250, help="Close predictor training lookback")
    parser.add_argument("--ticker", type=str, default=None, help="Run for single ticker only (NDX or SPX)")
    parser.add_argument("--period", type=str, default=None, help="Run for single period only (3mo, 1mo, 2wk)")
    args = parser.parse_args()

    logger = get_logger("gate_backtest", level=args.log_level)

    tickers = [args.ticker] if args.ticker else TICKERS
    periods = {args.period: PERIODS[args.period]} if args.period else PERIODS

    # Parse buffer
    from credit_spread_utils.close_predictor_gate import parse_close_predictor_buffer
    buf_pts, buf_pct = parse_close_predictor_buffer(args.buffer)

    print(f"Close Predictor Gate Backtest Grid")
    print(f"Buffer: {args.buffer} (points={buf_pts}, pct={buf_pct})")
    print(f"Risk cap: ${args.risk_cap:,.0f}")
    print(f"Lookback: {args.lookback} days")
    print(f"Tickers: {tickers}")
    print(f"Periods: {list(periods.keys())}")
    print()

    all_metrics = []

    # Cache: (ticker, period) -> results list
    backtest_cache: Dict[Tuple[str, str], List] = {}
    gate_cache: Dict[str, ClosePredictorGate] = {}

    for ticker in tickers:
        # Train gate once per ticker (lazy, reused across bands)
        print(f"\nTraining close predictor for {ticker}...")
        t0 = time.time()
        gate_config = ClosePredictorGateConfig(
            enabled=True,
            band_level="P95",  # Will be overridden per evaluation
            buffer_points=buf_pts,
            buffer_pct=buf_pct,
            mode="gate",
            lookback=args.lookback,
        )
        gate = ClosePredictorGate(gate_config, ticker, logger)
        if not gate.ensure_models_trained():
            print(f"  WARNING: Close predictor models failed to train for {ticker}, skipping.")
            continue
        gate_cache[ticker] = gate
        print(f"  Models trained in {time.time() - t0:.1f}s")

        for period_name, start_date in periods.items():
            cache_key = (ticker, period_name)
            end_date_str = "2026-02-07"  # Latest available data
            start_date_str = start_date.strftime("%Y-%m-%d")

            # Run backtest (cached per ticker+period)
            if cache_key not in backtest_cache:
                print(f"\n  Running backtest: {ticker} {period_name} ({start_date_str} to {end_date_str})...")
                t0 = time.time()
                results = await run_backtest_for_period(
                    ticker, start_date_str, end_date_str, logger, args.risk_cap
                )
                backtest_cache[cache_key] = results
                print(f"    {len(results)} results in {time.time() - t0:.1f}s")
            else:
                results = backtest_cache[cache_key]

            # Evaluate gate at each band level
            for band in BAND_LEVELS:
                print(f"    Evaluating {ticker} {period_name} {band}...", end=" ", flush=True)
                t0 = time.time()
                evaluated = evaluate_gate_on_results(gate, results, band, logger)

                # Compute metrics per option type
                for opt_type in OPTION_TYPES:
                    m = compute_metrics(ticker, opt_type, band, period_name, evaluated)
                    all_metrics.append(m)

                print(f"{time.time() - t0:.1f}s")

    # Print results
    print_main_grid(all_metrics)
    print_confusion_matrix_summary(all_metrics)
    print_hourly_analysis(all_metrics)
    print_insights_and_suggestions(all_metrics)

    # Also dump raw metrics to JSON for further analysis
    output_path = f"close_predictor_gate_backtest_results_buffer{args.buffer}.json"
    json_data = []
    for m in all_metrics:
        json_data.append({
            "ticker": m.ticker, "option_type": m.option_type,
            "band_level": m.band_level, "period": m.period,
            "total_spreads": m.total_spreads,
            "gate_passed": m.gate_passed, "gate_rejected": m.gate_rejected,
            "rejection_rate": round(m.rejection_rate, 2),
            "all_total_pnl": round(m.all_total_pnl, 2),
            "all_win_rate": round(m.all_win_rate, 2),
            "all_roi": round(m.all_roi, 2),
            "passed_total_pnl": round(m.passed_total_pnl, 2),
            "passed_win_rate": round(m.passed_win_rate, 2),
            "passed_roi": round(m.passed_roi, 2),
            "pnl_saved": round(m.passed_total_pnl - m.all_total_pnl, 2),
            "precision": round(m.precision, 2),
            "recall": round(m.recall, 2),
            "all_max_drawdown": round(m.all_max_drawdown, 2),
            "passed_max_drawdown": round(m.passed_max_drawdown, 2),
            "true_positive": m.true_positive, "false_positive": m.false_positive,
            "true_negative": m.true_negative, "false_negative": m.false_negative,
        })

    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\nRaw metrics saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
