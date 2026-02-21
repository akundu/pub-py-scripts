"""
Output formatting for live display and backtest results.
"""

import os
from typing import Dict, List

from .models import UnifiedBand, UnifiedPrediction, UNIFIED_BAND_NAMES

from scripts.percentile_range_backtest import HOURS_TO_CLOSE


def _fmt_price(p: float) -> str:
    return f"{p:,.0f}"


def _hrs_label(hrs: float) -> str:
    return f"{hrs:.1f}h"


def print_live_display(pred: UnifiedPrediction, interval: int):
    """Print the formatted live display."""
    os.system('cls' if os.name == 'nt' else 'clear')

    ticker = pred.ticker.replace("I:", "") if pred.ticker.startswith("I:") else pred.ticker
    above_str = "Above" if pred.above_prev else "Below"
    vol_str = f"{pred.realized_vol:.2f}%" if pred.realized_vol is not None else "N/A"
    vix_str = f"{pred.vix1d:.1f}" if pred.vix1d is not None else "N/A"
    conf_str = pred.confidence if pred.confidence else "N/A"
    risk_str = f"{pred.risk_level}/10" if pred.risk_level is not None else "N/A"

    w = 82
    print("=" * w)
    print(f" {ticker} Live — {pred.time_label} ET ({_hrs_label(pred.hours_to_close)} to close)"
          f" | Current: {_fmt_price(pred.current_price)}"
          f" | {above_str} Prev Close ({_fmt_price(pred.prev_close)})")
    ivol_str = f"{pred.intraday_vol_factor:.2f}x" if pred.intraday_vol_factor != 1.0 else "1.00x"
    rev_str = f"{pred.reversal_blend * 100:.0f}%" if pred.reversal_blend > 0 else "0%"
    print(f" 5-day Vol: {vol_str}  |  VIX1D: {vix_str}"
          f"  |  Intraday Vol: {ivol_str}  |  Reversal: {rev_str}")
    print(f" Confidence: {conf_str}  |  Risk: {risk_str}")
    print("=" * w)

    print()
    print(" Predicted Close Range:")
    print(f"   {'Band':<6}  {'Low':>10}    {'High':>10}    {'±Pts':>8}    {'±%':>8}")
    print(f"   {'----':<6}  {'--------':>10}    {'--------':>10}    {'--------':>8}    {'--------':>8}")

    for name in UNIFIED_BAND_NAMES:
        band = pred.combined_bands.get(name)
        if band is None:
            continue
        half_width_pts = band.width_pts / 2.0
        half_width_pct = band.width_pct / 2.0
        print(f"   {name:<6}  {_fmt_price(band.lo_price):>10}  -  {_fmt_price(band.hi_price):>10}"
              f"    ±{half_width_pts:>6,.0f}    ±{half_width_pct:>5.2f}%")

    print()
    print(f" Source: {pred.data_source} | Refreshing every {interval}s | Ctrl+C to exit")
    print("=" * w)


def print_backtest_results(
    results: List[dict],
    ticker: str,
    display_labels: List[str],
    test_days: int,
):
    """Print three-column backtest comparison (Percentile / Statistical / Combined)."""
    display_ticker = ticker.replace("I:", "") if ticker.startswith("I:") else ticker

    for condition, cond_label in [(True, "ABOVE"), (False, "BELOW")]:
        cond_results = [r for r in results if r['above'] == condition]
        if not cond_results:
            continue

        # --- Accuracy table ---
        col_w = 8
        model_labels = ["Pctl", "Stat", "Comb"]
        band_names_display = UNIFIED_BAND_NAMES

        # Header
        hdr = f"{'HrsLeft':<8} {'Time':<6} {'N':>3}"
        for bn in band_names_display:
            for ml in model_labels:
                hdr += f" {ml:>{col_w}}"
            hdr += "  "  # spacer between bands
        sep = "-" * len(hdr)

        print(f"\n{'=' * len(hdr)}")
        print(f" {display_ticker} {cond_label} Prev Close — Unified Backtest Accuracy")
        print(f" (does actual close fall within predicted range?)")
        print(f"{'=' * len(hdr)}")

        # Band header row
        band_hdr = f"{'':>18}"
        for bn in band_names_display:
            span = col_w * len(model_labels) + len(model_labels) - 1
            band_hdr += f" {bn:^{span}}  "
        print(band_hdr)
        print(hdr)
        print(sep)

        for tl in display_labels:
            slot = [r for r in cond_results if r['time'] == tl]
            if not slot:
                continue
            n = len(slot)
            hrs = HOURS_TO_CLOSE.get(tl, 0)
            row = f"{_hrs_label(hrs):<8} {tl:<6} {n:>3}"
            for bn in band_names_display:
                for model_key in ["pct", "stat", "comb"]:
                    hits = sum(1 for r in slot if r.get(f'{model_key}_{bn}_hit', False))
                    acc = hits / n * 100 if n > 0 else 0
                    row += f" {acc:>{col_w - 1}.0f}%"
                row += "  "
            print(row)

        print(sep)

    # Overall summary
    total = len(results)
    if total == 0:
        return
    above_n = sum(1 for r in results if r['above'])
    below_n = total - above_n
    test_day_count = len(set(r['date'] for r in results))

    print(f"\nTotal predictions: {total} across {test_day_count} test days")
    print(f"Above prev close slots: {above_n}, Below: {below_n}")

    print(f"\nOverall accuracy (all slots):")
    row = "  "
    for bn in UNIFIED_BAND_NAMES:
        row += f"  {bn}: "
        for model_key, label in [("pct", "Pctl"), ("stat", "Stat"), ("comb", "Comb")]:
            hits = sum(1 for r in results if r.get(f'{model_key}_{bn}_hit', False))
            acc = hits / total * 100
            row += f"{label}={acc:.0f}% "
    print(row)
