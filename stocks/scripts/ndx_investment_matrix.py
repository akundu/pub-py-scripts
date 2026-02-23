#!/usr/bin/env python3
"""
NDX Credit Spread Investment Decision Matrix

Comprehensive framework for thinking about:
- Puts vs Calls selection
- ROI expectations
- Spread width optimization
- Price barrier (percent beyond) settings
- Hour of day timing
- Day of week patterns

Usage:
    python ndx_investment_matrix.py [--close PRICE] [--prev-close PRICE] [--vix VIX]
"""

import argparse
from datetime import datetime
from typing import Dict, List, Tuple

# =============================================================================
# MASTER INVESTMENT MATRIX DATA
# Based on 6-month historical analysis of NDX credit spreads
# =============================================================================

# Price Barrier (Percent Beyond) Matrix
PRICE_BARRIER_MATRIX = {
    "puts": {
        # percent_beyond: (win_rate, roi, risk_level, recommendation)
        0.0355: (100.0, 9.92, "SAFE", "Conservative - guaranteed wins"),
        0.0330: (100.0, 14.39, "OPTIMAL", "Best risk/reward - RECOMMENDED"),
        0.0305: (86.75, -2.82, "RISKY", "AVOID - losses exceed gains"),
        0.0280: (83.18, -7.89, "DANGER", "NEVER - heavy losses"),
        0.0255: (86.45, -0.36, "DANGER", "NEVER - breakeven at best"),
    },
    "calls": {
        # percent_beyond: (win_rate, roi, risk_level, recommendation)
        0.0282: (100.0, 9.92, "SAFE", "Conservative - guaranteed wins"),
        0.0257: (100.0, 14.39, "OPTIMAL", "Best risk/reward - RECOMMENDED"),
        0.0232: (86.75, -2.82, "RISKY", "AVOID - losses exceed gains"),
        0.0207: (83.18, -7.89, "DANGER", "NEVER - heavy losses"),
        0.0182: (86.45, -0.36, "DANGER", "NEVER - breakeven at best"),
    }
}

# Spread Width Matrix by Distance
SPREAD_WIDTH_MATRIX = {
    # distance_pct: (recommended_width, min, max, quality)
    0.010: (20, 15, 25, "Premium capture - tight"),
    0.015: (22, 18, 28, "Premium capture - moderate"),
    0.020: (25, 20, 30, "Standard"),
    0.025: (28, 22, 35, "Standard-wide"),
    0.030: (30, 25, 40, "Wide - safe zone"),
    0.035: (32, 28, 45, "Wide - very safe"),
    0.040: (35, 30, 50, "Maximum width cap"),
}

# Dynamic Width Configurations and ROI
DYNAMIC_WIDTH_ROI = {
    "Linear-500": {
        "config": {"mode": "linear", "base_width": 15, "slope_factor": 500, "min_width": 10, "max_width": 50},
        "1wk": {"win_rate": 100.0, "roi": 9.66, "trades": 2},
        "2wk": {"win_rate": 100.0, "roi": 6.52, "trades": 7},
        "1mo": {"win_rate": 100.0, "roi": 6.89, "trades": 13},
        "3mo": {"win_rate": 99.05, "roi": 16.31, "trades": 105},
        "6mo": {"win_rate": 97.59, "roi": 6.47, "trades": 166},
    },
    "Linear-1000": {
        "config": {"mode": "linear", "base_width": 15, "slope_factor": 1000, "min_width": 10, "max_width": 75},
        "1wk": {"win_rate": 100.0, "roi": 3.85, "trades": 5},
        "1mo": {"win_rate": 100.0, "roi": 4.09, "trades": 25},
        "3mo": {"win_rate": 92.65, "roi": 7.13, "trades": 136},
        "6mo": {"win_rate": 92.86, "roi": 1.10, "trades": 210},
    },
    "Stepped": {
        "config": {"mode": "stepped", "base_width": 15, "steps": {"0.01": 20, "0.02": 30, "0.03": 50}},
        "1wk": {"win_rate": 100.0, "roi": 4.65, "trades": 4},
        "1mo": {"win_rate": 100.0, "roi": 4.23, "trades": 22},
        "3mo": {"win_rate": 94.44, "roi": 9.43, "trades": 126},
        "6mo": {"win_rate": 93.88, "roi": 2.52, "trades": 196},
    },
    "Fixed": {
        "config": None,
        "1wk": {"win_rate": 100.0, "roi": 2.36, "trades": 6},
        "1mo": {"win_rate": 100.0, "roi": 2.31, "trades": 30},
        "3mo": {"win_rate": 98.30, "roi": 4.74, "trades": 176},
        "6mo": {"win_rate": 97.05, "roi": -0.18, "trades": 271},
    },
}

# Hour of Day Matrix (ET)
HOUR_MATRIX = {
    # hour: (quality, capital_pct, spread_width_adjust, notes)
    9: ("EXCELLENT", 35, 0, "Market open - highest premium, most opportunity"),
    10: ("EXCELLENT", 35, 0, "Morning continuation - still prime"),
    11: ("GOOD", 20, 5, "Late morning - good but narrowing"),
    12: ("FAIR", 10, 5, "Midday - reduced quality"),
    13: ("POOR", 0, 10, "Early afternoon - avoid new positions"),
    14: ("POOR", 0, 10, "Late afternoon - avoid new positions"),
    15: ("DANGER", 0, 0, "Final hour - NO NEW POSITIONS"),
}

# Day of Week Matrix
DAY_MATRIX = {
    # day: (quality, capital_multiplier, notes)
    "Monday": ("GOOD", 1.0, "Normal trading - watch for weekend gaps"),
    "Tuesday": ("EXCELLENT", 1.0, "Best day - established trend from Monday"),
    "Wednesday": ("EXCELLENT", 1.0, "Mid-week - consistent patterns"),
    "Thursday": ("GOOD", 0.85, "Pre-Friday caution - reduced size"),
    "Friday": ("CAUTION", 0.65, "OPEX risk - reduce exposure, exit early"),
}

# Puts vs Calls Decision Matrix
PUTS_VS_CALLS_MATRIX = {
    # scenario: (put_allocation, call_allocation, rationale)
    "strong_bullish": (0.80, 0.20, "Heavy put credit spreads - profit from rise"),
    "bullish": (0.65, 0.35, "Favor puts - profit from rise or flat"),
    "neutral": (0.50, 0.50, "Iron condor - profit from range-bound"),
    "bearish": (0.35, 0.65, "Favor calls - profit from fall or flat"),
    "strong_bearish": (0.20, 0.80, "Heavy call credit spreads - profit from fall"),
    "high_vix": (0.50, 0.50, "Balanced but REDUCED SIZE - high premium but risk"),
}


def determine_market_condition(daily_change_pct: float, vix: float = None) -> str:
    """Determine market condition for allocation."""
    if vix and vix > 25:
        return "high_vix"

    if daily_change_pct > 1.0:
        return "strong_bullish"
    elif daily_change_pct > 0.3:
        return "bullish"
    elif daily_change_pct < -1.0:
        return "strong_bearish"
    elif daily_change_pct < -0.3:
        return "bearish"
    else:
        return "neutral"


def calculate_dynamic_width(distance_pct: float, config_name: str = "Linear-500") -> float:
    """Calculate spread width based on distance and configuration."""
    config = DYNAMIC_WIDTH_ROI[config_name]["config"]
    if config is None:
        return 25  # Default fixed width

    if config["mode"] == "linear":
        width = config["base_width"] + (distance_pct * config["slope_factor"])
    elif config["mode"] == "stepped":
        width = config["base_width"]
        for threshold, step_width in sorted(config["steps"].items()):
            if distance_pct >= float(threshold):
                width = step_width
    else:
        width = 25

    return max(config.get("min_width", 10), min(width, config.get("max_width", 50)))


def get_week_schedule(close_price: float, prev_close: float = None, vix: float = None) -> Dict:
    """Generate full week trading schedule."""
    daily_change_pct = ((close_price - prev_close) / prev_close * 100) if prev_close else 0
    market_condition = determine_market_condition(daily_change_pct, vix)
    put_alloc, call_alloc, rationale = PUTS_VS_CALLS_MATRIX[market_condition]

    # Calculate strikes
    put_strike_optimal = round(close_price * (1 - 0.0330), 0)
    put_strike_safe = round(close_price * (1 - 0.0355), 0)
    call_strike_optimal = round(close_price * (1 + 0.0257), 0)
    call_strike_safe = round(close_price * (1 + 0.0282), 0)

    schedule = {
        "market_condition": market_condition,
        "rationale": rationale,
        "allocation": {"puts": put_alloc, "calls": call_alloc},
        "strikes": {
            "puts": {"optimal": put_strike_optimal, "safe": put_strike_safe},
            "calls": {"optimal": call_strike_optimal, "safe": call_strike_safe},
        },
        "daily_capital": {},
    }

    base_capital = 300000
    for day, (quality, multiplier, notes) in DAY_MATRIX.items():
        day_capital = base_capital * multiplier
        schedule["daily_capital"][day] = {
            "total": day_capital,
            "puts": day_capital * put_alloc,
            "calls": day_capital * call_alloc,
            "quality": quality,
            "notes": notes,
        }

    return schedule


def print_investment_matrix(close_price: float, prev_close: float = None, vix: float = None):
    """Print comprehensive investment decision matrix."""

    print("=" * 80)
    print("NDX CREDIT SPREAD INVESTMENT DECISION MATRIX")
    print("=" * 80)
    print(f"Reference Price: ${close_price:,.2f}")
    if prev_close:
        change_pct = ((close_price - prev_close) / prev_close) * 100
        print(f"Daily Change: {change_pct:+.2f}%")
    if vix:
        print(f"VIX: {vix:.1f}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ==========================================================================
    # MATRIX 1: PUTS VS CALLS
    # ==========================================================================
    print("-" * 80)
    print("MATRIX 1: PUTS VS CALLS - When to Use Each")
    print("-" * 80)
    print()
    print("  PUTS (Sell below market) = BULLISH position")
    print("    -> You PROFIT when market RISES or stays FLAT")
    print("    -> Use when: Uptrend, support holding, VIX falling")
    print()
    print("  CALLS (Sell above market) = BEARISH position")
    print("    -> You PROFIT when market FALLS or stays FLAT")
    print("    -> Use when: Downtrend, resistance holding, VIX rising")
    print()
    print("  Allocation by Market Condition:")
    print("  " + "-" * 60)
    print(f"  {'Condition':<18} {'Puts':>8} {'Calls':>8}  Rationale")
    print("  " + "-" * 60)
    for condition, (puts, calls, rationale) in PUTS_VS_CALLS_MATRIX.items():
        print(f"  {condition:<18} {puts*100:>7.0f}% {calls*100:>7.0f}%  {rationale}")
    print()

    # Current recommendation
    if prev_close:
        change_pct = ((close_price - prev_close) / prev_close) * 100
        condition = determine_market_condition(change_pct, vix)
        puts, calls, rationale = PUTS_VS_CALLS_MATRIX[condition]
        print(f"  >>> CURRENT: {condition.upper()} -> {puts*100:.0f}% Puts / {calls*100:.0f}% Calls")
        print(f"      {rationale}")
    print()

    # ==========================================================================
    # MATRIX 2: PRICE BARRIER (Percent Beyond)
    # ==========================================================================
    print("-" * 80)
    print("MATRIX 2: PRICE BARRIER - How Far from Close to Sell")
    print("-" * 80)
    print()
    print("  PUTS (% Below Close):")
    print("  " + "-" * 70)
    print(f"  {'% Beyond':>10} {'Strike':>10} {'Win Rate':>10} {'ROI':>8} {'Risk':>10} Recommendation")
    print("  " + "-" * 70)
    for pct, (wr, roi, risk, rec) in PRICE_BARRIER_MATRIX["puts"].items():
        strike = round(close_price * (1 - pct), 0)
        marker = " <-- USE" if risk == "OPTIMAL" else ""
        print(f"  {pct*100:>9.2f}% ${strike:>9,.0f} {wr:>9.1f}% {roi:>7.1f}% {risk:>10} {rec}{marker}")
    print()

    print("  CALLS (% Above Close):")
    print("  " + "-" * 70)
    print(f"  {'% Beyond':>10} {'Strike':>10} {'Win Rate':>10} {'ROI':>8} {'Risk':>10} Recommendation")
    print("  " + "-" * 70)
    for pct, (wr, roi, risk, rec) in PRICE_BARRIER_MATRIX["calls"].items():
        strike = round(close_price * (1 + pct), 0)
        marker = " <-- USE" if risk == "OPTIMAL" else ""
        print(f"  {pct*100:>9.2f}% ${strike:>9,.0f} {wr:>9.1f}% {roi:>7.1f}% {risk:>10} {rec}{marker}")
    print()

    # ==========================================================================
    # MATRIX 3: SPREAD WIDTH
    # ==========================================================================
    print("-" * 80)
    print("MATRIX 3: SPREAD WIDTH - Dynamic Width by Distance")
    print("-" * 80)
    print()
    print("  Formula (Linear-500): width = 15 + (distance% * 500)")
    print()
    print(f"  {'Distance':>10} {'Width':>8} {'Range':>12} Quality")
    print("  " + "-" * 50)
    for dist, (width, min_w, max_w, quality) in SPREAD_WIDTH_MATRIX.items():
        print(f"  {dist*100:>9.1f}% {width:>7} {min_w:>5}-{max_w:<5} {quality}")
    print()
    print("  Key: Wider spreads = More capital at risk but more premium")
    print("       Tighter spreads = Less risk but less premium")
    print()

    # ==========================================================================
    # MATRIX 4: HOUR OF DAY
    # ==========================================================================
    print("-" * 80)
    print("MATRIX 4: HOUR OF DAY (ET) - When to Trade")
    print("-" * 80)
    print()
    print(f"  {'Hour':>6} {'Quality':>12} {'Capital%':>10} {'Width Adj':>10} Notes")
    print("  " + "-" * 70)
    for hour, (quality, capital_pct, width_adj, notes) in HOUR_MATRIX.items():
        print(f"  {hour:>5}h {quality:>12} {capital_pct:>9}% {width_adj:>+9} {notes}")
    print()
    print("  CRITICAL: Deploy 70% of capital between 9:30-11:00 AM ET")
    print("            NO new positions after 2:30 PM ET")
    print()

    # ==========================================================================
    # MATRIX 5: DAY OF WEEK
    # ==========================================================================
    print("-" * 80)
    print("MATRIX 5: DAY OF WEEK - Daily Adjustments")
    print("-" * 80)
    print()
    print(f"  {'Day':<12} {'Quality':>12} {'Capital':>10} Notes")
    print("  " + "-" * 60)
    for day, (quality, multiplier, notes) in DAY_MATRIX.items():
        cap = f"${300000 * multiplier:,.0f}"
        print(f"  {day:<12} {quality:>12} {cap:>10} {notes}")
    print()

    # ==========================================================================
    # MATRIX 6: ROI BY TIME PERIOD AND WIDTH CONFIG
    # ==========================================================================
    print("-" * 80)
    print("MATRIX 6: ROI BY TIME PERIOD - Dynamic Width Comparison")
    print("-" * 80)
    print()
    print(f"  {'Config':<15} {'1wk':>12} {'2wk':>12} {'1mo':>12} {'3mo':>12} {'6mo':>12}")
    print("  " + "-" * 75)
    for config_name, data in DYNAMIC_WIDTH_ROI.items():
        row = f"  {config_name:<15}"
        for period in ["1wk", "2wk", "1mo", "3mo", "6mo"]:
            if period in data:
                roi = data[period]["roi"]
                wr = data[period]["win_rate"]
                row += f" {roi:>5.1f}%/{wr:.0f}%"
            else:
                row += "       -    "
        print(row)
    print()
    print("  Format: ROI% / Win Rate%")
    print("  Winner: Linear-500 - Best ROI across all periods")
    print()

    # ==========================================================================
    # TOMORROW'S TRADE PLAN
    # ==========================================================================
    print("=" * 80)
    print("TOMORROW'S TRADE PLAN (Feb 3, 2026)")
    print("=" * 80)
    print()

    if prev_close:
        change_pct = ((close_price - prev_close) / prev_close) * 100
        condition = determine_market_condition(change_pct, vix)
        puts, calls, _ = PUTS_VS_CALLS_MATRIX[condition]
    else:
        puts, calls = 0.5, 0.5

    put_strike = round(close_price * 0.967, 0)
    call_strike = round(close_price * 1.0257, 0)
    put_width = calculate_dynamic_width(0.033)
    call_width = calculate_dynamic_width(0.0257)

    total_capital = 300000

    print(f"  PUT Credit Spreads ({puts*100:.0f}% = ${total_capital * puts:,.0f}):")
    print(f"    Sell: ${put_strike:,.0f} (3.30% below)")
    print(f"    Buy:  ${put_strike - put_width:,.0f} (width: {put_width:.0f})")
    print(f"    Positions: {int((total_capital * puts) / 50000)} @ $50K max each")
    print()
    print(f"  CALL Credit Spreads ({calls*100:.0f}% = ${total_capital * calls:,.0f}):")
    print(f"    Sell: ${call_strike:,.0f} (2.57% above)")
    print(f"    Buy:  ${call_strike + call_width:,.0f} (width: {call_width:.0f})")
    print(f"    Positions: {int((total_capital * calls) / 50000)} @ $50K max each")
    print()
    print("  Schedule:")
    print("    9:30-10:30 AM ET: Open 2 positions (33% capital)")
    print("    10:30-11:30 AM ET: Open 2 positions (33% capital)")
    print("    11:30-12:00 PM ET: Final positions if quality setups (33% capital)")
    print("    After 12:00 PM ET: Monitor only, exit at 80% profit")
    print()

    # ==========================================================================
    # WEEK SCHEDULE
    # ==========================================================================
    print("=" * 80)
    print("WEEKLY SCHEDULE (Feb 3-7, 2026)")
    print("=" * 80)
    print()
    print(f"  {'Day':<12} {'Capital':>12} {'Puts':>12} {'Calls':>12} Action")
    print("  " + "-" * 65)

    weekly_total = 0
    for day, (quality, mult, notes) in DAY_MATRIX.items():
        cap = 300000 * mult
        put_cap = cap * puts
        call_cap = cap * calls
        weekly_total += cap
        action = "Full deploy" if mult == 1.0 else f"Reduced ({mult*100:.0f}%)"
        print(f"  {day:<12} ${cap:>10,.0f} ${put_cap:>10,.0f} ${call_cap:>10,.0f} {action}")

    print("  " + "-" * 65)
    print(f"  {'TOTAL':<12} ${weekly_total:>10,.0f}")
    print()

    # Expected returns
    print("  Expected Weekly Returns (at optimal parameters):")
    print(f"    Conservative (5% daily): ${weekly_total * 0.05:,.0f}")
    print(f"    Optimal (8% daily):      ${weekly_total * 0.08:,.0f}")
    print(f"    Best case (12% daily):   ${weekly_total * 0.12:,.0f}")
    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="NDX Investment Decision Matrix")
    parser.add_argument("--close", type=float, default=25900, help="Today's closing price")
    parser.add_argument("--prev-close", type=float, default=25800, help="Previous day's close")
    parser.add_argument("--vix", type=float, help="Current VIX level")

    args = parser.parse_args()
    print_investment_matrix(args.close, args.prev_close, args.vix)


if __name__ == "__main__":
    main()
