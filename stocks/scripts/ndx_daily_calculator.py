#!/usr/bin/env python3
"""
NDX Daily Strike Calculator and Trend Advisor

Calculates optimal strike levels for tomorrow based on today's close
and provides guidance on whether to trade with or against the trend.

Usage:
    python ndx_daily_calculator.py [--close PRICE]
    python ndx_daily_calculator.py --fetch  # Fetch current price from DB
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from common.questdb_db import StockQuestDB
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


# Optimal parameters from 6-month analysis
OPTIMAL_PARAMS = {
    "put": {
        "percent_beyond": 0.0330,  # 3.30% below close
        "safe_point": 0.0355,      # 3.55% below close (100% win rate)
        "aggressive": 0.0255,      # 2.55% below (AVOID - 86% win rate)
        "spread_width_min": 15,
        "spread_width_max": 20,
    },
    "call": {
        "percent_beyond": 0.0257,  # 2.57% above close
        "safe_point": 0.0282,      # 2.82% above close (100% win rate)
        "aggressive": 0.0182,      # 1.82% above (AVOID - 86% win rate)
        "spread_width_min": 20,
        "spread_width_max": 25,
    }
}

# Linear-500 dynamic width formula
DYNAMIC_WIDTH = {
    "base_width": 15,
    "slope_factor": 500,
    "min_width": 10,
    "max_width": 50,
}


def calculate_dynamic_width(distance_pct: float) -> float:
    """Calculate spread width based on distance from close."""
    width = DYNAMIC_WIDTH["base_width"] + (distance_pct * DYNAMIC_WIDTH["slope_factor"])
    return max(DYNAMIC_WIDTH["min_width"], min(width, DYNAMIC_WIDTH["max_width"]))


def calculate_strikes(close_price: float) -> dict:
    """Calculate all strike levels based on closing price."""
    results = {
        "close_price": close_price,
        "timestamp": datetime.now().isoformat(),
    }

    # PUT strikes (below close)
    results["puts"] = {
        "optimal": {
            "sell_strike": round(close_price * (1 - OPTIMAL_PARAMS["put"]["percent_beyond"]), 0),
            "pct_below": OPTIMAL_PARAMS["put"]["percent_beyond"] * 100,
            "spread_width": OPTIMAL_PARAMS["put"]["spread_width_max"],
            "dynamic_width": calculate_dynamic_width(OPTIMAL_PARAMS["put"]["percent_beyond"]),
        },
        "safe": {
            "sell_strike": round(close_price * (1 - OPTIMAL_PARAMS["put"]["safe_point"]), 0),
            "pct_below": OPTIMAL_PARAMS["put"]["safe_point"] * 100,
        },
        "danger_zone": {
            "never_sell_below": round(close_price * (1 - OPTIMAL_PARAMS["put"]["aggressive"]), 0),
            "pct_below": OPTIMAL_PARAMS["put"]["aggressive"] * 100,
        }
    }

    # CALL strikes (above close)
    results["calls"] = {
        "optimal": {
            "sell_strike": round(close_price * (1 + OPTIMAL_PARAMS["call"]["percent_beyond"]), 0),
            "pct_above": OPTIMAL_PARAMS["call"]["percent_beyond"] * 100,
            "spread_width": OPTIMAL_PARAMS["call"]["spread_width_max"],
            "dynamic_width": calculate_dynamic_width(OPTIMAL_PARAMS["call"]["percent_beyond"]),
        },
        "safe": {
            "sell_strike": round(close_price * (1 + OPTIMAL_PARAMS["call"]["safe_point"]), 0),
            "pct_above": OPTIMAL_PARAMS["call"]["safe_point"] * 100,
        },
        "danger_zone": {
            "never_sell_above": round(close_price * (1 + OPTIMAL_PARAMS["call"]["aggressive"]), 0),
            "pct_above": OPTIMAL_PARAMS["call"]["aggressive"] * 100,
        }
    }

    return results


def analyze_trend(
    current_price: float,
    prev_close: float,
    week_change_pct: float = None,
    vix: float = None
) -> dict:
    """
    Analyze market trend and recommend strategy.

    Returns recommendation on whether to trade with trend (favor one side)
    or neutral (both sides equally).
    """
    daily_change_pct = ((current_price - prev_close) / prev_close) * 100

    analysis = {
        "daily_change_pct": daily_change_pct,
        "week_change_pct": week_change_pct,
        "vix": vix,
    }

    # Determine trend direction
    if abs(daily_change_pct) < 0.3:
        trend = "neutral"
    elif daily_change_pct > 0:
        trend = "bullish"
    else:
        trend = "bearish"

    analysis["trend"] = trend

    # Generate recommendation
    if trend == "bullish":
        analysis["recommendation"] = "FAVOR PUTS"
        analysis["rationale"] = [
            "Market is trending UP",
            "PUT credit spreads benefit when market rises or stays flat",
            "Selling puts = BULLISH position (you want market above your strike)",
            "Consider 60-70% allocation to puts, 30-40% to calls",
        ]
        analysis["allocation"] = {"puts": 0.65, "calls": 0.35}
    elif trend == "bearish":
        analysis["recommendation"] = "FAVOR CALLS"
        analysis["rationale"] = [
            "Market is trending DOWN",
            "CALL credit spreads benefit when market falls or stays flat",
            "Selling calls = BEARISH position (you want market below your strike)",
            "Consider 60-70% allocation to calls, 30-40% to puts",
        ]
        analysis["allocation"] = {"puts": 0.35, "calls": 0.65}
    else:
        analysis["recommendation"] = "NEUTRAL - BOTH SIDES"
        analysis["rationale"] = [
            "Market is SIDEWAYS/NEUTRAL",
            "Both put and call credit spreads work well in range-bound markets",
            "Iron condor strategy (both sides) captures premium on both ends",
            "Equal 50-50 allocation recommended",
        ]
        analysis["allocation"] = {"puts": 0.50, "calls": 0.50}

    # VIX-based adjustments
    if vix:
        if vix > 25:
            analysis["vix_warning"] = "HIGH VIX (>25): Consider reducing position sizes by 25%"
        elif vix > 20:
            analysis["vix_note"] = "Elevated VIX (>20): Good premium but increased risk"
        else:
            analysis["vix_note"] = "Normal VIX: Standard position sizing"

    return analysis


def print_strategy_report(close_price: float, prev_close: float = None, vix: float = None):
    """Print comprehensive strategy report."""
    strikes = calculate_strikes(close_price)

    print("=" * 70)
    print("NDX CREDIT SPREAD STRATEGY - DAILY CALCULATOR")
    print("=" * 70)
    print(f"Reference Close: ${close_price:,.2f}")
    print(f"Calculated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Trend analysis if we have previous close
    if prev_close:
        trend = analyze_trend(close_price, prev_close, vix=vix)
        print("-" * 70)
        print("TREND ANALYSIS & RECOMMENDATION")
        print("-" * 70)
        print(f"Daily Change: {trend['daily_change_pct']:+.2f}%")
        print(f"Trend: {trend['trend'].upper()}")
        print(f"\n>>> RECOMMENDATION: {trend['recommendation']} <<<\n")
        for point in trend['rationale']:
            print(f"  - {point}")
        print(f"\nSuggested Allocation:")
        print(f"  PUTS:  {trend['allocation']['puts']*100:.0f}% (${300000*trend['allocation']['puts']:,.0f} of $300K)")
        print(f"  CALLS: {trend['allocation']['calls']*100:.0f}% (${300000*trend['allocation']['calls']:,.0f} of $300K)")
        if 'vix_warning' in trend:
            print(f"\n  WARNING: {trend['vix_warning']}")
        elif 'vix_note' in trend:
            print(f"\n  Note: {trend['vix_note']}")
        print()

    # PUT STRIKES
    print("-" * 70)
    print("PUT CREDIT SPREADS (Sell below market - BULLISH position)")
    print("-" * 70)
    put_data = strikes["puts"]
    print(f"  OPTIMAL Sell Strike:  ${put_data['optimal']['sell_strike']:,.0f}  ({put_data['optimal']['pct_below']:.2f}% below close)")
    print(f"    -> Buy strike at: ${put_data['optimal']['sell_strike'] - put_data['optimal']['dynamic_width']:,.0f} (width: {put_data['optimal']['dynamic_width']:.0f})")
    print(f"  SAFE Sell Strike:     ${put_data['safe']['sell_strike']:,.0f}  ({put_data['safe']['pct_below']:.2f}% below close)")
    print(f"  DANGER ZONE:          Below ${put_data['danger_zone']['never_sell_below']:,.0f} - AVOID!")
    print()

    # CALL STRIKES
    print("-" * 70)
    print("CALL CREDIT SPREADS (Sell above market - BEARISH position)")
    print("-" * 70)
    call_data = strikes["calls"]
    print(f"  OPTIMAL Sell Strike:  ${call_data['optimal']['sell_strike']:,.0f}  ({call_data['optimal']['pct_above']:.2f}% above close)")
    print(f"    -> Buy strike at: ${call_data['optimal']['sell_strike'] + call_data['optimal']['dynamic_width']:,.0f} (width: {call_data['optimal']['dynamic_width']:.0f})")
    print(f"  SAFE Sell Strike:     ${call_data['safe']['sell_strike']:,.0f}  ({call_data['safe']['pct_above']:.2f}% above close)")
    print(f"  DANGER ZONE:          Above ${call_data['danger_zone']['never_sell_above']:,.0f} - AVOID!")
    print()

    # Trading schedule
    print("-" * 70)
    print("TRADING SCHEDULE (All times ET)")
    print("-" * 70)
    print("  9:30-10:30 AM:  Deploy $100K (first 2 positions, tight widths 15-20)")
    print("  10:30-12:00 PM: Deploy $100K (next 2 positions, moderate widths 20-25)")
    print("  12:00-2:30 PM:  Final $100K (selective, wider widths 25-30)")
    print("  After 2:30 PM:  NO NEW POSITIONS")
    print("  Exit Target:    80% of max premium")
    print()

    # Key reminders
    print("-" * 70)
    print("KEY REMINDERS")
    print("-" * 70)
    print("  1. Only 'Safe' and 'Optimal' levels maintain 100% win rate")
    print("  2. Anything closer than optimal = potential losses (85-90% win rate NOT enough!)")
    print("  3. Check VIX before trading - if >20, reduce position sizes by 25%")
    print("  4. Use Linear-500 dynamic width: width = 15 + (distance% * 500)")
    print("  5. Max per trade: $50K | Max concurrent: 6 positions")
    print("=" * 70)


async def fetch_current_price(ticker: str = "NDX") -> tuple:
    """Fetch current and previous close from database."""
    if not DB_AVAILABLE:
        print("ERROR: Database module not available")
        return None, None

    db = StockQuestDB()
    await db.connect()

    try:
        # Get most recent price
        async with db.connection.get_connection() as conn:
            # Try realtime first
            rows = await conn.fetch(
                """SELECT price, ts FROM realtime_prices
                   WHERE ticker = $1 ORDER BY ts DESC LIMIT 1""",
                ticker
            )

            if rows:
                current = float(rows[0]['price'])
            else:
                # Fall back to daily
                rows = await conn.fetch(
                    """SELECT close, date FROM daily_prices
                       WHERE ticker = $1 ORDER BY date DESC LIMIT 1""",
                    ticker
                )
                current = float(rows[0]['close']) if rows else None

            # Get previous close
            rows = await conn.fetch(
                """SELECT close FROM daily_prices
                   WHERE ticker = $1 ORDER BY date DESC LIMIT 2""",
                ticker
            )
            prev_close = float(rows[1]['close']) if len(rows) > 1 else None

            return current, prev_close
    finally:
        await db.close()


def main():
    parser = argparse.ArgumentParser(description="NDX Daily Strike Calculator")
    parser.add_argument("--close", type=float, help="Today's closing price")
    parser.add_argument("--prev-close", type=float, help="Previous day's closing price (for trend analysis)")
    parser.add_argument("--vix", type=float, help="Current VIX level")
    parser.add_argument("--fetch", action="store_true", help="Fetch current price from database")

    args = parser.parse_args()

    if args.fetch:
        current, prev_close = asyncio.run(fetch_current_price())
        if current:
            print_strategy_report(current, prev_close, args.vix)
        else:
            print("ERROR: Could not fetch price from database")
            sys.exit(1)
    elif args.close:
        print_strategy_report(args.close, args.prev_close, args.vix)
    else:
        # Default example with approximate current NDX price
        print("No price provided. Using example with NDX at 25,900...")
        print_strategy_report(25900, 25800)


if __name__ == "__main__":
    main()
