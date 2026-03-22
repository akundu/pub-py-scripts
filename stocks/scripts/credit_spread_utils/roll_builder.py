"""Shared roll spread builder — used by both backtesting and live advisor.

Builds a credit spread for rolling a position, with width expansion to
cover the debit of closing the current position.

Logic:
1. Find nearest available strike to the target (percentile-derived OTM strike)
2. Try base_width first — if credit >= btc_cost, accept immediately
3. Expand width by 5pt steps up to base_width × max_width_multiplier
4. Accept the narrowest width where credit >= btc_cost
5. If no width covers btc_cost, return the widest (best credit found)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class RollSpreadResult:
    """Result of a roll spread build attempt."""
    short_strike: float
    long_strike: float
    credit: float           # per-share net credit
    width: float            # spread width in points
    short_price: float      # sell leg price (bid)
    long_price: float       # buy leg price (ask)
    covers_btc: bool        # True if credit >= btc_cost


def build_roll_spread(
    options_df: pd.DataFrame,
    option_type: str,
    target_strike: float,
    base_width: float,
    max_width_multiplier: float,
    btc_cost: float,
    use_mid: bool,
    min_credit: float,
    prev_close: float,
    min_volume: Optional[int] = None,
    max_credit_width_ratio: float = 0.80,
    min_otm_pct: float = 0.0,
) -> Optional[RollSpreadResult]:
    """Build a credit spread for rolling, expanding width to cover btc_cost.

    Note: min_otm_pct defaults to 0 for rolls (we're already rolling to a
    percentile boundary, the OTM constraint is on the new entry not the roll).

    Args:
        options_df: Options chain DataFrame with strike, type, bid, ask columns.
        option_type: "put" or "call".
        target_strike: Target short strike (from percentile calculation).
        base_width: Starting spread width in points.
        max_width_multiplier: Max expansion factor (e.g., 2.0 = try up to 2× base).
        btc_cost: Per-share cost to close the current position (debit).
        use_mid: If True, use mid prices. If False, sell at bid, buy at ask.
        min_credit: Minimum per-share credit to accept.
        prev_close: Previous close price (for spread builder).
        min_volume: Minimum volume per leg (None = no filter).
        max_credit_width_ratio: Max credit/width ratio filter (default 0.80).

    Returns:
        RollSpreadResult or None if no valid spread found.
    """
    if options_df is None or options_df.empty:
        return None
    if "strike" not in options_df.columns:
        return None

    # Filter to matching option type
    type_col = "type" if "type" in options_df.columns else None
    if type_col:
        opt_data = options_df[options_df[type_col].str.lower() == option_type.lower()]
    else:
        opt_data = options_df

    if opt_data.empty:
        return None

    strikes = sorted(opt_data["strike"].unique())
    if not strikes:
        return None

    # Find nearest strike to target
    nearest_strike = min(strikes, key=lambda s: abs(s - target_strike))

    # Width expansion: try base first, then progressively wider
    max_width = int(base_width * max_width_multiplier)
    width_step = 5
    widths_to_try = [int(base_width)]
    w = int(base_width) + width_step
    while w <= max_width:
        widths_to_try.append(w)
        w += width_step

    # Import the spread builder
    try:
        from scripts.credit_spread_utils.spread_builder import (
            build_credit_spreads,
        )
    except ImportError:
        return None

    best: Optional[RollSpreadResult] = None

    for try_width in widths_to_try:
        # Filter options near the target strike
        margin = try_width + 5
        if option_type.lower() == "put":
            filtered = options_df[
                (options_df["strike"] >= nearest_strike - margin)
                & (options_df["strike"] <= nearest_strike + 5)
            ]
        else:
            filtered = options_df[
                (options_df["strike"] >= nearest_strike - 5)
                & (options_df["strike"] <= nearest_strike + margin)
            ]

        if type_col and len(filtered) > 20:
            if "bid" in filtered.columns:
                filtered = filtered.sort_values("bid", ascending=False)
            filtered = filtered.drop_duplicates(
                subset=["strike", "type"], keep="first"
            )

        if filtered.empty:
            continue

        spreads = build_credit_spreads(
            options_df=filtered,
            option_type=option_type,
            prev_close=prev_close,
            percent_beyond=(0.0, 0.0),
            min_width=5,
            max_width=(try_width, try_width),
            use_mid=use_mid,
            percentile_target_strike=nearest_strike,
            max_credit_width_ratio=max_credit_width_ratio,
            min_volume=min_volume,
            min_otm_pct=min_otm_pct,
        )

        if not spreads:
            continue

        # Pick spread with highest credit
        spread = max(spreads, key=lambda s: s["net_credit"])
        credit = spread["net_credit"]

        if credit < min_credit:
            continue

        result = RollSpreadResult(
            short_strike=spread["short_strike"],
            long_strike=spread["long_strike"],
            credit=credit,
            width=spread["width"],
            short_price=spread.get("short_price", 0),
            long_price=spread.get("long_price", 0),
            covers_btc=credit >= btc_cost,
        )

        # If credit covers BTC cost, accept immediately (narrowest sufficient width)
        if result.covers_btc:
            return result

        # Track best so far
        if best is None or credit > best.credit:
            best = result

    return best
