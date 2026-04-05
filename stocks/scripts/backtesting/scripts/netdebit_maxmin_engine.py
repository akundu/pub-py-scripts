"""NetDebitMaxMin — Contrarian Intraday Debit Spread Strategy (Core Engine).

Buys OTM debit spreads contrarian to the intraday trend:
  - New HOD → buy bear put debit spread (bet on reversal down)
  - New LOD → buy bull call debit spread (bet on reversal up)

Layers accumulate throughout the day; old layers are abandoned (max loss = debit
paid). The winning layer at settlement covers all dead layers.

Configurable dimensions:
  - leg_placement: "otm", "just_otm", "atm", "itm", "best_value"
  - spread_width_points / min_width / max_width: width control
  - depth_pct: how far from extreme to place the long leg
  - max_debit_pct_of_width: reject if debit is too expensive
  - check_times_pacific: when to scan for new HOD/LOD
  - max_daily_debit / max_concurrent_layers: risk caps
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Reuse data-loading and time helpers from vmaxmin_engine
from scripts.backtesting.scripts.vmaxmin_engine import (
    _time_to_mins,
    _utc_to_pacific,
    filter_valid_quotes,
    get_hod_lod_in_range,
    get_prev_close,
    get_price_at_time,
    get_trading_dates,
    load_0dte_options,
    load_equity_bars_df,
    load_equity_prices,
    snap_options_to_time,
)


# ── Default Config ──────────────────────────────────────────────────────────

NETDEBIT_DEFAULT_CONFIG: Dict[str, Any] = {
    "entry_time_pacific": "06:35",
    "check_times_pacific": ["07:35", "08:35", "09:35", "10:35", "11:35"],
    "settlement_time_pacific": "13:00",
    "num_contracts": 1,
    "commission_per_transaction": 10,

    # Width control
    "spread_width_points": 5,       # target width if min/max not set
    "min_width": None,              # allow flexible range [min_width, max_width]
    "max_width": None,
    "min_width_steps": {"SPX": 5, "NDX": 10, "RUT": 5},

    # Leg placement
    "leg_placement": "otm",         # "otm", "just_otm", "atm", "itm", "best_value"
    "depth_pct": 0.003,             # % distance from extreme for strike placement

    # Debit bounds
    "max_debit_pct_of_width": 0.50, # reject if debit > this * width
    "min_debit": 0.20,              # minimum debit per share

    # Trigger threshold
    "breach_min_points": None,      # None = use min_width_step

    # Risk caps
    "max_daily_debit": 5000,        # max total $ spent on debit per day
    "max_concurrent_layers": 10,    # max open positions

    # Entry
    "open_entry_spread": True,      # open contrarian spread at entry time

    # Filters (v2)
    "direction_filter": "both",     # "both", "puts_only", "calls_only"
    "single_layer_per_direction": False,  # max 1 spread per direction per day
    "min_range_points": 0,          # require HOD-LOD >= this before layering
    "range_observation_time": None,  # wait until this time to measure range (e.g. "09:30")

    # Data sources
    "equity_dir": "equities_output",
    "options_dir": "options_csv_output_full_5",
}

# Ticker-specific date ranges for data availability
TICKER_START_DATES = {
    "RUT": "2025-09-09",
    "SPX": "2025-09-09",
    "NDX": "2025-09-09",
}


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class DebitSpreadPosition:
    direction: str          # "put" (bear put spread) or "call" (bull call spread)
    long_strike: float      # leg we BUY (closer to money)
    short_strike: float     # leg we SELL (further OTM)
    width: float
    debit_per_share: float  # what we paid
    num_contracts: int
    entry_time: str         # Pacific time HH:MM
    entry_price: float      # underlying price at entry
    trigger: str = ""       # "hod", "lod", "entry"

    @property
    def total_debit(self) -> float:
        return self.debit_per_share * 100 * self.num_contracts

    @property
    def max_profit(self) -> float:
        return (self.width - self.debit_per_share) * 100 * self.num_contracts

    @property
    def max_loss(self) -> float:
        return self.total_debit


@dataclass
class DebitTradeRecord:
    event: str              # "entry", "settlement"
    time_pacific: str
    direction: str          # "put" or "call"
    long_strike: float = 0.0
    short_strike: float = 0.0
    width: float = 0.0
    debit: float = 0.0      # total $ paid (entry) or payout received (settlement)
    payout: float = 0.0     # settlement value
    num_contracts: int = 0
    commission: float = 0.0
    underlying_price: float = 0.0
    trigger: str = ""       # "hod", "lod", "entry"
    notes: str = ""


@dataclass
class DebitDayResult:
    ticker: str
    date: str
    trades: List[DebitTradeRecord] = field(default_factory=list)
    positions: List[DebitSpreadPosition] = field(default_factory=list)
    total_debits_paid: float = 0.0
    total_payouts: float = 0.0
    total_commissions: float = 0.0
    num_layers: int = 0
    close_price: float = 0.0
    open_price: float = 0.0
    hod: float = 0.0
    lod: float = 0.0
    failure_reason: str = ""

    @property
    def net_pnl(self) -> float:
        return self.total_payouts - self.total_debits_paid - self.total_commissions


# ── P&L Calculation ─────────────────────────────────────────────────────────

def calculate_debit_spread_pnl(
    debit_per_share: float,
    long_strike: float,
    short_strike: float,
    underlying_price: float,
    option_type: str,
) -> float:
    """Calculate P&L per share for a debit spread at settlement.

    For a bear put debit spread (option_type="put"):
      - Long put at long_strike (higher), short put at short_strike (lower)
      - width = long_strike - short_strike
      - Profit when price falls below long_strike

    For a bull call debit spread (option_type="call"):
      - Long call at long_strike (lower), short call at short_strike (higher)
      - width = short_strike - long_strike
      - Profit when price rises above long_strike

    Returns:
        P&L per share (positive = profit, negative = loss).
    """
    if option_type.lower() == "put":
        # Bear put: long put higher strike, short put lower strike
        width = long_strike - short_strike
        if underlying_price >= long_strike:
            spread_value = 0.0  # both OTM
        elif underlying_price <= short_strike:
            spread_value = width  # max payout
        else:
            spread_value = long_strike - underlying_price
    else:  # call
        # Bull call: long call lower strike, short call higher strike
        width = short_strike - long_strike
        if underlying_price <= long_strike:
            spread_value = 0.0  # both OTM
        elif underlying_price >= short_strike:
            spread_value = width  # max payout
        else:
            spread_value = underlying_price - long_strike

    return spread_value - debit_per_share


# ── Options Loading ─────────────────────────────────────────────────────────

def load_options_0dte_only(ticker: str, trade_date: str,
                           options_dir: str) -> Optional[pd.DataFrame]:
    """Load options and filter to 0DTE expiration only.

    The options_csv_output_full_5 files contain 7 days of expirations.
    We only want same-day (0DTE) contracts.
    """
    df = load_0dte_options(ticker, trade_date, options_dir)
    if df is None or df.empty:
        return None
    # Filter to 0DTE: expiration == trade_date
    if "expiration" in df.columns:
        df["expiration"] = df["expiration"].astype(str).str[:10]
        df = df[df["expiration"] == trade_date].copy()
    if df.empty:
        return None
    return df


# ── Spread Construction ─────────────────────────────────────────────────────

def find_debit_spread(
    options_snap: pd.DataFrame,
    extreme_price: float,
    spread_type: str,
    min_step: float,
    target_width: float,
    leg_placement: str = "otm",
    depth_pct: float = 0.003,
    max_debit_pct: float = 0.50,
    min_debit: float = 0.20,
    min_width: Optional[float] = None,
    max_width: Optional[float] = None,
) -> Optional[Dict]:
    """Find a debit spread anchored near an intraday extreme.

    For bear put debit spread (spread_type="put", triggered by HOD):
      - Long put (buy, higher strike) near the extreme
      - Short put (sell, lower strike) further OTM
      - Profits when price reverses DOWN past the long strike

    For bull call debit spread (spread_type="call", triggered by LOD):
      - Long call (buy, lower strike) near the extreme
      - Short call (sell, higher strike) further OTM
      - Profits when price reverses UP past the long strike

    Leg placement modes:
      - "otm": Both legs OTM. Cheapest debit, needs largest reversal.
      - "just_otm": Long leg 1 strike from money. Slightly more expensive.
      - "atm": Long leg at-the-money (nearest to extreme).
      - "itm": Long leg ITM (past the extreme). Already has intrinsic value.
      - "best_value": Scan all pairs, pick best (max_payout / debit) ratio.

    Returns dict with long_strike, short_strike, long_ask, short_bid, debit, width
    or None if no valid spread found.
    """
    side = filter_valid_quotes(options_snap, spread_type)
    if len(side) < 2:
        return None

    # Determine width range
    effective_min_width = min_width if min_width is not None else target_width
    effective_max_width = max_width if max_width is not None else target_width

    if leg_placement == "best_value":
        return _find_best_value_spread(
            side, extreme_price, spread_type, min_step,
            effective_min_width, effective_max_width,
            depth_pct, max_debit_pct, min_debit)
    elif leg_placement == "itm":
        return _find_debit_spread_itm(
            side, extreme_price, spread_type, min_step,
            effective_min_width, effective_max_width,
            depth_pct, max_debit_pct, min_debit)
    elif leg_placement == "atm":
        return _find_debit_spread_atm(
            side, extreme_price, spread_type, min_step,
            effective_min_width, effective_max_width,
            max_debit_pct, min_debit)
    elif leg_placement == "just_otm":
        return _find_debit_spread_just_otm(
            side, extreme_price, spread_type, min_step,
            effective_min_width, effective_max_width,
            max_debit_pct, min_debit)
    else:  # "otm" (default)
        return _find_debit_spread_otm(
            side, extreme_price, spread_type, min_step,
            effective_min_width, effective_max_width,
            depth_pct, max_debit_pct, min_debit)


def _build_debit_result(long_strike, short_strike, long_ask, short_bid,
                        spread_type) -> Optional[Dict]:
    """Construct result dict for a debit spread."""
    debit = long_ask - short_bid
    if spread_type == "put":
        width = long_strike - short_strike
    else:
        width = short_strike - long_strike
    return {
        "long_strike": long_strike,
        "short_strike": short_strike,
        "long_ask": long_ask,
        "short_bid": short_bid,
        "debit": debit,
        "width": width,
    }


def _validate_debit(debit, width, max_debit_pct, min_debit) -> bool:
    """Check debit bounds."""
    if debit <= 0 or width <= 0:
        return False
    if debit < min_debit:
        return False
    if debit > width * max_debit_pct:
        return False
    return True


def _get_strike_lookup(side: pd.DataFrame) -> Tuple[list, dict, dict]:
    """Build strike → bid/ask lookup."""
    strikes = sorted(side["strike"].unique())
    bids = {}
    asks = {}
    for _, row in side.iterrows():
        s = float(row["strike"])
        bids[s] = float(row["bid"])
        asks[s] = float(row["ask"])
    return strikes, bids, asks


def _width_range(min_width, max_width, min_step):
    """Yield widths from min to max in min_step increments."""
    w = min_width
    while w <= max_width + 0.01:
        yield w
        w += min_step


def _find_debit_spread_otm(
    side, extreme_price, spread_type, min_step,
    min_width, max_width, depth_pct, max_debit_pct, min_debit,
) -> Optional[Dict]:
    """Both legs OTM from the extreme. Cheapest debit."""
    strikes, bids, asks = _get_strike_lookup(side)

    if spread_type == "put":
        # Bear put: long put below HOD, short put further below
        # Long strike just below extreme with depth offset
        min_long = extreme_price * (1 - depth_pct) if depth_pct else extreme_price
        # Long candidates: strikes <= min_long, sorted descending (closest first)
        long_candidates = [s for s in strikes if s <= min_long]
        long_candidates.sort(reverse=True)

        for long_strike in long_candidates[:10]:
            long_ask = asks.get(long_strike)
            if long_ask is None or long_ask <= 0:
                continue
            for width in _width_range(min_width, max_width, min_step):
                short_strike = long_strike - width
                if short_strike not in bids:
                    continue
                short_bid = bids[short_strike]
                if short_bid <= 0:
                    continue
                debit = long_ask - short_bid
                if _validate_debit(debit, width, max_debit_pct, min_debit):
                    return _build_debit_result(
                        long_strike, short_strike, long_ask, short_bid, spread_type)

    else:  # call
        # Bull call: long call above LOD, short call further above
        max_long = extreme_price * (1 + depth_pct) if depth_pct else extreme_price
        long_candidates = [s for s in strikes if s >= max_long]
        long_candidates.sort()  # closest first

        for long_strike in long_candidates[:10]:
            long_ask = asks.get(long_strike)
            if long_ask is None or long_ask <= 0:
                continue
            for width in _width_range(min_width, max_width, min_step):
                short_strike = long_strike + width
                if short_strike not in bids:
                    continue
                short_bid = bids[short_strike]
                if short_bid <= 0:
                    continue
                debit = long_ask - short_bid
                if _validate_debit(debit, width, max_debit_pct, min_debit):
                    return _build_debit_result(
                        long_strike, short_strike, long_ask, short_bid, spread_type)

    return None


def _find_debit_spread_just_otm(
    side, extreme_price, spread_type, min_step,
    min_width, max_width, max_debit_pct, min_debit,
) -> Optional[Dict]:
    """Long leg 1 strike from money (barely OTM)."""
    strikes, bids, asks = _get_strike_lookup(side)

    if spread_type == "put":
        # First strike below extreme
        long_candidates = [s for s in strikes if s < extreme_price]
        long_candidates.sort(reverse=True)
        long_candidates = long_candidates[:3]  # just the nearest few
    else:
        long_candidates = [s for s in strikes if s > extreme_price]
        long_candidates.sort()
        long_candidates = long_candidates[:3]

    for long_strike in long_candidates:
        long_ask = asks.get(long_strike)
        if long_ask is None or long_ask <= 0:
            continue
        for width in _width_range(min_width, max_width, min_step):
            if spread_type == "put":
                short_strike = long_strike - width
            else:
                short_strike = long_strike + width
            if short_strike not in bids:
                continue
            short_bid = bids[short_strike]
            if short_bid <= 0:
                continue
            debit = long_ask - short_bid
            if _validate_debit(debit, width, max_debit_pct, min_debit):
                return _build_debit_result(
                    long_strike, short_strike, long_ask, short_bid, spread_type)

    return None


def _find_debit_spread_atm(
    side, extreme_price, spread_type, min_step,
    min_width, max_width, max_debit_pct, min_debit,
) -> Optional[Dict]:
    """Long leg at-the-money (nearest strike to extreme)."""
    strikes, bids, asks = _get_strike_lookup(side)

    # Find nearest strike to extreme
    nearest = min(strikes, key=lambda s: abs(s - extreme_price))
    # Also try 1 strike on each side
    near_idx = strikes.index(nearest)
    candidates = [nearest]
    if near_idx > 0:
        candidates.append(strikes[near_idx - 1])
    if near_idx < len(strikes) - 1:
        candidates.append(strikes[near_idx + 1])
    # Sort by distance to extreme
    candidates.sort(key=lambda s: abs(s - extreme_price))

    for long_strike in candidates:
        long_ask = asks.get(long_strike)
        if long_ask is None or long_ask <= 0:
            continue
        for width in _width_range(min_width, max_width, min_step):
            if spread_type == "put":
                short_strike = long_strike - width
            else:
                short_strike = long_strike + width
            if short_strike not in bids:
                continue
            short_bid = bids[short_strike]
            if short_bid <= 0:
                continue
            debit = long_ask - short_bid
            if _validate_debit(debit, width, max_debit_pct, min_debit):
                return _build_debit_result(
                    long_strike, short_strike, long_ask, short_bid, spread_type)

    return None


def _find_debit_spread_itm(
    side, extreme_price, spread_type, min_step,
    min_width, max_width, depth_pct, max_debit_pct, min_debit,
) -> Optional[Dict]:
    """Long leg ITM (past the extreme). Already has intrinsic value."""
    strikes, bids, asks = _get_strike_lookup(side)

    if spread_type == "put":
        # Long put ITM = strike > extreme (put is ITM when strike > price)
        min_long = extreme_price * (1 + depth_pct) if depth_pct else extreme_price
        long_candidates = [s for s in strikes if s >= min_long]
        long_candidates.sort()  # nearest ITM first
    else:
        # Long call ITM = strike < extreme (call is ITM when strike < price)
        max_long = extreme_price * (1 - depth_pct) if depth_pct else extreme_price
        long_candidates = [s for s in strikes if s <= max_long]
        long_candidates.sort(reverse=True)  # nearest ITM first

    for long_strike in long_candidates[:10]:
        long_ask = asks.get(long_strike)
        if long_ask is None or long_ask <= 0:
            continue
        for width in _width_range(min_width, max_width, min_step):
            if spread_type == "put":
                short_strike = long_strike - width
            else:
                short_strike = long_strike + width
            if short_strike not in bids:
                continue
            short_bid = bids[short_strike]
            if short_bid <= 0:
                continue
            debit = long_ask - short_bid
            if _validate_debit(debit, width, max_debit_pct, min_debit):
                return _build_debit_result(
                    long_strike, short_strike, long_ask, short_bid, spread_type)

    return None


def _find_best_value_spread(
    side, extreme_price, spread_type, min_step,
    min_width, max_width, depth_pct, max_debit_pct, min_debit,
) -> Optional[Dict]:
    """Scan all valid pairs near extreme, pick best (max_payout / debit) ratio."""
    strikes, bids, asks = _get_strike_lookup(side)

    # Consider strikes within a proximity band of the extreme
    proximity_pts = extreme_price * max(depth_pct * 5, 0.02)  # wider scan
    near_strikes = [s for s in strikes if abs(s - extreme_price) <= proximity_pts]

    candidates = []
    for long_strike in near_strikes:
        long_ask = asks.get(long_strike)
        if long_ask is None or long_ask <= 0:
            continue
        for width in _width_range(min_width, max_width, min_step):
            if spread_type == "put":
                short_strike = long_strike - width
            else:
                short_strike = long_strike + width
            if short_strike not in bids:
                continue
            short_bid = bids[short_strike]
            if short_bid <= 0:
                continue
            debit = long_ask - short_bid
            if not _validate_debit(debit, width, max_debit_pct, min_debit):
                continue
            # Value = max_payout / debit = (width - debit) / debit
            value = (width - debit) / debit if debit > 0 else 0
            candidates.append({
                "long_strike": long_strike,
                "short_strike": short_strike,
                "long_ask": long_ask,
                "short_bid": short_bid,
                "debit": debit,
                "width": width,
                "_value": value,
            })

    if not candidates:
        return None

    # Best value ratio, then narrowest width as tiebreaker
    candidates.sort(key=lambda c: (-c["_value"], c["width"]))
    best = candidates[0]
    del best["_value"]
    return best


# ── Core Engine ─────────────────────────────────────────────────────────────

class NetDebitMaxMinEngine:
    """Runs the NetDebitMaxMin strategy for a single trading day."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = {**NETDEBIT_DEFAULT_CONFIG, **(config or {})}

    def _get_min_step(self, ticker: str) -> float:
        steps = self.config["min_width_steps"]
        return steps.get(ticker, 5)

    def _get_width_params(self, ticker: str) -> Tuple[float, float, float]:
        """Return (target_width, min_width, max_width) in points."""
        min_step = self._get_min_step(ticker)
        target = self.config.get("spread_width_points") or min_step
        cfg_min = self.config.get("min_width")
        cfg_max = self.config.get("max_width")
        return (
            target,
            cfg_min if cfg_min is not None else None,
            cfg_max if cfg_max is not None else None,
        )

    def run_single_day(
        self,
        ticker: str,
        trade_date: str,
        equity_df: pd.DataFrame,
        equity_prices: dict,
        options_all: Optional[pd.DataFrame],
        all_dates: List[str],
        prev_close: Optional[float],
    ) -> DebitDayResult:
        """Run the strategy for one trading day.

        Returns DebitDayResult with all positions and P&L.
        """
        result = DebitDayResult(ticker=ticker, date=trade_date)
        commission = self.config["commission_per_transaction"]
        entry_time = self.config["entry_time_pacific"]
        min_step = self._get_min_step(ticker)
        target_width, cfg_min_w, cfg_max_w = self._get_width_params(ticker)
        leg_placement = self.config.get("leg_placement", "otm")
        depth_pct = self.config.get("depth_pct", 0.003)
        max_debit_pct = self.config.get("max_debit_pct_of_width", 0.50)
        min_debit = self.config.get("min_debit", 0.20)
        max_daily_debit = self.config.get("max_daily_debit", 5000)
        max_layers = self.config.get("max_concurrent_layers", 10)
        num_contracts = self.config.get("num_contracts", 1)

        # v2 filters
        direction_filter = self.config.get("direction_filter", "both")
        single_layer = self.config.get("single_layer_per_direction", False)
        min_range_pts = self.config.get("min_range_points", 0)
        range_obs_time = self.config.get("range_observation_time")
        layers_opened_put = 0
        layers_opened_call = 0

        # --- Entry price ---
        entry_price = get_price_at_time(equity_prices, entry_time)
        if entry_price is None:
            result.failure_reason = f"No price at {entry_time}"
            return result

        sorted_times = sorted(equity_prices.keys(), key=_time_to_mins)
        result.close_price = equity_prices[sorted_times[-1]] if sorted_times else 0
        result.open_price = entry_price

        if prev_close is None:
            result.failure_reason = "No previous close"
            return result

        if options_all is None or options_all.empty:
            result.failure_reason = "No options data"
            return result

        # --- Filter to 0DTE ---
        if "expiration" in options_all.columns:
            options_all = options_all.copy()
            options_all["expiration"] = options_all["expiration"].astype(str).str[:10]
            options_0dte = options_all[options_all["expiration"] == trade_date]
        else:
            options_0dte = options_all

        if options_0dte.empty:
            result.failure_reason = "No 0DTE options for this date"
            return result

        positions: List[DebitSpreadPosition] = []
        daily_debit_spent = 0.0

        def _can_add_layer(debit_total: float) -> bool:
            if len(positions) >= max_layers:
                return False
            if daily_debit_spent + debit_total > max_daily_debit:
                return False
            return True

        def _open_debit_spread(
            snap: pd.DataFrame, anchor_price: float,
            spread_type: str, trigger: str,
            current_price: float, time_str: str,
        ) -> Optional[DebitSpreadPosition]:
            nonlocal daily_debit_spent
            spread = find_debit_spread(
                snap, anchor_price, spread_type, min_step, target_width,
                leg_placement=leg_placement, depth_pct=depth_pct,
                max_debit_pct=max_debit_pct, min_debit=min_debit,
                min_width=cfg_min_w, max_width=cfg_max_w,
            )
            if spread is None:
                return None

            pos = DebitSpreadPosition(
                direction=spread_type,
                long_strike=spread["long_strike"],
                short_strike=spread["short_strike"],
                width=spread["width"],
                debit_per_share=spread["debit"],
                num_contracts=num_contracts,
                entry_time=time_str,
                entry_price=current_price,
                trigger=trigger,
            )

            if not _can_add_layer(pos.total_debit):
                return None

            positions.append(pos)
            daily_debit_spent += pos.total_debit
            result.total_debits_paid += pos.total_debit
            result.total_commissions += commission
            result.num_layers += 1
            result.trades.append(DebitTradeRecord(
                event="entry", time_pacific=time_str,
                direction=spread_type,
                long_strike=spread["long_strike"],
                short_strike=spread["short_strike"],
                width=spread["width"],
                debit=pos.total_debit,
                num_contracts=num_contracts,
                commission=commission,
                underlying_price=current_price,
                trigger=trigger,
                notes=(f"{'Bear put' if spread_type == 'put' else 'Bull call'} "
                       f"debit spread, {leg_placement} placement, "
                       f"debit=${spread['debit']:.2f}/sh"),
            ))
            return pos

        def _direction_allowed(spread_type: str) -> bool:
            if direction_filter == "puts_only" and spread_type != "put":
                return False
            if direction_filter == "calls_only" and spread_type != "call":
                return False
            return True

        def _single_layer_ok(spread_type: str) -> bool:
            if not single_layer:
                return True
            if spread_type == "put" and layers_opened_put >= 1:
                return False
            if spread_type == "call" and layers_opened_call >= 1:
                return False
            return True

        # --- Initial entry spread ---
        entry_mins = _time_to_mins(entry_time)
        if self.config.get("open_entry_spread", True):
            snap = snap_options_to_time(options_0dte, entry_time, tolerance_mins=5)
            if not snap.empty:
                # Contrarian: gap up → buy puts (expect reversal down)
                #             gap down → buy calls (expect reversal up)
                if entry_price >= prev_close:
                    entry_dir = "put"
                    anchor = entry_price
                else:
                    entry_dir = "call"
                    anchor = entry_price
                if _direction_allowed(entry_dir) and _single_layer_ok(entry_dir):
                    pos = _open_debit_spread(snap, anchor, entry_dir, "entry",
                                             entry_price, entry_time)
                    if pos is not None:
                        if entry_dir == "put":
                            layers_opened_put += 1
                        else:
                            layers_opened_call += 1

        # --- HOD/LOD tracking ---
        hod, lod = entry_price, entry_price
        prev_hod, prev_lod = hod, lod

        # --- Intraday checks: layer on new extremes ---
        check_times = self.config.get("check_times_pacific",
                                      ["07:35", "08:35", "09:35", "10:35", "11:35"])
        check_times = [t for t in check_times if _time_to_mins(t) > entry_mins]
        breach_min = self.config.get("breach_min_points") or min_step

        for check_time in check_times:
            check_mins = _time_to_mins(check_time)

            # Update HOD/LOD from bars
            if not equity_df.empty:
                h, l = get_hod_lod_in_range(equity_df, entry_mins, check_mins)
                if h > 0:
                    hod = max(hod, h)
                if l < float("inf"):
                    lod = min(lod, l)

            current_price = get_price_at_time(equity_prices, check_time)
            if current_price is None:
                continue

            snap = snap_options_to_time(options_0dte, check_time, tolerance_mins=5)
            if snap.empty:
                continue

            new_hod = hod >= prev_hod + breach_min
            new_lod = lod <= prev_lod - breach_min

            # Min range gate: only layer if intraday range is wide enough
            if min_range_pts > 0:
                if range_obs_time:
                    obs_mins = _time_to_mins(range_obs_time)
                    if check_mins < obs_mins:
                        # Before observation time: update HOD/LOD but don't trade
                        if new_hod:
                            prev_hod = hod
                        if new_lod:
                            prev_lod = lod
                        continue
                current_range = hod - lod
                if current_range < min_range_pts:
                    # Range too narrow — update tracking but skip layer
                    if new_hod:
                        prev_hod = hod
                    if new_lod:
                        prev_lod = lod
                    continue

            # New HOD → bear put debit spread (contrarian: bet on reversal down)
            if new_hod and _direction_allowed("put") and _single_layer_ok("put"):
                pos = _open_debit_spread(snap, hod, "put", "hod",
                                         current_price, check_time)
                if pos is not None:
                    prev_hod = hod
                    layers_opened_put += 1
                else:
                    prev_hod = hod  # still ratchet to avoid stale re-triggers

            # New LOD → bull call debit spread (contrarian: bet on reversal up)
            if new_lod and _direction_allowed("call") and _single_layer_ok("call"):
                pos = _open_debit_spread(snap, lod, "call", "lod",
                                         current_price, check_time)
                if pos is not None:
                    prev_lod = lod
                    layers_opened_call += 1
                else:
                    prev_lod = lod

        # --- Settlement ---
        settlement_time = self.config.get("settlement_time_pacific", "13:00")
        settlement_price = get_price_at_time(equity_prices, settlement_time)
        if settlement_price is None:
            # Fallback to last known price
            settlement_price = result.close_price

        # Compute HOD/LOD over entire day for reporting
        if not equity_df.empty:
            day_h, day_l = get_hod_lod_in_range(equity_df, entry_mins, 999)
            if day_h > 0:
                result.hod = day_h
            if day_l < float("inf"):
                result.lod = day_l

        result.positions = list(positions)

        for pos in positions:
            pnl_per_share = calculate_debit_spread_pnl(
                pos.debit_per_share,
                pos.long_strike,
                pos.short_strike,
                settlement_price,
                pos.direction,
            )
            payout_per_share = max(0, pnl_per_share + pos.debit_per_share)
            payout_total = payout_per_share * 100 * pos.num_contracts
            result.total_payouts += payout_total

            result.trades.append(DebitTradeRecord(
                event="settlement", time_pacific=settlement_time,
                direction=pos.direction,
                long_strike=pos.long_strike,
                short_strike=pos.short_strike,
                width=pos.width,
                debit=0,
                payout=payout_total,
                num_contracts=pos.num_contracts,
                commission=0,
                underlying_price=settlement_price,
                trigger=pos.trigger,
                notes=(f"P&L/sh=${pnl_per_share:+.2f}, "
                       f"payout=${payout_total:,.0f}, "
                       f"{'WIN' if pnl_per_share > 0 else 'LOSS'}"),
            ))

        return result
