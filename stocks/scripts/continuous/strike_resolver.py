#!/usr/bin/env python3
"""
Strike Resolver for Continuous Mode

Resolves grid config opportunities into specific option strikes and prices
by computing band boundaries and looking up real option chain data.
"""

import sys
from pathlib import Path
from datetime import date as date_type, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Band percentile thresholds (same as comprehensive_backtest.py)
BAND_CONFIGS = {
    'P95':  (2.5,  97.5),
    'P97':  (1.5,  98.5),
    'P98':  (1.0,  99.0),
    'P99':  (0.5,  99.5),
    'P100': (0.0, 100.0),
}

# Normal distribution z-scores for percentile boundaries
BAND_Z_SCORES = {
    'P95':  1.960,
    'P97':  2.170,
    'P98':  2.326,
    'P99':  2.576,
    'P100': 3.291,
}

MULTIPLIER = 100

# Option chain CSV directories (in priority order)
# options_csv_output_full has clean 15-min market-hours data (small files)
# options_csv_output has 0DTE chains
# csv_exports has large multi-snapshot files
CHAIN_DIRS = [
    ('options_csv_output_full', '{ticker}_options_{date}.csv'),
    ('options_csv_output', '{ticker}_options_{date}.csv'),
    ('csv_exports/options/{ticker}', '{date}.csv'),
]


@dataclass
class TradeLeg:
    """A single option leg in a spread."""
    option_type: str    # 'put' or 'call'
    side: str           # 'sell' or 'buy'
    strike: float
    price: float        # bid for sell, ask for buy (per share)
    expiration: str = ''


@dataclass
class ResolvedTrade:
    """Fully resolved trade with specific strikes and prices."""
    spread_type: str
    legs: List[TradeLeg]
    total_credit: float      # per contract ($)
    max_risk: float          # per contract ($)
    lo_price: float          # band lower boundary
    hi_price: float          # band upper boundary
    expiration: str = ''
    data_source: str = ''    # 'chain' or 'estimated'

    def to_dict(self) -> Dict:
        return {
            'spread_type': self.spread_type,
            'legs': [
                {
                    'option_type': leg.option_type,
                    'side': leg.side,
                    'strike': leg.strike,
                    'price': leg.price,
                    'expiration': leg.expiration,
                }
                for leg in self.legs
            ],
            'total_credit': self.total_credit,
            'max_risk': self.max_risk,
            'lo_price': self.lo_price,
            'hi_price': self.hi_price,
            'expiration': self.expiration,
            'data_source': self.data_source,
        }

    def instruction_text(self) -> str:
        """Human-readable trade instruction."""
        parts = []
        for leg in self.legs:
            action = 'Sell' if leg.side == 'sell' else 'Buy'
            parts.append(f"{action} {leg.strike:.0f} {leg.option_type.title()}")

        credit_str = f"${self.total_credit / MULTIPLIER:.2f}"
        instruction = ' / '.join(parts) + f" @ {credit_str} credit"
        if self.expiration:
            instruction += f" (exp {self.expiration})"
        return instruction


def estimate_band_prices(
    current_price: float,
    band: str,
    dte: int,
    vix: float,
) -> Tuple[float, float]:
    """
    Estimate band boundary prices using VIX-based approximation.

    Uses: expected_move = price * VIX/100 * sqrt(DTE/252)
    Then applies z-score for the band percentile.
    """
    z = BAND_Z_SCORES.get(band, 2.0)
    effective_dte = max(dte, 1)
    period_vol = (vix / 100.0) * np.sqrt(effective_dte / 252.0)

    lo_price = current_price * (1 - z * period_vol)
    hi_price = current_price * (1 + z * period_vol)

    return lo_price, hi_price


def find_option_chain_file(
    ticker: str,
    target_date: date_type,
    base_dir: Path = None,
) -> Optional[Path]:
    """Find the option chain CSV file for a given date and ticker."""
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent

    date_str = target_date.strftime('%Y-%m-%d')

    for dir_pattern, file_pattern in CHAIN_DIRS:
        dir_path = base_dir / dir_pattern.format(ticker=ticker, date=date_str)
        file_name = file_pattern.format(ticker=ticker, date=date_str)

        # Try with ticker subdirectory
        for candidate in [
            dir_path / file_name,
            dir_path / ticker / file_name,
            base_dir / dir_pattern.format(ticker=ticker) / ticker / file_name,
        ]:
            if candidate.exists():
                return candidate

    return None


def load_option_chain(
    chain_file: Path,
    entry_time_et: str = None,
    expiration_date: str = None,
    max_rows: int = 500000,
) -> Optional[Dict[str, List[Tuple[float, float, float]]]]:
    """
    Load option chain from CSV file into lightweight format.

    Returns:
        Dict with 'put' and 'call' keys, each containing
        [(strike, bid, ask), ...] sorted by strike ascending.
    """
    try:
        # Only load needed columns to save memory on large files
        usecols = ['timestamp', 'type', 'strike', 'bid', 'ask']
        if expiration_date:
            usecols.append('expiration')

        df = pd.read_csv(chain_file, usecols=usecols, nrows=max_rows)
    except Exception as e:
        print(f"  Error loading chain {chain_file}: {e}")
        return None

    if df.empty:
        return None

    # Parse timestamps
    df['ts_utc'] = pd.to_datetime(df['timestamp'], utc=True)
    df['ts_et'] = df['ts_utc'].dt.tz_convert('America/New_York')
    df['ts_str'] = df['ts_et'].dt.strftime('%H:%M')

    # Filter by expiration if specified
    if expiration_date and 'expiration' in df.columns:
        exp_matches = df[df['expiration'].str.contains(expiration_date, na=False)]
        if not exp_matches.empty:
            df = exp_matches

    # Prefer market-hours data; if entry_time_et given, find nearest
    if entry_time_et:
        available_times = sorted(df['ts_str'].unique())
        if entry_time_et in available_times:
            df = df[df['ts_str'] == entry_time_et]
        elif available_times:
            def to_min(s):
                h, m = map(int, s.split(':'))
                return h * 60 + m
            target_min = to_min(entry_time_et)
            nearest = min(available_times, key=lambda s: abs(to_min(s) - target_min))
            if abs(to_min(nearest) - target_min) <= 60:
                df = df[df['ts_str'] == nearest]
    else:
        # Default: use the latest available market-hours snapshot
        available_times = sorted(df['ts_str'].unique())
        market_times = [t for t in available_times if '09:30' <= t <= '16:00']
        if market_times:
            df = df[df['ts_str'] == market_times[-1]]
        elif available_times:
            df = df[df['ts_str'] == available_times[-1]]

    # Build chain dict
    chain = {}
    for opt_type in ('put', 'call'):
        rows = df[df['type'].str.lower() == opt_type][['strike', 'bid', 'ask']].copy()
        rows = rows.dropna(subset=['strike'])
        rows['bid'] = pd.to_numeric(rows['bid'], errors='coerce').fillna(0)
        rows['ask'] = pd.to_numeric(rows['ask'], errors='coerce').fillna(0)
        rows = rows[(rows['bid'] > 0) | (rows['ask'] > 0)]

        chain[opt_type] = sorted(
            [(float(r.strike), float(r.bid), float(r.ask))
             for _, r in rows.iterrows()],
            key=lambda x: x[0]
        )

    return chain if (chain.get('put') or chain.get('call')) else None


def _nearest_below(opts: List[Tuple], target: float):
    """Find the highest strike <= target. opts sorted ascending."""
    best = None
    for o in opts:
        if o[0] <= target:
            best = o
    return best


def _nearest_above(opts: List[Tuple], target: float):
    """Find the lowest strike >= target. opts sorted ascending."""
    for o in opts:
        if o[0] >= target:
            return o
    return None


def _infer_underlying_from_chain(chain: Dict[str, List[Tuple]]) -> Optional[float]:
    """Estimate the underlying price from the option chain.

    Uses the put-call parity approximation: underlying ~ strike where
    put_mid ≈ call_mid (at-the-money).
    """
    puts = chain.get('put', [])
    calls = chain.get('call', [])
    if not puts or not calls:
        return None

    # Find strikes that exist in both chains
    put_strikes = {o[0]: (o[1] + o[2]) / 2 for o in puts if o[1] > 0 and o[2] > 0}
    call_strikes = {o[0]: (o[1] + o[2]) / 2 for o in calls if o[1] > 0 and o[2] > 0}

    common = set(put_strikes.keys()) & set(call_strikes.keys())
    if not common:
        # Fall back: use median of all put strikes
        all_strikes = [o[0] for o in puts]
        return np.median(all_strikes) if all_strikes else None

    # Find strike where |put_mid - call_mid| is smallest (ATM)
    best_strike = min(common, key=lambda s: abs(put_strikes[s] - call_strikes[s]))
    return best_strike


def resolve_trade(
    current_price: float,
    band: str,
    dte: int,
    spread_type: str,
    spread_width: int,
    vix: float,
    chain: Optional[Dict[str, List[Tuple[float, float, float]]]] = None,
    expiration: str = '',
) -> Optional[ResolvedTrade]:
    """
    Resolve a grid config into specific strikes and prices.

    If an option chain is provided and its strikes are compatible with the
    current price, uses real bid/ask prices. Otherwise, estimates strikes
    from VIX-based band boundaries.
    """
    # Compute band boundaries
    lo_price, hi_price = estimate_band_prices(current_price, band, dte, vix)

    if chain:
        # Validate chain is compatible with current price
        chain_underlying = _infer_underlying_from_chain(chain)
        if chain_underlying and abs(chain_underlying - current_price) / current_price > 0.10:
            # Chain is for a different price level (>10% mismatch)
            # Skip chain and use estimated strikes instead
            pass
        else:
            result = _resolve_from_chain(
                chain, lo_price, hi_price, spread_type, spread_width,
                current_price, expiration,
            )
            if result:
                return result

    # Fall back to estimated
    return _resolve_estimated(
        lo_price, hi_price, spread_type, spread_width,
        current_price, expiration,
    )


def _resolve_from_chain(
    chain: Dict[str, List[Tuple]],
    lo_price: float,
    hi_price: float,
    spread_type: str,
    spread_width: int,
    underlying: float,
    expiration: str,
) -> Optional[ResolvedTrade]:
    """Resolve using real option chain data (adapted from build_spread_fast)."""
    puts = chain.get('put', [])
    calls = chain.get('call', [])
    legs = []
    put_credit = call_credit = 0.0

    # Put side
    if spread_type in ('put_spread', 'iron_condor'):
        sp_row = _nearest_below(puts, lo_price)
        if sp_row is None:
            if spread_type == 'put_spread':
                return None
        else:
            sp = sp_row[0]
            lp_target = sp - spread_width
            lp_row = _nearest_below([o for o in puts if o[0] < sp], lp_target)
            if lp_row is None:
                sub = [o for o in puts if o[0] < sp]
                lp_row = min(sub, key=lambda o: abs(o[0] - lp_target)) if sub else None
            if lp_row is None:
                if spread_type == 'put_spread':
                    return None
            else:
                lp = lp_row[0]
                raw = sp_row[1] - lp_row[2]  # short_bid - long_ask
                if raw > 0:
                    put_credit = raw * MULTIPLIER
                    legs.append(TradeLeg('put', 'sell', sp, sp_row[1], expiration))
                    legs.append(TradeLeg('put', 'buy', lp, lp_row[2], expiration))
                elif spread_type == 'put_spread':
                    return None

    # Call side
    if spread_type in ('call_spread', 'iron_condor'):
        sc_row = _nearest_above(calls, hi_price)
        if sc_row is None:
            if spread_type == 'call_spread':
                return None
        else:
            sc = sc_row[0]
            lc_target = sc + spread_width
            lc_row = _nearest_above([o for o in calls if o[0] > sc], lc_target)
            if lc_row is None:
                sub = [o for o in calls if o[0] > sc]
                lc_row = min(sub, key=lambda o: abs(o[0] - lc_target)) if sub else None
            if lc_row is None:
                if spread_type == 'call_spread':
                    return None
            else:
                lc = lc_row[0]
                raw = sc_row[1] - lc_row[2]
                if raw > 0:
                    call_credit = raw * MULTIPLIER
                    legs.append(TradeLeg('call', 'sell', sc, sc_row[1], expiration))
                    legs.append(TradeLeg('call', 'buy', lc, lc_row[2], expiration))
                elif spread_type == 'call_spread':
                    return None

    total_credit = put_credit + call_credit
    if total_credit <= 0 or not legs:
        return None

    # Calculate max risk
    if spread_type == 'put_spread':
        max_risk = (legs[0].strike - legs[1].strike) * MULTIPLIER - put_credit
    elif spread_type == 'call_spread':
        max_risk = (legs[1].strike - legs[0].strike) * MULTIPLIER - call_credit
    else:  # iron_condor
        put_legs = [l for l in legs if l.option_type == 'put']
        call_legs = [l for l in legs if l.option_type == 'call']
        pr = (put_legs[0].strike - put_legs[1].strike) * MULTIPLIER - put_credit if len(put_legs) == 2 else 0
        cr = (call_legs[1].strike - call_legs[0].strike) * MULTIPLIER - call_credit if len(call_legs) == 2 else 0
        max_risk = max(pr, cr)

    if max_risk <= 0:
        return None

    return ResolvedTrade(
        spread_type=spread_type,
        legs=legs,
        total_credit=total_credit,
        max_risk=max_risk,
        lo_price=lo_price,
        hi_price=hi_price,
        expiration=expiration,
        data_source='chain',
    )


def _resolve_estimated(
    lo_price: float,
    hi_price: float,
    spread_type: str,
    spread_width: int,
    underlying: float,
    expiration: str,
) -> Optional[ResolvedTrade]:
    """
    Estimate strikes without real option chain data.

    Snaps to nearest 5-point strikes (NDX standard interval).
    Estimates credit using typical OTM spread pricing heuristics.
    """
    interval = 5
    legs = []
    put_credit = call_credit = 0.0

    def snap_down(price):
        return int(price / interval) * interval

    def snap_up(price):
        return int(np.ceil(price / interval)) * interval

    if spread_type in ('put_spread', 'iron_condor'):
        short_put = float(snap_down(lo_price))
        long_put = short_put - spread_width

        # Estimate per-share prices using distance from underlying
        dist_pct = (underlying - short_put) / underlying
        # OTM put premium decays with distance; use simplified model
        short_bid = max(0.10, spread_width * 0.30 * (1 - dist_pct * 5))
        long_ask = max(0.05, short_bid * 0.4)
        net = short_bid - long_ask
        if net > 0:
            put_credit = net * MULTIPLIER
            legs.append(TradeLeg('put', 'sell', short_put, short_bid, expiration))
            legs.append(TradeLeg('put', 'buy', long_put, long_ask, expiration))

    if spread_type in ('call_spread', 'iron_condor'):
        short_call = float(snap_up(hi_price))
        long_call = short_call + spread_width

        dist_pct = (short_call - underlying) / underlying
        short_bid = max(0.10, spread_width * 0.30 * (1 - dist_pct * 5))
        long_ask = max(0.05, short_bid * 0.4)
        net = short_bid - long_ask
        if net > 0:
            call_credit = net * MULTIPLIER
            legs.append(TradeLeg('call', 'sell', short_call, short_bid, expiration))
            legs.append(TradeLeg('call', 'buy', long_call, long_ask, expiration))

    total_credit = put_credit + call_credit
    if total_credit <= 0 or not legs:
        return None

    if spread_type == 'iron_condor':
        max_risk = spread_width * MULTIPLIER - max(put_credit, call_credit)
    else:
        max_risk = spread_width * MULTIPLIER - total_credit

    if max_risk <= 0:
        return None

    return ResolvedTrade(
        spread_type=spread_type,
        legs=legs,
        total_credit=total_credit,
        max_risk=max_risk,
        lo_price=lo_price,
        hi_price=hi_price,
        expiration=expiration,
        data_source='estimated',
    )


def get_expiration_date(entry_date: date_type, dte: int) -> date_type:
    """Calculate expiration date (skips weekends)."""
    current = entry_date
    days_added = 0
    while days_added < dte:
        current += timedelta(days=1)
        if current.weekday() < 5:
            days_added += 1
    return current


def pst_to_et(time_pst: str) -> str:
    """Convert PST time (HH:MM) to ET (HH:MM) by adding 3 hours."""
    h, m = map(int, time_pst.split(':'))
    return f"{h + 3:02d}:{m:02d}"


if __name__ == '__main__':
    """Test strike resolver."""
    print("=" * 80)
    print("STRIKE RESOLVER TEST")
    print("=" * 80)

    price = 20075.0
    vix = 16.26

    for band in ['P95', 'P97', 'P98', 'P99']:
        for dte in [0, 1, 3]:
            lo, hi = estimate_band_prices(price, band, dte, vix)
            print(f"{band} {dte}DTE: ${lo:,.0f} - ${hi:,.0f} (width: {hi-lo:,.0f} pts)")

    # Test with real option chain
    print("\n" + "=" * 80)
    print("OPTION CHAIN RESOLUTION")
    print("=" * 80)

    chain_file = find_option_chain_file('NDX', date_type(2026, 2, 20))
    if chain_file:
        print(f"Found chain: {chain_file}")
        chain = load_option_chain(chain_file, entry_time_et='10:30')
        if chain:
            n_puts = len(chain.get('put', []))
            n_calls = len(chain.get('call', []))
            print(f"  Puts: {n_puts} strikes, Calls: {n_calls} strikes")

            chain_und = _infer_underlying_from_chain(chain)
            print(f"  Inferred underlying: ${chain_und:,.0f}" if chain_und else "  Could not infer underlying")

            trade = resolve_trade(
                current_price=price, band='P97', dte=3,
                spread_type='iron_condor', spread_width=10,
                vix=vix, chain=chain,
                expiration='2026-02-25',
            )
            if trade:
                print(f"\n  Trade: {trade.instruction_text()}")
                print(f"  Source: {trade.data_source}")
                print(f"  Credit: ${trade.total_credit:,.0f} | Risk: ${trade.max_risk:,.0f}")
                for leg in trade.legs:
                    print(f"    {leg.side.upper()} {leg.strike:.0f} {leg.option_type} @ ${leg.price:.2f}")
            else:
                print("  No chain-based trade resolved")
    else:
        print("No chain file found for 2026-02-20")

    # Test estimated fallback
    print("\n" + "=" * 80)
    print("ESTIMATED RESOLUTION (no chain)")
    print("=" * 80)

    trade = resolve_trade(
        current_price=price, band='P97', dte=3,
        spread_type='iron_condor', spread_width=10,
        vix=vix, chain=None,
        expiration='2026-02-25',
    )
    if trade:
        print(f"  Trade: {trade.instruction_text()}")
        print(f"  Source: {trade.data_source}")
        for leg in trade.legs:
            print(f"    {leg.side.upper()} {leg.strike:.0f} {leg.option_type} @ ${leg.price:.2f}")
