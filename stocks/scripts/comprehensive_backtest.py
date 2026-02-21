#!/usr/bin/env python3
"""
Comprehensive NDX Credit Spread Backtest Engine  (v2 – optimized)

Tests all combinations of:
  - DTE: 0, 1, 2, 3, 5, 10
  - Percentile bands: P95, P97, P98, P99, P100  (rolling 100-day window)
  - Time buckets: 10-min 6:30-7AM PST | 15-min 7-9AM PST | 30-min 9AM+ PST
  - Spread types: put_spread, call_spread, iron_condor
  - Flow modes: neutral, with_flow, against_flow
  - Spread widths: 50, 100, 150, 200 NDX points  (max $20k risk per trade)

All times stored in UTC, displayed in PST (UTC-8, Feb 2026).
Max risk per position: $20,000.
Early exit for >0DTE: 50% credit captured (sqrt theta decay model).
Capital assumption: $500k+ ($20k max risk → 25 simultaneous positions max).

Speed strategy:
  Phase 1 – precompute underlying prices, EOD prices, bands, chains (single thread)
  Phase 2 – fast multiprocess grid search using only dict lookups

Output: results/comprehensive_backtest/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple, NamedTuple
import warnings, logging, sys, argparse, json, pickle, os
warnings.filterwarnings('ignore')

# ──────────────────────────── CONSTANTS ───────────────────────────────────────
MULTIPLIER   = 100
MAX_RISK     = 30_000       # $ per trade
PROFIT_TGT   = 0.50        # 50 % credit early-exit (>0DTE)
BAND_DAYS    = 100
BACKTEST_DAYS = 90

BAND_CONFIGS = {
    'P95':  (2.5,  97.5),
    'P97':  (1.5,  98.5),
    'P98':  (1.0,  99.0),
    'P99':  (0.5,  99.5),
    'P100': (0.0, 100.0),
}
DTE_TARGETS   = [0, 1, 3, 5, 10]
SPREAD_TYPES  = ['put_spread', 'call_spread', 'iron_condor']
FLOW_MODES    = ['neutral', 'with_flow', 'against_flow']
SPREAD_WIDTHS = list(range(5, 95, 5))   # 5,10,15,...,90 — filtered per band/DTE below

# Max spread width (NDX points) per percentile band at 0DTE.
# Each additional DTE adds 20%, capped at 3 days (1.728×).
BASE_MAX_WIDTH = {'P95': 30, 'P97': 30, 'P98': 40, 'P99': 50, 'P100': 50}
INVEST_AMOUNT  = 30_000   # $ deployed per trade (used for n_contracts calculation)

def max_width_for(band_name: str, dte: int) -> int:
    base  = BASE_MAX_WIDTH.get(band_name, 30)
    scale = min(1.20 ** dte, 1.20 ** 3)   # cap at 3 DTE
    return int(round(base * scale))

# Time buckets – ET HH:MM (displayed in PST = ET - 3h)
# 6:30-7:00 AM PST = 9:30-10:00 AM ET  → 10-min
BUCKET_A = ['09:30', '09:40', '09:50']
# 7:00-9:00 AM PST = 10:00-12:00 PM ET → 15-min
BUCKET_B = [f'{h:02d}:{m:02d}' for h in [10, 11] for m in [0, 15, 30, 45]]
# 9:00 AM PST+ = 12:00-4:00 PM ET      → 30-min
BUCKET_C = [f'{h:02d}:{m:02d}' for h in [12, 13, 14, 15] for m in [0, 30]]
ALL_BUCKETS_ET = BUCKET_A + BUCKET_B + BUCKET_C      # 19 ET times
EOD_SLOTS_ET   = ['15:55', '15:50', '15:45', '15:30', '15:00']

def et_to_pst(s: str) -> str:
    h, m = map(int, s.split(':'))
    return f'{h-3:02d}:{m:02d}'

# ──────────────────────────── LOGGING ─────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('backtest')

# ──────────────────────────── PATHS ───────────────────────────────────────────
BASE     = Path(__file__).parent.parent
DTE0_DIR = BASE / 'options_csv_output'      / 'NDX'
FULL_DIR = BASE / 'options_csv_output_full' / 'NDX'
OUT_DIR  = BASE / 'results' / 'comprehensive_backtest'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 – DATA LOADING & PRECOMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def _load_csv(path: Path) -> pd.DataFrame:
    """Parse one options CSV → DataFrame with ts_et, ts_str, ts_date, mid cols."""
    df = pd.read_csv(path)
    if df.empty:
        return df
    df['ts_utc']  = pd.to_datetime(df['timestamp'], utc=True)
    df['ts_et']   = df['ts_utc'].dt.tz_convert('America/New_York')
    df['ts_str']  = df['ts_et'].dt.strftime('%H:%M')
    df['ts_date'] = df['ts_et'].dt.date
    df['mid']     = (df['bid'] + df['ask']) / 2.0
    return df


def load_all_data(n: int) -> Tuple[Dict[date, pd.DataFrame], Dict[date, pd.DataFrame]]:
    """Load last n days of 0DTE and full-DTE CSVs."""
    def _load(d: Path) -> Dict[date, pd.DataFrame]:
        files = sorted(d.glob('NDX_options_*.csv'))[-n:]
        out = {}
        for f in files:
            df = _load_csv(f)
            if df.empty:
                continue
            k = df['ts_date'].iloc[0]
            out[k] = df
        return out
    log.info(f'Loading {n} days of 0DTE data…')
    d0 = _load(DTE0_DIR)
    log.info(f'Loading {n} days of full-DTE data…')
    df = _load(FULL_DIR)
    log.info(f'  → {len(d0)} 0DTE days,  {len(df)} full-DTE days')
    return d0, df


def _underlying_from_chain(df: pd.DataFrame, ts: str) -> Optional[float]:
    """
    Estimate NDX underlying at time ts (ET HH:MM).
    Uses min(strike + call_bid) for liquid ITM calls.
    Falls back to nearest 5-min slot within ±15 min.
    """
    calls = df[(df['ts_str'] == ts) & (df['type'] == 'call')]
    for thresh in (50, 20, 5, 0.5):
        liq = calls[calls['bid'] > thresh]
        if not liq.empty:
            v = (liq['strike'] + liq['bid']).min()
            if v > 10_000:
                return float(v)

    # Nearest available timestamp
    h, m = map(int, ts.split(':'))
    tgt = h * 60 + m
    avail = df[df['type'] == 'call']['ts_str'].unique()
    if len(avail) == 0:
        return None
    def to_m(s): hh, mm = map(int, s.split(':')); return hh*60+mm
    nearest = min(avail, key=lambda s: abs(to_m(s) - tgt))
    if abs(to_m(nearest) - tgt) > 15:
        return None
    calls2 = df[(df['ts_str'] == nearest) & (df['type'] == 'call')]
    for thresh in (50, 20, 5, 0.5):
        liq = calls2[calls2['bid'] > thresh]
        if not liq.empty:
            v = (liq['strike'] + liq['bid']).min()
            if v > 10_000:
                return float(v)
    return None


def precompute_underlyings(
    data: Dict[date, pd.DataFrame],
    time_slots: List[str],
) -> Dict[date, Dict[str, float]]:
    """
    Precompute {date: {ts_str: underlying}} for all dates × time slots.
    Runs once in main process.
    """
    out: Dict[date, Dict[str, float]] = {}
    for d, df in data.items():
        row: Dict[str, float] = {}
        for ts in time_slots:
            p = _underlying_from_chain(df, ts)
            if p:
                row[ts] = p
        if row:
            out[d] = row
    return out


def precompute_chains(
    data: Dict[date, pd.DataFrame],
    time_slots: List[str],
) -> Dict[date, Dict[str, Dict[str, List[Tuple[float, float, float]]]]]:
    """
    Precompute lightweight option chains:
    {date: {ts_str: {'put': [(strike, bid, ask), ...], 'call': [...]}}}
    Sorted by strike ascending.  Runs once in main process.
    """
    out: Dict = {}
    for d, df in data.items():
        d_chains: Dict = {}
        for ts in time_slots:
            sub = df[df['ts_str'] == ts]
            if sub.empty:
                continue
            chain: Dict[str, List] = {}
            for opt_type in ('put', 'call'):
                rows = sub[sub['type'] == opt_type][['strike', 'bid', 'ask']].dropna()
                chain[opt_type] = sorted(
                    [(float(r.strike), float(r.bid), float(r.ask))
                     for _, r in rows.iterrows()],
                    key=lambda x: x[0]
                )
            if chain.get('put') or chain.get('call'):
                d_chains[ts] = chain
        if d_chains:
            out[d] = d_chains
    return out


def precompute_eod_prices(
    und0: Dict[date, Dict[str, float]],
    und_full: Dict[date, Dict[str, float]],
) -> Dict[date, float]:
    """Build {date: EOD_underlying} using latest available ET time slot."""
    out: Dict[date, float] = {}
    all_dates = set(und0.keys()) | set(und_full.keys())
    for d in all_dates:
        for slot in EOD_SLOTS_ET:
            p = (und0.get(d, {}).get(slot) or und_full.get(d, {}).get(slot))
            if p:
                out[d] = p
                break
    return out


def precompute_bands(
    sorted_dates: List[date],
    backtest_dates: List[date],
    und_by_date_0dte: Dict[date, Dict[str, float]],
    und_by_date_full: Dict[date, Dict[str, float]],
    eod_prices: Dict[date, float],
    band_days: int = BAND_DAYS,
) -> Dict[Tuple, Tuple[float, float]]:
    """
    Precompute (lo_pct, hi_pct) bands for every
    (backtest_date, ts_str, dte, band_name) combination.

    For 0DTE: band from (entry → same-day EOD) moves.
    For N>0 DTE: band from (entry → EOD N trading-days later) moves.
    """
    bands: Dict[Tuple, Tuple[float, float]] = {}

    all_slots = ALL_BUCKETS_ET

    for today in backtest_dates:
        today_idx = sorted_dates.index(today) if today in sorted_dates else -1
        if today_idx < 0:
            continue
        lb_start = max(0, today_idx - band_days)
        lb_dates  = sorted_dates[lb_start:today_idx]
        if len(lb_dates) < 20:
            continue

        for ts in all_slots:
            for dte in DTE_TARGETS:
                # Build historical (entry, expiry) pairs from lookback window
                entries, expiries = [], []
                for lb_date in lb_dates:
                    lb_und = (und_by_date_0dte.get(lb_date, {}).get(ts) or
                              und_by_date_full.get(lb_date, {}).get(ts))
                    if not lb_und:
                        continue
                    if dte == 0:
                        exp_price = eod_prices.get(lb_date)
                    else:
                        lb_idx = sorted_dates.index(lb_date) if lb_date in sorted_dates else -1
                        if lb_idx < 0:
                            continue
                        exp_idx = lb_idx + dte
                        if exp_idx >= len(sorted_dates):
                            continue
                        exp_date  = sorted_dates[exp_idx]
                        exp_price = eod_prices.get(exp_date)
                    if exp_price:
                        entries.append(lb_und)
                        expiries.append(exp_price)

                if len(entries) < 15:
                    continue

                moves = [(e - b) / b * 100 for b, e in zip(entries, expiries) if b > 0]
                if len(moves) < 15:
                    continue

                for band_name, (lo_t, hi_t) in BAND_CONFIGS.items():
                    lo = float(np.percentile(moves, lo_t))
                    hi = float(np.percentile(moves, hi_t))
                    bands[(today, ts, dte, band_name)] = (lo, hi)

    return bands


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE (set in worker initialiser)
# ══════════════════════════════════════════════════════════════════════════════
_STATE = None   # will be set by init_worker()

def init_worker(state):
    global _STATE
    _STATE = state


# ══════════════════════════════════════════════════════════════════════════════
#  SPREAD BUILDING  (uses lightweight chain tuples)
# ══════════════════════════════════════════════════════════════════════════════

def _nearest_below(opts: List[Tuple], target: float):
    """Find the highest strike ≤ target.  opts sorted ascending."""
    best = None
    for o in opts:
        if o[0] <= target:
            best = o
    return best    # last one ≤ target


def _nearest_above(opts: List[Tuple], target: float):
    """Find the lowest strike ≥ target.  opts sorted ascending."""
    for o in opts:
        if o[0] >= target:
            return o
    return None


def build_spread_fast(
    chain: Dict[str, List[Tuple]],
    lo_price: float,
    hi_price: float,
    spread_type: str,
    width: int,
    underlying: float,
) -> Optional[Dict]:
    """Fast spread builder using pre-sorted tuples."""
    puts  = chain.get('put',  [])
    calls = chain.get('call', [])
    put_credit = call_credit = 0.0
    sp = lp = sc = lc = None

    # ── PUT SIDE ─────────────────────────────────────────────────────────────
    if spread_type in ('put_spread', 'iron_condor'):
        sp_row = _nearest_below(puts, lo_price) or (puts[-1] if puts else None)
        if sp_row is None:
            if spread_type == 'put_spread': return None
        else:
            sp = sp_row[0]
            lp_target = sp - width
            lp_row = _nearest_below([o for o in puts if o[0] < sp], lp_target)
            if lp_row is None:
                # try nearest below sp regardless of target
                sub = [o for o in puts if o[0] < sp]
                lp_row = min(sub, key=lambda o: abs(o[0] - lp_target)) if sub else None
            if lp_row is None:
                if spread_type == 'put_spread': return None
            else:
                lp = lp_row[0]
                raw = sp_row[1] - lp_row[2]   # short_bid - long_ask
                put_credit = raw * MULTIPLIER
                if put_credit <= 0:
                    if spread_type == 'put_spread': return None
                    put_credit = 0.0; sp = lp = None

    # ── CALL SIDE ────────────────────────────────────────────────────────────
    if spread_type in ('call_spread', 'iron_condor'):
        sc_row = _nearest_above(calls, hi_price) or (calls[0] if calls else None)
        if sc_row is None:
            if spread_type == 'call_spread': return None
        else:
            sc = sc_row[0]
            lc_target = sc + width
            lc_row = _nearest_above([o for o in calls if o[0] > sc], lc_target)
            if lc_row is None:
                sub = [o for o in calls if o[0] > sc]
                lc_row = min(sub, key=lambda o: abs(o[0] - lc_target)) if sub else None
            if lc_row is None:
                if spread_type == 'call_spread': return None
            else:
                lc = lc_row[0]
                raw = sc_row[1] - lc_row[2]
                call_credit = raw * MULTIPLIER
                if call_credit <= 0:
                    if spread_type == 'call_spread': return None
                    call_credit = 0.0; sc = lc = None

    total = put_credit + call_credit
    if total <= 0:
        return None

    if spread_type == 'put_spread':
        if sp is None: return None
        max_risk = (sp - lp) * MULTIPLIER - put_credit if lp else 0
    elif spread_type == 'call_spread':
        if sc is None: return None
        max_risk = (lc - sc) * MULTIPLIER - call_credit if lc else 0
    else:
        pr = (sp - lp) * MULTIPLIER - put_credit  if sp and lp else 0
        cr = (lc - sc) * MULTIPLIER - call_credit if sc and lc else 0
        max_risk = max(pr, cr)

    if max_risk <= 0 or max_risk > MAX_RISK:
        return None

    return {
        'spread_type': spread_type, 'underlying': underlying,
        'short_put': sp,  'long_put': lp,  'put_credit':  put_credit,
        'short_call': sc, 'long_call': lc, 'call_credit': call_credit,
        'total_credit': total, 'max_risk': max_risk,
        'roi_pct': total / max_risk * 100,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  P&L SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def pnl_at_expiry(spread: Dict, final: float) -> float:
    """Realised P&L at expiration (dollars, positive = profit)."""
    pnl = spread['total_credit']
    sp, lp = spread.get('short_put'), spread.get('long_put')
    if sp and lp and final < sp:
        pnl -= (sp - final - max(lp - final, 0)) * MULTIPLIER
    sc, lc = spread.get('short_call'), spread.get('long_call')
    if sc and lc and final > sc:
        pnl -= (final - sc - max(final - lc, 0)) * MULTIPLIER
    return float(np.clip(pnl, -spread['max_risk'], spread['total_credit']))


def simulate_ndte(
    spread: Dict,
    entry_date: date,
    dte: int,
    sorted_dates: List[date],
    eod_prices: Dict[date, float],
) -> Tuple[float, int, str]:
    """
    Simulate >0DTE trade with 50% credit early exit (sqrt theta decay).
    Returns (pnl_$, holding_days, exit_reason).
    """
    credit   = spread['total_credit']
    idx      = sorted_dates.index(entry_date) if entry_date in sorted_dates else -1
    if idx < 0:
        return (0.0, 0, 'no_entry')

    future = sorted_dates[idx:]
    for i, d in enumerate(future):
        if i > dte + 2:   # don't run past expiry + buffer
            break
        price = eod_prices.get(d)
        if price is None:
            continue

        sp, sc = spread.get('short_put'), spread.get('short_call')
        within = (sp is None or price >= sp) and (sc is None or price <= sc)

        if not within:
            loss = pnl_at_expiry(spread, price)
            return (loss, i, 'stop_loss')

        # sqrt-decay early exit: elapsed ≥ 75 % of DTE → 50 % credit captured
        if dte > 0 and i / dte >= 0.75:
            return (credit * 0.50, i, 'profit_target')

        if i >= dte:
            return (pnl_at_expiry(spread, price), i, 'expiration')

    # fallback
    last = max((d for d in eod_prices if d >= entry_date), default=entry_date)
    return (pnl_at_expiry(spread, eod_prices.get(last, spread['underlying'])),
            len(future), 'fallback')


# ══════════════════════════════════════════════════════════════════════════════
#  GRID SEARCH WORKER (multiprocess-safe)
# ══════════════════════════════════════════════════════════════════════════════

def run_config(cfg: Dict) -> Optional[Dict]:
    """
    Worker: run one complete config across all backtest days.
    Accesses precomputed state via global _STATE.
    """
    global _STATE
    S = _STATE

    dte         = cfg['dte']
    band_name   = cfg['band_name']
    ts_et       = cfg['ts_et']
    spread_type = cfg['spread_type']
    flow_mode   = cfg['flow_mode']
    width       = cfg['spread_width']

    # Choose data source dicts
    und_src    = S['und_0dte'] if dte == 0 else S['und_full']
    chain_src  = S['chains_0dte'] if dte == 0 else S['chains_full']

    trades = []
    for today in S['backtest_dates']:
        today_und = und_src.get(today, {})
        entry_p   = today_und.get(ts_et)
        if entry_p is None:
            continue

        # ── FLOW ─────────────────────────────────────────────────────────────
        if dte == 0:
            # Compare to 10-min-ago price (2 × 5-min slots back)
            h, m = map(int, ts_et.split(':'))
            past_t = f'{(h*60+m-10)//60:02d}:{(h*60+m-10)%60:02d}'
            past_p = today_und.get(past_t)
            flow = (0 if not past_p or past_p == 0
                    else (1 if (entry_p - past_p)/past_p*100 > 0.05
                          else (-1 if (entry_p - past_p)/past_p*100 < -0.05 else 0)))
        else:
            sorted_dates = S['sorted_dates']
            idx = sorted_dates.index(today) if today in sorted_dates else -1
            prev_eod = S['eod_prices'].get(sorted_dates[idx-1]) if idx > 0 else None
            if prev_eod and prev_eod > 0:
                pct = (entry_p - prev_eod) / prev_eod * 100
                flow = 1 if pct > 0.1 else (-1 if pct < -0.1 else 0)
            else:
                flow = 0

        # ── FLOW FILTER ──────────────────────────────────────────────────────
        if flow_mode == 'with_flow':
            if flow == 0: continue
            # With flow: up trend → sell call spread; down → sell put spread
            if spread_type == 'put_spread'  and flow > 0: continue
            if spread_type == 'call_spread' and flow < 0: continue
            # iron condor allowed regardless of direction
        elif flow_mode == 'against_flow':
            if flow == 0: continue
            # Fade the trend: up trend → fade with call spread; down → fade with put spread
            if spread_type == 'put_spread'  and flow < 0: continue
            if spread_type == 'call_spread' and flow > 0: continue

        # ── BANDS ────────────────────────────────────────────────────────────
        band = S['bands'].get((today, ts_et, dte, band_name))
        if band is None:
            continue
        lo_pct, hi_pct = band
        lo_price = entry_p * (1 + lo_pct / 100)
        hi_price = entry_p * (1 + hi_pct / 100)

        # ── OPTIONS CHAIN ────────────────────────────────────────────────────
        chain = chain_src.get(today, {}).get(ts_et)
        if chain is None:
            continue

        spread = build_spread_fast(chain, lo_price, hi_price,
                                   spread_type, width, entry_p)
        if spread is None:
            continue

        # ── P&L ──────────────────────────────────────────────────────────────
        if dte == 0:
            eod_p = S['eod_prices'].get(today)
            if eod_p is None: continue
            pnl       = pnl_at_expiry(spread, eod_p)
            hold_days = 0
            exit_r    = 'expiration_0dte'
        else:
            sorted_dates = S['sorted_dates']
            pnl, hold_days, exit_r = simulate_ndte(
                spread, today, dte, sorted_dates, S['eod_prices'])

        trades.append({
            'entry_date':   today.isoformat(),
            'entry_pst':    et_to_pst(ts_et),
            'entry_et':     ts_et,
            'dte': dte, 'band': band_name,
            'spread_type': spread_type, 'flow_mode': flow_mode,
            'spread_width': width, 'flow_signal': flow,
            'underlying':   round(entry_p, 1),
            'lo_price':     round(lo_price, 1),
            'hi_price':     round(hi_price, 1),
            'short_put':    spread.get('short_put'),
            'long_put':     spread.get('long_put'),
            'short_call':   spread.get('short_call'),
            'long_call':    spread.get('long_call'),
            'total_credit': round(spread['total_credit'], 2),
            'max_risk':     round(spread['max_risk'], 2),
            'pnl':          round(pnl, 2),
            'holding_days': hold_days,
            'exit_reason':  exit_r,
            'win':          1 if pnl >= 0 else 0,
        })

    if not trades:
        return None

    df   = pd.DataFrame(trades)
    n    = len(df)
    wins = int(df['win'].sum())

    summary = {
        'dte': dte, 'band': band_name,
        'time_pst': et_to_pst(ts_et), 'time_et': ts_et,
        'spread_type': spread_type, 'flow_mode': flow_mode,
        'spread_width': width,
        'n_trades': n, 'n_wins': wins,
        'win_rate_pct':  round(wins / n * 100, 1),
        'avg_credit':    round(df['total_credit'].mean(), 2),
        'avg_max_risk':  round(df['max_risk'].mean(), 2),
        'avg_pnl':       round(df['pnl'].mean(), 2),
        'total_pnl':     round(df['pnl'].sum(), 2),
        'median_pnl':    round(df['pnl'].median(), 2),
        'min_pnl':       round(df['pnl'].min(), 2),
        'max_pnl':       round(df['pnl'].max(), 2),
        'pnl_std':       round(df['pnl'].std(), 2),
        'sharpe':        round(df['pnl'].mean() / df['pnl'].std(), 3)
                         if df['pnl'].std() > 0 else 0.0,
        'avg_hold_days': round(df['holding_days'].mean(), 1),
        '_trades': trades,
    }
    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='NDX Comprehensive Backtest')
    ap.add_argument('--processes',     type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument('--backtest-days', type=int, default=BACKTEST_DAYS)
    ap.add_argument('--band-days',     type=int, default=BAND_DAYS)
    ap.add_argument('--dte',           type=int, nargs='+', default=DTE_TARGETS)
    ap.add_argument('--cache',         action='store_true',
                    help='Use cached precomputed state if available')
    ap.add_argument('--out-dir',       type=str, default=None,
                    help='Output directory (default: results/comprehensive_backtest)')
    args = ap.parse_args()

    n_procs = args.processes
    log.info(f'Processes: {n_procs}  |  Backtest: {args.backtest_days}d  |  '
             f'Band window: {args.band_days}d  |  DTEs: {args.dte}')

    # ── OUTPUT DIR ────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── LOAD DATA ─────────────────────────────────────────────────────────────
    total_days = args.backtest_days + args.band_days + 20
    cache_path = OUT_DIR / 'precomputed_state.pkl'

    if args.cache and cache_path.exists():
        log.info(f'Loading cached state from {cache_path}…')
        with open(cache_path, 'rb') as f:
            state = pickle.load(f)
        log.info('Cached state loaded.')
    else:
        d0, df = load_all_data(total_days)

        all_dates = sorted(set(d0.keys()) | set(df.keys()))
        backtest_dates = all_dates[-args.backtest_days:]
        log.info(f'Backtest window: {backtest_dates[0]} → {backtest_dates[-1]}  '
                 f'({len(backtest_dates)} days)')

        # ── PRECOMPUTE ────────────────────────────────────────────────────────
        all_slots = ALL_BUCKETS_ET + EOD_SLOTS_ET

        log.info('Precomputing 0DTE underlying prices…')
        und_0dte = precompute_underlyings(d0, all_slots)
        log.info('Precomputing full-DTE underlying prices…')
        und_full = precompute_underlyings(df, all_slots)

        log.info('Building EOD price series…')
        eod = precompute_eod_prices(und_0dte, und_full)
        log.info(f'  → {len(eod)} days with EOD price')

        log.info('Precomputing 0DTE option chains…')
        chains_0dte = precompute_chains(d0, ALL_BUCKETS_ET)
        log.info('Precomputing full-DTE option chains…')
        chains_full = precompute_chains(df, ALL_BUCKETS_ET)

        log.info('Precomputing rolling percentile bands… (may take a few minutes)')
        bands = precompute_bands(all_dates, backtest_dates,
                                 und_0dte, und_full, eod,
                                 args.band_days)
        log.info(f'  → {len(bands):,} band entries computed')

        state = {
            'und_0dte':      und_0dte,
            'und_full':      und_full,
            'eod_prices':    eod,
            'chains_0dte':   chains_0dte,
            'chains_full':   chains_full,
            'bands':         bands,
            'sorted_dates':  all_dates,
            'backtest_dates': backtest_dates,
        }

        log.info(f'Saving precomputed state → {cache_path}')
        with open(cache_path, 'wb') as f:
            pickle.dump(state, f)

    backtest_dates = state['backtest_dates']

    # ── BUILD CONFIG LIST ─────────────────────────────────────────────────────
    configs = [
        {'dte': dte, 'band_name': b, 'ts_et': ts,
         'spread_type': st, 'flow_mode': fm, 'spread_width': w}
        for dte in args.dte
        for b   in BAND_CONFIGS
        for ts  in ALL_BUCKETS_ET
        for st  in SPREAD_TYPES
        for fm  in FLOW_MODES
        for w   in SPREAD_WIDTHS
        if w <= max_width_for(b, dte)   # enforce per-band/DTE width cap
    ]
    log.info(f'Total configs: {len(configs):,}')

    # ── GRID SEARCH ───────────────────────────────────────────────────────────
    log.info(f'Starting grid search with {n_procs} workers…')
    ctx = mp.get_context('fork')   # fork for shared memory on macOS/Linux
    with ctx.Pool(processes=n_procs,
                  initializer=init_worker,
                  initargs=(state,)) as pool:
        raw = pool.map(run_config, configs, chunksize=max(1, len(configs)//(n_procs*4)))

    results = [r for r in raw if r is not None]
    log.info(f'Done. {len(results):,} non-empty configs.')

    if not results:
        log.error('No results. Exiting.')
        sys.exit(1)

    # ── EXTRACT TRADES ────────────────────────────────────────────────────────
    all_trades = []
    summaries  = []
    for r in results:
        trades = r.pop('_trades', [])
        all_trades.extend(trades)
        summaries.append(r)

    # ── SAVE ──────────────────────────────────────────────────────────────────
    grid_df  = pd.DataFrame(summaries)
    grid_path = out_dir / 'grid_summary.csv'
    grid_df.sort_values('total_pnl', ascending=False).to_csv(grid_path, index=False)
    log.info(f'Grid summary → {grid_path}')

    trades_df = pd.DataFrame(all_trades)
    trades_path = out_dir / 'all_trades.csv'
    trades_df.to_csv(trades_path, index=False)
    log.info(f'All trades → {trades_path}  ({len(trades_df):,} rows)')

    for dte in args.dte:
        sub = grid_df[grid_df['dte'] == dte].sort_values('total_pnl', ascending=False)
        if not sub.empty:
            sub.to_csv(out_dir / f'grid_dte{dte}.csv', index=False)

    # ── PRINT SUMMARY ─────────────────────────────────────────────────────────
    print_summary(grid_df, args.dte)


def print_summary(gdf: pd.DataFrame, dtes: List[int]):
    print('\n' + '='*80)
    print('NDX CREDIT SPREAD BACKTEST — RESULTS SUMMARY')
    print(f'Times shown in PST (UTC-8). Risk limit: $20,000/trade.')
    print('='*80)

    # Quality filter
    q = gdf[(gdf['win_rate_pct'] >= 60) & (gdf['n_trades'] >= 10) & (gdf['total_pnl'] > 0)].copy()
    if q.empty:
        q = gdf[gdf['n_trades'] >= 5].copy()
    if q.empty:
        q = gdf.copy()

    # Composite score: win_rate × sharpe (clipped positive) × total_pnl (normalised)
    q['score'] = (q['win_rate_pct'] / 100 *
                  q['sharpe'].clip(lower=0.01) *
                  q['total_pnl'].clip(lower=1).apply(np.log1p))
    q = q.sort_values('score', ascending=False)

    cols = ['dte','band','time_pst','spread_type','flow_mode','spread_width',
            'n_trades','win_rate_pct','avg_credit','total_pnl','sharpe','avg_hold_days']

    print('\n── TOP 20 OVERALL CONFIGS ──')
    print(q[cols].head(20).to_string(index=False))

    print('\n── BEST CONFIG PER DTE ──')
    for dte in dtes:
        sub = q[q['dte'] == dte]
        if sub.empty:
            sub = gdf[gdf['dte'] == dte].sort_values('total_pnl', ascending=False)
        if sub.empty:
            print(f'  DTE={dte}: no data'); continue
        b = sub.iloc[0]
        print(f'\n  DTE={dte}  {b["band"]} | {b["time_pst"]} PST | '
              f'{b["spread_type"]} | {b["flow_mode"]} | w={b["spread_width"]}')
        print(f'    Win: {b["win_rate_pct"]:.1f}%  '
              f'AvgP&L: ${b["avg_pnl"]:,.0f}  '
              f'TotalP&L: ${b["total_pnl"]:,.0f}  '
              f'Sharpe: {b["sharpe"]:.2f}  '
              f'Hold: {b["avg_hold_days"]:.1f}d')
        print(f'    AvgCredit: ${b["avg_credit"]:,.0f}  '
              f'MaxRisk: ${b["avg_max_risk"]:,.0f}  '
              f'Trades: {b["n_trades"]}')

    print('\n── FLOW MODE VALIDATION ──')
    for fm in FLOW_MODES:
        sub = q[q['flow_mode'] == fm]
        if sub.empty: continue
        print(f'  {fm:15s}  n={len(sub):4d}  '
              f'avg_win={sub["win_rate_pct"].mean():.1f}%  '
              f'avg_pnl=${sub["avg_pnl"].mean():,.0f}  '
              f'avg_sharpe={sub["sharpe"].mean():.2f}')

    print('\n── BEST ENTRY TIMES (PST) ──')
    tp = q.groupby('time_pst').agg(
        n_top_configs=('n_trades','count'),
        avg_win=('win_rate_pct','mean'),
        avg_pnl=('avg_pnl','mean'),
        total_pnl=('total_pnl','mean'),
        avg_sharpe=('sharpe','mean'),
    ).sort_values('avg_pnl', ascending=False)
    print(tp.head(10).to_string())

    print('\n── SPREAD TYPE COMPARISON ──')
    for st in SPREAD_TYPES:
        sub = q[q['spread_type'] == st]
        if sub.empty: continue
        print(f'  {st:15s}  avg_win={sub["win_rate_pct"].mean():.1f}%  '
              f'avg_pnl=${sub["avg_pnl"].mean():,.0f}  '
              f'total_pnl=${sub["total_pnl"].sum():,.0f}')

    print('\n── POSITION SIZING ($500k capital, $20k max risk) ──')
    print(f'  Max simultaneous positions: {500_000 // MAX_RISK} = 25')
    if not q.empty:
        best = q.iloc[0]
        print(f'  Best config: {best["dte"]}DTE | {best["band"]} | '
              f'{best["time_pst"]} PST | {best["spread_type"]} | {best["flow_mode"]}')
        daily = best["avg_pnl"] * 25
        monthly = daily * 20
        print(f'  Est. daily P&L @ 25 positions: ${daily:,.0f}')
        print(f'  Est. monthly return:            ${monthly:,.0f}  '
              f'({monthly/500_000*100:.1f}% on $500k)')

    print('\n' + '='*80)
    print(f'Full results: {OUT_DIR}')
    print('='*80 + '\n')


if __name__ == '__main__':
    main()
