"""Greedy ROI backtest: pick the single best trade per 5-min interval.

Two modes compared side-by-side:
  STATIC:  $50K max per trade, $1M daily cap, no multiplier
  DYNAMIC: $50K base, multiplied by ROI tier (1x/2x/4x), $200K cap per trade, $1M daily cap

ROI tier multipliers:
  norm_roi < 6%  → 1x  ($50K)
  norm_roi 6-9%  → 2x  ($100K)
  norm_roi > 9%  → 4x  ($200K)

For each 5-min interval, picks the single best-ROI spread across all tickers
and DTEs that has volume, then allocates capital per the mode's rules.

Usage:
  python -m scripts.backtesting.scripts.greedy_roi_backtest \
      --tickers SPX NDX RUT --days 180
"""

import argparse
import glob
import math
import os
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

UTC_TO_PST = -8
TARGET_DTES = [0, 1, 2, 5]
DTE_FLEX = {0: [0], 1: [1], 2: [2, 3], 5: [5, 4, 6]}
PERCENTILE_BY_DTE = {0: 95, 1: 90, 2: 90, 5: 90}

# Multiplier schedule
TIER_THRESHOLDS = [(9.0, 4.0), (6.0, 2.0)]  # (min_roi%, multiplier) descending
BASE_ALLOC = 50_000
MAX_ALLOC = 200_000
DAILY_CAP = 1_000_000


def load_equity_closes(ticker, equity_dir, n):
    d = os.path.join(equity_dir, f"I:{ticker}")
    files = sorted(glob.glob(os.path.join(d, "*.csv")))
    closes = {}
    for f in files[-(n + 80):]:
        try:
            df = pd.read_csv(f, usecols=["close"])
            if not df.empty:
                dt = os.path.basename(f).split("_equities_")[1].replace(".csv", "")
                closes[dt] = float(df["close"].iloc[-1])
        except Exception:
            continue
    return closes


def pct_strikes(closes, trading_date, prev_close, lookback, dte, percentile):
    sorted_d = sorted(closes.keys())
    lb = [d for d in sorted_d if d < trading_date][-lookback:]
    if len(lb) < 20:
        return {}
    arr = np.array([closes[d] for d in lb])
    w = dte + 1
    if len(arr) <= w:
        return {}
    rets = (arr[w:] - arr[:-w]) / arr[:-w]
    up, down = rets[rets > 0], rets[rets < 0]
    r = {}
    if len(up) > 0:
        r["call"] = round(prev_close * (1 + np.percentile(up, percentile)), 2)
    if len(down) > 0:
        r["put"] = round(prev_close * (1 - np.percentile(np.abs(down), percentile)), 2)
    return r


def best_spread(opts, option_type, target_strike, spread_width, min_volume):
    side = opts[opts["type"] == option_type].copy()
    if side.empty:
        return None
    margin = spread_width + 10
    if option_type == "put":
        side = side[(side["strike"] >= target_strike - margin) & (side["strike"] <= target_strike + 5)]
    else:
        side = side[(side["strike"] >= target_strike - 5) & (side["strike"] <= target_strike + margin)]
    side = side[side["bid"].notna() & side["ask"].notna() & ((side["bid"] > 0) | (side["ask"] > 0))]
    if len(side) < 2:
        return None
    if min_volume > 0 and "volume" in side.columns:
        if side[side["volume"] >= min_volume].empty:
            return None
    side = side.sort_values("bid", ascending=False).drop_duplicates("strike", keep="first")
    strikes = sorted(side["strike"].unique())
    best = None
    for short_s in strikes:
        sr = side[side["strike"] == short_s].iloc[0]
        sb = float(sr["bid"])
        if sb <= 0:
            continue
        if option_type == "put":
            longs = [s for s in strikes if s < short_s and 5 <= short_s - s <= spread_width]
        else:
            longs = [s for s in strikes if s > short_s and 5 <= s - short_s <= spread_width]
        for long_s in longs:
            lr = side[side["strike"] == long_s].iloc[0]
            la = float(lr["ask"])
            if la <= 0:
                continue
            w = abs(short_s - long_s)
            cr = sb - la
            if cr <= 0.05 or cr >= w * 0.80:
                continue
            sv = int(sr.get("volume", 0) or 0)
            lv = int(lr.get("volume", 0) or 0)
            mlv = min(sv, lv)
            if min_volume > 0 and mlv < min_volume:
                continue
            ml = (w - cr) * 100
            if ml <= 50:
                continue
            roi = (cr / (w - cr)) * 100
            if best is None or roi > best["roi_pct"]:
                best = {"short_strike": short_s, "long_strike": long_s, "credit": cr,
                        "width": w, "max_loss_pc": ml, "min_leg_volume": mlv,
                        "option_type": option_type, "roi_pct": roi}
    return best


def get_multiplier(norm_roi):
    for threshold, mult in TIER_THRESHOLDS:
        if norm_roi >= threshold:
            return mult
    return 1.0


def scan_day(tickers_data, date_str, all_closes, lookback, spread_width, min_volume, min_credit):
    """Scan all tickers/DTEs for a single day, return list of best spreads per 5-min interval."""
    all_spreads = []

    for ticker, (closes, opts_path) in tickers_data.items():
        sorted_dates = sorted(closes.keys())
        idx = sorted_dates.index(date_str) if date_str in sorted_dates else -1
        if idx <= 0:
            continue
        prev_close = closes[sorted_dates[idx - 1]]
        day_close = closes[date_str]

        # Compute percentile strikes
        strikes_map = {}
        for td in TARGET_DTES:
            pct = PERCENTILE_BY_DTE[td]
            s = pct_strikes(closes, date_str, prev_close, lookback, td, pct)
            if s:
                strikes_map[td] = s

        if not strikes_map:
            continue

        # Load options
        try:
            opts = pd.read_csv(opts_path)
        except Exception:
            continue
        if opts.empty:
            continue

        opts["timestamp"] = pd.to_datetime(opts["timestamp"], utc=True)
        opts["bid"] = pd.to_numeric(opts["bid"], errors="coerce").fillna(0)
        opts["ask"] = pd.to_numeric(opts["ask"], errors="coerce").fillna(0)
        opts["volume"] = pd.to_numeric(opts.get("volume", 0), errors="coerce").fillna(0)
        opts["strike"] = pd.to_numeric(opts["strike"], errors="coerce")
        td_date = pd.Timestamp(date_str).date()
        if "expiration" in opts.columns:
            opts["expiration_date"] = pd.to_datetime(opts["expiration"]).dt.date
            opts["dte_raw"] = opts["expiration_date"].apply(lambda x: (x - td_date).days)
        else:
            opts["dte_raw"] = 0

        # Group by 5-min intervals
        opts["iv5"] = opts["timestamp"].dt.floor("5min")

        for iv_ts in sorted(opts["iv5"].unique()):
            iv_opts = opts[opts["iv5"] == iv_ts]
            h, m = iv_ts.hour, iv_ts.minute
            hp = (h + UTC_TO_PST) % 24
            iv_pst = f"{hp:02d}:{m:02d}"

            for target_dte, strikes in strikes_map.items():
                flex = DTE_FLEX.get(target_dte, [target_dte])
                dte_opts = iv_opts[iv_opts["dte_raw"].isin(flex)]
                if dte_opts.empty:
                    continue
                actual_dte = int(dte_opts["dte_raw"].mode().iloc[0])

                for ot in ["put", "call"]:
                    tgt = strikes.get(ot)
                    if tgt is None:
                        continue
                    sp = best_spread(dte_opts, ot, tgt, spread_width, min_volume)
                    if sp is None or sp["credit"] < min_credit:
                        continue
                    # Compute actual PnL using day_close
                    credit = sp["credit"]
                    ss = sp["short_strike"]
                    w = sp["width"]
                    if ot == "put":
                        intrinsic = max(0, ss - day_close)
                    else:
                        intrinsic = max(0, day_close - ss)
                    pnl_per_share = credit - intrinsic
                    max_loss_ps = w - credit
                    if max_loss_ps <= 0.005:
                        continue
                    roi = (pnl_per_share / max_loss_ps) * 100
                    norm_roi = roi / (actual_dte + 1)

                    all_spreads.append({
                        "date": date_str, "interval_pst": iv_pst, "iv_ts": iv_ts,
                        "ticker": ticker, "dte": target_dte, "actual_dte": actual_dte,
                        "option_type": ot, "credit": credit, "width": w,
                        "pnl_per_share": pnl_per_share, "max_loss_ps": max_loss_ps,
                        "roi_pct": roi, "norm_roi_pct": norm_roi,
                        "min_leg_volume": sp["min_leg_volume"],
                        "day_close": day_close,
                    })

    return all_spreads


def run_backtest(day_spreads_by_date, mode="static"):
    """Run greedy backtest: pick best spread per 5-min interval.

    mode="static":  $50K per trade, no multiplier
    mode="dynamic": $50K base × tier multiplier, capped at $200K
    """
    results = []
    daily_summaries = []

    for date_str in sorted(day_spreads_by_date.keys()):
        spreads = day_spreads_by_date[date_str]
        if not spreads:
            continue

        df = pd.DataFrame(spreads)
        # Sort intervals chronologically
        intervals = sorted(df["iv_ts"].unique())

        daily_used = 0.0
        daily_trades = []

        for iv_ts in intervals:
            if daily_used >= DAILY_CAP:
                break

            iv_df = df[df["iv_ts"] == iv_ts]
            # Pick the single best norm_roi spread
            best_row = iv_df.loc[iv_df["norm_roi_pct"].idxmax()]

            norm_roi = best_row["norm_roi_pct"]
            max_loss_ps = best_row["max_loss_ps"]  # per share
            pnl_ps = best_row["pnl_per_share"]
            width = best_row["width"]

            # How much to allocate
            if mode == "dynamic":
                mult = get_multiplier(norm_roi)
                alloc = min(BASE_ALLOC * mult, MAX_ALLOC)
            else:
                alloc = BASE_ALLOC

            # Don't exceed daily cap
            alloc = min(alloc, DAILY_CAP - daily_used)
            if alloc <= 0:
                break

            # How many contracts can we buy?
            risk_per_contract = max_loss_ps * 100  # max_loss per contract in $
            if risk_per_contract <= 0:
                continue
            num_contracts = max(1, int(alloc / risk_per_contract))
            actual_risk = num_contracts * risk_per_contract
            actual_pnl = num_contracts * pnl_ps * 100

            daily_used += actual_risk

            trade = {
                "date": date_str,
                "interval_pst": best_row["interval_pst"],
                "ticker": best_row["ticker"],
                "dte": best_row["dte"],
                "option_type": best_row["option_type"],
                "credit": best_row["credit"],
                "width": width,
                "norm_roi_pct": norm_roi,
                "roi_pct": best_row["roi_pct"],
                "num_contracts": num_contracts,
                "risk": actual_risk,
                "pnl": actual_pnl,
                "multiplier": get_multiplier(norm_roi) if mode == "dynamic" else 1.0,
                "alloc": alloc if mode == "dynamic" else BASE_ALLOC,
                "won": pnl_ps > 0,
                "volume": best_row["min_leg_volume"],
            }
            daily_trades.append(trade)
            results.append(trade)

        if daily_trades:
            tdf = pd.DataFrame(daily_trades)
            daily_summaries.append({
                "date": date_str,
                "trades": len(tdf),
                "pnl": tdf["pnl"].sum(),
                "risk": tdf["risk"].sum(),
                "wins": tdf["won"].sum(),
                "losses": (~tdf["won"]).sum(),
            })

    return results, daily_summaries


def print_results(results, daily_summaries, mode):
    if not results:
        print(f"  {mode}: No trades")
        return {}

    rdf = pd.DataFrame(results)
    ddf = pd.DataFrame(daily_summaries)

    total_trades = len(rdf)
    wins = rdf["won"].sum()
    losses = total_trades - wins
    wr = wins / total_trades * 100
    net_pnl = rdf["pnl"].sum()
    total_risk = rdf["risk"].sum()
    roi = (net_pnl / total_risk * 100) if total_risk > 0 else 0
    avg_pnl = rdf["pnl"].mean()

    # Daily stats
    daily_pnls = ddf["pnl"]
    sharpe = (daily_pnls.mean() / daily_pnls.std() * np.sqrt(252)) if daily_pnls.std() > 0 else 0
    max_dd = 0
    peak = 0
    cum = 0
    for p in daily_pnls:
        cum += p
        peak = max(peak, cum)
        dd = peak - cum
        max_dd = max(max_dd, dd)

    pf = abs(rdf[rdf["pnl"] > 0]["pnl"].sum() / rdf[rdf["pnl"] < 0]["pnl"].sum()) if (rdf["pnl"] < 0).any() else float("inf")

    print(f"\n  {'─'*70}")
    print(f"  {mode.upper()} MODE")
    print(f"  {'─'*70}")
    print(f"  Trades:        {total_trades:,}")
    print(f"  Wins/Losses:   {wins}/{losses}  ({wr:.1f}% win rate)")
    print(f"  Net P&L:       ${net_pnl:>12,.0f}")
    print(f"  Total Risk:    ${total_risk:>12,.0f}")
    print(f"  ROI:           {roi:>11.1f}%")
    print(f"  Avg P&L/Trade: ${avg_pnl:>12,.0f}")
    print(f"  Sharpe:        {sharpe:>11.2f}")
    print(f"  Max Drawdown:  ${max_dd:>12,.0f}")
    print(f"  Profit Factor: {pf:>11.2f}")
    print(f"  Days traded:   {len(ddf)}")
    print(f"  Avg trades/day:{total_trades/len(ddf):>11.1f}")

    # Multiplier distribution (dynamic only)
    if mode == "dynamic":
        print(f"\n  Multiplier distribution:")
        for m in sorted(rdf["multiplier"].unique()):
            mc = rdf[rdf["multiplier"] == m]
            print(f"    {m:.0f}x: {len(mc):>5} trades ({len(mc)/len(rdf)*100:.1f}%), "
                  f"P&L ${mc['pnl'].sum():>10,.0f}, WR {mc['won'].mean()*100:.0f}%")

    # Per-ticker breakdown
    print(f"\n  Per-ticker:")
    for t in sorted(rdf["ticker"].unique()):
        td = rdf[rdf["ticker"] == t]
        print(f"    {t}: {len(td):>4} trades, P&L ${td['pnl'].sum():>10,.0f}, "
              f"WR {td['won'].mean()*100:.0f}%, avg_risk ${td['risk'].mean():>7,.0f}")

    # Per-DTE breakdown
    print(f"\n  Per-DTE:")
    for d in sorted(rdf["dte"].unique()):
        dd = rdf[rdf["dte"] == d]
        print(f"    DTE={d}: {len(dd):>4} trades, P&L ${dd['pnl'].sum():>10,.0f}, "
              f"WR {dd['won'].mean()*100:.0f}%")

    return {"net_pnl": net_pnl, "trades": total_trades, "wr": wr, "sharpe": sharpe,
            "max_dd": max_dd, "pf": pf, "roi": roi}


def main():
    parser = argparse.ArgumentParser(
        description="Greedy ROI backtest: static vs dynamic allocation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --tickers SPX NDX RUT --days 180
  %(prog)s --tickers RUT --days 90
        """,
    )
    parser.add_argument("--tickers", nargs="+", default=["SPX", "NDX", "RUT"])
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--lookback", type=int, default=180)
    parser.add_argument("--spread-width", type=int, default=50)
    parser.add_argument("--min-volume", type=int, default=5)
    parser.add_argument("--min-credit", type=float, default=0.30)
    parser.add_argument("--options-dir", default="options_csv_output_full")
    parser.add_argument("--equity-dir", default="equities_output")
    parser.add_argument("--output-dir", default="results/greedy_roi_backtest")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load equity data for all tickers
    all_closes = {}
    for ticker in args.tickers:
        all_closes[ticker] = load_equity_closes(ticker, args.equity_dir, args.lookback + args.days)

    # Find common trading dates
    all_dates_sets = [set(c.keys()) for c in all_closes.values()]
    common_dates = sorted(set.intersection(*all_dates_sets))[-args.days:]
    print(f"Analyzing {len(common_dates)} trading days: {common_dates[0]} to {common_dates[-1]}")
    print(f"Tickers: {args.tickers}")
    print(f"Base allocation: ${BASE_ALLOC:,}, max per trade: ${MAX_ALLOC:,}, daily cap: ${DAILY_CAP:,}")
    print(f"Multipliers: <6% → 1x, 6-9% → 2x, >9% → 4x")

    # Scan all days
    day_spreads = {}
    for i, dt in enumerate(common_dates):
        # Build per-ticker data for this date
        tickers_data = {}
        for ticker in args.tickers:
            closes = all_closes[ticker]
            op = os.path.join(args.options_dir, ticker, f"{ticker}_options_{dt}.csv")
            if os.path.exists(op) and dt in closes:
                tickers_data[ticker] = (closes, op)

        if not tickers_data:
            continue

        spreads = scan_day(tickers_data, dt, all_closes, args.lookback,
                           args.spread_width, args.min_volume, args.min_credit)
        if spreads:
            day_spreads[dt] = spreads

        if (i + 1) % 30 == 0:
            total = sum(len(v) for v in day_spreads.values())
            print(f"  ... {i+1}/{len(common_dates)} days scanned, {total:,} candidate spreads")

    total = sum(len(v) for v in day_spreads.values())
    print(f"\nTotal: {len(day_spreads)} days with spreads, {total:,} candidate spreads")

    # Run both modes
    print(f"\n{'='*75}")
    print(f"  BACKTEST RESULTS — {common_dates[0]} to {common_dates[-1]}")
    print(f"{'='*75}")

    static_results, static_daily = run_backtest(day_spreads, mode="static")
    static_metrics = print_results(static_results, static_daily, "static")

    dynamic_results, dynamic_daily = run_backtest(day_spreads, mode="dynamic")
    dynamic_metrics = print_results(dynamic_results, dynamic_daily, "dynamic")

    # Side-by-side comparison
    print(f"\n  {'='*70}")
    print(f"  COMPARISON: STATIC vs DYNAMIC")
    print(f"  {'='*70}")
    print(f"  {'Metric':<20s}  {'STATIC':>15s}  {'DYNAMIC':>15s}  {'Delta':>12s}")
    print(f"  {'-'*20}  {'-'*15}  {'-'*15}  {'-'*12}")
    for key, label, fmt in [
        ("net_pnl", "Net P&L", "${:>12,.0f}"),
        ("trades", "Trades", "{:>12,d}"),
        ("wr", "Win Rate", "{:>11.1f}%"),
        ("sharpe", "Sharpe", "{:>12.2f}"),
        ("max_dd", "Max Drawdown", "${:>11,.0f}"),
        ("pf", "Profit Factor", "{:>12.2f}"),
        ("roi", "ROI", "{:>11.1f}%"),
    ]:
        sv = static_metrics.get(key, 0)
        dv = dynamic_metrics.get(key, 0)
        delta = dv - sv
        sf = fmt.format(sv)
        df = fmt.format(dv)
        if key in ("net_pnl", "roi", "sharpe", "pf"):
            ds = f"+{delta:,.0f}" if delta > 0 else f"{delta:,.0f}" if isinstance(delta, (int, float)) and not isinstance(delta, bool) else str(delta)
            if key == "roi":
                ds = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
            elif key in ("sharpe", "pf"):
                ds = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
        else:
            ds = f"{delta:+,}" if isinstance(delta, int) else f"{delta:+,.0f}"
        print(f"  {label:<20s}  {sf:>15s}  {df:>15s}  {ds:>12s}")

    # Save trade logs
    if static_results:
        pd.DataFrame(static_results).to_csv(os.path.join(args.output_dir, "trades_static.csv"), index=False)
    if dynamic_results:
        pd.DataFrame(dynamic_results).to_csv(os.path.join(args.output_dir, "trades_dynamic.csv"), index=False)
    if static_daily:
        pd.DataFrame(static_daily).to_csv(os.path.join(args.output_dir, "daily_static.csv"), index=False)
    if dynamic_daily:
        pd.DataFrame(dynamic_daily).to_csv(os.path.join(args.output_dir, "daily_dynamic.csv"), index=False)
    print(f"\nTrade logs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
