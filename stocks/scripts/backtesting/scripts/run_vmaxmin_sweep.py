"""VMaxMin.1 Parameter Sweep — find optimal configuration.

Layered sweep testing depth, width, stop loss, sizing, and roll mode.

Usage:
  python -m scripts.backtesting.scripts.run_vmaxmin_sweep --tickers SPX --lookback-days 30

  python -m scripts.backtesting.scripts.run_vmaxmin_sweep --tickers SPX --lookback-days 30 \\
      --layer 1    # depth + width only

  python -m scripts.backtesting.scripts.run_vmaxmin_sweep --tickers SPX --lookback-days 30 \\
      --layer all  # run all layers sequentially
"""

import argparse
import os
import sys
from datetime import date
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, BASE_DIR)

NUM_WORKERS = min(8, cpu_count())


def run_single_config(args):
    """Execute one VMaxMin config in a subprocess. Returns (label, metrics)."""
    label, config_dict, ticker, lookback_days, equity_dir, options_dir, dte1_dir = args

    import sys as _sys
    _sys.path.insert(0, BASE_DIR)
    from scripts.backtesting.scripts.vmaxmin_engine import (
        VMaxMinEngine, TICKER_START_DATES, DayResult,
        get_trading_dates, load_equity_bars_df, load_0dte_options,
        get_prev_close, _time_to_mins,
    )

    engine = VMaxMinEngine(config_dict)
    today_str = date.today().isoformat()
    start_date = TICKER_START_DATES.get(ticker, "2026-02-15")

    all_dates = get_trading_dates(ticker, equity_dir, start_date, today_str)
    if len(all_dates) < 2:
        return (label, None)

    eval_dates = all_dates[-lookback_days:] if len(all_dates) > lookback_days else all_dates

    results = []
    for dt in eval_dates:
        eq_df = load_equity_bars_df(ticker, dt, equity_dir)
        eq_prices = {}
        if not eq_df.empty:
            for _, row in eq_df.iterrows():
                eq_prices[row["time_pacific"]] = float(row["close"])
        opts = load_0dte_options(ticker, dt, options_dir)
        prev = get_prev_close(ticker, dt, all_dates, equity_dir)
        r = engine.run_single_day(ticker, dt, eq_df, eq_prices, opts, all_dates, prev)
        results.append(r)

    # Compute metrics
    valid = [r for r in results if not r.failure_reason]
    if not valid:
        return (label, None)

    pnls = [r.net_pnl for r in valid]
    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / len(pnls) * 100

    daily_std = np.std(pnls) if len(pnls) > 1 else 0
    daily_mean = np.mean(pnls)
    sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0

    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    max_dd = float((cumulative - running_max).min())

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_credits = sum(r.total_credits for r in valid)
    total_debits = sum(r.total_debits for r in valid)

    # Avg credit as % of width
    entry_trades = []
    for r in valid:
        for t in r.trades:
            if t.event == "entry" and t.width > 0 and t.num_contracts > 0:
                cr_per_share = t.credit_or_debit / (t.num_contracts * 100)
                entry_trades.append(cr_per_share / t.width * 100)
    avg_credit_pct = np.mean(entry_trades) if entry_trades else 0

    avg_contracts = np.mean([
        t.num_contracts for r in valid for t in r.trades if t.event == "entry"
    ]) if valid else 0

    # Avg width
    widths = [t.width for r in valid for t in r.trades if t.event == "entry"]
    avg_width = np.mean(widths) if widths else 0

    dte1_days = sum(1 for r in valid if r.eod_rolled_to_dte1)
    sl_days = sum(1 for r in valid if r.stopped_out)

    return (label, {
        "label": label,
        "trading_days": len(valid),
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "avg_daily_pnl": daily_mean,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "total_credits": total_credits,
        "total_debits": total_debits,
        "avg_credit_pct": avg_credit_pct,
        "avg_contracts": avg_contracts,
        "avg_width": avg_width,
        "dte1_roll_pct": dte1_days / len(valid) * 100,
        "stop_loss_pct": sl_days / len(valid) * 100,
    })


def build_layer1_configs(ticker, equity_dir, options_dir, dte1_dir):
    """Layer 1: leg_placement × depth_pct × min_spread_width."""
    combos = []
    for lp in ["otm", "itm"]:
        for depth in [None, 0.005, 0.01, 0.015, 0.02, 0.03]:
            for msw in [None, 25, 50]:
                d_label = f"d{depth}" if depth else "d0"
                w_label = f"w{msw}" if msw else "wmin"
                label = f"L1_{lp}_{d_label}_{w_label}"
                config = {
                    "leg_placement": lp,
                    "depth_pct": depth,
                    "min_spread_width": msw,
                    "roll_mode": "none",
                    "stop_loss_mode": None,
                    "sizing_mode": "budget",
                    "equity_dir": equity_dir,
                    "options_0dte_dir": options_dir,
                    "options_dte1_dir": dte1_dir,
                }
                combos.append((label, config, ticker, None, equity_dir, options_dir, dte1_dir))
    return combos


def build_layer2_configs(best_l1, ticker, equity_dir, options_dir, dte1_dir):
    """Layer 2: stop loss sweep on top of best L1."""
    combos = []
    for sl_mode, sl_val in [
        (None, None),
        ("credit_multiple", 1), ("credit_multiple", 2),
        ("credit_multiple", 3), ("credit_multiple", 5),
        ("width_pct", 0.25), ("width_pct", 0.50), ("width_pct", 0.75),
    ]:
        sl_label = "noSL" if sl_mode is None else f"SL_{sl_mode}_{sl_val}"
        label = f"L2_{best_l1['_base']}_{sl_label}"
        config = {**best_l1["_config"],
                  "stop_loss_mode": sl_mode, "stop_loss_value": sl_val}
        combos.append((label, config, ticker, None, equity_dir, options_dir, dte1_dir))
    return combos


def build_layer3_configs(best_l2, ticker, equity_dir, options_dir, dte1_dir):
    """Layer 3: sizing mode sweep."""
    combos = []
    for sizing, mult in [("budget", None), ("credit_multiple", 5),
                         ("credit_multiple", 10), ("credit_multiple", 20)]:
        s_label = "budg" if sizing == "budget" else f"cr{mult}x"
        label = f"L3_{best_l2['_base']}_{s_label}"
        config = {**best_l2["_config"],
                  "sizing_mode": sizing, "sizing_credit_multiple": mult}
        combos.append((label, config, ticker, None, equity_dir, options_dir, dte1_dir))
    return combos


def build_layer4_configs(best_l3, ticker, equity_dir, options_dir, dte1_dir):
    """Layer 4: roll mode sweep."""
    combos = []
    for rm in ["none", "eod_itm", "midday_dte1", "conditional_dte1"]:
        label = f"L4_{best_l3['_base']}_{rm}"
        config = {**best_l3["_config"], "roll_mode": rm}
        combos.append((label, config, ticker, None, equity_dir, options_dir, dte1_dir))
    return combos


def run_layer(combos, lookback_days, output_dir, layer_name):
    """Run all combos in parallel, return sorted DataFrame."""
    # Patch lookback_days into each combo
    patched = []
    for c in combos:
        label, config, ticker, _, eq, opt, dte1 = c
        patched.append((label, config, ticker, lookback_days, eq, opt, dte1))

    print(f"\n{'='*80}")
    print(f"  {layer_name}: {len(patched)} configs × {lookback_days} days, {NUM_WORKERS} workers")
    print(f"{'='*80}")

    with Pool(processes=NUM_WORKERS) as pool:
        raw_results = pool.map(run_single_config, patched)

    rows = []
    for label, metrics in raw_results:
        if metrics is not None:
            rows.append(metrics)
        else:
            print(f"  SKIP {label}: no valid trades")

    if not rows:
        print(f"  No results for {layer_name}!")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("total_pnl", ascending=False)

    # Save
    layer_dir = os.path.join(output_dir, layer_name)
    os.makedirs(layer_dir, exist_ok=True)
    df.to_csv(os.path.join(layer_dir, "summary.csv"), index=False)

    # Print top 10
    print(f"\n  Top 10 by P&L:")
    print(f"  {'Label':<50s} {'P&L':>12s} {'WR':>6s} {'Sharpe':>7s} {'PF':>6s} {'DD':>12s} {'Cr%':>5s} {'Ctrs':>5s} {'W':>4s}")
    for _, r in df.head(10).iterrows():
        print(f"  {r['label']:<50s} ${r['total_pnl']:>11,.0f} {r['win_rate']:>5.0f}% {r['sharpe']:>7.2f} "
              f"{r['profit_factor']:>5.2f} ${r['max_drawdown']:>11,.0f} {r['avg_credit_pct']:>4.1f}% {r['avg_contracts']:>5.0f} ${r['avg_width']:>3.0f}")

    return df


def pick_best(df, base_configs_map):
    """Pick best config from a layer's results."""
    if df.empty:
        return None
    best_label = df.iloc[0]["label"]
    return {
        "_base": best_label,
        "_config": base_configs_map[best_label],
        **df.iloc[0].to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(
        description='''
VMaxMin.1 Parameter Sweep — systematic search for optimal configuration.

Layer 1: leg placement × depth × width (hold-to-expiration baseline)
Layer 2: stop loss modes (on best L1)
Layer 3: position sizing (on best L2)
Layer 4: roll modes (on best L3)
        ''',
        epilog='''
Examples:
  %(prog)s --tickers SPX --lookback-days 30
      Run all layers for SPX, 30 days

  %(prog)s --tickers SPX --lookback-days 30 --layer 1
      Run only Layer 1 (depth + width sweep)

  %(prog)s --tickers SPX NDX --lookback-days 30
      Run for multiple tickers (sequentially)
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tickers", nargs="+", default=["SPX"],
                        help="Tickers (default: SPX)")
    parser.add_argument("--lookback-days", type=int, default=30,
                        help="Days to backtest (default: 30)")
    parser.add_argument("--output-dir", default="results/vmaxmin_sweep",
                        help="Output directory")
    parser.add_argument("--layer", default="all",
                        help="Which layer to run: 1, 2, 3, 4, or all")
    parser.add_argument("--equity-dir", default="equities_output")
    parser.add_argument("--options-dir", default="options_csv_output_full")
    parser.add_argument("--dte1-dir", default="csv_exports/options")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for ticker in args.tickers:
        print(f"\n{'#'*80}")
        print(f"  VMAXMIN SWEEP — {ticker}")
        print(f"{'#'*80}")

        # ── Layer 1: Depth + Width ──
        if args.layer in ("1", "all"):
            l1_combos = build_layer1_configs(
                ticker, args.equity_dir, args.options_dir, args.dte1_dir)
            l1_configs = {c[0]: c[1] for c in l1_combos}
            l1_df = run_layer(l1_combos, args.lookback_days, args.output_dir,
                              f"{ticker}_L1_depth_width")

            if args.layer == "1":
                continue

        # ── Layer 2: Stop Loss ──
        if args.layer in ("2", "all"):
            if args.layer == "2":
                # Load L1 results
                l1_path = os.path.join(args.output_dir, f"{ticker}_L1_depth_width", "summary.csv")
                l1_df = pd.read_csv(l1_path)
                # Reconstruct configs from labels
                l1_combos = build_layer1_configs(
                    ticker, args.equity_dir, args.options_dir, args.dte1_dir)
                l1_configs = {c[0]: c[1] for c in l1_combos}

            best_l1 = pick_best(l1_df, l1_configs)
            if best_l1 is None:
                print(f"  No L1 results, skipping L2+")
                continue
            print(f"\n  Best L1: {best_l1['_base']} (P&L=${best_l1['total_pnl']:,.0f}, WR={best_l1['win_rate']:.0f}%)")

            l2_combos = build_layer2_configs(
                best_l1, ticker, args.equity_dir, args.options_dir, args.dte1_dir)
            l2_configs = {c[0]: c[1] for c in l2_combos}
            l2_df = run_layer(l2_combos, args.lookback_days, args.output_dir,
                              f"{ticker}_L2_stop_loss")

            if args.layer == "2":
                continue

        # ── Layer 3: Sizing ──
        if args.layer in ("3", "all"):
            if args.layer == "3":
                l2_path = os.path.join(args.output_dir, f"{ticker}_L2_stop_loss", "summary.csv")
                l2_df = pd.read_csv(l2_path)
                l2_combos = build_layer2_configs(
                    {"_base": "manual", "_config": {}},
                    ticker, args.equity_dir, args.options_dir, args.dte1_dir)
                l2_configs = {c[0]: c[1] for c in l2_combos}

            best_l2 = pick_best(l2_df, l2_configs)
            if best_l2 is None:
                print(f"  No L2 results, skipping L3+")
                continue
            print(f"\n  Best L2: {best_l2['_base']} (P&L=${best_l2['total_pnl']:,.0f}, WR={best_l2['win_rate']:.0f}%)")

            l3_combos = build_layer3_configs(
                best_l2, ticker, args.equity_dir, args.options_dir, args.dte1_dir)
            l3_configs = {c[0]: c[1] for c in l3_combos}
            l3_df = run_layer(l3_combos, args.lookback_days, args.output_dir,
                              f"{ticker}_L3_sizing")

            if args.layer == "3":
                continue

        # ── Layer 4: Roll Mode ──
        if args.layer in ("4", "all"):
            if args.layer == "4":
                l3_path = os.path.join(args.output_dir, f"{ticker}_L3_sizing", "summary.csv")
                l3_df = pd.read_csv(l3_path)
                l3_combos = build_layer3_configs(
                    {"_base": "manual", "_config": {}},
                    ticker, args.equity_dir, args.options_dir, args.dte1_dir)
                l3_configs = {c[0]: c[1] for c in l3_combos}

            best_l3 = pick_best(l3_df, l3_configs)
            if best_l3 is None:
                print(f"  No L3 results, skipping L4")
                continue
            print(f"\n  Best L3: {best_l3['_base']} (P&L=${best_l3['total_pnl']:,.0f}, WR={best_l3['win_rate']:.0f}%)")

            l4_combos = build_layer4_configs(
                best_l3, ticker, args.equity_dir, args.options_dir, args.dte1_dir)
            l4_configs = {c[0]: c[1] for c in l4_combos}
            l4_df = run_layer(l4_combos, args.lookback_days, args.output_dir,
                              f"{ticker}_L4_roll_mode")

    print(f"\nSweep complete. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
