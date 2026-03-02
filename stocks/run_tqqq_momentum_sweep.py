"""TQQQ Momentum Scalper -- full parameter sweep with multiprocessing.

Sweeps across signal modes (ORB, consecutive-day, gap fade, combined),
OTM distances, and profit targets. Runs all configs in parallel using
up to 8 workers.

Usage:
    python run_tqqq_momentum_sweep.py
    python run_tqqq_momentum_sweep.py --workers 4
    python run_tqqq_momentum_sweep.py --dry-run

Examples:
    # Full sweep (all signal modes × OTM distances × profit targets)
    python run_tqqq_momentum_sweep.py

    # Quick test with 4 workers
    python run_tqqq_momentum_sweep.py --workers 4

    # Dry run to see all configs
    python run_tqqq_momentum_sweep.py --dry-run
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

# ── Configuration ──────────────────────────────────────────────────────────
START_DATE = "2025-03-01"
END_DATE = "2026-02-28"
LOOKBACK = 30

# Sweep axes
SIGNAL_MODES = ["orb", "consecutive", "gap_fade", "combined"]
PERCENT_BEYOND_VALUES = ["0.01:0.01", "0.02:0.02", "0.03:0.03"]
PROFIT_TARGETS = [0.30, 0.50, 0.70]

# Mode-specific axes
MIN_CONSECUTIVE_DOWN = [2, 3]  # Only for consecutive mode
MAX_GAP_PCT = [0.003, 0.005]   # Only for gap_fade mode
# ───────────────────────────────────────────────────────────────────────────


def build_combos():
    """Build all (signal_mode, percent_beyond, profit_target, extra_params) combos."""
    combos = []

    for mode in SIGNAL_MODES:
        for pb in PERCENT_BEYOND_VALUES:
            for pt in PROFIT_TARGETS:
                if mode == "consecutive":
                    for mcd in MIN_CONSECUTIVE_DOWN:
                        label = f"{mode}_pb{pb.replace(':','')}_pt{int(pt*100)}_cd{mcd}"
                        combos.append((label, mode, pb, pt, {"min_consecutive_down": mcd}))
                elif mode == "gap_fade":
                    for mgp in MAX_GAP_PCT:
                        label = f"{mode}_pb{pb.replace(':','')}_pt{int(pt*100)}_gp{int(mgp*1000)}"
                        combos.append((label, mode, pb, pt, {"max_gap_pct": mgp}))
                else:
                    label = f"{mode}_pb{pb.replace(':','')}_pt{int(pt*100)}"
                    combos.append((label, mode, pb, pt, {}))

    return combos


def run_single_config(args):
    """Run a single backtest config in a subprocess."""
    label, mode, pb, pt, extra = args

    import sys
    import logging
    import time as _time
    sys.path.insert(0, '.')
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(f'bt_{label}')

    from scripts.backtesting.config import BacktestConfig
    from scripts.backtesting.engine import BacktestEngine
    import scripts.backtesting.providers.csv_equity_provider    # noqa
    import scripts.backtesting.providers.csv_options_provider    # noqa
    import scripts.backtesting.instruments.credit_spread         # noqa
    import scripts.backtesting.strategies.credit_spread.tqqq_momentum_scalper  # noqa

    config = BacktestConfig.load(
        'scripts/backtesting/configs/tqqq_momentum_scalper.yaml')

    # Apply sweep params
    config.strategy.params['signal_mode'] = mode
    config.strategy.params['percent_beyond'] = pb
    config.strategy.params.update(extra)

    config.infra.start_date = START_DATE
    config.infra.end_date = END_DATE
    config.infra.lookback_days = LOOKBACK
    config.infra.output_dir = f'results/tqqq_sweep_{label}'

    # Set profit target
    config.constraints.exit_rules.profit_target_pct = pt

    engine = BacktestEngine(config, logger)
    t0 = _time.time()

    try:
        results = engine.run()
    except Exception as e:
        print(f"  FAILED {label}: {e}", flush=True)
        return (label, {
            'mode': mode, 'percent_beyond': pb, 'profit_target': pt,
            'extra': extra, 'metrics': {}, 'error': str(e),
            'csv': '', 'elapsed': _time.time() - t0,
        })

    elapsed = _time.time() - t0
    metrics = results.get('metrics', {})
    trades = metrics.get('total_trades', 0)
    print(f"  done {label}: {trades} trades in {elapsed:.0f}s "
          f"(roi={metrics.get('roi', 0):.1f}%, "
          f"win={metrics.get('win_rate', 0):.1f}%, "
          f"sharpe={metrics.get('sharpe', 0):.2f})", flush=True)

    return (label, {
        'mode': mode, 'percent_beyond': pb, 'profit_target': pt,
        'extra': extra, 'metrics': metrics,
        'csv': f'results/tqqq_sweep_{label}/trades.csv',
        'elapsed': elapsed,
    })


def analyze_trades(csv_path):
    """Analyze a trades CSV for detailed metrics."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])

    a = {}
    a['total'] = len(df)
    a['avg_credit'] = float(df['initial_credit'].mean())

    # Exit breakdown
    for reason in ['profit_target', 'stop_loss', 'eod_close']:
        a[reason] = int(df['exit_reason'].str.startswith(reason, na=False).sum())

    # Win/loss
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]
    a['wins'] = len(wins)
    a['losses'] = len(losses)
    a['total_pnl'] = float(df['pnl'].sum())
    a['avg_pnl'] = float(df['pnl'].mean()) if len(df) else 0
    a['total_loss_damage'] = float(losses['pnl'].sum()) if len(losses) else 0
    a['max_single_loss'] = float(losses['pnl'].min()) if len(losses) else 0

    # Signal source breakdown
    if 'signal_source' in df.columns:
        a['signal_sources'] = df['signal_source'].value_counts().to_dict()
    else:
        a['signal_sources'] = {}

    # Put vs Call
    for ot in ['put', 'call']:
        sub = df[df['option_type'] == ot]
        if len(sub):
            a[f'{ot}_n'] = len(sub)
            a[f'{ot}_win'] = float((sub['pnl'] > 0).mean() * 100)
            a[f'{ot}_pnl'] = float(sub['pnl'].sum())
        else:
            a[f'{ot}_n'] = 0
            a[f'{ot}_win'] = 0
            a[f'{ot}_pnl'] = 0

    # Hold time
    hold_minutes = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60
    a['avg_hold_min'] = float(hold_minutes.mean())
    a['median_hold_min'] = float(hold_minutes.median())

    return a


def print_report(all_results, all_analysis, combos, total_elapsed):
    """Print comprehensive sweep report."""
    W = 140

    print(f"\n\n{'=' * W}")
    print(f"  TQQQ MOMENTUM SCALPER -- FULL SWEEP  ({START_DATE} -> {END_DATE})")
    print(f"  Signal modes: {SIGNAL_MODES}")
    print(f"  OTM distances: {PERCENT_BEYOND_VALUES}")
    print(f"  Profit targets: {PROFIT_TARGETS}")
    print(f"{'=' * W}")

    # ── 1. Comparison Matrix ──
    print(f"\n{'-' * W}")
    print(f"  1. COMPARISON MATRIX (sorted by Net P&L)")
    print(f"{'-' * W}")
    hdr = (f"{'Config':<42} {'Mode':<12} {'Trades':>6} {'Win%':>6} "
           f"{'Net P&L':>12} {'ROI':>8} {'Sharpe':>7} {'MaxDD':>10} "
           f"{'PF':>6} {'AvgHold':>8}")
    print(hdr)
    print('-' * len(hdr))

    # Sort by net P&L descending
    sorted_labels = sorted(
        all_results.keys(),
        key=lambda l: all_results[l]['metrics'].get('net_pnl', 0),
        reverse=True
    )

    for label in sorted_labels:
        r = all_results[label]
        m = r['metrics']
        a = all_analysis.get(label) or {}
        if not m:
            continue

        pf = m.get('profit_factor', 0)
        pf_s = f"{pf:.1f}" if pf != float('inf') else "inf"
        hold = f"{a.get('avg_hold_min', 0):.0f}m" if a else "N/A"

        print(f"{label:<42} {r['mode']:<12} {m.get('total_trades', 0):>6} "
              f"{m.get('win_rate', 0):>5.1f}% ${m.get('net_pnl', 0):>11,.0f} "
              f"{m.get('roi', 0):>7.1f}% {m.get('sharpe', 0):>7.2f} "
              f"${m.get('max_drawdown', 0):>9,.0f} {pf_s:>6} {hold:>8}")

    # ── 2. Signal Mode Summary ──
    print(f"\n{'-' * W}")
    print(f"  2. SIGNAL MODE SUMMARY (aggregated across all params)")
    print(f"{'-' * W}")

    for mode in SIGNAL_MODES:
        mode_results = {l: r for l, r in all_results.items() if r['mode'] == mode}
        if not mode_results:
            continue

        total_trades = sum(r['metrics'].get('total_trades', 0) for r in mode_results.values())
        avg_win = np.mean([r['metrics'].get('win_rate', 0) for r in mode_results.values()
                          if r['metrics'].get('total_trades', 0) > 0]) if total_trades > 0 else 0
        avg_roi = np.mean([r['metrics'].get('roi', 0) for r in mode_results.values()
                          if r['metrics'].get('total_trades', 0) > 0]) if total_trades > 0 else 0
        total_pnl = sum(r['metrics'].get('net_pnl', 0) for r in mode_results.values())
        best_label = max(mode_results, key=lambda l: mode_results[l]['metrics'].get('net_pnl', 0))
        best_m = mode_results[best_label]['metrics']

        print(f"\n  {mode.upper()}: {len(mode_results)} configs, "
              f"{total_trades} total trades across all configs")
        print(f"    Avg win rate: {avg_win:.1f}%, Avg ROI: {avg_roi:.1f}%")
        print(f"    Total P&L across all configs: ${total_pnl:,.0f}")
        print(f"    Best config: {best_label}")
        print(f"      ROI={best_m.get('roi', 0):.1f}%, "
              f"Win={best_m.get('win_rate', 0):.1f}%, "
              f"Net P&L=${best_m.get('net_pnl', 0):,.0f}, "
              f"Sharpe={best_m.get('sharpe', 0):.2f}")

    # ── 3. Put vs Call Breakdown ──
    print(f"\n{'-' * W}")
    print(f"  3. PUT vs CALL BREAKDOWN (top 10 configs by net P&L)")
    print(f"{'-' * W}")
    hdr2 = (f"{'Config':<42} {'Put#':>5} {'PutWin%':>8} {'PutP&L':>10}  "
            f"{'Call#':>5} {'CallWin%':>9} {'CallP&L':>10}")
    print(hdr2)
    print('-' * len(hdr2))

    for label in sorted_labels[:10]:
        a = all_analysis.get(label)
        if not a:
            continue
        print(f"{label:<42} "
              f"{a.get('put_n', 0):>5} {a.get('put_win', 0):>7.1f}% "
              f"${a.get('put_pnl', 0):>9,.0f}  "
              f"{a.get('call_n', 0):>5} {a.get('call_win', 0):>8.1f}% "
              f"${a.get('call_pnl', 0):>9,.0f}")

    # ── 4. Signal Source Analysis ──
    print(f"\n{'-' * W}")
    print(f"  4. SIGNAL SOURCE BREAKDOWN (combined mode configs)")
    print(f"{'-' * W}")

    for label in sorted_labels:
        r = all_results[label]
        a = all_analysis.get(label)
        if r['mode'] != 'combined' or not a:
            continue
        sources = a.get('signal_sources', {})
        if sources:
            src_str = "  ".join(f"{k}: {v}" for k, v in sorted(sources.items()))
            print(f"  {label}: {src_str}")

    # ── Summary ──
    print(f"\n{'=' * W}")
    print(f"  SWEEP COMPLETE -- {len(all_results)} configs in "
          f"{total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    print(f"{'=' * W}")

    # Best configs
    valid = {l: r for l, r in all_results.items()
             if r['metrics'].get('total_trades', 0) > 0}
    if valid:
        best_roi = max(valid, key=lambda l: valid[l]['metrics'].get('roi', 0))
        best_sharpe = max(valid, key=lambda l: valid[l]['metrics'].get('sharpe', 0))
        best_pnl = max(valid, key=lambda l: valid[l]['metrics'].get('net_pnl', 0))
        safest = min(valid, key=lambda l: valid[l]['metrics'].get('max_drawdown', 0))

        print(f"  Best Net P&L:  {best_pnl}  "
              f"(${valid[best_pnl]['metrics'].get('net_pnl', 0):,.0f})")
        print(f"  Best ROI:      {best_roi}  "
              f"({valid[best_roi]['metrics'].get('roi', 0):.1f}%)")
        print(f"  Best Sharpe:   {best_sharpe}  "
              f"({valid[best_sharpe]['metrics'].get('sharpe', 0):.2f})")
        m_safe = valid[safest]['metrics']
        print(f"  Lowest DD:     {safest}  "
              f"(${m_safe.get('max_drawdown', 0):,.0f} drawdown, "
              f"{m_safe.get('roi', 0):.1f}% ROI)")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''
TQQQ Momentum Scalper -- full parameter sweep.

Sweeps across signal modes (ORB, consecutive-day mean reversion, gap fade,
combined), OTM distances, and profit targets. All configs run in parallel.
        ''',
        epilog='''
Examples:
  %(prog)s
      Run full sweep with default 8 workers

  %(prog)s --workers 4
      Limit to 4 parallel workers

  %(prog)s --dry-run
      Show all configs without running
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--workers', type=int, default=min(8, cpu_count()),
                        help='Number of parallel workers (default: min(8, cpu_count))')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show configs without running')
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)
    combos = build_combos()

    print(f"TQQQ Momentum Scalper Sweep")
    print(f"  Period: {START_DATE} -> {END_DATE}")
    print(f"  Signal modes: {SIGNAL_MODES}")
    print(f"  OTM distances: {PERCENT_BEYOND_VALUES}")
    print(f"  Profit targets: {PROFIT_TARGETS}")
    print(f"  Total configs: {len(combos)}")
    print(f"  Workers: {args.workers}")
    print()

    if args.dry_run:
        for i, (label, mode, pb, pt, extra) in enumerate(combos):
            extra_str = f"  {extra}" if extra else ""
            print(f"  [{i + 1:>3}] {label:<42} mode={mode:<12} "
                  f"pb={pb}  pt={pt}{extra_str}")
        print(f"\n  {len(combos)} configs total (dry run, nothing executed)")
        sys.exit(0)

    t0_total = time.time()

    with Pool(processes=args.workers) as pool:
        results_list = pool.map(run_single_config, combos)

    total_elapsed = time.time() - t0_total
    all_results = dict(results_list)

    # Save raw results JSON
    with open('results/tqqq_sweep_results.json', 'w') as f:
        json.dump(
            {k: {kk: vv for kk, vv in v.items() if kk != 'csv'}
             for k, v in all_results.items()},
            f, indent=2, default=str,
        )

    # Analyze trade CSVs
    all_analysis = {}
    for label, r in all_results.items():
        csv_path = r.get('csv', '')
        if csv_path:
            all_analysis[label] = analyze_trades(csv_path)

    print_report(all_results, all_analysis, combos, total_elapsed)
