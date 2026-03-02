"""Full percentile sweep: P75/P90/P95/P98/P99/P100 × DTE 0/2/5/10.

Runs all 24 configurations in PARALLEL using multiprocessing (up to 8 workers).

Produces:
  1. Comparison matrix (ROI, Sharpe, win%, rolls, etc.)
  2. Roll analysis (count, time-of-day, loss % at roll)
  3. Exit time-of-day analysis (when profit targets, rolls, stop losses fire)
  4. Win % distribution per trade
  5. Put vs Call breakdown
"""
import sys, os, time, json
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

# ── Configuration ──────────────────────────────────────────────────────────
PERCENTILES = [75, 90, 95, 98, 99, 100]
DTE_VALUES  = [0, 2, 5, 10]
LOOKBACK    = 180          # calendar days
START_DATE  = "2025-03-01"
END_DATE    = "2026-02-27"
NUM_WORKERS = min(8, cpu_count())  # Cap at 8 to avoid memory pressure
# ───────────────────────────────────────────────────────────────────────────


def run_single_config(args):
    """Run a single backtest config in a subprocess. Returns (label, result_dict)."""
    pct, dte = args
    label = f"P{pct}_DTE{dte}"

    # Each subprocess must do its own imports (fresh interpreter state)
    import sys, time, logging
    sys.path.insert(0, '.')
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(f'bt_{label}')

    from scripts.backtesting.config import BacktestConfig
    from scripts.backtesting.engine import BacktestEngine
    import scripts.backtesting.providers.csv_equity_provider   # noqa
    import scripts.backtesting.providers.csv_options_provider   # noqa
    import scripts.backtesting.instruments.credit_spread        # noqa
    import scripts.backtesting.strategies.credit_spread.percentile_entry  # noqa

    config = BacktestConfig.load(
        'scripts/backtesting/configs/percentile_entry_ndx.yaml')
    config.strategy.params['dte'] = dte
    config.strategy.params['percentile'] = pct
    config.strategy.params['lookback'] = LOOKBACK
    config.infra.start_date = START_DATE
    config.infra.end_date = END_DATE
    config.infra.lookback_days = LOOKBACK
    config.infra.output_dir = f'results/sweep_{label.lower()}'

    if dte > 0:
        for p in config.providers.providers:
            if p.name == 'csv_options':
                p.params['dte_buckets'] = list(range(max(dte + 2, 5)))

    engine = BacktestEngine(config, logger)
    t0 = time.time()
    results = engine.run()
    elapsed = time.time() - t0

    metrics = results.get('metrics', {})
    trades = metrics.get('total_trades', 0)
    print(f"  done {label}: {trades} trades in {elapsed:.0f}s "
          f"(roi={metrics.get('roi',0):.1f}%, sharpe={metrics.get('sharpe',0):.2f})",
          flush=True)

    return (label, {
        'pct': pct, 'dte': dte,
        'metrics': metrics,
        'csv': f'results/sweep_{label.lower()}/trades.csv',
        'elapsed': elapsed,
    })


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    # Build all (pct, dte) combos
    combos = [(pct, dte) for pct in PERCENTILES for dte in DTE_VALUES]
    combo_count = len(combos)

    print(f"Running {combo_count} configs with {NUM_WORKERS} parallel workers...")
    print(f"Percentiles: {PERCENTILES}")
    print(f"DTEs: {DTE_VALUES}")
    print(f"Period: {START_DATE} -> {END_DATE} ({LOOKBACK}-day lookback)")
    print(f"Min credit filter: $0.75/contract")
    print()

    t0_total = time.time()

    with Pool(processes=NUM_WORKERS) as pool:
        results_list = pool.map(run_single_config, combos)

    total_elapsed = time.time() - t0_total
    all_results = dict(results_list)

    # Save full results JSON
    with open('results/sweep_progress.json', 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'csv'}
                   for k, v in all_results.items()}, f, indent=2, default=str)

    print(f"\nAll {combo_count} configs completed in {total_elapsed:.0f}s "
          f"({total_elapsed/60:.1f} min)")


    # ══════════════════════════════════════════════════════════════════════════
    #  ANALYSIS HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def analyze(csv_path):
        if not os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path)
        if df.empty:
            return None

        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time']  = pd.to_datetime(df['exit_time'])

        max_profit = df['initial_credit'] * 100
        df['win_pct'] = (df['pnl'] / max_profit * 100).clip(-500, 100)

        a = {}
        a['total'] = len(df)
        a['avg_credit'] = float(df['initial_credit'].mean())
        a['min_credit'] = float(df['initial_credit'].min())

        # Exit breakdown
        a['profit_target'] = int((df['exit_reason'].str.startswith('profit_target', na=False)).sum())
        a['roll_trigger']  = int((df['exit_reason'].str.startswith('roll_trigger', na=False)).sum())
        a['stop_loss']     = int((df['exit_reason'].str.startswith('stop_loss', na=False)).sum())
        a['eod_close']     = int((df['exit_reason'].str.startswith('eod_close', na=False)).sum())
        a['expiration']    = int((df['exit_reason'].str.startswith('expiration', na=False)).sum())

        # Exit time-of-day breakdown (hour:minute UTC)
        for reason_key, reason_prefix in [
            ('pt_times', 'profit_target'),
            ('sl_times', 'stop_loss'),
            ('rt_times', 'roll_trigger'),
            ('eod_times', 'eod_close'),
        ]:
            subset = df[df['exit_reason'].str.startswith(reason_prefix, na=False)]
            if len(subset):
                # Group by hour UTC
                hours = subset['exit_time'].dt.hour
                a[reason_key] = hours.value_counts().sort_index().to_dict()
            else:
                a[reason_key] = {}

        # Rolls
        rolled = df[df['roll_count'] > 0]
        a['total_rolls'] = int(rolled['roll_count'].sum()) if len(rolled) else 0
        a['trades_rolled'] = len(rolled)

        # Roll timing (precise)
        rt = df[df['exit_reason'].str.startswith('roll_trigger', na=False)]
        if len(rt):
            a['roll_times_detail'] = rt['exit_time'].dt.strftime('%H:%M').value_counts().to_dict()
            rt_loss_pct = (rt['pnl'] / (rt['initial_credit'] * 100) * 100)
            a['roll_loss_pct'] = {
                'min': float(rt_loss_pct.min()),
                'median': float(rt_loss_pct.median()),
                'max': float(rt_loss_pct.max()),
                'mean': float(rt_loss_pct.mean()),
            }
        else:
            a['roll_times_detail'] = {}
            a['roll_loss_pct'] = {}

        # Roll chain outcomes
        if len(rolled):
            a['roll_chain_pnl_mean'] = float(rolled['total_chain_pnl'].mean())
            a['roll_chain_pnl_min']  = float(rolled['total_chain_pnl'].min())
            a['roll_chain_pnl_max']  = float(rolled['total_chain_pnl'].max())
        else:
            a['roll_chain_pnl_mean'] = 0
            a['roll_chain_pnl_min'] = 0
            a['roll_chain_pnl_max'] = 0

        # Win/loss counts and damage
        wins   = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]
        a['wins']   = len(wins)
        a['losses'] = len(losses)
        a['total_loss_damage'] = float(losses['pnl'].sum()) if len(losses) else 0

        # Stop loss damage
        sl = df[df['exit_reason'].str.startswith('stop_loss', na=False)]
        a['sl_total_damage'] = float(sl['pnl'].sum()) if len(sl) else 0
        a['sl_avg_credit'] = float(sl['initial_credit'].mean()) if len(sl) else 0
        a['sl_option_types'] = sl['option_type'].value_counts().to_dict() if len(sl) else {}

        # Win % distribution
        if len(wins):
            wp = wins['win_pct']
            a['win_dist'] = {
                'min': float(wp.min()), 'p10': float(np.percentile(wp,10)),
                'p25': float(np.percentile(wp,25)), 'median': float(wp.median()),
                'p75': float(np.percentile(wp,75)), 'p90': float(np.percentile(wp,90)),
                'max': float(wp.max()), 'mean': float(wp.mean()),
            }
            buckets = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            hist, _ = np.histogram(wp.clip(0, 100), bins=buckets)
            a['win_buckets'] = {f"{buckets[i]}-{buckets[i+1]}%": int(hist[i])
                               for i in range(len(hist))}
        else:
            a['win_dist'] = {}
            a['win_buckets'] = {}

        if len(losses):
            lp = losses['win_pct']
            a['loss_dist'] = {
                'min': float(lp.min()), 'median': float(lp.median()),
                'max': float(lp.max()), 'mean': float(lp.mean()),
            }
        else:
            a['loss_dist'] = {}

        # Put vs Call
        for ot in ['put', 'call']:
            sub = df[df['option_type'] == ot]
            if len(sub):
                a[f'{ot}_n']    = len(sub)
                a[f'{ot}_win']  = float((sub['pnl'] > 0).mean() * 100)
                a[f'{ot}_pnl']  = float(sub['pnl'].mean())
                a[f'{ot}_rolls'] = int(sub[sub['roll_count'] > 0]['roll_count'].sum())
                a[f'{ot}_avg_credit'] = float(sub['initial_credit'].mean())
            else:
                a[f'{ot}_n'] = 0
                a[f'{ot}_win'] = 0
                a[f'{ot}_pnl'] = 0
                a[f'{ot}_rolls'] = 0
                a[f'{ot}_avg_credit'] = 0

        return a


    all_a = {label: analyze(r['csv']) for label, r in all_results.items()}


    # ══════════════════════════════════════════════════════════════════════════
    #  REPORT
    # ══════════════════════════════════════════════════════════════════════════

    W = 130

    print(f"\n\n{'='*W}")
    print(f"  PERCENTILE ENTRY STRATEGY -- FULL SWEEP  (NDX {START_DATE} -> {END_DATE}, {LOOKBACK}-day lookback)")
    print(f"  Min credit filter: $0.75/contract")
    print(f"{'='*W}")

    # -- 1. Comparison matrix
    print(f"\n{'-'*W}")
    print(f"  1. COMPARISON MATRIX")
    print(f"{'-'*W}")
    hdr = (f"{'Config':<14} {'Trades':>6} {'Win%':>6} {'Net P&L':>14} {'ROI':>8} "
           f"{'Sharpe':>7} {'MaxDD':>11} {'PF':>7} {'AvgCr':>6} {'AvgP&L':>9} "
           f"{'Rolls':>5} {'StopL':>5} {'SL$':>12}")
    print(hdr)
    print('-' * len(hdr))

    for pct in PERCENTILES:
        for dte in DTE_VALUES:
            label = f"P{pct}_DTE{dte}"
            m = all_results[label]['metrics']
            a = all_a.get(label) or {}
            pf = m.get('profit_factor', 0)
            pf_s = f"{pf:.1f}" if pf != float('inf') else "inf"
            sl_dmg = a.get('sl_total_damage', 0)
            print(f"{label:<14} {m.get('total_trades',0):>6} {m.get('win_rate',0):>5.1f}% "
                  f"${m.get('net_pnl',0):>13,.0f} {m.get('roi',0):>7.1f}% "
                  f"{m.get('sharpe',0):>7.2f} ${m.get('max_drawdown',0):>10,.0f} "
                  f"{pf_s:>7} ${a.get('avg_credit',0):>5.1f} ${m.get('avg_pnl',0):>8,.0f} "
                  f"{a.get('total_rolls',0):>5} {a.get('stop_loss',0):>5} "
                  f"${sl_dmg:>11,.0f}")
        print()

    # -- 2. Exit time-of-day analysis
    print(f"\n{'-'*W}")
    print(f"  2. EXIT TIME-OF-DAY ANALYSIS  (hour UTC -> PST)")
    print(f"{'-'*W}")

    for pct in PERCENTILES:
        has_any = False
        for dte in DTE_VALUES:
            label = f"P{pct}_DTE{dte}"
            a = all_a.get(label)
            if not a or a['total'] == 0:
                continue
            # Only print configs with interesting exit patterns
            if a.get('roll_trigger', 0) == 0 and a.get('stop_loss', 0) == 0:
                continue
            has_any = True

            print(f"\n  -- {label} ({a['total']} trades) --")

            for reason_label, time_key, count_key in [
                ('Profit Target', 'pt_times', 'profit_target'),
                ('Roll Trigger',  'rt_times', 'roll_trigger'),
                ('Stop Loss',     'sl_times', 'stop_loss'),
                ('EOD Close',     'eod_times', 'eod_close'),
            ]:
                times = a.get(time_key, {})
                count = a.get(count_key, 0)
                if not times or count == 0:
                    continue
                parts = []
                for hour_utc in sorted(times.keys()):
                    n = times[hour_utc]
                    hour_pst = (hour_utc - 7) % 24
                    parts.append(f"{hour_utc:02d}:xx UTC ({hour_pst:02d}:xx PST): {n}")
                print(f"    {reason_label} ({count} total):")
                for p in parts:
                    print(f"      {p}")

        if not has_any:
            # Print a summary for configs with only profit targets
            for dte in DTE_VALUES:
                label = f"P{pct}_DTE{dte}"
                a = all_a.get(label)
                if a and a['total'] > 0:
                    pt_times = a.get('pt_times', {})
                    if pt_times:
                        peak_hour = max(pt_times, key=pt_times.get)
                        peak_pst = (peak_hour - 7) % 24
                        print(f"  {label}: all {a['profit_target']} exits via profit_target, "
                              f"peak hour {peak_hour:02d}:xx UTC ({peak_pst:02d}:xx PST)")

    # -- 3. Roll analysis
    print(f"\n{'-'*W}")
    print(f"  3. ROLL ANALYSIS  (when, how often, at what loss %)")
    print(f"{'-'*W}")

    any_rolls = False
    for label in sorted(all_results, key=lambda x: (all_results[x]['pct'], all_results[x]['dte'])):
        a = all_a.get(label)
        if not a:
            continue
        if a['total_rolls'] == 0 and a['roll_trigger'] == 0 and a['stop_loss'] == 0:
            continue
        any_rolls = True

        print(f"\n  +-- {label} --+")
        print(f"  | Exit breakdown:  profit_target={a['profit_target']}  "
              f"roll_trigger={a['roll_trigger']}  stop_loss={a['stop_loss']}  "
              f"eod_close={a['eod_close']}  expiration={a['expiration']}")
        print(f"  | Rolls: {a['total_rolls']} total across {a['trades_rolled']} position chains")

        if a.get('roll_times_detail'):
            times_str = "  ".join(
                f"{t} UTC ({(int(t[:2])-7)%24:02d}:{t[3:]} PST) x{c}"
                for t, c in sorted(a['roll_times_detail'].items()))
            print(f"  | Roll times: {times_str}")

        rlp = a.get('roll_loss_pct', {})
        if rlp:
            print(f"  | Loss % at roll:  mean={rlp['mean']:.1f}%  "
                  f"median={rlp['median']:.1f}%  range=[{rlp['min']:.1f}%, {rlp['max']:.1f}%]")

        if a['total_rolls']:
            print(f"  | Roll chain P&L:  mean=${a['roll_chain_pnl_mean']:,.0f}  "
                  f"range=[${a['roll_chain_pnl_min']:,.0f}, ${a['roll_chain_pnl_max']:,.0f}]")

        if a.get('stop_loss', 0) > 0:
            print(f"  | Stop loss damage: ${a['sl_total_damage']:,.0f}  "
                  f"(avg credit at SL: ${a['sl_avg_credit']:.2f})  "
                  f"types: {a.get('sl_option_types', {})}")

        for ot in ['put', 'call']:
            r = a.get(f'{ot}_rolls', 0)
            if r:
                print(f"  | {ot.upper()} side: {r} rolls  "
                      f"(win rate {a.get(f'{ot}_win',0):.1f}%, avg P&L ${a.get(f'{ot}_pnl',0):,.0f})")
        print(f"  +{'-'*(W-4)}+")

    if not any_rolls:
        print("  No rolls or stop-losses triggered in any configuration.")

    # -- 4. Win % distribution
    print(f"\n{'-'*W}")
    print(f"  4. WIN % DISTRIBUTION PER TRADE  (pnl / max_credit x 100)")
    print(f"{'-'*W}")

    for pct in PERCENTILES:
        print(f"\n  -- P{pct} --")
        for dte in DTE_VALUES:
            label = f"P{pct}_DTE{dte}"
            a = all_a.get(label)
            if not a:
                continue

            wd = a.get('win_dist', {})
            if not wd:
                print(f"  {label}: no winning trades")
                continue

            print(f"  {label} ({a['wins']}W/{a['losses']}L)  "
                  f"min={wd['min']:.0f}%  P10={wd['p10']:.0f}%  P25={wd['p25']:.0f}%  "
                  f"med={wd['median']:.0f}%  P75={wd['p75']:.0f}%  P90={wd['p90']:.0f}%  "
                  f"max={wd['max']:.0f}%  mean={wd['mean']:.0f}%")

            bk = a.get('win_buckets', {})
            if bk:
                total = max(sum(bk.values()), 1)
                for bucket, count in bk.items():
                    pct_of_total = count / total * 100
                    bar = '#' * int(pct_of_total / 2)
                    print(f"      {bucket:>8} {count:>5} ({pct_of_total:>5.1f}%) {bar}")

            ld = a.get('loss_dist', {})
            if ld:
                print(f"      Losses: mean={ld['mean']:.0f}%  "
                      f"median={ld['median']:.0f}%  "
                      f"worst={ld['min']:.0f}%  best={ld['max']:.0f}%")

    # -- 5. Put vs Call
    print(f"\n{'-'*W}")
    print(f"  5. PUT vs CALL BREAKDOWN")
    print(f"{'-'*W}")
    print(f"{'Config':<14} {'Put#':>5} {'PutWin%':>8} {'PutCr':>7} {'PutAvgP&L':>11} {'PutRolls':>8}  "
          f"{'Call#':>5} {'CallWin%':>9} {'CallCr':>7} {'CallAvgP&L':>11} {'CallRolls':>9}")
    print('-' * 110)

    for pct in PERCENTILES:
        for dte in DTE_VALUES:
            label = f"P{pct}_DTE{dte}"
            a = all_a.get(label)
            if not a:
                continue
            print(f"{label:<14} "
                  f"{a.get('put_n',0):>5} {a.get('put_win',0):>7.1f}% "
                  f"${a.get('put_avg_credit',0):>5.1f} ${a.get('put_pnl',0):>10,.0f} {a.get('put_rolls',0):>8}  "
                  f"{a.get('call_n',0):>5} {a.get('call_win',0):>8.1f}% "
                  f"${a.get('call_avg_credit',0):>5.1f} ${a.get('call_pnl',0):>10,.0f} {a.get('call_rolls',0):>9}")
        print()

    print(f"\n{'='*W}")
    print(f"  SWEEP COMPLETE -- {combo_count} configurations evaluated in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*W}")

    # Best configs
    best_roi    = max(all_results, key=lambda l: all_results[l]['metrics'].get('roi', 0))
    best_sharpe = max(all_results, key=lambda l: all_results[l]['metrics'].get('sharpe', 0))
    safest      = min(all_results, key=lambda l: all_results[l]['metrics'].get('max_drawdown', 0))
    print(f"  Best ROI:     {best_roi}  ({all_results[best_roi]['metrics'].get('roi',0):.1f}%)")
    print(f"  Best Sharpe:  {best_sharpe}  ({all_results[best_sharpe]['metrics'].get('sharpe',0):.2f})")
    m_safe = all_results[safest]['metrics']
    print(f"  Lowest DD:    {safest}  (${m_safe.get('max_drawdown',0):,.0f} drawdown, "
          f"{m_safe.get('roi',0):.1f}% ROI)")
    print()
