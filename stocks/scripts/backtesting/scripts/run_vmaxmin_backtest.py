"""VMaxMin.1 Backtest Runner — CLI entry point.

Runs the dynamic mean-reversion credit spread strategy across multiple
tickers and days. Produces per-day trade logs, summary tables, and CSV output.

Usage:
  python scripts/backtesting/scripts/run_vmaxmin_backtest.py --tickers SPX --lookback-days 10

  python scripts/backtesting/scripts/run_vmaxmin_backtest.py --lookback-days 30

  python scripts/backtesting/scripts/run_vmaxmin_backtest.py --lookback-days 180 \\
      --output-dir results/vmaxmin_v1_180d

  python scripts/backtesting/scripts/run_vmaxmin_backtest.py --tickers NDX --lookback-days 10 \\
      --max-per-transaction 25000 --commission 5
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.backtesting.scripts.vmaxmin_engine import (
    DEFAULT_CONFIG,
    TICKER_START_DATES,
    DayResult,
    RolledPosition,
    VMaxMinEngine,
    get_prev_close,
    get_trading_dates,
    load_0dte_options,
    load_equity_bars_df,
    load_equity_prices,
)


def print_day_summary(result: DayResult) -> None:
    """Print a one-line summary for a single day."""
    pnl = result.net_pnl
    pnl_str = f"{'+'if pnl>=0 else ''}${pnl:,.0f}"
    rolls_str = f"{result.num_rolls}R"
    dte1_str = " DTE1" if result.eod_rolled_to_dte1 else ""
    fail_str = f" [{result.failure_reason}]" if result.failure_reason else ""
    carried_count = sum(1 for t in result.trades if t.event == "carried_position")
    carry_str = f" +{carried_count}carry" if carried_count else ""

    print(f"  {result.date}  {result.direction or '—':>4s}  "
          f"open={result.open_price:>8.1f}  close={result.close_price:>8.1f}  "
          f"hod={result.hod:>8.1f}  lod={result.lod:>8.1f}  "
          f"{rolls_str:>3s}{dte1_str}{carry_str}  "
          f"cr=${result.total_credits:>7,.0f}  db=${result.total_debits:>7,.0f}  "
          f"comm=${result.total_commissions:>4,.0f}  "
          f"pnl={pnl_str:>8s}{fail_str}")


def print_ticker_summary(ticker: str, results: list[DayResult]) -> None:
    """Print aggregate metrics for a ticker."""
    valid = [r for r in results if not r.failure_reason]
    if not valid:
        print(f"\n  {ticker}: No valid trading days")
        return

    pnls = [r.net_pnl for r in valid]
    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    win_rate = wins / len(pnls) * 100 if pnls else 0
    avg_rolls = np.mean([r.num_rolls for r in valid])
    dte1_count = sum(1 for r in valid if r.eod_rolled_to_dte1)
    total_credits = sum(r.total_credits for r in valid)
    total_debits = sum(r.total_debits for r in valid)
    total_commissions = sum(r.total_commissions for r in valid)

    # Max drawdown (running P&L)
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else 0

    # Sharpe (annualized from daily)
    daily_std = np.std(pnls) if len(pnls) > 1 else 0
    daily_mean = np.mean(pnls) if pnls else 0
    sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0

    # Profit factor
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    print(f"\n{'='*90}")
    print(f"  {ticker} — VMaxMin.1 Summary ({len(valid)} trading days)")
    print(f"{'='*90}")
    print(f"  Total P&L:       {'+'if total_pnl>=0 else ''}${total_pnl:>12,.2f}")
    print(f"  Win Rate:         {win_rate:>10.1f}% ({wins}W / {losses}L)")
    print(f"  Avg Daily P&L:   {'+'if daily_mean>=0 else ''}${daily_mean:>12,.2f}")
    print(f"  Sharpe Ratio:     {sharpe:>10.2f}")
    print(f"  Profit Factor:    {profit_factor:>10.2f}")
    print(f"  Max Drawdown:     ${max_dd:>10,.2f}")
    print(f"  Total Credits:    ${total_credits:>12,.2f}")
    print(f"  Total Debits:     ${total_debits:>12,.2f}")
    print(f"  Total Commissions:${total_commissions:>11,.2f}")
    print(f"  Avg Rolls/Day:    {avg_rolls:>10.1f}")
    print(f"  DTE+1 Roll Days:  {dte1_count:>10d} ({dte1_count/len(valid)*100:.1f}%)")
    print(f"  Skipped Days:     {len(results) - len(valid):>10d}")


def results_to_dataframe(all_results: dict[str, list[DayResult]]) -> pd.DataFrame:
    """Convert results to a DataFrame for CSV export."""
    rows = []
    for ticker, results in all_results.items():
        for r in results:
            rows.append({
                "ticker": ticker,
                "date": r.date,
                "direction": r.direction,
                "open_price": r.open_price,
                "close_price": r.close_price,
                "hod": r.hod,
                "lod": r.lod,
                "total_credits": r.total_credits,
                "total_debits": r.total_debits,
                "total_commissions": r.total_commissions,
                "net_pnl": r.net_pnl,
                "num_rolls": r.num_rolls,
                "eod_rolled": r.eod_rolled_to_dte1,
                "num_trades": len(r.trades),
                "failure_reason": r.failure_reason,
            })
    return pd.DataFrame(rows)


def trades_to_dataframe(all_results: dict[str, list[DayResult]]) -> pd.DataFrame:
    """Convert all trades to a DataFrame for detailed CSV export."""
    rows = []
    for ticker, results in all_results.items():
        for r in results:
            for t in r.trades:
                rows.append({
                    "ticker": ticker,
                    "date": r.date,
                    "event": t.event,
                    "time_pacific": t.time_pacific,
                    "direction": t.direction,
                    "short_strike": t.short_strike,
                    "long_strike": t.long_strike,
                    "width": t.width,
                    "credit_or_debit": t.credit_or_debit,
                    "num_contracts": t.num_contracts,
                    "commission": t.commission,
                    "underlying_price": t.underlying_price,
                    "dte": t.dte,
                    "notes": t.notes,
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description='''
VMaxMin.1 — Dynamic Mean-Reversion Credit Spread Tracker

Sells a 0DTE OTM credit spread at market open (6:35 AM Pacific), then
dynamically rolls throughout the day when price sets new extremes (HOD/LOD).
At end-of-day, if the position is threatened, it rolls to DTE+1 for safety.

The thesis: selling into new extremes is a mean-reversion bet — markets
tend to revert after setting new intraday highs/lows.
        ''',
        epilog='''
Examples:
  %(prog)s --tickers SPX --lookback-days 10
      Quick test with SPX only, 10 days

  %(prog)s --lookback-days 30
      Full 30-day run (all tickers: SPX, NDX, RUT)

  %(prog)s --lookback-days 180 --output-dir results/vmaxmin_v1_180d
      Extended backtest with custom output

  %(prog)s --tickers NDX --max-per-transaction 25000
      NDX with smaller position budget

  %(prog)s --tickers SPX NDX --entry-time 06:40 --commission 5
      Custom entry time and commission

  %(prog)s --tickers SPX --lookback-days 30 --leg-placement itm
      Run with both legs ITM (deeper credit, higher intrinsic)

  %(prog)s --tickers SPX --lookback-days 30 --leg-placement otm
      Run with both legs OTM (default, lower credit, lower risk)

  %(prog)s --tickers SPX --lookback-days 30 --call-track
      Call-track mode: always sell call spread, roll up every 3 hours
      when price exceeds short strike. Roll budget = 25%% of original credit.

  %(prog)s --tickers SPX --lookback-days 30 --call-track --roll-interval 1
      Call-track with 1-hour roll checks

  %(prog)s --tickers RUT --lookback-days 30 --layer --num-contracts 1
      Layer mode with dual entry + multi-day roll tracking

  %(prog)s --tickers RUT --lookback-days 30 --layer --single-entry --max-rolls 2
      Layer mode, single-direction entry, max 2 rolls per position

  %(prog)s --tickers RUT --layer --entry-directions call --num-contracts 1
      Layer mode, only sell call spreads

  %(prog)s --tickers NDX --layer --entry-directions put --check-times 08:35 10:35 12:00
      Layer mode, only sell put spreads, custom check times

  %(prog)s --tickers SPX --layer --eod-exit-pct 0.10 --eod-time 12:30 --max-rolls 1
      Layer mode, only roll if breached > 10%% of width, early EOD, max 1 roll

  %(prog)s --tickers RUT --layer --daily-budget 100000 --risk-cap 500000
      Layer mode with $100K daily budget and $500K total exposure cap
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tickers", nargs="+", default=["SPX", "NDX", "RUT"],
                        help="Tickers to backtest (default: SPX NDX RUT)")
    parser.add_argument("--lookback-days", type=int, default=30,
                        help="Number of trading days to backtest (default: 30)")
    parser.add_argument("--output-dir", default="results/vmaxmin_v1",
                        help="Output directory (default: results/vmaxmin_v1)")
    parser.add_argument("--max-per-transaction", type=float, default=50000,
                        help="Max budget per position in $ (default: 50000)")
    parser.add_argument("--commission", type=float, default=10,
                        help="Commission per transaction in $ (default: 10)")
    parser.add_argument("--num-contracts", type=int, default=None,
                        help="Fixed number of contracts (default: auto from budget)")
    parser.add_argument("--entry-time", default="06:35",
                        help="Entry time in Pacific HH:MM (default: 06:35)")
    parser.add_argument("--proximity-threshold", type=float, default=0.005,
                        help="EOD proximity threshold as fraction (default: 0.005 = 0.5%%)")
    parser.add_argument("--max-width-multiplier", type=int, default=10,
                        help="Max width as multiple of min step (default: 10)")
    parser.add_argument("--leg-placement", choices=["otm", "itm"], default="otm",
                        help="Both legs OTM or ITM (default: otm)")
    parser.add_argument("--intraday-rolls", action="store_true",
                        help="Enable intraday close+reopen on new HOD/LOD (default: off)")
    parser.add_argument("--no-eod-roll", action="store_true",
                        help="Disable EOD roll to DTE+1 — let spreads expire (default: EOD roll on)")
    parser.add_argument("--roll-times", nargs="+",
                        default=["07:35", "08:35", "09:35", "10:35", "11:35"],
                        help="Intraday roll check times in Pacific HH:MM")
    parser.add_argument("--equity-dir", default="equities_output",
                        help="Equity data directory (default: equities_output)")
    parser.add_argument("--options-dir", default="options_csv_output_full",
                        help="0DTE options directory (default: options_csv_output_full)")
    parser.add_argument("--dte1-dir", default="csv_exports/options",
                        help="DTE+1 options directory (default: csv_exports/options)")
    parser.add_argument("--call-track", action="store_true",
                        help="Call-track mode: always sell call spread, roll up when price > short strike")
    parser.add_argument("--layer", action="store_true",
                        help="Layer mode: accumulate spreads on new HOD/LOD, roll ITM at 12:45")
    parser.add_argument("--single-entry", action="store_true",
                        help="Layer mode: single direction entry instead of dual call+put (default: dual)")
    parser.add_argument("--entry-directions", choices=["both", "call", "put"], default="both",
                        help="Layer mode: which spreads to open at entry (default: both)")
    parser.add_argument("--check-times", nargs="+", default=None,
                        help="Layer mode: intraday check times in Pacific HH:MM (default: 08:35 10:35)")
    parser.add_argument("--eod-time", default="12:45",
                        help="Layer mode: EOD roll check time in Pacific HH:MM (default: 12:45)")
    parser.add_argument("--eod-exit-pct", type=float, default=0.0,
                        help="Layer mode: min breach %% of width to trigger EOD roll (default: 0 = any ITM)")
    parser.add_argument("--max-rolls", type=int, default=3,
                        help="Max number of rolls per position before forcing settlement (default: 3)")
    parser.add_argument("--roll-recovery-threshold", type=float, default=1.0,
                        help="Stop rolling if cumulative cost >= original_credit * this (default: 1.0)")
    parser.add_argument("--roll-snowball", action="store_true",
                        help="Allow roll sizing to increase contracts (default: match original count)")
    parser.add_argument("--roll-max-width-mult", type=float, default=5,
                        help="Max roll width = original_width * this (default: 5)")
    parser.add_argument("--roll-max-contract-mult", type=float, default=2,
                        help="Max roll contracts = original_count * this (default: 2)")
    parser.add_argument("--roll-interval", type=int, default=3,
                        help="Roll check interval in hours for call-track mode (default: 3)")
    parser.add_argument("--roll-budget-pct", type=float, default=0.25,
                        help="Max fraction of original credit for roll costs (default: 0.25 = 25%%)")
    parser.add_argument("--eod-proximity-pct", type=float, default=0.003,
                        help="EOD proximity threshold for DTE+1 roll in call-track mode (default: 0.003 = 0.3%%)")
    parser.add_argument("--unlimited-roll-budget", action="store_true",
                        help="No cap on roll costs — assume unlimited capital for rolls")
    parser.add_argument("--daily-budget", type=float, default=None,
                        help="Daily risk budget in $ (e.g. 100000). Limits total new positions per day.")
    parser.add_argument("--risk-cap", type=float, default=None,
                        help="Max total exposure in $ (e.g. 500000). Hard cap on aggregate open risk.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-trade details")
    args = parser.parse_args()

    config = {
        **DEFAULT_CONFIG,
        "entry_time_pacific": args.entry_time,
        "max_per_transaction": args.max_per_transaction,
        "commission_per_transaction": args.commission,
        "num_contracts": args.num_contracts,
        "roll_check_times_pacific": args.roll_times,
        "proximity_threshold_pct": args.proximity_threshold,
        "max_width_multiplier": args.max_width_multiplier,
        "leg_placement": args.leg_placement,
        "intraday_rolls": args.intraday_rolls,
        "eod_roll": not args.no_eod_roll,
        "equity_dir": args.equity_dir,
        "options_0dte_dir": args.options_dir,
        "options_dte1_dir": args.dte1_dir,
    }

    if args.daily_budget is not None:
        config["daily_budget"] = args.daily_budget
    if args.risk_cap is not None:
        config["max_total_exposure"] = args.risk_cap

    if args.layer:
        config["strategy_mode"] = "layer"
        config["call_track_unlimited_budget"] = True
        config["layer_dual_entry"] = not args.single_entry
        config["layer_entry_directions"] = args.entry_directions
        config["layer_eod_exit_pct"] = args.eod_exit_pct
        config["call_track_eod_time_pacific"] = args.eod_time
        config["max_roll_count"] = args.max_rolls
        config["roll_recovery_threshold"] = args.roll_recovery_threshold
        config["roll_match_contracts"] = not args.roll_snowball
        config["roll_max_width_mult"] = args.roll_max_width_mult
        config["roll_max_contract_mult"] = args.roll_max_contract_mult
        if args.check_times is not None:
            config["call_track_check_times_pacific"] = args.check_times
    elif args.call_track:
        config["strategy_mode"] = "call_track"
        config["call_track_roll_interval_hours"] = args.roll_interval
        config["call_track_roll_budget_pct"] = args.roll_budget_pct
        config["call_track_eod_proximity_pct"] = args.eod_proximity_pct
        config["call_track_unlimited_budget"] = args.unlimited_roll_budget

    engine = VMaxMinEngine(config)
    os.makedirs(args.output_dir, exist_ok=True)
    today_str = date.today().isoformat()

    all_results: dict[str, list[DayResult]] = {}

    for ticker in args.tickers:
        start_date = TICKER_START_DATES.get(ticker, "2026-02-15")
        if config.get("strategy_mode") == "layer":
            dirs = config.get("layer_entry_directions", "both")
            eod_t = config.get("call_track_eod_time_pacific", "12:45")
            max_r = config.get("max_roll_count", 3)
            eod_pct = config.get("layer_eod_exit_pct", 0.0)
            chk = config.get("call_track_check_times_pacific", ["08:35", "10:35"])
            mode_label = (f"LAYER dirs={dirs}, checks={','.join(chk)}, "
                         f"eod={eod_t}, exit>{eod_pct*100:.0f}%w, max_rolls={max_r}")
        elif config.get("strategy_mode") == "call_track":
            mode_label = f"CALL-TRACK, {args.roll_interval}hr, budget={args.roll_budget_pct*100:.0f}%"
        else:
            eod_label = "EOD-roll" if config.get("eod_roll", True) else "hold-to-exp"
            mode_label = f"{config['leg_placement'].upper()}, {eod_label}"
        print(f"\n{'─'*90}")
        print(f"  {ticker} — VMaxMin.1 ({mode_label}, start={start_date})")
        print(f"{'─'*90}")

        all_dates = get_trading_dates(ticker, args.equity_dir, start_date, today_str)
        if len(all_dates) < 2:
            print(f"  Skipping {ticker}: not enough trading dates")
            continue

        eval_dates = all_dates[-args.lookback_days:] if len(all_dates) > args.lookback_days else all_dates
        # Need at least 1 date before eval range for prev_close
        if eval_dates[0] == all_dates[0] and len(all_dates) > len(eval_dates):
            pass  # prev_close will be None for first day, that's fine

        print(f"  {len(eval_dates)} eval days: {eval_dates[0]} to {eval_dates[-1]}")

        ticker_results = []
        active_carries: list[RolledPosition] = []

        for i, dt in enumerate(eval_dates):
            equity_df = load_equity_bars_df(ticker, dt, args.equity_dir)
            equity_prices = {}
            if not equity_df.empty:
                for _, row in equity_df.iterrows():
                    equity_prices[row["time_pacific"]] = float(row["close"])

            options_0dte = load_0dte_options(ticker, dt, args.options_dir)
            prev_close = get_prev_close(ticker, dt, all_dates, args.equity_dir)

            # Filter carries expiring today
            today_carries = [rp for rp in active_carries if rp.expiration_date == dt]

            result, new_carries = engine.run_single_day(
                ticker, dt, equity_df, equity_prices,
                options_0dte, all_dates, prev_close,
                carried_positions=today_carries)

            # Update active carries: remove expired, add new
            active_carries = [rp for rp in active_carries if rp.expiration_date > dt]
            active_carries.extend(new_carries)

            ticker_results.append(result)

            if args.verbose:
                print_day_summary(result)
                if result.trades:
                    for t in result.trades:
                        cr_str = f"{'+'if t.credit_or_debit>=0 else ''}${t.credit_or_debit:,.0f}"
                        print(f"    {t.event:<16s} {t.time_pacific:>5s} "
                              f"K={t.short_strike:>7.0f}/{t.long_strike:>7.0f} "
                              f"w={t.width:>4.0f} x{t.num_contracts:>2d} "
                              f"{cr_str:>10s} {t.notes}")
            elif (i + 1) % 10 == 0 or (i + 1) == len(eval_dates):
                done = sum(1 for r in ticker_results if not r.failure_reason)
                running_pnl = sum(r.net_pnl for r in ticker_results if not r.failure_reason)
                print(f"  ... {i+1}/{len(eval_dates)} days, "
                      f"{done} traded, running P&L: {'+'if running_pnl>=0 else ''}${running_pnl:,.0f}")

        # Report remaining carries
        if active_carries:
            print(f"\n  {len(active_carries)} carried position(s) still active after last eval date:")
            for rp in active_carries:
                print(f"    {rp.direction} {rp.short_strike}/{rp.long_strike} "
                      f"exp={rp.expiration_date} roll#{rp.roll_count} "
                      f"cum_cost=${rp.cumulative_roll_cost:.0f}")

        all_results[ticker] = ticker_results
        if not args.verbose:
            # Print all day summaries
            for r in ticker_results:
                print_day_summary(r)

        print_ticker_summary(ticker, ticker_results)

    # --- Cross-ticker summary ---
    if len(all_results) > 1:
        print(f"\n{'='*90}")
        print(f"  COMBINED SUMMARY — All Tickers")
        print(f"{'='*90}")
        all_pnls = []
        for ticker, results in all_results.items():
            valid = [r for r in results if not r.failure_reason]
            ticker_pnl = sum(r.net_pnl for r in valid)
            wins = sum(1 for r in valid if r.net_pnl > 0)
            days = len(valid)
            wr = wins / days * 100 if days else 0
            print(f"  {ticker:>5s}:  {days:>3d} days  "
                  f"{'+'if ticker_pnl>=0 else ''}${ticker_pnl:>10,.2f}  "
                  f"WR={wr:.0f}%")
            all_pnls.extend(r.net_pnl for r in valid)

        total = sum(all_pnls)
        wins = sum(1 for p in all_pnls if p > 0)
        wr = wins / len(all_pnls) * 100 if all_pnls else 0
        print(f"  {'TOTAL':>5s}:  {len(all_pnls):>3d} days  "
              f"{'+'if total>=0 else ''}${total:>10,.2f}  "
              f"WR={wr:.0f}%")

    # --- Save CSVs ---
    summary_df = results_to_dataframe(all_results)
    summary_path = os.path.join(args.output_dir, "vmaxmin_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved {len(summary_df)} day summaries to {summary_path}")

    trades_df = trades_to_dataframe(all_results)
    trades_path = os.path.join(args.output_dir, "vmaxmin_trades.csv")
    trades_df.to_csv(trades_path, index=False)
    print(f"Saved {len(trades_df)} trade records to {trades_path}")


if __name__ == "__main__":
    main()
