"""NetDebitMaxMin Backtest Runner — CLI entry point.

Runs the contrarian intraday debit spread strategy across multiple
tickers and days. Buys OTM debit spreads at new HOD/LOD extremes,
betting on mean reversion. No rolling — dead layers are abandoned.

Usage:
  python scripts/backtesting/scripts/run_netdebit_maxmin_backtest.py --tickers RUT --lookback-days 30

  python scripts/backtesting/scripts/run_netdebit_maxmin_backtest.py --tickers RUT --lookback-days 60 \\
      --leg-placement best_value --spread-width 10

  python scripts/backtesting/scripts/run_netdebit_maxmin_backtest.py --tickers RUT --lookback-days 30 \\
      --leg-placement itm --max-debit-pct 0.80 --num-contracts 5
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.backtesting.scripts.netdebit_maxmin_engine import (
    NETDEBIT_DEFAULT_CONFIG,
    TICKER_START_DATES,
    DebitDayResult,
    DebitTradeRecord,
    NetDebitMaxMinEngine,
    load_options_0dte_only,
)
from scripts.backtesting.scripts.vmaxmin_engine import (
    get_prev_close,
    get_trading_dates,
    load_0dte_options,
    load_equity_bars_df,
    load_equity_prices,
)


def print_day_summary(result: DebitDayResult) -> None:
    """Print a one-line summary for a single day."""
    pnl = result.net_pnl
    pnl_str = f"{'+'if pnl>=0 else ''}${pnl:,.0f}"
    fail_str = f" [{result.failure_reason}]" if result.failure_reason else ""

    print(f"  {result.date}  {result.num_layers:>2d}L  "
          f"open={result.open_price:>8.1f}  close={result.close_price:>8.1f}  "
          f"hod={result.hod:>8.1f}  lod={result.lod:>8.1f}  "
          f"debit=${result.total_debits_paid:>7,.0f}  "
          f"payout=${result.total_payouts:>7,.0f}  "
          f"comm=${result.total_commissions:>4,.0f}  "
          f"pnl={pnl_str:>8s}{fail_str}")


def print_ticker_summary(ticker: str, results: list[DebitDayResult]) -> None:
    """Print aggregate metrics for a ticker."""
    valid = [r for r in results if not r.failure_reason]
    if not valid:
        print(f"\n  {ticker}: No valid trading days")
        return

    pnls = [r.net_pnl for r in valid]
    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    flat = sum(1 for p in pnls if p == 0)
    win_rate = wins / len(pnls) * 100 if pnls else 0
    avg_layers = np.mean([r.num_layers for r in valid])
    total_debits = sum(r.total_debits_paid for r in valid)
    total_payouts = sum(r.total_payouts for r in valid)
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

    # Avg debit per day
    avg_debit = np.mean([r.total_debits_paid for r in valid])

    print(f"\n{'='*90}")
    print(f"  {ticker} — NetDebitMaxMin Summary ({len(valid)} trading days)")
    print(f"{'='*90}")
    print(f"  Total P&L:       {'+'if total_pnl>=0 else ''}${total_pnl:>12,.2f}")
    print(f"  Win Rate:         {win_rate:>10.1f}% ({wins}W / {losses}L / {flat}F)")
    print(f"  Avg Daily P&L:   {'+'if daily_mean>=0 else ''}${daily_mean:>12,.2f}")
    print(f"  Sharpe Ratio:     {sharpe:>10.2f}")
    print(f"  Profit Factor:    {profit_factor:>10.2f}")
    print(f"  Max Drawdown:     ${max_dd:>10,.2f}")
    print(f"  Total Debits:     ${total_debits:>12,.2f}")
    print(f"  Total Payouts:    ${total_payouts:>12,.2f}")
    print(f"  Total Commissions:${total_commissions:>11,.2f}")
    print(f"  Avg Layers/Day:   {avg_layers:>10.1f}")
    print(f"  Avg Debit/Day:    ${avg_debit:>10,.2f}")
    print(f"  Skipped Days:     {len(results) - len(valid):>10d}")


def results_to_dataframe(all_results: dict[str, list[DebitDayResult]]) -> pd.DataFrame:
    rows = []
    for ticker, results in all_results.items():
        for r in results:
            rows.append({
                "ticker": ticker,
                "date": r.date,
                "open_price": r.open_price,
                "close_price": r.close_price,
                "hod": r.hod,
                "lod": r.lod,
                "num_layers": r.num_layers,
                "total_debits_paid": r.total_debits_paid,
                "total_payouts": r.total_payouts,
                "total_commissions": r.total_commissions,
                "net_pnl": r.net_pnl,
                "num_trades": len(r.trades),
                "failure_reason": r.failure_reason,
            })
    return pd.DataFrame(rows)


def trades_to_dataframe(all_results: dict[str, list[DebitDayResult]]) -> pd.DataFrame:
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
                    "long_strike": t.long_strike,
                    "short_strike": t.short_strike,
                    "width": t.width,
                    "debit": t.debit,
                    "payout": t.payout,
                    "num_contracts": t.num_contracts,
                    "commission": t.commission,
                    "underlying_price": t.underlying_price,
                    "trigger": t.trigger,
                    "notes": t.notes,
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description='''
NetDebitMaxMin — Contrarian Intraday Debit Spread Strategy

Buys OTM debit spreads contrarian to the intraday trend:
  - New HOD → buy bear put debit spread (bet on reversal down)
  - New LOD → buy bull call debit spread (bet on reversal up)

Layers accumulate; old layers are abandoned (max loss = debit paid).
A single winning layer at settlement can cover all dead layers.

No rolling, no margin, no bid-ask slippage from closing positions.
        ''',
        epilog='''
Examples:
  %(prog)s --tickers RUT --lookback-days 30
      Quick 30-day backtest with RUT

  %(prog)s --tickers RUT --lookback-days 120 -v
      Extended backtest with per-trade details

  %(prog)s --tickers RUT --lookback-days 60 --leg-placement best_value
      Scan all strikes for best value (highest payout/debit ratio)

  %(prog)s --tickers RUT --lookback-days 60 --leg-placement itm --max-debit-pct 0.80
      ITM placement (already has intrinsic value, higher debit)

  %(prog)s --tickers RUT --spread-width 10 --num-contracts 5
      10-point wide spreads, 5 contracts per layer

  %(prog)s --tickers RUT --min-width 5 --max-width 15
      Flexible width range: engine picks best from 5-15 points

  %(prog)s --tickers RUT --check-times 08:35 10:35 12:35
      Custom check times (3 checks instead of default 5)

  %(prog)s --tickers RUT --max-daily-debit 2000 --max-layers 5
      Conservative: cap $2K/day spend, max 5 layers

  %(prog)s --tickers RUT --no-entry-spread
      Skip the opening entry spread, only layer on new extremes

  %(prog)s --tickers RUT --check-times 10:35 11:35 12:00 12:30 --no-entry-spread
      Afternoon-only: skip morning, layer on late-day reversals

  %(prog)s --tickers RUT --direction-filter puts_only
      Only buy put debit spreads (no LOD call spreads)

  %(prog)s --tickers RUT --single-layer --check-times 10:35 11:35 12:00
      Max 1 spread per direction per day (eliminates dead layer drag)

  %(prog)s --tickers RUT --min-range 15 --range-observation-time 09:30
      Require 15pt range established by 09:30 before layering
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tickers", nargs="+", default=["RUT"],
                        help="Tickers to backtest (default: RUT)")
    parser.add_argument("--lookback-days", type=int, default=30,
                        help="Number of trading days to backtest (default: 30)")
    parser.add_argument("--start-date", default=None,
                        help="Start date YYYY-MM-DD (overrides lookback-days)")
    parser.add_argument("--end-date", default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--output-dir", default="results/netdebit_maxmin",
                        help="Output directory (default: results/netdebit_maxmin)")
    parser.add_argument("--num-contracts", type=int, default=1,
                        help="Contracts per layer (default: 1)")
    parser.add_argument("--commission", type=float, default=10,
                        help="Commission per transaction in $ (default: 10)")

    # Width control
    parser.add_argument("--spread-width", type=float, default=5,
                        help="Target spread width in points (default: 5)")
    parser.add_argument("--min-width", type=float, default=None,
                        help="Minimum width (enables flexible range)")
    parser.add_argument("--max-width", type=float, default=None,
                        help="Maximum width (enables flexible range)")

    # Leg placement
    parser.add_argument("--leg-placement",
                        choices=["otm", "just_otm", "atm", "itm", "best_value"],
                        default="otm",
                        help="Where to place the long leg (default: otm)")
    parser.add_argument("--depth-pct", type=float, default=0.003,
                        help="OTM distance from extreme as fraction (default: 0.003 = 0.3%%)")

    # Debit bounds
    parser.add_argument("--max-debit-pct", type=float, default=0.50,
                        help="Max debit as fraction of width (default: 0.50 = 50%%)")
    parser.add_argument("--min-debit", type=float, default=0.20,
                        help="Minimum debit per share (default: 0.20)")

    # Risk caps
    parser.add_argument("--max-daily-debit", type=float, default=5000,
                        help="Max total $ spent on debit per day (default: 5000)")
    parser.add_argument("--max-layers", type=int, default=10,
                        help="Max concurrent layers (default: 10)")

    # Timing
    parser.add_argument("--check-times", nargs="+",
                        default=["07:35", "08:35", "09:35", "10:35", "11:35"],
                        help="Intraday check times in Pacific HH:MM")
    parser.add_argument("--entry-time", default="06:35",
                        help="Entry time in Pacific HH:MM (default: 06:35)")
    parser.add_argument("--breach-min-points", type=float, default=None,
                        help="Min points for HOD/LOD breach (default: min_step)")
    parser.add_argument("--no-entry-spread", action="store_true",
                        help="Skip the opening entry spread")

    # v2 filters
    parser.add_argument("--direction-filter",
                        choices=["both", "puts_only", "calls_only"],
                        default="both",
                        help="Direction filter: both, puts_only, calls_only (default: both)")
    parser.add_argument("--single-layer", action="store_true",
                        help="Max 1 spread per direction per day (eliminates dead layer drag)")
    parser.add_argument("--min-range", type=float, default=0,
                        help="Require HOD-LOD >= this points before layering (default: 0)")
    parser.add_argument("--range-observation-time", default=None,
                        help="Wait until this time to measure range (Pacific HH:MM, e.g. 09:30)")

    # Data sources
    parser.add_argument("--equity-dir", default="equities_output",
                        help="Equity data directory (default: equities_output)")
    parser.add_argument("--options-dir", default="options_csv_output_full_5",
                        help="Options data directory (default: options_csv_output_full_5)")

    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-trade details")
    args = parser.parse_args()

    config = {
        **NETDEBIT_DEFAULT_CONFIG,
        "entry_time_pacific": args.entry_time,
        "check_times_pacific": args.check_times,
        "num_contracts": args.num_contracts,
        "commission_per_transaction": args.commission,
        "spread_width_points": args.spread_width,
        "min_width": args.min_width,
        "max_width": args.max_width,
        "leg_placement": args.leg_placement,
        "depth_pct": args.depth_pct,
        "max_debit_pct_of_width": args.max_debit_pct,
        "min_debit": args.min_debit,
        "max_daily_debit": args.max_daily_debit,
        "max_concurrent_layers": args.max_layers,
        "open_entry_spread": not args.no_entry_spread,
        "direction_filter": args.direction_filter,
        "single_layer_per_direction": args.single_layer,
        "min_range_points": args.min_range,
        "range_observation_time": args.range_observation_time,
        "equity_dir": args.equity_dir,
        "options_dir": args.options_dir,
    }

    if args.breach_min_points is not None:
        config["breach_min_points"] = args.breach_min_points

    engine = NetDebitMaxMinEngine(config)
    os.makedirs(args.output_dir, exist_ok=True)
    today_str = args.end_date or date.today().isoformat()

    all_results: dict[str, list[DebitDayResult]] = {}

    for ticker in args.tickers:
        start_date = TICKER_START_DATES.get(ticker, "2025-09-09")

        print(f"\n{'─'*90}")
        print(f"  {ticker} — NetDebitMaxMin ({args.leg_placement.upper()}, "
              f"w={args.spread_width}pt, debit<={args.max_debit_pct*100:.0f}%w)")
        print(f"{'─'*90}")

        all_dates = get_trading_dates(ticker, args.equity_dir, start_date, today_str)
        if len(all_dates) < 2:
            print(f"  Skipping {ticker}: not enough trading dates")
            continue

        if args.start_date:
            eval_dates = [d for d in all_dates if d >= args.start_date]
            if args.end_date:
                eval_dates = [d for d in eval_dates if d <= args.end_date]
        else:
            eval_dates = all_dates[-args.lookback_days:] if len(all_dates) > args.lookback_days else all_dates

        print(f"  {len(eval_dates)} eval days: {eval_dates[0]} to {eval_dates[-1]}")

        ticker_results = []

        for i, dt in enumerate(eval_dates):
            equity_df = load_equity_bars_df(ticker, dt, args.equity_dir)
            equity_prices = {}
            if not equity_df.empty:
                for _, row in equity_df.iterrows():
                    equity_prices[row["time_pacific"]] = float(row["close"])

            # Load options — filter to 0DTE happens inside the engine
            options_all = load_0dte_options(ticker, dt, args.options_dir)
            prev_close = get_prev_close(ticker, dt, all_dates, args.equity_dir)

            result = engine.run_single_day(
                ticker, dt, equity_df, equity_prices,
                options_all, all_dates, prev_close)

            ticker_results.append(result)

            if args.verbose:
                print_day_summary(result)
                if result.trades:
                    for t in result.trades:
                        val_str = f"${t.debit:,.0f}" if t.event == "entry" else f"${t.payout:,.0f}"
                        print(f"    {t.event:<12s} {t.time_pacific:>5s} "
                              f"{t.direction:>4s} "
                              f"L={t.long_strike:>7.0f} S={t.short_strike:>7.0f} "
                              f"w={t.width:>4.0f} x{t.num_contracts:>2d} "
                              f"{val_str:>10s} {t.trigger:>5s} {t.notes}")
            elif (i + 1) % 10 == 0 or (i + 1) == len(eval_dates):
                done = sum(1 for r in ticker_results if not r.failure_reason)
                running_pnl = sum(r.net_pnl for r in ticker_results if not r.failure_reason)
                print(f"  ... {i+1}/{len(eval_dates)} days, "
                      f"{done} traded, running P&L: {'+'if running_pnl>=0 else ''}${running_pnl:,.0f}")

        all_results[ticker] = ticker_results
        if not args.verbose:
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
    summary_path = os.path.join(args.output_dir, "netdebit_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved {len(summary_df)} day summaries to {summary_path}")

    trades_df = trades_to_dataframe(all_results)
    trades_path = os.path.join(args.output_dir, "netdebit_trades.csv")
    trades_df.to_csv(trades_path, index=False)
    print(f"Saved {len(trades_df)} trade records to {trades_path}")


if __name__ == "__main__":
    main()
