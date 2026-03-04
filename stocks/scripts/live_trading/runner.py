"""CLI entry point for the live trading platform.

Usage:
    python -m scripts.live_trading.runner --config <yaml>           # Run live/paper trading
    python -m scripts.live_trading.runner --config <yaml> --dry-run # Preview config
    python -m scripts.live_trading.runner --performance --days 30   # Performance report
    python -m scripts.live_trading.runner --positions               # Show open positions
    python -m scripts.live_trading.runner --journal --days 7        # Recent journal entries
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def setup_logging(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("live_trading")


def main():
    parser = argparse.ArgumentParser(
        description="""
Live Paper Trading Platform

Run strategies during market hours with paper execution. Reuses the
backtesting framework's abstractions (DataProvider, SignalGenerator,
Constraint, ExitRule, Instrument) with live data from QuestDB and
csv_exports/options/.

Supports the NDX credit spread playbook with P80 percentile strikes,
multi-day position tracking, dynamic rolling, and performance reporting.
        """,
        epilog="""
Examples:
  %(prog)s --config scripts/live_trading/configs/ndx_credit_spread_paper.yaml
      Run NDX credit spread paper trading

  %(prog)s --config scripts/live_trading/configs/ndx_credit_spread_paper.yaml --dry-run
      Preview configuration without trading

  %(prog)s --performance --days 30
      Show performance report for last 30 days

  %(prog)s --positions
      Show all open positions

  %(prog)s --journal --days 7
      Show recent journal entries

  %(prog)s --daily-summary
      Show today's trading summary

  %(prog)s --daily-summary --date 2026-03-01
      Show summary for a specific date
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Run mode
    parser.add_argument(
        "--config",
        help="Path to YAML config file for live trading",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview config and component setup without trading",
    )

    # Report modes
    parser.add_argument(
        "--performance", action="store_true",
        help="Show performance metrics",
    )
    parser.add_argument(
        "--positions", action="store_true",
        help="Show open positions",
    )
    parser.add_argument(
        "--journal", action="store_true",
        help="Show recent journal entries",
    )
    parser.add_argument(
        "--daily-summary", action="store_true",
        help="Show daily trading summary",
    )

    # Common options
    parser.add_argument(
        "--days", type=int, default=30,
        help="Number of days for performance/journal reports (default: 30)",
    )
    parser.add_argument(
        "--date",
        help="Specific date for daily summary (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--ticker",
        help="Override ticker from config",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    parser.add_argument(
        "--position-db",
        default="data/live_trading/positions.json",
        help="Path to position store (default: data/live_trading/positions.json)",
    )
    parser.add_argument(
        "--journal-path",
        default="data/live_trading/journal.jsonl",
        help="Path to trade journal (default: data/live_trading/journal.jsonl)",
    )

    args = parser.parse_args()
    logger = setup_logging(args.log_level)

    # Import providers/instruments to trigger auto-registration
    import scripts.backtesting.providers.csv_equity_provider  # noqa: F401
    import scripts.backtesting.providers.csv_options_provider  # noqa: F401
    import scripts.backtesting.instruments.credit_spread  # noqa: F401
    import scripts.live_trading.providers.realtime_equity  # noqa: F401
    import scripts.live_trading.providers.realtime_options  # noqa: F401

    # Report modes (no config required)
    if args.performance:
        _show_performance(args, logger)
        return

    if args.positions:
        _show_positions(args, logger)
        return

    if args.journal:
        _show_journal(args, logger)
        return

    if args.daily_summary:
        _show_daily_summary(args, logger)
        return

    # Run mode (config required)
    if not args.config:
        parser.error("--config is required for live trading (or use --performance/--positions/--journal)")

    from scripts.live_trading.config import LiveConfig
    from scripts.live_trading.engine import LiveEngine

    config = LiveConfig.from_yaml(args.config)

    # Apply CLI overrides
    if args.ticker:
        config.infra.ticker = args.ticker
    if args.position_db != "data/live_trading/positions.json":
        config.live.position_db_path = args.position_db
    if args.journal_path != "data/live_trading/journal.jsonl":
        config.live.journal_path = args.journal_path

    engine = LiveEngine(config, logger)
    results = engine.start(dry_run=args.dry_run)

    if not args.dry_run and results:
        logger.info("=== Final Performance ===")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")


def _show_performance(args, logger):
    """Show performance metrics."""
    from scripts.live_trading.position_store import PositionStore

    store = PositionStore(args.position_db)
    end_date = date.today()
    start_date = end_date - timedelta(days=args.days)

    metrics = store.get_performance(start_date, end_date)

    print(f"\n=== Performance Report ({start_date} to {end_date}) ===")
    print(f"  Total trades:    {metrics['total_trades']}")
    print(f"  Win rate:        {metrics['win_rate']:.1f}%")
    print(f"  Net P&L:         ${metrics['net_pnl']:,.2f}")
    print(f"  Total credits:   ${metrics['total_credits']:,.2f}")
    print(f"  Total gains:     ${metrics['total_gains']:,.2f}")
    print(f"  Total losses:    ${metrics['total_losses']:,.2f}")
    print(f"  ROI:             {metrics['roi']:.1f}%")
    print(f"  Profit factor:   {metrics['profit_factor']:.2f}")
    print(f"  Sharpe ratio:    {metrics['sharpe']:.2f}")
    print(f"  Max drawdown:    ${metrics['max_drawdown']:,.2f}")
    print(f"  Avg P&L/trade:   ${metrics['avg_pnl']:,.2f}")


def _show_positions(args, logger):
    """Show open positions."""
    from scripts.live_trading.position_store import PositionStore

    store = PositionStore(args.position_db)
    positions = store.get_open_positions()

    if not positions:
        print("\nNo open positions.")
        return

    print(f"\n=== Open Positions ({len(positions)}) ===")
    for p in positions:
        pnl_str = f"${p.get('current_pnl', 0) or 0:,.2f}" if p.get('current_pnl') is not None else "N/A"
        print(
            f"  {p['position_id']}: {p['option_type'].upper()} "
            f"{p['short_strike']}/{p['long_strike']} "
            f"x{p['num_contracts']} "
            f"credit={p['initial_credit']:.4f} "
            f"DTE={p.get('dte', '?')} "
            f"P&L={pnl_str} "
            f"exp={p.get('expiration_date', '?')}"
        )

    total_risk = sum(p.get("max_loss", 0) for p in positions)
    total_unrealized = sum(p.get("current_pnl", 0) or 0 for p in positions)
    print(f"\n  Total risk:       ${total_risk:,.2f}")
    print(f"  Unrealized P&L:   ${total_unrealized:,.2f}")


def _show_journal(args, logger):
    """Show recent journal entries."""
    from scripts.live_trading.trade_journal import TradeJournal

    journal = TradeJournal(args.journal_path)
    end_date = date.today()
    start_date = end_date - timedelta(days=args.days)

    entries = journal.get_entries(start_date, end_date)

    if not entries:
        print(f"\nNo journal entries in last {args.days} days.")
        return

    print(f"\n=== Journal Entries ({len(entries)}) ===")
    for e in entries[-50:]:  # Last 50
        ts = e.timestamp.strftime("%Y-%m-%d %H:%M")
        detail_str = ""
        if e.details:
            if "pnl" in e.details:
                detail_str = f" P&L=${e.details['pnl']:,.2f}"
            elif "position_id" in e.details:
                detail_str = f" pos={e.details['position_id']}"
        reasoning = f" ({e.reasoning})" if e.reasoning else ""
        print(f"  {ts} [{e.event_type}] {e.ticker}{detail_str}{reasoning}")


def _show_daily_summary(args, logger):
    """Show daily trading summary."""
    from scripts.live_trading.position_store import PositionStore

    store = PositionStore(args.position_db)

    if args.date:
        from datetime import datetime
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = date.today()

    summary = store.get_daily_summary(target_date)

    print(f"\n=== Daily Summary ({summary['date']}) ===")
    print(f"  Positions opened:  {summary['positions_opened']}")
    print(f"  Positions closed:  {summary['positions_closed']}")
    print(f"  Positions open:    {summary['positions_open']}")
    print(f"  Realized P&L:      ${summary['realized_pnl']:,.2f}")
    print(f"  Unrealized P&L:    ${summary['unrealized_pnl']:,.2f}")
    print(f"  Total P&L:         ${summary['total_pnl']:,.2f}")


if __name__ == "__main__":
    main()
