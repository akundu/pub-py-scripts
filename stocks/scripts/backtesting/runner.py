"""CLI entry point for the backtesting framework."""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.backtesting.config import BacktestConfig
from scripts.backtesting.engine import BacktestEngine


def setup_logging(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("backtesting")


def main():
    parser = argparse.ArgumentParser(
        description="""
Modular Backtesting Framework

Run backtests for options trading strategies using YAML/JSON configs.
Supports credit spreads (0DTE, multi-day, scale-in, tiered),
composable constraints, exit rules, and grid sweeps.
        """,
        epilog="""
Examples:
  %(prog)s --config configs/credit_spread_0dte_ndx.yaml
      Run a 0DTE credit spread backtest on NDX

  %(prog)s --config configs/grid_sweep_comprehensive.yaml
      Run a grid sweep over multiple parameter combinations

  %(prog)s --config configs/credit_spread_0dte_ndx.yaml --dry-run
      Preview what the backtest would do without executing

  %(prog)s --config configs/credit_spread_0dte_ndx.yaml --ticker SPX
      Override the ticker from the config file

  %(prog)s --config configs/credit_spread_0dte_ndx.yaml --start-date 2026-01-01 --end-date 2026-02-28
      Override the date range
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", required=True,
        help="Path to YAML or JSON config file"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview backtest without executing"
    )
    parser.add_argument(
        "--ticker",
        help="Override ticker from config"
    )
    parser.add_argument(
        "--start-date",
        help="Override start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        help="Override end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--lookback-days", type=int,
        help="Override lookback days from config"
    )
    parser.add_argument(
        "--num-processes", type=int,
        help="Override number of parallel processes"
    )
    parser.add_argument(
        "--log-level", default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: from config)"
    )
    parser.add_argument(
        "--output-dir",
        help="Override output directory"
    )

    args = parser.parse_args()

    # Load config
    config = BacktestConfig.load(args.config)

    # Apply CLI overrides
    if args.ticker:
        config.infra.ticker = args.ticker
    if args.start_date:
        config.infra.start_date = args.start_date
    if args.end_date:
        config.infra.end_date = args.end_date
    if args.lookback_days is not None:
        config.infra.lookback_days = args.lookback_days
    if args.num_processes is not None:
        config.infra.num_processes = args.num_processes
    if args.output_dir:
        config.infra.output_dir = args.output_dir

    log_level = args.log_level or config.infra.log_level
    logger = setup_logging(log_level)

    # Import providers and strategies to trigger auto-registration
    import scripts.backtesting.providers.csv_equity_provider  # noqa: F401
    import scripts.backtesting.providers.csv_options_provider  # noqa: F401
    import scripts.backtesting.instruments.credit_spread  # noqa: F401
    import scripts.backtesting.instruments.iron_condor  # noqa: F401
    import scripts.backtesting.strategies.credit_spread.zero_dte  # noqa: F401
    import scripts.backtesting.strategies.credit_spread.multi_day  # noqa: F401
    import scripts.backtesting.strategies.credit_spread.scale_in  # noqa: F401
    import scripts.backtesting.strategies.credit_spread.tiered  # noqa: F401
    import scripts.backtesting.strategies.credit_spread.time_allocated  # noqa: F401
    import scripts.backtesting.strategies.credit_spread.gate_filtered  # noqa: F401
    import scripts.backtesting.strategies.credit_spread.percentile_entry  # noqa: F401

    # Check for grid sweep mode
    if config.report.grid_sweep:
        _run_grid_sweep(config, logger, args.dry_run)
    else:
        _run_single(config, logger, args.dry_run)


def _run_single(config: BacktestConfig, logger: logging.Logger, dry_run: bool):
    engine = BacktestEngine(config, logger)
    results = engine.run(dry_run=dry_run)

    if dry_run:
        logger.info("Dry run complete.")
    else:
        metrics = results.get("metrics", {})
        logger.info(f"Backtest complete. {metrics.get('total_trades', 0)} trades processed.")


def _run_grid_sweep(config: BacktestConfig, logger: logging.Logger, dry_run: bool):
    from scripts.backtesting.results.grid_sweep import GridSweep

    gs = GridSweep(config, config.report.grid_sweep.param_grid, executor=None)
    configs = gs.generate_configs()

    logger.info(f"Grid sweep: {len(configs)} parameter combinations")

    if dry_run:
        for i, c in enumerate(configs[:5]):
            logger.info(f"  Config {i+1}: {c.strategy.params}")
        if len(configs) > 5:
            logger.info(f"  ... and {len(configs) - 5} more")
        return

    results = gs.run()
    ranked = gs.rank(results, sort_by=config.report.metrics[:1] or ["roi"])
    logger.info(f"Grid sweep complete. Top config: {ranked.iloc[0].to_dict() if len(ranked) > 0 else 'N/A'}")


if __name__ == "__main__":
    main()
