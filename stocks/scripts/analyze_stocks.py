import os
import sys
import asyncio
import argparse
import warnings
import logging
import multiprocessing
import signal
import pandas as pd
from pathlib import Path

# Project Path Setup
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.stock_db import get_stock_db, StockDBBase
from common.analysis.stocks import STRATEGY_CONFIG, analyze_stocks

warnings.filterwarnings('ignore')

def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    log_level_upper = log_level.upper()
    numeric_level = getattr(logging, log_level_upper, logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s [PID:%(process)d] [%(name)s] [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set level for this module's logger
    logger = logging.getLogger("StrategyEngine")
    logger.setLevel(numeric_level)
    return logger

# --- TICKER-SPECIFIC ANALYSIS ---
def print_ticker_analysis(final_df: pd.DataFrame, tickers: list, args, strategy_details_map: dict):
    """Print detailed analysis for specific tickers across all strategies."""
    print(f"\n{'='*80}")
    print(f"📊 DETAILED ANALYSIS FOR TICKER(S): {', '.join(tickers)}")
    print(f"{'='*80}")
    
    found_tickers = []
    missing_tickers = []
    
    for ticker in tickers:
        ticker_data = final_df[final_df['ticker'] == ticker]
        
        if ticker_data.empty:
            missing_tickers.append(ticker)
            continue
        
        found_tickers.append(ticker)
        row = ticker_data.iloc[0]
        strategy_details = strategy_details_map.get(ticker, {})
        
        print(f"\n{'─'*80}")
        print(f"📈 {ticker} - DETAILED STRATEGY ANALYSIS")
        print(f"{'─'*80}")
        print(f"Price: ${row['price']:.2f} | IV Rank: {row['iv_rank']:.1f} | Sector: {row['sector']} | Sector Z: {row.get('sector_z', 0):.1f}")
        print(f"Conviction Score: {row['conviction_score']} | Active Strategies: {row['strategies']}")
        print(f"\n{'Strategy':<25} {'Status':<12} {'Details'}")
        print(f"{'-'*80}")
        
        # BACKWARDATION
        back = strategy_details.get('BACKWARDATION', {})
        status = "✅ ACTIVE" if back.get('active') else "❌ INACTIVE"
        details = f"Spike: {back.get('spike_score', 0):.3f} (need >{back.get('threshold', 0):.2f}) | IV30: {back.get('iv_30', 0):.1f}% | IV90: {back.get('iv_90', 0):.1f}%"
        print(f"{'BACKWARDATION':<25} {status:<12} {details}")
        
        # WHALE SQUEEZE
        whale = strategy_details.get('WHALE SQUEEZE', {})
        status = "✅ ACTIVE" if whale.get('active') else "❌ INACTIVE"
        vol_oi = whale.get('vol_oi_ratio', 0)
        threshold = whale.get('threshold', 0)
        base_threshold = whale.get('base_threshold', 0)
        time_factor = whale.get('time_fill_factor', 0)
        delta = whale.get('max_oi_delta', 0)
        details = f"Vol/OI: {vol_oi:.2f} (need >{threshold:.2f}, base={base_threshold:.2f}, time={time_factor:.2f}) | Delta: {delta:.3f} (need 0.15-0.55)"
        print(f"{'WHALE SQUEEZE':<25} {status:<12} {details}")
        
        # SECTOR RELATIVE
        sector_rel = strategy_details.get('SECTOR RELATIVE', {})
        status = "✅ ACTIVE" if sector_rel.get('active') else "❌ INACTIVE"
        sector_z = sector_rel.get('sector_z', 0)
        sector_avg = sector_rel.get('sector_avg_rank', 0)
        details = f"Sector Z: {sector_z:.1f} (need <-15.0) | Sector Avg IV Rank: {sector_avg:.1f} | IV Rank: {sector_rel.get('iv_rank', 0):.1f}"
        print(f"{'SECTOR RELATIVE':<25} {status:<12} {details}")
        
        # CASH FLOW KING
        cf = strategy_details.get('CASH FLOW KING', {})
        status = "✅ ACTIVE" if cf.get('active') else "❌ INACTIVE"
        fcf = cf.get('fcf_ratio')
        fcf_str = f"{fcf:.2f}" if fcf else "N/A"
        details = f"FCF Ratio: {fcf_str} (need <{cf.get('fcf_cap', 0):.1f}) | IV Rank: {cf.get('iv_rank', 0):.1f} (need <55)"
        print(f"{'CASH FLOW KING':<25} {status:<12} {details}")
        
        # MEAN REVERSION
        mean = strategy_details.get('MEAN REVERSION', {})
        status = "✅ ACTIVE" if mean.get('active') else "❌ INACTIVE"
        ma_50 = mean.get('ma_50')
        ma_str = f"${ma_50:.2f}" if ma_50 else "N/A"
        price_vs_ma = mean.get('price_vs_ma', 0)
        details = f"Price: ${mean.get('price', 0):.2f} | MA50: {ma_str} | Ratio: {price_vs_ma:.3f} (need 0.80-1.02) | IV Rank: {mean.get('iv_rank', 0):.1f} (need <45)"
        print(f"{'MEAN REVERSION':<25} {status:<12} {details}")
        
        # ACCUMULATION
        accum = strategy_details.get('ACCUMULATION', {})
        status = "✅ ACTIVE" if accum.get('active') else "❌ INACTIVE"
        ma_50 = accum.get('ma_50')
        ma_str = f"${ma_50:.2f}" if ma_50 else "N/A"
        above_ma = "Yes" if accum.get('price_above_ma') else "No"
        details = f"Price: ${accum.get('price', 0):.2f} | MA50: {ma_str} | Above MA: {above_ma} | IV Rank: {accum.get('iv_rank', 0):.1f} (need <{accum.get('iv_rank_cap', 0):.1f} or <18)"
        print(f"{'ACCUMULATION':<25} {status:<12} {details}")
        
        print(f"\n💡 Action Plan: {row['action_plan']}")
    
    if missing_tickers:
        print(f"\n⚠️  Tickers not found in database: {', '.join(missing_tickers)}")
    
    if found_tickers:
        print(f"\n{'='*80}")
        print(f"📊 SUMMARY FOR {len(found_tickers)} TICKER(S)")
        print(f"{'='*80}")
        summary_df = final_df[final_df['ticker'].isin(found_tickers)][
            ['ticker', 'price', 'iv_rank', 'sector_z', 'conviction_score', 'strategies', 'sector']
        ].sort_values('conviction_score', ascending=False)
        print(summary_df.to_string(index=False))

# --- REPORT GENERATION ---
async def generate_report(db_instance: StockDBBase, args, shutdown_event):
    """Generate and display stock analysis report."""
    logger = logging.getLogger("StrategyEngine")
    
    # Prepare configuration from args
    worker_cfg = vars(args)
    
    # Expand directory path (handles ~/, environment variables, etc.)
    expanded_path = os.path.expandvars(os.path.expanduser(args.symbols_dir))
    symbols_dir = str(Path(expanded_path).resolve())
    
    # Call the library function to perform analysis
    print(f"Analyzing stocks with {args.workers} workers...")
    print(f"Symbols directory: {symbols_dir}")
    final_df, strategy_details_map = await analyze_stocks(
        db_instance=db_instance,
        symbols_dir=symbols_dir,
        config=worker_cfg,
        workers=args.workers,
        shutdown_event=shutdown_event
    )
    
    if final_df.empty:
        logger.error("No analysis results returned.")
        return

    # Handle ticker-specific analysis if requested
    if args.ticker:
        tickers_to_analyze = [t.upper() for t in args.ticker]
        print_ticker_analysis(final_df, tickers_to_analyze, args, strategy_details_map)
        return

    # Categorical Output
    cat_config = [
        ('BACKWARDATION', '⚠️  BACKWARDATION', '\033[91m'),
        ('WHALE SQUEEZE', '🐳 WHALE SQUEEZE', '\033[94m'),
        ('SECTOR RELATIVE', '📊 SECTOR RELATIVE', '\033[95m'),
        ('CASH FLOW KING', '👑 CASH FLOW KINGS', '\033[93m'),
        ('MEAN REVERSION', '📈 MEAN REVERSION', '\033[96m'),
        ('ACCUMULATION', '🟢 ACCUMULATION', '\033[92m')
    ]

    print(f"\n✅ ANALYZED {len(final_df)} TICKERS")
    for strat, label, color in cat_config:
        subset = final_df[final_df['strategies'].str.contains(strat)].sort_values('iv_rank').head(args.top_n)
        if subset.empty: continue
        print(f"\n{color}{label} (Top {args.top_n})\033[0m")
        print(f"   {'Ticker':<8} {'Price':<10} {'IV Rank':<10} {'Sector':<20}")
        for _, row in subset.iterrows():
            print(f"   {row['ticker']:<8} ${row['price']:<9.2f} {row['iv_rank']:<10.1f} {row['sector']:<20}")

    # Final Combined Ranking
    print("\n" + "="*80 + "\n🏆 FINAL RANKED OPPORTUNITIES (CONVICTION RANK)\n" + "="*80)
    top_picks = final_df[final_df['conviction_score'] > 0].sort_values(['conviction_score', 'iv_rank'], ascending=[False, True]).head(args.top_n)
    if not top_picks.empty:
        print(f"{'Ticker':<8} {'Score':<6} {'Price':<10} {'Strategies':<35} {'Action Plan'}")
        for _, row in top_picks.iterrows():
            print(f"{row['ticker']:<8} {row['conviction_score']:<6} ${row['price']:<9.2f} {row['strategies']:<35} {row['action_plan']}")
    
    if args.csv: final_df.to_csv(args.csv, index=False)

# --- MAIN & CLI ---
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-core options strategy analysis with sector-relative value and time-normalized volume.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=None,
        help="QuestDB connection string (e.g., questdb://user:password@localhost:8812/stock_data). "
             "If not provided, uses default from environment or repo rules."
    )
    parser.add_argument(
        '--db-file',
        type=str,
        default=None,
        help="Alias for --db-path (for compatibility)."
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Logging level (default: INFO)."
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help="Disable Redis caching for QuestDB operations (default: cache enabled)."
    )
    parser.add_argument(
        '--symbols-dir',
        type=str,
        default="~/var/US-Stock-Symbols",
        help="Local Git directory for sector metadata (default: ~/var/US-Stock-Symbols)."
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help="Number of results to print per category (default: 10)."
    )
    parser.add_argument(
        '--csv',
        type=str,
        help="Export results to CSV file."
    )
    parser.add_argument(
        '--ticker',
        type=str,
        nargs='+',
        help="Specific ticker(s) to analyze in detail. Shows how each ticker performs across all strategies. "
             "Can specify multiple tickers: --ticker AAPL MSFT TSLA"
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=max(1, int(multiprocessing.cpu_count() * 0.9)),
        help="Number of worker processes for parallel processing (default: 90%% of CPU count)."
    )
    
    # Strategy Parameters
    parser.add_argument(
        '--vol-oi-ratio',
        type=float,
        default=1.5,
        help="Whale Squeeze trigger - Vol/OI ratio threshold (default: 1.5)."
    )
    parser.add_argument(
        '--spike-threshold',
        type=float,
        default=1.12,
        help="Backwardation trigger - IV spike threshold (default: 1.12)."
    )
    parser.add_argument(
        '--fcf-cap',
        type=float,
        default=20.0,
        help="FCF ratio cap for cash flow king (default: 20.0)."
    )
    parser.add_argument(
        '--iv-rank-cap',
        type=float,
        default=45.0,
        help="IV rank cap for accumulation (default: 45.0)."
    )
    parser.add_argument(
        '--ma-floor',
        type=float,
        default=0.85,
        help="MA floor for mean reversion (default: 0.85)."
    )

    # Strategy Toggles
    for k in ['whale', 'cf', 'mean', 'accum', 'income', 'back']:
        parser.add_argument(
            f'--no-{k}',
            action='store_true',
            help=f"Disable {STRATEGY_CONFIG.get(k, {}).get('name', k.upper())} strategy."
        )
    parser.add_argument(
        '--no-sector-rel',
        action='store_true',
        help="Disable SECTOR RELATIVE strategy."
    )

    return parser.parse_args()

def setup_database(args) -> StockDBBase:
    """Create and initialize database instance.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Database instance
    """
    # Use --db-file if provided, otherwise --db-path, otherwise default
    db_path = args.db_file or args.db_path
    
    # Default to the connection string from repo rules if not provided
    if db_path is None:
        db_path = "questdb://stock_user:stock_password@ms1.kundu.dev:8812/stock_data"
    
    enable_cache = not args.no_cache
    
    # Determine database type from connection string
    if db_path.startswith('questdb://'):
        db_instance = get_stock_db(
            "questdb", 
            db_path, 
            log_level=args.log_level, 
            enable_cache=enable_cache, 
            redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
        )
    elif db_path.startswith('postgresql://'):
        db_instance = get_stock_db("postgresql", db_path, log_level=args.log_level)
    else:
        # Assume QuestDB if no prefix
        db_instance = get_stock_db(
            "questdb", 
            db_path, 
            log_level=args.log_level, 
            enable_cache=enable_cache, 
            redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
        )
    
    return db_instance

async def main():
    """Main async entry point."""
    args = parse_args()
    
    # Set up logging with the specified log level
    logger = setup_logging(args.log_level)
    
    # Create a shutdown event for graceful termination
    shutdown_event = multiprocessing.Event()
    
    # Set up signal handlers for graceful shutdown
    if sys.platform != 'win32':
        try:
            loop = asyncio.get_running_loop()
            def signal_handler():
                logger.warning("\n⚠️  Received SIGINT (Ctrl-C). Initiating graceful shutdown...")
                shutdown_event.set()
                for task in asyncio.all_tasks(loop):
                    if task != asyncio.current_task(loop):
                        task.cancel()
            loop.add_signal_handler(signal.SIGINT, signal_handler)
            loop.add_signal_handler(signal.SIGTERM, signal_handler)
        except (RuntimeError, NotImplementedError):
            # Fallback for platforms that don't support add_signal_handler
            pass
    
    db_instance = setup_database(args)
    
    try:
        await generate_report(db_instance, args, shutdown_event)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received in main")
        shutdown_event.set()
    except asyncio.CancelledError:
        logger.warning("Tasks cancelled due to shutdown signal")
        shutdown_event.set()
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        raise
    finally:
        # Ensure proper cleanup
        shutdown_event.set()
        if hasattr(db_instance, 'close'):
            await db_instance.close()
        logger.debug("Cleanup complete")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    asyncio.run(main())
