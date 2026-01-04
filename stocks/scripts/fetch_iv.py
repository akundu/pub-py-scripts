import os
import sys
import json
import argparse
import logging
import multiprocessing
import asyncio
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path so `common` can be imported when running from any cwd
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.symbol_loader import add_symbol_arguments, get_symbols_from_args
from common.iv_analysis import IVAnalyzer
from common.logging_utils import get_logger

# --- Configuration ---
DATA_DIR = "data"

# --- WORKER FUNCTION ---
def worker_task(ticker, cal_days, force_api, config, log_level_str="ERROR", server_url=None, use_polygon=False, data_dir="data", db_config=None):
    """Worker task for processing a single ticker's IV analysis."""
    # Setup logger for this worker process
    worker_logger = get_logger(f"IVAnalyzer.{os.getpid()}")
    log_level = getattr(logging, log_level_str.upper(), logging.ERROR)
    worker_logger.setLevel(log_level)
    
    # Create async worker function
    async def _async_worker():
        from common.stock_db import get_stock_db
        
        # Initialize database if config provided
        db_instance = None
        if db_config:
            try:
                db = get_stock_db('questdb', db_config=db_config, enable_cache=True,
                                 redis_url=config.get('redis_url'), log_level=log_level_str,
                                 auto_init=False)
                await db._init_db()
                db_instance = db
            except Exception as e:
                worker_logger.warning(f"[{ticker}] Failed to initialize database: {e}, will use HTTP server only")
        
        try:
            # Initialize IV Analyzer
            analyzer = IVAnalyzer(
                polygon_api_key=config['poly_key'],
                data_dir=data_dir,
                redis_url=config.get('redis_url'),
                server_url=server_url,
                db_instance=db_instance,  # Pass database for direct access
                use_polygon=False,  # Never use Polygon for price history
                logger=worker_logger
            )
            
            # Get IV analysis (now async)
            result, needs_update = await analyzer.get_iv_analysis(
                ticker=ticker,
                calendar_days=cal_days,
                force_refresh=force_api
            )
            
            return result, needs_update
        finally:
            # Close database if we opened it
            if db_instance:
                try:
                    await db_instance.close()
                except Exception:
                    pass
    
    try:
        # Run async worker in new event loop
        return asyncio.run(_async_worker())
    except Exception as e:
        worker_logger.error(f"[{ticker}] Worker error: {e}", exc_info=True)
        return {ticker: f"Error: {e}"}, False

async def main():
    parser = argparse.ArgumentParser()
    
    # Add symbol loading arguments
    add_symbol_arguments(parser, required=False, allow_positional=False, include_symbols_list=True)
    
    parser.add_argument("-c", "--calendar-days", type=int, default=90)
    parser.add_argument("--dont-sync", action="store_true", help="Don't force API refresh (default: sync/refresh)")
    parser.add_argument("--dont-save", action="store_true", help="Don't save IV analysis to database (default: save)")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='ERROR',
                        help="Logging level (default: ERROR)")
    parser.add_argument("-w", "--workers", type=int, help="Number of processes")
    parser.add_argument("--server-url", type=str, default="http://localhost:9100",
                        help="URL of local db_server endpoint (default: http://localhost:9100). Will auto-add http:// if missing.")
    parser.add_argument("--use-polygon", action="store_true",
                        help="Force using Polygon API instead of local server for historical data")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Data directory for symbol lists (default: data)")
    parser.add_argument("--db-config", type=str, default=None,
                        help="Database connection string (default: from QUESTDB_URL env var or questdb://user:password@localhost:8812/stock_data)")
    args = parser.parse_args()
    
    # Normalize server URL - add http:// if missing
    if args.server_url and not args.server_url.startswith(('http://', 'https://')):
        args.server_url = f"http://{args.server_url}"
    
    # Determine sync flag: default is True (sync/refresh), unless --dont-sync is provided (then sync=False)
    # --dont-sync means "don't sync" = use cache, so sync=False
    # Without --dont-sync, we default to sync=True (force refresh)
    args.sync = True  # Default: sync/refresh
    if args.dont_sync:
        # User explicitly said --dont-sync, which means don't sync (use cache)
        args.sync = False
    
    # Setup logger
    logger = get_logger("fetch_iv_debug")
    log_level = getattr(logging, args.log_level.upper(), logging.ERROR)
    logger.setLevel(log_level)
    
    # Ensure data directory exists
    data_dir = args.data_dir if hasattr(args, 'data_dir') else DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    
    # Get symbols from arguments - handle simple case first
    symbols = []
    if hasattr(args, 'symbols') and args.symbols:
        # Direct symbols provided - no need for async call
        symbols = [s.upper() for s in args.symbols]
        logger.info(f"Using {len(symbols)} symbols provided via --symbols: {', '.join(symbols)}")
    elif hasattr(args, 'symbols_list') and args.symbols_list:
        # Load from YAML file
        from common.symbol_loader import load_symbols_from_yaml
        symbols = load_symbols_from_yaml(args.symbols_list, quiet=(args.log_level == "ERROR"))
        if symbols:
            logger.info(f"Loaded {len(symbols)} symbols from YAML file: {args.symbols_list}")
    elif hasattr(args, 'types') and args.types:
        # Load from types - requires async
        try:
            symbols = await get_symbols_from_args(args, quiet=(args.log_level == "ERROR"))
        except Exception as e:
            logger.error(f"Error getting symbols from types: {e}", exc_info=True)
            return
    else:
        logger.error("No symbols provided. Use --symbols, --symbols-list, or --types to specify symbols.")
        return
    
    if not symbols:
        logger.error("No symbols found. Please check your input.")
        return
    
    # Always include VOO for relative ranking
    all_tickers = list(set(["VOO"] + [t.upper() for t in symbols]))
    
    worker_config = {
        'poly_key': os.getenv("POLYGON_API_KEY"),
        'redis_url': os.getenv("REDIS_URL")
    }
    
    # Get database config for worker
    db_config = args.db_config if hasattr(args, 'db_config') and args.db_config else (
        os.getenv("QUESTDB_URL") or "questdb://user:password@localhost:8812/stock_data"
    )
    
    total_cores = multiprocessing.cpu_count()
    max_workers = args.workers if args.workers else max(1, int(total_cores * 0.90))
    logger.info(f"Engine starting on {max_workers}/{total_cores} cores.")
    logger.info(f"Processing {len(symbols)} symbol(s): {', '.join(symbols)}")
    logger.info(f"Using database: {db_config[:50]}... for price history")
    if args.server_url:
        logger.info(f"HTTP server fallback: {args.server_url}")

    results_map = {}
    tickers_needing_update = []

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(worker_task, t, args.calendar_days, args.sync, worker_config, args.log_level, 
                               args.server_url, False, data_dir, db_config): t 
                for t in all_tickers
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data, needs_update = future.result()
                    results_map[ticker] = data
                    if needs_update: tickers_needing_update.append(ticker)
                except Exception as e: logger.error(f"Worker failed for {ticker}: {e}")

        voo_rank = 50.0
        if "VOO" in results_map and "metrics" in results_map["VOO"]:
            voo_rank = results_map["VOO"]["metrics"]["rank"]
        
        final_output = []
        for t in symbols:
            t_upper = t.upper()
            if t_upper in results_map:
                data = results_map[t_upper]
                if "metrics" in data:
                    ticker_rank = data['metrics']['rank']
                    # Calculate relative rank as ratio (ticker_rank / voo_rank)
                    # 1.0 = equal, >1.0 = higher, <1.0 = lower
                    if voo_rank > 0:
                        data['relative_rank'] = round(ticker_rank / voo_rank, 2)
                    else:
                        data['relative_rank'] = 1.0 if ticker_rank == 0 else None
                    final_output.append(data)
                else: final_output.append(data)

        print("\n" + json.dumps(final_output, indent=4))
        
        # Save IV analysis data to database and cache if enabled
        if not args.dont_save:
            await save_iv_analysis_to_db(final_output, args.db_config, logger, args.log_level)
        
        if tickers_needing_update:
            print(f"\n[INFO] Updating {len(tickers_needing_update)} stale records...")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(worker_task, t, args.calendar_days, True, worker_config, args.log_level,
                                   args.server_url, False, data_dir, db_config) 
                    for t in tickers_needing_update
                ]
                for f in as_completed(futures): pass
            print("[INFO] Background updates complete.")

    except KeyboardInterrupt: logger.warning("\n[!] Ctrl-C detected.")


async def save_iv_analysis_to_db(results: list, db_config: Optional[str], logger: logging.Logger, log_level: str):
    """Save IV analysis results to financial_info table and cache."""
    if not results:
        return
    
    try:
        # Get database config
        if not db_config:
            # Try environment variable first, then fallback to default
            db_config = os.getenv("QUESTDB_URL") or os.getenv("QUEST_DB_STRING") or "questdb://user:password@localhost:8812/stock_data"
        
        # Import database utilities
        from common.stock_db import get_stock_db
        
        # Initialize database connection
        db = get_stock_db('questdb', db_config=db_config, enable_cache=True, 
                         redis_url=os.getenv("REDIS_URL"), log_level=log_level)
        await db._init_db()
        
        # Get current date for financial_info
        from datetime import date
        today = date.today().isoformat()
        
        saved_count = 0
        for result in results:
            if "ticker" not in result or "metrics" not in result:
                continue
            
            ticker = result["ticker"]
            
            # Extract IV metrics
            metrics = result.get("metrics", {})
            strategy = result.get("strategy", {})
            
            # Parse IV_30d from percentage string (e.g., "40.41%")
            iv_30d_str = metrics.get("iv_30d", "")
            iv_30d = None
            if iv_30d_str and iv_30d_str.endswith("%"):
                try:
                    iv_30d = float(iv_30d_str.rstrip("%")) / 100.0
                except (ValueError, AttributeError):
                    pass
            
            # Get rank values
            iv_rank = metrics.get("rank")
            relative_rank = result.get("relative_rank")
            
            # Convert full result to JSON string
            iv_analysis_json = json.dumps(result)
            
            # Prepare financial_data dict
            financial_data = {
                'date': today,
                'iv_30d': iv_30d,
                'iv_rank': float(iv_rank) if iv_rank is not None else None,
                'relative_rank': float(relative_rank) if relative_rank is not None else None,
                'iv_analysis_json': iv_analysis_json,
                'iv_analysis_spare': None  # Spare column for future use
            }
            
            try:
                await db.save_financial_info(ticker, financial_data)
                saved_count += 1
                if log_level == "DEBUG":
                    logger.debug(f"Saved IV analysis for {ticker} to financial_info")
            except Exception as e:
                logger.error(f"Error saving IV analysis for {ticker}: {e}")
        
        if saved_count > 0:
            logger.info(f"Saved IV analysis for {saved_count} ticker(s) to financial_info table and cache")
        
        await db.close()
        
    except Exception as e:
        logger.error(f"Error saving IV analysis to database: {e}", exc_info=True)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    asyncio.run(main())
