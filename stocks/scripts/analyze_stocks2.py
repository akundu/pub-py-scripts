import pandas as pd
import json
import os
import sys
import asyncio
import argparse
import warnings
import logging
import multiprocessing
import signal
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Project Path Setup
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.stock_db import get_stock_db, StockDBBase

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

# --- THE QUANT HANDBOOK & CONFIG ---
STRATEGY_CONFIG = {
    'back': {
        'name': 'BACKWARDATION',
        'flag': '--spike-threshold',
        'default': 1.12,
        'range': '1.05 to 1.30 (Higher isolates extreme panic)',
        'desc': '30d IV > 90d IV. Short-term fear is significantly overpriced.',
        'plan': 'Sell 30-day calls against long positions.'
    },
    'whale': {
        'name': 'WHALE SQUEEZE',
        'flag': '--vol-oi-ratio',
        'default': 1.5,
        'range': '1.1 to 3.0 (Higher finds larger institutional sweeps)',
        'desc': 'Today\'s Volume over Previous Day\'s Open Interest.',
        'plan': 'Buy short-term OTM calls to ride Gamma pressure.'
    },
    'cf': {
        'name': 'CASH FLOW KING',
        'flag': '--fcf-cap',
        'default': 20.0,
        'range': '10.0 to 25.0 (Lower finds higher quality value)',
        'desc': 'High Free Cash Flow yield + Low IV Rank.',
        'plan': 'Buy ITM LEAPS and sell monthly calls (Diagonal).'
    },
    'mean': {
        'name': 'MEAN REVERSION',
        'flag': '--ma-floor',
        'default': 0.85,
        'range': '0.75 to 0.95 (Lower finds deeper oversold setups)',
        'desc': 'Oversold (Price < 50MA) with depressed volatility.',
        'plan': 'Buy near-term ITM calls for the mean reversion bounce.'
    },
    'accum': {
        'name': 'ACCUMULATION',
        'flag': '--iv-rank-cap',
        'default': 45.0,
        'range': '20.0 to 60.0 (Lower finds "clearance" pricing)',
        'desc': 'Intermediate uptrend with historically cheap volatility.',
        'plan': 'Systematically build core LEAP positions.'
    }
}

# --- WORKER TASK ---
def analyze_ticker_task(row, spy_rank, cfg):
    ticker = row['ticker']
    iv_rank = row.get('iv_rank', 50)
    price = row.get('current_price', 0)
    ma_50 = row.get('ma_50', 0)
    fcf_ratio = row.get('price_to_free_cash_flow')
    iv_30 = row.get('iv_30d', 0)
    iv_90 = row.get('iv_90d', 0)
    vol_oi_ratio = row.get('vol_oi_ratio', 0)
    max_oi_delta = row.get('max_oi_delta', 0)

    active_strats = []
    spike_score = iv_30 / iv_90 if iv_90 and iv_90 != 0 else 0
    
    # Detailed strategy evaluation (for ticker-specific analysis)
    strategy_details = {}

    # Evaluate Strategies using CLI overrides
    # BACKWARDATION
    back_active = not cfg['no_back'] and spike_score > cfg['spike_threshold']
    if back_active:
        active_strats.append('BACKWARDATION')
    strategy_details['BACKWARDATION'] = {
        'active': back_active,
        'spike_score': round(spike_score, 3),
        'threshold': cfg['spike_threshold'],
        'iv_30': round(iv_30, 2),
        'iv_90': round(iv_90, 2)
    }
    
    # WHALE SQUEEZE
    whale_active = not cfg['no_whale'] and vol_oi_ratio > cfg['vol_oi_ratio'] and 0.15 < max_oi_delta < 0.55
    if whale_active:
        active_strats.append('WHALE SQUEEZE')
    strategy_details['WHALE SQUEEZE'] = {
        'active': whale_active,
        'vol_oi_ratio': round(vol_oi_ratio, 2),
        'threshold': cfg['vol_oi_ratio'],
        'max_oi_delta': round(max_oi_delta, 3),
        'delta_range': '0.15-0.55'
    }
    
    # CASH FLOW KING
    cf_active = not cfg['no_cf'] and fcf_ratio and 0 < fcf_ratio < cfg['fcf_cap'] and iv_rank < 45
    if cf_active:
        active_strats.append('CASH FLOW KING')
    strategy_details['CASH FLOW KING'] = {
        'active': cf_active,
        'fcf_ratio': round(fcf_ratio, 2) if fcf_ratio else None,
        'fcf_cap': cfg['fcf_cap'],
        'iv_rank': round(iv_rank, 1),
        'iv_rank_threshold': 45
    }
    
    # MEAN REVERSION
    mean_active = not cfg['no_mean'] and ma_50 and (ma_50 * cfg['ma_floor']) < price < ma_50 and iv_rank < 40
    if mean_active:
        active_strats.append('MEAN REVERSION')
    strategy_details['MEAN REVERSION'] = {
        'active': mean_active,
        'price': round(price, 2),
        'ma_50': round(ma_50, 2) if ma_50 else None,
        'ma_floor': cfg['ma_floor'],
        'price_vs_ma': round((price / ma_50) if ma_50 else 0, 3),
        'iv_rank': round(iv_rank, 1),
        'iv_rank_threshold': 40
    }
    
    # ACCUMULATION
    accum_active = not cfg['no_accum'] and ((ma_50 and price > ma_50 and iv_rank < cfg['iv_rank_cap']) or (iv_rank < 15))
    if accum_active:
        active_strats.append('ACCUMULATION')
    strategy_details['ACCUMULATION'] = {
        'active': accum_active,
        'price': round(price, 2),
        'ma_50': round(ma_50, 2) if ma_50 else None,
        'price_above_ma': (ma_50 and price > ma_50) if ma_50 else False,
        'iv_rank': round(iv_rank, 1),
        'iv_rank_cap': cfg['iv_rank_cap'],
        'iv_rank_floor': 15
    }

    score = len(active_strats)
    plan = STRATEGY_CONFIG.get(active_strats[0].lower(), {'plan': 'Review'}).get('plan') if active_strats else "Hold"

    return {
        "ticker": ticker,
        "price": round(price, 2),
        "iv_rank": round(iv_rank, 1),
        "conviction_score": score,
        "strategies": ", ".join(active_strats) if active_strats else "None",
        "action_plan": plan,
        "signal_metric": round(vol_oi_ratio if 'WHALE SQUEEZE' in active_strats else iv_rank, 2),
        "sector": row.get('sector', 'Unknown'),
        "strategy_details": strategy_details  # Added for detailed analysis
    }

# --- DATA LAYER ---
def load_sector_data(data_dir: str):
    """Load sector data from JSON files. Returns DataFrame with 'ticker' and 'sector' columns."""
    metadata = []
    logger = logging.getLogger("StrategyEngine")
    
    # Check if directory exists
    if not os.path.isdir(data_dir):
        logger.warning(f"Sector data directory not found: {data_dir}. Using empty sector data.")
        return pd.DataFrame(columns=['ticker', 'sector'])
    
    # Files are in subdirectories (nasdaq/, nyse/, amex/) and named *_full_tickers.json (plural)
    # Try both patterns: recursive search and direct subdirectory search
    json_files = (
        glob.glob(os.path.join(data_dir, "**", "*_full_tickers.json"), recursive=True) +
        glob.glob(os.path.join(data_dir, "*_full_tickers.json")) +
        glob.glob(os.path.join(data_dir, "**", "*_full_ticker.json"), recursive=True) +
        glob.glob(os.path.join(data_dir, "*_full_ticker.json"))
    )
    
    # Remove duplicates
    json_files = list(set(json_files))
    
    if not json_files:
        # List what files are actually in the directory for debugging
        all_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.json'):
                    all_files.append(os.path.join(root, file))
        
        if all_files:
            logger.debug(f"Found JSON files in {data_dir}: {all_files[:5]}... (showing first 5)")
        else:
            logger.warning(f"No JSON files found in {data_dir}. Using empty sector data.")
        return pd.DataFrame(columns=['ticker', 'sector'])
    
    logger.debug(f"Found {len(json_files)} sector JSON files: {[os.path.basename(f) for f in json_files]}")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            ticker = item.get('symbol')
                            if ticker:  # Only add if ticker exists
                                metadata.append({
                                    'ticker': ticker,
                                    'sector': item.get('sector', 'Unknown')
                                })
        except Exception as e:
            logging.getLogger("StrategyEngine").debug(f"Could not parse {file_path}: {e}")
            continue
    
    if not metadata:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=['ticker', 'sector'])
    
    df = pd.DataFrame(metadata)
    if 'ticker' in df.columns:
        df = df.drop_duplicates('ticker')
    else:
        # If somehow we lost the ticker column, recreate it
        df = pd.DataFrame(columns=['ticker', 'sector'])
    
    return df

async def fetch_latest_market_data(db_instance: StockDBBase, symbols_dir: str):
    # Calculate lookback date in Python for QuestDB compatibility
    lookback_date = (datetime.now(timezone.utc) - timedelta(days=1)).replace(tzinfo=None)
    
    # Fix: Use GROUP BY instead of SAMPLE BY, and proper date arithmetic
    q_whale = f"SELECT ticker, sum(volume) as total_vol, sum(open_interest) as total_oi, avg(delta) as avg_delta FROM options_data WHERE timestamp > '{lookback_date}' AND option_type = 'call' GROUP BY ticker;"
    
    tasks = [
        db_instance.execute_select_sql("SELECT ticker, iv_rank, iv_30d, iv_90d, price_to_free_cash_flow FROM financial_info LATEST ON date PARTITION BY ticker;"),
        db_instance.execute_select_sql("SELECT ticker, price as current_price FROM realtime_data LATEST ON timestamp PARTITION BY ticker;"),
        db_instance.execute_select_sql("SELECT ticker, ma_50 FROM daily_prices LATEST ON date PARTITION BY ticker;"),
        db_instance.execute_select_sql(q_whale)
    ]
    df_f, df_rt, df_t, df_w = await asyncio.gather(*tasks)
    
    # Handle empty DataFrame from whale query
    if df_w.empty:
        # Create empty DataFrame with expected columns
        df_w = pd.DataFrame(columns=['ticker', 'total_vol', 'total_oi', 'avg_delta', 'vol_oi_ratio', 'max_oi_delta'])
    else:
        # Rename avg_delta to max_oi_delta for consistency
        if 'avg_delta' in df_w.columns:
            df_w = df_w.rename(columns={'avg_delta': 'max_oi_delta'})
        else:
            df_w['max_oi_delta'] = 0
        
        # Calculate vol_oi_ratio only if we have data
        df_w['vol_oi_ratio'] = df_w.apply(
            lambda r: (r['total_vol'] / r['total_oi']) if pd.notna(r['total_oi']) and r['total_oi'] > 0 else 0,
            axis=1
        )
    
    df_master = df_f.merge(df_rt, on='ticker', how='inner').merge(df_t, on='ticker', how='left').merge(df_w, on='ticker', how='left')
    # Debug: Print counts of pruned results after merging if log level is DEBUG
    logger = logging.getLogger("StrategyEngine")
    if logger.isEnabledFor(logging.INFO):
        orig_count = len(df_f)
        pruned_count = orig_count - len(df_master)
        logger.debug(f"fetch_latest_market_data: Pruned {pruned_count} tickers from {orig_count} (financial_info) after merging; {len(df_master)} remain in master set.")
    
    # Load sector data with error handling
    df_sec = load_sector_data(symbols_dir)
    
    # Ensure df_sec has the required columns before merging
    if df_sec.empty or 'ticker' not in df_sec.columns:
        # Create empty DataFrame with correct structure
        df_sec = pd.DataFrame(columns=['ticker', 'sector'])
    
    # Merge sector data, handling the case where it might be empty
    if not df_sec.empty:
        df_master = df_master.merge(df_sec, on='ticker', how='left')
    else:
        # If no sector data, add 'sector' column with 'Unknown' values
        df_master['sector'] = 'Unknown'
    
    # Fill any remaining NaN sector values
    df_master['sector'] = df_master['sector'].fillna('Unknown')
    
    return df_master

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
        print(f"Price: ${row['price']:.2f} | IV Rank: {row['iv_rank']:.1f} | Sector: {row['sector']}")
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
        delta = whale.get('max_oi_delta', 0)
        details = f"Vol/OI: {vol_oi:.2f} (need >{threshold:.2f}) | Delta: {delta:.3f} (need 0.15-0.55)"
        print(f"{'WHALE SQUEEZE':<25} {status:<12} {details}")
        
        # CASH FLOW KING
        cf = strategy_details.get('CASH FLOW KING', {})
        status = "✅ ACTIVE" if cf.get('active') else "❌ INACTIVE"
        fcf = cf.get('fcf_ratio')
        fcf_str = f"{fcf:.2f}" if fcf else "N/A"
        details = f"FCF Ratio: {fcf_str} (need <{cf.get('fcf_cap', 0):.1f}) | IV Rank: {cf.get('iv_rank', 0):.1f} (need <45)"
        print(f"{'CASH FLOW KING':<25} {status:<12} {details}")
        
        # MEAN REVERSION
        mean = strategy_details.get('MEAN REVERSION', {})
        status = "✅ ACTIVE" if mean.get('active') else "❌ INACTIVE"
        ma_50 = mean.get('ma_50')
        ma_str = f"${ma_50:.2f}" if ma_50 else "N/A"
        price_vs_ma = mean.get('price_vs_ma', 0)
        details = f"Price: ${mean.get('price', 0):.2f} | MA50: {ma_str} | Ratio: {price_vs_ma:.3f} | IV Rank: {mean.get('iv_rank', 0):.1f} (need <40)"
        print(f"{'MEAN REVERSION':<25} {status:<12} {details}")
        
        # ACCUMULATION
        accum = strategy_details.get('ACCUMULATION', {})
        status = "✅ ACTIVE" if accum.get('active') else "❌ INACTIVE"
        ma_50 = accum.get('ma_50')
        ma_str = f"${ma_50:.2f}" if ma_50 else "N/A"
        above_ma = "Yes" if accum.get('price_above_ma') else "No"
        details = f"Price: ${accum.get('price', 0):.2f} | MA50: {ma_str} | Above MA: {above_ma} | IV Rank: {accum.get('iv_rank', 0):.1f} (need <{accum.get('iv_rank_cap', 0):.1f} or <15)"
        print(f"{'ACCUMULATION':<25} {status:<12} {details}")
        
        print(f"\n💡 Action Plan: {row['action_plan']}")
    
    if missing_tickers:
        print(f"\n⚠️  Tickers not found in database: {', '.join(missing_tickers)}")
    
    if found_tickers:
        print(f"\n{'='*80}")
        print(f"📊 SUMMARY FOR {len(found_tickers)} TICKER(S)")
        print(f"{'='*80}")
        summary_df = final_df[final_df['ticker'].isin(found_tickers)][
            ['ticker', 'price', 'iv_rank', 'conviction_score', 'strategies', 'sector']
        ].sort_values('conviction_score', ascending=False)
        print(summary_df.to_string(index=False))

# --- REPORT GENERATION ---
async def generate_report(db_instance: StockDBBase, args, shutdown_event):
    logger = logging.getLogger("StrategyEngine")
    logger.info("Fetching latest market data...")
    df = await fetch_latest_market_data(db_instance, args.symbols_dir)
    
    if df.empty:
        logger.error("No data returned from database. Check table populations.")
        return

    spy_row = df[df['ticker'] == 'SPY']
    spy_rank = spy_row['iv_rank'].values[0] if not spy_row.empty else 50.0
    logger.debug(f"SPY rank: {spy_rank}")
    
    worker_cfg = vars(args)
    records = df.to_dict('records')
    logger.info(f"Processing {len(records)} tickers with {args.workers} workers...")
    results = []

    executor = None
    try:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        futures = []
        for i, rec in enumerate(records):
            if shutdown_event.is_set():
                logger.warning("Shutdown signal received, cancelling remaining tasks")
                break
            futures.append(executor.submit(analyze_ticker_task, rec, spy_rank, worker_cfg))
            if (i + 1) % 100 == 0:
                logger.debug(f"Submitted {i + 1}/{len(records)} tasks")
        
        completed_count = 0
        for f in as_completed(futures):
            if shutdown_event.is_set():
                logger.warning("Shutdown signal received, cancelling remaining futures")
                for future in futures:
                    future.cancel()
                break
            try:
                results.append(f.result())
                completed_count += 1
                if completed_count % 100 == 0:
                    logger.debug(f"Completed {completed_count}/{len(futures)} tasks")
            except Exception as e:
                logger.error(f"Error processing task: {e}")
    finally:
        if executor:
            wait_timeout = not shutdown_event.is_set()
            try:
                executor.shutdown(wait=wait_timeout, cancel_futures=not wait_timeout)
            except TypeError:
                executor.shutdown(wait=wait_timeout)

    # Convert results to DataFrame, handling strategy_details dict
    final_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'strategy_details'} 
        for r in results
    ])
    
    # Create a separate dict mapping ticker to strategy_details for detailed analysis
    strategy_details_map = {r['ticker']: r.get('strategy_details', {}) for r in results}

    # Handle ticker-specific analysis if requested
    if args.ticker:
        tickers_to_analyze = [t.upper() for t in args.ticker]
        print_ticker_analysis(final_df, tickers_to_analyze, args, strategy_details_map)
        return

    # 1. CATEGORICAL BREAKDOWN
    print(f"\n✅ SCANNED {len(final_df)} TICKERS")
    
    cat_config = [
        ('BACKWARDATION', '⚠️  BACKWARDATION ALERTS', '\033[91m'),
        ('WHALE SQUEEZE', '🐳 WHALE SQUEEZE ALERTS', '\033[94m'),
        ('CASH FLOW KING', '👑 CASH FLOW KINGS', '\033[93m'),
        ('MEAN REVERSION', '📈 MEAN REVERSION', '\033[96m'),
        ('ACCUMULATION', '🟢 ACCUMULATION ZONE', '\033[92m')
    ]

    for strat_tag, label, color in cat_config:
        subset = final_df[final_df['strategies'].str.contains(strat_tag)].sort_values('iv_rank').head(args.top_n)
        if not subset.empty:
            print(f"\n{color}{label} (Top {args.top_n})\033[0m")
            print(f"   {'Ticker':<8} {'Price':<10} {'IV Rank':<10} {'Sector':<20}")
            print("   " + "-"*55)
            for _, row in subset.iterrows():
                print(f"   {row['ticker']:<8} {row['price']:<10} {row['iv_rank']:<10} {row['sector']:<20}")

    # 2. COMBINED CONVICTION LIST
    print("\n" + "="*80)
    print("🏆 FINAL RANKED OPPORTUNITIES (COMBINED CONVICTION)")
    print("="*80)
    top_picks = final_df[final_df['conviction_score'] > 0].sort_values(['conviction_score', 'iv_rank'], ascending=[False, True]).head(args.top_n)
    
    if not top_picks.empty:
        print(f"{'Ticker':<8} {'Score':<6} {'Price':<10} {'Strategies':<35} {'Action Plan'}")
        print("-" * 110)
        for _, row in top_picks.iterrows():
            print(f"{row['ticker']:<8} {row['conviction_score']:<6} ${row['price']:<9.2f} {row['strategies']:<35} {row['action_plan']}")
    else:
        print("⚠️  No assets met current criteria. Loosen thresholds in CLI.")

    if args.csv:
        final_df.to_csv(args.csv, index=False)
        print(f"\n💾 Full analysis exported to {args.csv}")

# --- MAIN & CLI ---
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-core options strategy analysis with categorical breakdown and parallel processing.",
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
        default="../US-Stock-Symbols",
        help="Local Git directory for sector metadata (default: ../US-Stock-Symbols)."
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help="Number of results to print per category (default: 10)."
    )
    parser.add_argument(
        '--explain',
        action='store_true',
        help="Show Strategy Handbook."
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

    # Configurable Strategy Knobs
    parser.add_argument(
        '--vol-oi-ratio',
        type=float,
        default=1.5,
        help="Whale Squeeze Trigger (Range: 1.1 to 3.0. Higher = larger institutional sweeps, default: 1.5)."
    )
    parser.add_argument(
        '--spike-threshold',
        type=float,
        default=1.12,
        help="Backwardation Trigger (Range: 1.05 to 1.30. Higher = extreme panic, default: 1.12)."
    )
    parser.add_argument(
        '--fcf-cap',
        type=float,
        default=20.0,
        help="FCF Ratio Cap (Range: 10.0 to 25.0. Lower = higher quality value, default: 20.0)."
    )
    parser.add_argument(
        '--ma-floor',
        type=float,
        default=0.85,
        help="Mean Reversion Floor (Range: 0.75 to 0.95. Lower = deeper crashes, default: 0.85)."
    )
    parser.add_argument(
        '--iv-rank-cap',
        type=float,
        default=45.0,
        help="Accumulation Cap (Range: 20.0 to 60.0. Lower = clearance pricing, default: 45.0)."
    )

    # Strategy Toggles
    for k in STRATEGY_CONFIG.keys():
        parser.add_argument(
            f'--no-{k}',
            action='store_true',
            help=f"Disable {STRATEGY_CONFIG[k]['name']} strategy."
        )
    parser.add_argument(
        '--no-income',
        action='store_true',
        help="Disable INCOME ZONE strategy."
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
    
    # Handle --explain flag early
    if args.explain:
        print("\n=== 🧠 STRATEGY QUANT HANDBOOK ===")
        for k, v in STRATEGY_CONFIG.items():
            print(f"\n🔹 {v['name']}\n   Logic: {v['desc']}\n   Optimal Range: {v['range']}\n   Plan: {v['plan']}")
        return
    
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
