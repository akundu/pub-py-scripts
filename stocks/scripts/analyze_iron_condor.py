import os
import sys
import asyncio
import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

# Project Path Setup
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.stock_db import get_stock_db, StockDBBase

def generate_all_combinations(group, logger=None):
    """Generate all possible iron condor combinations from available options.
    
    An iron condor requires:
    - Short Put (lower strike)
    - Long Put (even lower strike)
    - Short Call (higher strike)
    - Long Call (even higher strike)
    
    Args:
        group: DataFrame with options data for a specific hour
        logger: Optional logger for debugging
        
    Returns:
        List of dictionaries, each containing the 4 legs of an iron condor
    """
    import itertools
    
    # Separate PUTs and CALLs
    puts = group[group['option_type'] == 'PUT'].copy()
    calls = group[group['option_type'] == 'CALL'].copy()
    
    if puts.empty or calls.empty:
        if logger:
            logger.debug(f"No PUTs ({len(puts)}) or CALLs ({len(calls)}) available")
        return []
    
    # Filter out options with missing bid/ask
    puts = puts[(puts['bid'].notna()) & (puts['ask'].notna())]
    calls = calls[(calls['bid'].notna()) & (calls['ask'].notna())]
    
    if len(puts) < 2 or len(calls) < 2:
        if logger:
            logger.debug(f"Not enough valid options: PUTs={len(puts)}, CALLs={len(calls)}")
        return []
    
    combinations = []
    
    # Generate all combinations of 2 PUTs (for short and long put)
    for short_put_idx, long_put_idx in itertools.combinations(puts.index, 2):
        short_put = puts.loc[short_put_idx]
        long_put = puts.loc[long_put_idx]
        
        # For iron condor: short put should have higher strike than long put
        if short_put['strike_price'] <= long_put['strike_price']:
            short_put, long_put = long_put, short_put
        
        # Generate all combinations of 2 CALLs (for short and long call)
        for short_call_idx, long_call_idx in itertools.combinations(calls.index, 2):
            short_call = calls.loc[short_call_idx]
            long_call = calls.loc[long_call_idx]
            
            # For iron condor: short call should have lower strike than long call
            if short_call['strike_price'] >= long_call['strike_price']:
                short_call, long_call = long_call, short_call
            
            # Verify iron condor structure: long_put < short_put < short_call < long_call
            if long_put['strike_price'] < short_put['strike_price'] < short_call['strike_price'] < long_call['strike_price']:
                combinations.append({
                    'short_put': short_put,
                    'long_put': long_put,
                    'short_call': short_call,
                    'long_call': long_call
                })
    
    if logger:
        logger.debug(f"Generated {len(combinations)} iron condor combinations from {len(puts)} PUTs and {len(calls)} CALLs")
    
    return combinations

def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    log_level_upper = log_level.upper()
    numeric_level = getattr(logging, log_level_upper, logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s [%(name)s] [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set level for this module's logger
    logger = logging.getLogger("IronCondorBacktest")
    logger.setLevel(numeric_level)
    return logger

async def run_backtest(args, db_instance: StockDBBase):
    logger = logging.getLogger("IronCondorBacktest")
    
    # Build timestamp filter if start_date and end_date are provided
    timestamp_filter = ""
    if args.start_date and args.end_date:
        start_dt = f"'{args.start_date}T{args.start_hour:02}:00:00.000000Z'"
        end_dt = f"'{args.end_date}T{args.end_hour:02}:00:00.000000Z'"
        timestamp_filter = f"AND o.timestamp BETWEEN {start_dt} AND {end_dt}"
        logger.info(f"Filtering by timestamp: {args.start_date} to {args.end_date}")

    # Build expiration date filter if provided
    expiration_filter = ""
    if args.expiration_date:
        # Options expire at end of day, so include the full day + 24 hours
        # expiration_date is stored as TIMESTAMP, so we need to filter for that date and next day
        exp_date = pd.to_datetime(args.expiration_date)
        exp_start = exp_date.strftime("'%Y-%m-%dT00:00:00.000000Z'")
        # Add 24 hours (1 day) to include the full expiration day
        exp_end = (exp_date + pd.Timedelta(days=1)).strftime("'%Y-%m-%dT23:59:59.999999Z'")
        expiration_filter = f"AND o.expiration_date >= {exp_start} AND o.expiration_date <= {exp_end}"
        logger.info(f"Filtering for expiration date: {args.expiration_date} (including full day + 24 hours)")

    # Optimized Query with ASOF JOIN to correlate Option Greeks with Spot Price
    query = f"""
    SELECT 
        o.timestamp, 
        o.option_type, 
        o.strike_price, 
        o.bid, 
        o.ask, 
        o.delta, 
        o.expiration_date,
        r.price as underlying_price
    FROM options_data o
    ASOF JOIN realtime_data r ON (ticker)
    WHERE o.ticker = '{args.ticker}'
      {timestamp_filter}
      {expiration_filter}
    """
    
    if args.expiration_date:
        logger.info(f"Executing query for ticker {args.ticker} with expiration date {args.expiration_date}")
    else:
        logger.info(f"Executing query for ticker {args.ticker} from {args.start_date} to {args.end_date}")
    df = await db_instance.execute_select_sql(query)
    if df.empty:
        logger.warning("No data returned from query. Check ticker, date range, and database connection.")
        return
    
    logger.info(f"Retrieved {len(df)} option data records")
    
    # Ensure timestamp column is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.floor('h')  # Use 'h' instead of deprecated 'H'
    else:
        logger.error("Timestamp column not found in query results")
        return
    
    # Ensure expiration_date is datetime if present
    if 'expiration_date' in df.columns:
        df['expiration_date'] = pd.to_datetime(df['expiration_date'])
    
    all_trades = []
    
    # Track statistics for debugging
    total_hours = 0
    skipped_hour_filter = 0
    total_combinations_evaluated = 0

    for hour, group in df.groupby('hour'):
        total_hours += 1
        
        # Only filter by hour if start_date/end_date are provided (otherwise use all hours)
        if args.start_date and args.end_date and not (args.start_hour <= hour.hour <= args.end_hour):
            skipped_hour_filter += 1
            continue

        if args.no_filter:
            # When --no-filter is set, evaluate ALL combinations
            combinations = generate_all_combinations(group, logger)
            total_combinations_evaluated += len(combinations)
            
            for legs in combinations:
                entry_credit = calculate_credit(legs, args.slippage)
                
                # Max Risk = (Width of the Wings) - (Premium Received)
                wing_width = legs['short_call']['strike_price'] - legs['long_call']['strike_price']
                max_risk = abs(wing_width) - entry_credit
                
                # Risk-to-Reward Ratio (Reward / Risk)
                rr_ratio = entry_credit / max_risk if max_risk > 0 else 0
                
                # Get expiration date from the legs (should be same for all legs in an iron condor)
                expiration_date = None
                if 'expiration_date' in group.columns and 'expiration_date' in legs['short_call'].index:
                    expiration_date = legs['short_call']['expiration_date']
                
                # Get underlying price (use first available)
                underlying_price = None
                if 'underlying_price' in legs['short_call'].index:
                    underlying_price = legs['short_call']['underlying_price']
                elif 'underlying_price' in group.columns:
                    underlying_price = group['underlying_price'].iloc[0] if not group['underlying_price'].empty else None
                
                all_trades.append({
                    "time": hour,
                    "expiration_date": expiration_date,
                    "spot": underlying_price,
                    "credit": entry_credit,
                    "risk": max_risk,
                    "rr_ratio": rr_ratio,
                    "short_call": legs['short_call']['strike_price'],
                    "short_put": legs['short_put']['strike_price'],
                    "long_call": legs['long_call']['strike_price'],
                    "long_put": legs['long_put']['strike_price'],
                    "short_call_delta": legs['short_call'].get('delta', None) if isinstance(legs['short_call'], pd.Series) else None,
                    "short_put_delta": legs['short_put'].get('delta', None) if isinstance(legs['short_put'], pd.Series) else None,
                    "long_call_delta": legs['long_call'].get('delta', None) if isinstance(legs['long_call'], pd.Series) else None,
                    "long_put_delta": legs['long_put'].get('delta', None) if isinstance(legs['long_put'], pd.Series) else None,
                })
        else:
            # Original behavior: filter by delta targets
            legs = select_legs(group, args.short_delta, args.long_delta, None)
            if not legs:
                continue
            
            entry_credit = calculate_credit(legs, args.slippage)
            
            # Max Risk = (Width of the Wings) - (Premium Received)
            wing_width = legs['short_call']['strike_price'] - legs['long_call']['strike_price']
            max_risk = abs(wing_width) - entry_credit
            
            # Risk-to-Reward Ratio (Reward / Risk)
            rr_ratio = entry_credit / max_risk if max_risk > 0 else 0
            
            # Get expiration date from the legs (should be same for all legs in an iron condor)
            expiration_date = None
            if 'expiration_date' in group.columns and 'expiration_date' in legs['short_call'].index:
                expiration_date = legs['short_call']['expiration_date']
            
            # FEATURE: Risk-to-Reward Filter
            if rr_ratio < args.min_rr:
                # Skip trade if it doesn't pay enough for the risk taken
                continue

            all_trades.append({
                "time": hour,
                "expiration_date": expiration_date,
                "spot": legs['short_call']['underlying_price'],
                "credit": entry_credit,
                "risk": max_risk,
                "rr_ratio": rr_ratio,
                "short_call": legs['short_call']['strike_price'],
                "short_put": legs['short_put']['strike_price'],
                "long_call": legs['long_call']['strike_price'],
                "long_put": legs['long_put']['strike_price']
            })

    # Log filtering statistics
    logger.info(f"Processing statistics:")
    logger.info(f"  Total hour groups: {total_hours}")
    logger.info(f"  Skipped by hour filter: {skipped_hour_filter}")
    if args.no_filter:
        logger.info(f"  Total combinations evaluated: {total_combinations_evaluated}")
    logger.info(f"  Valid trades found: {len(all_trades)}")

    summarize_results(all_trades, args.tp_pct, args.sort_by)

def calculate_credit(legs, slippage):
    def get_price(leg, is_sell):
        mid = (leg['bid'] + leg['ask']) / 2
        spread = leg['ask'] - leg['bid']
        return mid - (slippage * 0.5 * spread) if is_sell else mid + (slippage * 0.5 * spread)

    return (get_price(legs['short_put'], True) + get_price(legs['short_call'], True)) - \
           (get_price(legs['long_put'], False) + get_price(legs['long_call'], False))

def select_legs(group, s_delta, l_delta, logger=None):
    """Select iron condor legs based on delta targets."""
    try:
        def find_closest(target, type_str):
            subset = group[group['option_type'] == type_str].copy()
            if subset.empty:
                return None
            # Check if delta column exists and has valid values
            if 'delta' not in subset.columns or subset['delta'].isna().all():
                return None
            subset['diff'] = (subset['delta'].abs() - target).abs()
            return subset.sort_values('diff').iloc[0]
        
        short_put = find_closest(s_delta, 'PUT')
        long_put = find_closest(l_delta, 'PUT')
        short_call = find_closest(s_delta, 'CALL')
        long_call = find_closest(l_delta, 'CALL')
        
        if any(x is None for x in [short_put, long_put, short_call, long_call]):
            if logger:
                missing = []
                if short_put is None:
                    missing.append(f"short_put (delta={s_delta})")
                if long_put is None:
                    missing.append(f"long_put (delta={l_delta})")
                if short_call is None:
                    missing.append(f"short_call (delta={s_delta})")
                if long_call is None:
                    missing.append(f"long_call (delta={l_delta})")
                logger.debug(f"Could not find legs: {', '.join(missing)}")
            return None
            
        return {
            'short_put': short_put,
            'long_put': long_put,
            'short_call': short_call,
            'long_call': long_call
        }
    except Exception as e:
        if logger:
            logger.debug(f"Error selecting legs: {e}")
        return None

def summarize_results(trades, tp_pct, sort_by="rr_ratio"):
    res_df = pd.DataFrame(trades)
    if res_df.empty:
        print("No trades found matching the criteria.")
        return

    # Sort results - handle missing values by putting them last
    ascending = sort_by in ["time", "risk", "expiration_date"]
    if sort_by in res_df.columns:
        res_df = res_df.sort_values(by=sort_by, ascending=ascending, na_position='last')
    else:
        logger = logging.getLogger("IronCondorBacktest")
        logger.warning(f"Sort column '{sort_by}' not found, using default sort by rr_ratio")
        res_df = res_df.sort_values(by="rr_ratio", ascending=False, na_position='last')

    print("\n--- STRATEGY BACKTEST RESULTS ---")
    print(f"Total Trades Found: {len(res_df)}")
    print(f"Sorted by: {sort_by} ({'ascending' if ascending else 'descending'})")
    print(f"\n{'Time':<20} {'Exp Date':<12} {'Spot':<10} {'Credit':<10} {'Risk':<10} {'R:R':<8} {'Short Call':<12} {'Short Put':<12} {'Long Call':<12} {'Long Put':<12}")
    print("-" * 120)
    
    for _, row in res_df.iterrows():
        time_str = str(row['time']) if pd.notna(row['time']) else "N/A"
        if 'expiration_date' in row and pd.notna(row['expiration_date']):
            if isinstance(row['expiration_date'], pd.Timestamp):
                exp_str = row['expiration_date'].strftime('%Y-%m-%d')
            else:
                exp_str = str(row['expiration_date'])[:10]  # Take first 10 chars for date
        else:
            exp_str = "N/A"
        print(f"{time_str:<20} {exp_str:<12} ${row['spot']:<9.2f} ${row['credit']:<9.2f} ${row['risk']:<9.2f} {row['rr_ratio']:<8.3f} ${row['short_call']:<11.2f} ${row['short_put']:<11.2f} ${row['long_call']:<11.2f} ${row['long_put']:<11.2f}")
    
    print("\n--- SUMMARY STATISTICS ---")
    print(f"Avg R:R Ratio: {res_df['rr_ratio'].mean():.2f}")
    print(f"Median R:R Ratio: {res_df['rr_ratio'].median():.2f}")
    print(f"Min R:R Ratio: {res_df['rr_ratio'].min():.2f}")
    print(f"Max R:R Ratio: {res_df['rr_ratio'].max():.2f}")
    print(f"Total Credit Harvested: ${res_df['credit'].sum():.2f}")
    print(f"Total Risk: ${res_df['risk'].sum():.2f}")
    print(f"Target Profit (Realized): ${res_df['credit'].sum() * tp_pct:.2f}")

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
    
    # Validate that either expiration_date is set OR both start_date and end_date are set
    if not args.expiration_date and (not args.start_date or not args.end_date):
        import sys
        print("Error: Either --expiration-date must be set, OR both --start-date and --end-date must be set.", file=sys.stderr)
        sys.exit(1)
    
    # Set up logging with the specified log level
    logger = setup_logging(args.log_level)
    
    db_instance = setup_database(args)
    
    try:
        await run_backtest(args, db_instance)
    except Exception as e:
        logger.error(f"Error in backtest: {e}", exc_info=True)
        raise
    finally:
        # Ensure proper cleanup
        if hasattr(db_instance, 'close'):
            await db_instance.close()
        logger.debug("Cleanup complete")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Iron Condor strategy backtest with risk-to-reward filtering.",
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
    parser.add_argument("--start-date", type=str, required=False,
                        dest="start_date",
                        help="Start date for backtest (YYYY-MM-DD). Not required if --expiration-date is set.")
    parser.add_argument("--end-date", type=str, required=False,
                        dest="end_date",
                        help="End date for backtest (YYYY-MM-DD). Not required if --expiration-date is set.")
    parser.add_argument("--ticker", type=str, default="VOO",
                        help="Ticker symbol to analyze (default: VOO)")
    parser.add_argument("--short-delta", type=float, default=0.15,
                        dest="short_delta",
                        help="Target delta for short legs (default: 0.15)")
    parser.add_argument("--long-delta", type=float, default=0.05,
                        dest="long_delta",
                        help="Target delta for long legs (default: 0.05)")
    parser.add_argument("--tp-pct", type=float, default=0.50,
                        dest="tp_pct",
                        help="Target profit percentage (default: 0.50)")
    parser.add_argument("--slippage", type=float, default=0.5,
                        help="Slippage factor (default: 0.5)")
    parser.add_argument("--min-rr", type=float, default=0.15,
                        dest="min_rr",
                        help="Minimum Credit/MaxRisk ratio to accept trade (default: 0.15)")
    parser.add_argument("--start-hour", type=int, default=10,
                        dest="start_hour",
                        help="Start hour for trading window (default: 10)")
    parser.add_argument("--end-hour", type=int, default=15,
                        dest="end_hour",
                        help="End hour for trading window (default: 15)")
    parser.add_argument("--expiration-date", type=str, default=None,
                        dest="expiration_date",
                        help="Filter for options expiring on this date (YYYY-MM-DD). "
                             "Includes full expiration day + 24 hours since options expire at end of day.")
    parser.add_argument("--no-filter", action='store_true',
                        dest="no_filter",
                        help="Disable R:R filtering - show all results regardless of risk-to-reward ratio")
    parser.add_argument("--sort-by", type=str, default="rr_ratio",
                        choices=["rr_ratio", "credit", "risk", "time", "expiration_date"],
                        help="Sort results by this field (default: rr_ratio)")
    return parser.parse_args()

if __name__ == "__main__":
    asyncio.run(main())
