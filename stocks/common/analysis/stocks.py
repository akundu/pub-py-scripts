"""Core stock analysis functionality for options strategy evaluation."""

import json
import logging
import os
import glob
import pandas as pd
import asyncio
from datetime import datetime, timedelta, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Optional

from common.stock_db import StockDBBase

# --- THE QUANT HANDBOOK & CONFIG ---
STRATEGY_CONFIG = {
    'back': {'name': 'BACKWARDATION', 'plan': 'Sell 30-day calls against long positions.'},
    'whale': {'name': 'WHALE SQUEEZE', 'plan': 'Buy short-term OTM calls for Gamma pressure.'},
    'sector_z': {'name': 'SECTOR RELATIVE', 'plan': 'Buy LEAPS on sector-lagging volatility.'},
    'cf': {'name': 'CASH FLOW KING', 'plan': 'Buy ITM LEAPS and sell monthly calls (Diagonal).'},
    'mean': {'name': 'MEAN REVERSION', 'plan': 'Buy near-term ITM calls for the snap-back.'},
    'accum': {'name': 'ACCUMULATION', 'plan': 'Systematically build core LEAP positions.'}
}


def analyze_ticker_task(row: Dict[str, Any], spy_rank: float, sector_avgs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single ticker for strategy opportunities.
    
    Args:
        row: Ticker data row with market metrics
        spy_rank: SPY IV rank for sector comparison
        sector_avgs: Dictionary mapping sector names to average IV ranks
        cfg: Configuration dictionary with strategy parameters
        
    Returns:
        Dictionary with ticker analysis results including strategies and conviction score
    """
    ticker = row['ticker']
    sector = row.get('sector', 'Unknown')
    iv_rank = row.get('iv_rank', 50)
    # Pricing Trap fix: use current_price if exists, else fallback to financial_info price
    # Priority: realtime_data > hourly_prices > financial_info.price
    # Note: realtime_data and hourly_prices are merged as 'current_price' in fetch_latest_market_data
    # This means current_price could be from realtime (most recent quote) or hourly (most recent hourly bar close)
    # which may differ from the daily close price shown in fetch_symbol_data.py
    current_price_val = row.get('current_price')
    financial_price_val = row.get('price', 0)
    price = current_price_val if pd.notnull(current_price_val) else financial_price_val
    
    # Log price source for specific tickers (when DEBUG is enabled)
    logger = logging.getLogger("StrategyEngine")
    if logger.isEnabledFor(logging.DEBUG) and ticker in ['WDC', 'SPY', 'AAPL']:  # Log for common tickers
        price_source = 'realtime/hourly' if pd.notnull(current_price_val) else 'financial_info'
        logger.debug(f"[PRICE SOURCE] {ticker}: Using {price_source} - current_price={current_price_val}, financial_price={financial_price_val}, final={price}")
    ma_50 = row.get('ma_50', 0)
    fcf_ratio = row.get('price_to_free_cash_flow')
    iv_30 = row.get('iv_30d', 0)
    iv_90 = row.get('iv_90d', 0)
    vol_oi_ratio = row.get('vol_oi_ratio', 0)
    max_oi_delta = row.get('max_oi_delta', 0)

    # --- 1. SECTOR RELATIVE INTELLIGENCE ---
    sector_avg_rank = sector_avgs.get(sector, spy_rank)
    sector_z = iv_rank - sector_avg_rank  # Negative means cheaper than peers

    # --- 2. TIME-NORMALIZED VOLUME (Speedometer Fix) ---
    now = datetime.now(timezone.utc)
    # Market opens 14:30 UTC. Assume 390 minute trading day.
    market_start = now.replace(hour=14, minute=30, second=0, microsecond=0)
    minutes_elapsed = max(5, (now - market_start).total_seconds() / 60)
    time_fill_factor = min(1.0, minutes_elapsed / 390)
    # Require less volume in the morning to trigger Whale Squeeze
    dynamic_vol_threshold = cfg['vol_oi_ratio'] * time_fill_factor

    active_strats = []
    spike_score = iv_30 / iv_90 if iv_90 and iv_90 != 0 else 0
    
    # Detailed strategy evaluation (for ticker-specific analysis)
    strategy_details = {}
    
    # Evaluate Strategies with "Elastic" bounds
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
    
    # WHALE SQUEEZE (Using dynamic time-based threshold)
    whale_active = not cfg['no_whale'] and vol_oi_ratio > dynamic_vol_threshold and 0.15 < max_oi_delta < 0.55
    if whale_active:
        active_strats.append('WHALE SQUEEZE')
    strategy_details['WHALE SQUEEZE'] = {
        'active': whale_active,
        'vol_oi_ratio': round(vol_oi_ratio, 2),
        'threshold': round(dynamic_vol_threshold, 2),
        'base_threshold': cfg['vol_oi_ratio'],
        'time_fill_factor': round(time_fill_factor, 3),
        'max_oi_delta': round(max_oi_delta, 3),
        'delta_range': '0.15-0.55'
    }

    # SECTOR RELATIVE (New: Cheap vs Peers)
    sector_rel_active = not cfg.get('no_sector_rel', False) and sector_z < -15.0
    if sector_rel_active:
        active_strats.append('SECTOR RELATIVE')
    strategy_details['SECTOR RELATIVE'] = {
        'active': sector_rel_active,
        'sector_z': round(sector_z, 1),
        'threshold': -15.0,
        'sector_avg_rank': round(sector_avg_rank, 1),
        'iv_rank': round(iv_rank, 1)
    }

    # CASH FLOW KING (Relaxed IV Rank for regime)
    cf_active = not cfg['no_cf'] and fcf_ratio and 0 < fcf_ratio < cfg['fcf_cap'] and iv_rank < 55
    if cf_active:
        active_strats.append('CASH FLOW KING')
    strategy_details['CASH FLOW KING'] = {
        'active': cf_active,
        'fcf_ratio': round(fcf_ratio, 2) if fcf_ratio else None,
        'fcf_cap': cfg['fcf_cap'],
        'iv_rank': round(iv_rank, 1),
        'iv_rank_threshold': 55
    }
    
    # MEAN REVERSION (Relaxed distance to MA to capture transition zones)
    mean_active = not cfg['no_mean'] and ma_50 and (ma_50 * 0.80) < price < (ma_50 * 1.02) and iv_rank < 45
    if mean_active:
        active_strats.append('MEAN REVERSION')
    strategy_details['MEAN REVERSION'] = {
        'active': mean_active,
        'price': round(price, 2),
        'ma_50': round(ma_50, 2) if ma_50 else None,
        'ma_floor': 0.80,
        'ma_ceiling': 1.02,
        'price_vs_ma': round((price / ma_50) if ma_50 else 0, 3),
        'iv_rank': round(iv_rank, 1),
        'iv_rank_threshold': 45
    }
    
    # ACCUMULATION
    accum_active = not cfg['no_accum'] and ((ma_50 and price > ma_50 and iv_rank < cfg['iv_rank_cap']) or (iv_rank < 18))
    if accum_active:
        active_strats.append('ACCUMULATION')
    strategy_details['ACCUMULATION'] = {
        'active': accum_active,
        'price': round(price, 2),
        'ma_50': round(ma_50, 2) if ma_50 else None,
        'price_above_ma': (ma_50 and price > ma_50) if ma_50 else False,
        'iv_rank': round(iv_rank, 1),
        'iv_rank_cap': cfg['iv_rank_cap'],
        'iv_rank_floor': 18
    }

    score = len(active_strats)
    plan = STRATEGY_CONFIG.get(active_strats[0].lower(), {'plan': 'Review'}).get('plan') if active_strats else "Hold"

    return {
        "ticker": ticker,
        "sector": sector,
        "price": round(price, 2),
        "iv_rank": round(iv_rank, 1),
        "sector_z": round(sector_z, 1),
        "conviction_score": score,
        "strategies": ", ".join(active_strats) if active_strats else "None",
        "action_plan": plan,
        "strategy_details": strategy_details  # Added for detailed analysis
    }


def load_sector_data(data_dir: str) -> pd.DataFrame:
    """Load sector metadata from JSON files in the symbols directory.
    
    Args:
        data_dir: Directory path containing symbol JSON files
        
    Returns:
        DataFrame with ticker and sector columns
    """
    metadata = []
    json_files = glob.glob(os.path.join(data_dir, "**", "*_full_tickers.json"), recursive=True)
    for f_path in json_files:
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    metadata.append({'ticker': item.get('symbol'), 'sector': item.get('sector', 'Unknown')})
        except Exception:
            continue
    return pd.DataFrame(metadata).drop_duplicates('ticker') if metadata else pd.DataFrame(columns=['ticker', 'sector'])


async def fetch_latest_market_data(db_instance: StockDBBase, symbols_dir: str) -> pd.DataFrame:
    """Fetch and merge latest market data from database.
    
    Args:
        db_instance: Database instance for querying
        symbols_dir: Directory path for sector metadata
        
    Returns:
        Merged DataFrame with all market data
    """
    logger = logging.getLogger("StrategyEngine")
    logger.debug("Fetching financial data...")
    
    lookback = (datetime.now(timezone.utc) - timedelta(days=1)).replace(tzinfo=None)
    q_whale = f"SELECT ticker, sum(volume) as total_vol, sum(open_interest) as total_oi, avg(delta) as avg_delta FROM options_data WHERE timestamp > '{lookback}' AND option_type = 'call' GROUP BY ticker;"
    
    tasks = [
        db_instance.execute_select_sql("SELECT ticker, price, iv_rank, iv_30d, iv_90d, price_to_free_cash_flow FROM financial_info LATEST ON date PARTITION BY ticker;"),
        db_instance.execute_select_sql("SELECT ticker, price as current_price FROM realtime_data LATEST ON timestamp PARTITION BY ticker;"),
        db_instance.execute_select_sql("SELECT ticker, close as current_price FROM hourly_prices LATEST ON datetime PARTITION BY ticker;"),
        db_instance.execute_select_sql("SELECT ticker, ma_50 FROM daily_prices LATEST ON date PARTITION BY ticker;"),
        db_instance.execute_select_sql(q_whale)
    ]
    df_f, df_rt, df_hr, df_t, df_w = await asyncio.gather(*tasks)
    
    logger.debug(f"Financial data: {len(df_f)} rows, Realtime: {len(df_rt)} rows, Hourly: {len(df_hr)} rows, Trend: {len(df_t)} rows, Whale: {len(df_w)} rows")

    # Combine Prices (realtime has priority over hourly)
    # Ensure both DataFrames have 'ticker' column before concatenating
    if df_rt.empty and df_hr.empty:
        df_p = pd.DataFrame(columns=['ticker', 'current_price'])
    elif df_rt.empty:
        df_p = df_hr.copy()
    elif df_hr.empty:
        df_p = df_rt.copy()
    else:
        df_p = pd.concat([df_rt.assign(s=1), df_hr.assign(s=2)]).sort_values('s').drop_duplicates('ticker').drop(columns='s')
    logger.debug(f"Combined price data: {len(df_p)} rows")
    
    # Log price source for debugging (only if we have data)
    if logger.isEnabledFor(logging.DEBUG) and not df_p.empty and 'ticker' in df_p.columns:
        for ticker in df_p['ticker'].head(10):  # Log first 10 for debugging
            rt_price = None
            hr_price = None
            fin_price = None
            final_price = None
            
            # Safely check df_rt
            if not df_rt.empty and 'ticker' in df_rt.columns:
                rt_match = df_rt[df_rt['ticker'] == ticker]
                rt_price = rt_match['current_price'].values[0] if not rt_match.empty else None
            
            # Safely check df_hr
            if not df_hr.empty and 'ticker' in df_hr.columns:
                hr_match = df_hr[df_hr['ticker'] == ticker]
                hr_price = hr_match['current_price'].values[0] if not hr_match.empty else None
            
            # Safely check df_f
            if not df_f.empty and 'ticker' in df_f.columns:
                fin_match = df_f[df_f['ticker'] == ticker]
                fin_price = fin_match['price'].values[0] if not fin_match.empty else None
            
            # Get final price
            final_match = df_p[df_p['ticker'] == ticker]
            final_price = final_match['current_price'].values[0] if not final_match.empty else None
            
            logger.debug(f"Price sources for {ticker}: realtime={rt_price}, hourly={hr_price}, financial={fin_price}, final={final_price}")
    
    # Process Whale Metrics
    if not df_w.empty:
        df_w['vol_oi_ratio'] = df_w.apply(
            lambda r: (r['total_vol'] / r['total_oi']) if pd.notna(r['total_oi']) and r['total_oi'] > 0 else 0,
            axis=1
        )
        df_w = df_w.rename(columns={'avg_delta': 'max_oi_delta'})
    else:
        df_w = pd.DataFrame(columns=['ticker', 'vol_oi_ratio', 'max_oi_delta'])

    # Merging Strategy: Change inner join to left join to prevent pricing deletion trap
    logger.debug("Merging datasets...")
    df_master = df_f.merge(df_p, on='ticker', how='left').merge(df_t, on='ticker', how='left').merge(df_w, on='ticker', how='left')
    
    # Debug: Print counts of pruned results after merging if log level is DEBUG
    if logger.isEnabledFor(logging.DEBUG):
        orig_count = len(df_f)
        final_count = len(df_master)
        logger.debug(f"fetch_latest_market_data: Started with {orig_count} tickers (financial_info), {final_count} remain after merging.")
    
    df_sec = load_sector_data(symbols_dir)
    logger.debug(f"Sector data: {len(df_sec)} rows")
    
    if not df_sec.empty and 'ticker' in df_sec.columns:
        df_master = df_master.merge(df_sec, on='ticker', how='left')
    else:
        df_master['sector'] = 'Unknown'
    
    df_master['sector'] = df_master['sector'].fillna('Unknown')
    logger.debug(f"Final merged dataset: {len(df_master)} rows")
    
    return df_master


async def analyze_stocks(
    db_instance: StockDBBase,
    symbols_dir: str,
    config: Dict[str, Any],
    workers: int,
    shutdown_event: Optional[Any] = None
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Perform stock analysis across all tickers.
    
    Args:
        db_instance: Database instance for querying
        symbols_dir: Directory path for sector metadata
        config: Configuration dictionary with strategy parameters
        workers: Number of worker processes for parallel processing
        shutdown_event: Optional multiprocessing Event for graceful shutdown
        
    Returns:
        Tuple of (results DataFrame, strategy_details_map dictionary)
    """
    logger = logging.getLogger("StrategyEngine")
    logger.info("Fetching latest market data...")
    df = await fetch_latest_market_data(db_instance, symbols_dir)
    
    if df.empty:
        logger.error("No data returned from database. Check table populations.")
        return pd.DataFrame(), {}

    # Pre-calculate Sector Averages for the Elastic Filter
    logger.debug("Calculating sector averages...")
    sector_avgs = df.groupby('sector')['iv_rank'].mean().to_dict()
    spy_row = df[df['ticker'] == 'SPY']
    spy_rank = spy_row['iv_rank'].values[0] if not spy_row.empty else 50.0
    logger.debug(f"SPY rank: {spy_rank}, Found {len(sector_avgs)} sectors")
    
    records = df.to_dict('records')
    logger.info(f"Processing {len(records)} tickers with {workers} workers...")
    results = []

    executor = None
    try:
        executor = ProcessPoolExecutor(max_workers=workers)
        futures = []
        for i, rec in enumerate(records):
            if shutdown_event and shutdown_event.is_set():
                logger.warning("Shutdown signal received, cancelling remaining tasks")
                break
            futures.append(executor.submit(analyze_ticker_task, rec, spy_rank, sector_avgs, config))
            if (i + 1) % 100 == 0:
                logger.debug(f"Submitted {i + 1}/{len(records)} tasks")
        
        completed_count = 0
        for f in as_completed(futures):
            if shutdown_event and shutdown_event.is_set():
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
            wait_timeout = not (shutdown_event and shutdown_event.is_set()) if shutdown_event else True
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

    return final_df, strategy_details_map

