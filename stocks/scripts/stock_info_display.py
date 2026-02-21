#!/usr/bin/env python3
"""
Display comprehensive stock information including price, options, and financial ratios.

This script combines functionality from fetch_symbol_data.py and fetch_options.py
to provide a unified view of stock information.
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fetch_symbol_data import (
    get_current_price,
    get_financial_ratios,
    get_latest_news,
    get_latest_iv,
    process_symbol_data,
    _format_price_block,
    _normalize_timezone_string,
    _get_et_now,
    get_default_db_path,
    StockDBBase,
    # New imports for stock info functions
    get_price_info,
    get_options_info,
    get_financial_info,
    get_news_info,
    get_iv_info,
    get_stock_info_parallel,
)
from common.stock_db import get_stock_db
import aiohttp

logger = logging.getLogger(__name__)

# Try to import Polygon client
try:
    from polygon.rest import RESTClient as PolygonRESTClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    print("Warning: polygon-api-client not installed. Polygon.io data source will not be available.", file=sys.stderr)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Display comprehensive stock information (price, options, financial ratios)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display latest info for a single symbol (default behavior)
  python stock_info_display.py AAPL

  # Explicitly request latest price only
  python stock_info_display.py AAPL --latest

  # Display info for multiple symbols
  python stock_info_display.py AAPL MSFT GOOGL

  # Display with date range for price data
  python stock_info_display.py AAPL --start-date 2024-01-01 --end-date 2024-12-31

  # Display options for next 90 days (instead of default 180)
  python stock_info_display.py AAPL --options-days 90

  # Force fetch from API (bypass DB/cache)
  python stock_info_display.py AAPL --force-fetch

  # Use specific database
  python stock_info_display.py AAPL --db-path questdb://localhost:9000
        """
    )
    
    parser.add_argument(
        "symbols",
        nargs="+",
        help="Stock symbol(s) to display (e.g., AAPL MSFT GOOGL)"
    )
    
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Show only the latest price information (overrides --start-date and --end-date)"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for price data (YYYY-MM-DD). Default: latest only. Ignored if --latest is set."
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for price data (YYYY-MM-DD). Default: today. Ignored if --latest is set."
    )
    
    parser.add_argument(
        "--options-days",
        type=int,
        default=180,
        help="Number of days ahead to fetch options data (default: 180)"
    )
    
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Force fetch all data from API (bypass DB/cache)"
    )
    
    parser.add_argument(
        "--db-type",
        type=str,
        default="questdb",
        choices=["sqlite", "duckdb", "questdb", "postgresql"],
        help="Database type (default: questdb)"
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Database path or connection string (e.g., questdb://localhost:9000, postgresql://user:pass@host:port/db)"
    )
    
    parser.add_argument(
        "--data-source",
        choices=["polygon", "alpaca"],
        default="polygon",
        help="Data source for fetching (default: polygon)"
    )
    
    parser.add_argument(
        "--timezone",
        type=str,
        default="America/New_York",
        help="Timezone for displaying timestamps (default: America/New_York)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Redis caching for QuestDB operations"
    )
    
    parser.add_argument(
        "--show-price-history",
        action="store_true",
        help="Show price history table (only if date range specified)"
    )
    
    parser.add_argument(
        "--options-type",
        choices=["all", "call", "put"],
        default="all",
        help="Filter options by type (default: all)"
    )
    
    parser.add_argument(
        "--strike-range-percent",
        type=int,
        default=None,
        help="Filter options by strike range (±percent from stock price, e.g., 20 for ±20%%)"
    )
    
    parser.add_argument(
        "--max-options-per-expiry",
        type=int,
        default=10,
        help="Maximum number of options to show per expiration date (default: 10)"
    )
    
    parser.add_argument(
        "--show-news",
        action="store_true",
        help="Show latest news articles for the symbol"
    )
    
    parser.add_argument(
        "--show-iv",
        action="store_true",
        help="Show latest implied volatility statistics"
    )
    
    parser.add_argument(
        "--api-server",
        type=str,
        default=None,
        help="API server URL to fetch data from (e.g., http://localhost:8080). If not provided, uses --db-path directly."
    )
    
    return parser.parse_args()


async def get_stock_info_from_api(
    symbol: str,
    api_server: str,
    **kwargs
) -> Dict[str, Any]:
    """Get stock information from API server.
    
    Args:
        symbol: Stock symbol
        api_server: Base URL of the API server (e.g., http://localhost:8080)
        **kwargs: Additional query parameters to pass to the API
    
    Returns:
        Dictionary with stock information (same format as get_stock_info_parallel)
    """
    import time
    fetch_start = time.time()
    
    # Build URL
    url = f"{api_server.rstrip('/')}/api/stock_info/{symbol}"
    
    # Build query parameters from kwargs
    params = {}
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool):
                params[key] = "true" if value else "false"
            else:
                params[key] = str(value)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    fetch_time = (time.time() - fetch_start) * 1000
                    data["fetch_time_ms"] = fetch_time
                    return data
                else:
                    error_text = await response.text()
                    logger.error(f"API request failed with status {response.status}: {error_text}")
                    return {
                        "symbol": symbol,
                        "error": f"API request failed: {response.status} - {error_text}",
                        "price_info": {"error": f"API error: {response.status}"},
                        "options_info": {"error": f"API error: {response.status}"},
                        "financial_info": {"error": f"API error: {response.status}"},
                        "news_info": {"error": f"API error: {response.status}"},
                        "iv_info": {"error": f"API error: {response.status}"},
                    }
    except aiohttp.ClientError as e:
        logger.error(f"API request error for {symbol}: {e}")
        return {
            "symbol": symbol,
            "error": f"API request error: {str(e)}",
            "price_info": {"error": f"API connection error: {str(e)}"},
            "options_info": {"error": f"API connection error: {str(e)}"},
            "financial_info": {"error": f"API connection error: {str(e)}"},
            "news_info": {"error": f"API connection error: {str(e)}"},
            "iv_info": {"error": f"API connection error: {str(e)}"},
        }
    except Exception as e:
        logger.error(f"Unexpected error fetching from API for {symbol}: {e}")
        return {
            "symbol": symbol,
            "error": f"Unexpected error: {str(e)}",
            "price_info": {"error": str(e)},
            "options_info": {"error": str(e)},
            "financial_info": {"error": str(e)},
            "news_info": {"error": str(e)},
            "iv_info": {"error": str(e)},
        }


def format_price_display(price_info: Dict[str, Any], timezone_str: Optional[str] = None, latest_only: bool = False) -> List[str]:
    """Format price information for display."""
    lines = []
    
    symbol = price_info.get("symbol", "N/A")
    current_price = price_info.get("current_price")
    price_data = price_info.get("price_data")
    error = price_info.get("error")
    
    lines.append(f"\n{'='*80}")
    lines.append(f"PRICE INFORMATION: {symbol}")
    if latest_only:
        lines.append("(Latest price only - no historical data)")
    lines.append(f"{'='*80}")
    
    if error:
        lines.append(f"Error: {error}")
        return lines
    
    # Display current/latest price
    if current_price:
        lines.append("\nCurrent/Latest Price:")
        price_lines = _format_price_block(current_price, timezone_str or "America/New_York")
        lines.extend(price_lines)
        
        # Show data source and cache info
        source = current_price.get("source", "N/A")
        cache_hit = current_price.get("cache_hit", False)
        fetch_time = current_price.get("fetch_time_ms")
        
        source_display = source
        if cache_hit:
            source_display += " (CACHED)"
        elif source == "database":
            source_display += " (DB)"
        elif source in ["polygon_quote", "polygon_trade", "polygon_daily"]:
            source_display += " (API)"
        
        lines.append(f"Source: {source_display}")
        
        # Show timing info if available
        if fetch_time is not None:
            timing_info = f"Fetch time: {fetch_time:.1f}ms"
            if current_price.get("cache_check_time_ms"):
                timing_info += f" (cache check: {current_price.get('cache_check_time_ms'):.1f}ms)"
            if current_price.get("db_check_time_ms"):
                timing_info += f" (db check: {current_price.get('db_check_time_ms'):.1f}ms)"
            if current_price.get("api_fetch_time_ms"):
                timing_info += f" (api: {current_price.get('api_fetch_time_ms'):.1f}ms)"
            lines.append(timing_info)
    
    # Display historical price data if available
    if price_data is not None and not price_data.empty:
        lines.append(f"\nHistorical Price Data ({len(price_data)} rows):")
        display_cols = ['open', 'high', 'low', 'close']
        if 'volume' in price_data.columns:
            display_cols.append('volume')
        
        available_cols = [col for col in display_cols if col in price_data.columns]
        if available_cols:
            # Show summary stats
            lines.append(f"\nDate Range: {price_data.index.min()} to {price_data.index.max()}")
            lines.append(f"\nSummary Statistics:")
            summary = price_data[available_cols].describe()
            lines.append(str(summary))
            
            # Show first and last few rows
            lines.append(f"\nFirst 5 rows:")
            lines.append(str(price_data[available_cols].head()))
            lines.append(f"\nLast 5 rows:")
            lines.append(str(price_data[available_cols].tail()))
    
    return lines


def format_options_display(options_info: Dict[str, Any], max_options_per_expiry: int = 10) -> List[str]:
    """Format options information for display."""
    lines = []
    
    symbol = options_info.get("symbol", "N/A")
    options_data = options_info.get("options_data")
    error = options_info.get("error")
    source = options_info.get("source", "unknown")
    fetch_time = options_info.get("fetch_time_ms")
    
    lines.append(f"\n{'='*80}")
    lines.append(f"OPTIONS INFORMATION: {symbol}")
    lines.append(f"{'='*80}")
    
    # Show source and timing
    if source:
        source_display = source.upper()
        if source == "cache":
            source_display += " (CACHED)"
        elif source == "database":
            source_display += " (DB)"
        elif source == "api":
            source_display += " (API)"
        lines.append(f"Source: {source_display}")
    if fetch_time is not None:
        lines.append(f"Fetch time: {fetch_time:.1f}ms")
    if source or fetch_time is not None:
        lines.append("")
    
    if error:
        lines.append(f"Error: {error}")
        return lines
    
    if not options_data:
        lines.append("No options data available")
        return lines
    
    if not options_data.get("success", False):
        error_msg = options_data.get("error", "Unknown error")
        lines.append(f"Error fetching options: {error_msg}")
        return lines
    
    contracts = options_data.get("data", {}).get("contracts", [])
    
    if not contracts:
        lines.append("No options contracts found")
        return lines
    
    lines.append(f"\nFound {len(contracts)} options contracts")
    
    # Group by expiration date
    by_expiry = {}
    for contract in contracts:
        exp = contract.get("expiration", "Unknown")
        if exp not in by_expiry:
            by_expiry[exp] = []
        by_expiry[exp].append(contract)
    
    # Display by expiration
    for exp_date in sorted(by_expiry.keys())[:20]:  # Limit to first 20 expirations
        contracts_for_exp = by_expiry[exp_date]
        lines.append(f"\n--- Expiration: {exp_date} ({len(contracts_for_exp)} contracts) ---")
        
        # Sort by type and strike
        contracts_for_exp.sort(key=lambda x: (x.get("type", ""), x.get("strike", 0)))
        
        # Limit to max_options_per_expiry
        display_contracts = contracts_for_exp[:max_options_per_expiry]
        
        # Create table
        table_data = []
        for c in display_contracts:
            table_data.append([
                c.get("ticker", "N/A"),
                c.get("type", "N/A").upper(),
                f"${c.get('strike', 0):.2f}",
                f"${c.get('bid', 0):.2f}" if c.get("bid") is not None else "N/A",
                f"${c.get('ask', 0):.2f}" if c.get("ask") is not None else "N/A",
                f"${c.get('day_close', 0):.2f}" if c.get("day_close") is not None else "N/A",
                f"{c.get('delta', 0):.3f}" if c.get("delta") is not None else "N/A",
                f"{c.get('gamma', 0):.3f}" if c.get("gamma") is not None else "N/A",
                f"{c.get('theta', 0):.3f}" if c.get("theta") is not None else "N/A",
                f"{c.get('vega', 0):.3f}" if c.get("vega") is not None else "N/A",
                f"{c.get('implied_volatility', 0):.3f}" if c.get("implied_volatility") is not None else "N/A",
            ])
        
        if table_data:
            try:
                from tabulate import tabulate
                lines.append(tabulate(
                    table_data,
                    headers=["Ticker", "Type", "Strike", "Bid", "Ask", "Close", "Delta", "Gamma", "Theta", "Vega", "IV"],
                    tablefmt="grid"
                ))
            except ImportError:
                # Fallback if tabulate not available
                lines.append("Ticker | Type | Strike | Bid | Ask | Close | Delta | Gamma | Theta | Vega | IV")
                lines.append("-" * 80)
                for row in table_data:
                    lines.append(" | ".join(str(x) for x in row))
        
        if len(contracts_for_exp) > max_options_per_expiry:
            lines.append(f"... and {len(contracts_for_exp) - max_options_per_expiry} more contracts for this expiration")
    
    if len(by_expiry) > 20:
        lines.append(f"\n... and {len(by_expiry) - 20} more expiration dates")
    
    return lines


def format_financial_display(financial_info: Dict[str, Any]) -> List[str]:
    """Format financial ratios information for display."""
    lines = []
    
    symbol = financial_info.get("symbol", "N/A")
    financial_data = financial_info.get("financial_data")
    error = financial_info.get("error")
    source = financial_info.get("source", "unknown")
    fetch_time = financial_info.get("fetch_time_ms")
    
    lines.append(f"\n{'='*80}")
    lines.append(f"FINANCIAL RATIOS: {symbol}")
    lines.append(f"{'='*80}")
    
    # Show source and timing
    source_display = source.upper() if source else "UNKNOWN"
    if source == "cache":
        source_display += " (CACHED)"
    elif source == "database":
        source_display += " (DB)"
    elif source == "api":
        source_display += " (API)"
    lines.append(f"Source: {source_display}")
    if fetch_time is not None:
        lines.append(f"Fetch time: {fetch_time:.1f}ms")
    lines.append("")
    
    if error:
        lines.append(f"Error: {error}")
        return lines
    
    if not financial_data:
        lines.append("No financial ratios data available")
        return lines
    
    # Display key ratios
    key_ratios = [
        ("P/E Ratio", "price_to_earnings"),
        ("P/B Ratio", "price_to_book"),
        ("P/S Ratio", "price_to_sales"),
        ("PEG Ratio", "peg_ratio"),
        ("Debt-to-Equity", "debt_to_equity"),
        ("Return on Equity", "return_on_equity"),
        ("Return on Assets", "return_on_assets"),
        ("Current Ratio", "current"),
        ("Quick Ratio", "quick"),
        ("Cash Ratio", "cash"),
        ("Dividend Yield", "dividend_yield"),
        ("Market Cap", "market_cap"),
        ("Enterprise Value", "enterprise_value"),
        ("Free Cash Flow", "free_cash_flow"),
        ("EV to Sales", "ev_to_sales"),
        ("EV to EBITDA", "ev_to_ebitda"),
        ("Price to Cash Flow", "price_to_cash_flow"),
        ("Price to Free Cash Flow", "price_to_free_cash_flow"),
    ]
    
    table_data = []
    for label, key in key_ratios:
        value = financial_data.get(key)
        if value is not None:
            if isinstance(value, (int, float)):
                if abs(value) >= 1e9:
                    value_str = f"${value/1e9:.2f}B"
                elif abs(value) >= 1e6:
                    value_str = f"${value/1e6:.2f}M"
                elif abs(value) >= 1e3:
                    value_str = f"${value/1e3:.2f}K"
                else:
                    value_str = f"{value:.2f}"
            else:
                value_str = str(value)
        else:
            value_str = "N/A"
        table_data.append([label, value_str])
    
    if table_data:
        try:
            from tabulate import tabulate
            lines.append(tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))
        except ImportError:
            # Fallback if tabulate not available
            lines.append("Metric | Value")
            lines.append("-" * 80)
            for row in table_data:
                lines.append(" | ".join(str(x) for x in row))
    
    return lines

def format_news_display(news_info: Dict[str, Any]) -> List[str]:
    """Format news information for display."""
    lines = []
    
    symbol = news_info.get("symbol", "N/A")
    news_data = news_info.get("news_data")
    error = news_info.get("error")
    freshness = news_info.get("freshness")
    
    lines.append(f"\n{'='*80}")
    lines.append(f"LATEST NEWS: {symbol}")
    lines.append(f"{'='*80}")
    
    if error:
        lines.append(f"Error: {error}")
        return lines
    
    if not news_data or not news_data.get("articles"):
        lines.append("No news articles available")
        return lines
    
    # Display freshness info
    if freshness:
        age_min = freshness.get("age_minutes", 0)
        status = "🟢 FRESH" if freshness.get("is_fresh") else "🟡 STALE" if not freshness.get("needs_refetch") else "🔴 NEEDS REFETCH"
        lines.append(f"Data Status: {status} (Age: {age_min:.1f} minutes)")
        if freshness.get("needs_refetch"):
            lines.append("⚠️  Recommendation: Refetch from source (data is >2 hours old)")
        lines.append("")
    
    articles = news_data.get("articles", [])
    lines.append(f"Found {len(articles)} news articles (from {news_data.get('date_range', {}).get('start', 'N/A')} to {news_data.get('date_range', {}).get('end', 'N/A')})")
    lines.append(f"Fetched at: {news_data.get('fetched_at', 'N/A')}")
    lines.append("")
    
    # Display top articles
    for i, article in enumerate(articles[:10], 1):  # Show top 10
        lines.append(f"{i}. {article.get('title', 'No title')}")
        if article.get('published_utc'):
            lines.append(f"   Published: {article['published_utc']}")
        if article.get('publisher', {}).get('name'):
            lines.append(f"   Source: {article['publisher']['name']}")
        if article.get('description'):
            desc = article['description'][:200] + "..." if len(article.get('description', '')) > 200 else article.get('description', '')
            lines.append(f"   {desc}")
        if article.get('article_url'):
            lines.append(f"   URL: {article['article_url']}")
        lines.append("")
    
    if len(articles) > 10:
        lines.append(f"... and {len(articles) - 10} more articles")
    
    return lines

def format_iv_display(iv_info: Dict[str, Any]) -> List[str]:
    """Format IV information for display."""
    lines = []
    
    symbol = iv_info.get("symbol", "N/A")
    iv_data = iv_info.get("iv_data")
    error = iv_info.get("error")
    freshness = iv_info.get("freshness")
    
    lines.append(f"\n{'='*80}")
    lines.append(f"LATEST IMPLIED VOLATILITY: {symbol}")
    lines.append(f"{'='*80}")
    
    if error:
        lines.append(f"Error: {error}")
        return lines
    
    if not iv_data:
        lines.append("No IV data available (options data may not be available)")
        return lines
    
    # Display freshness info
    if freshness:
        age_min = freshness.get("age_minutes", 0)
        status = "🟢 FRESH" if freshness.get("is_fresh") else "🟡 STALE" if not freshness.get("needs_refetch") else "🔴 NEEDS REFETCH"
        lines.append(f"Data Status: {status} (Age: {age_min:.1f} minutes)")
        if freshness.get("needs_refetch"):
            lines.append("⚠️  Recommendation: Refetch from source (data is >10 minutes old)")
        lines.append("")
    
    stats = iv_data.get("statistics", {})
    lines.append(f"Data timestamp: {iv_data.get('data_timestamp', 'N/A')}")
    lines.append(f"Fetched at: {iv_data.get('fetched_at', 'N/A')}")
    if iv_data.get('current_price'):
        lines.append(f"Current price: ${iv_data['current_price']:.2f}")
    lines.append("")
    
    lines.append("IV Statistics:")
    lines.append(f"  Count: {stats.get('count', 'N/A')}")
    if stats.get('mean') is not None:
        lines.append(f"  Mean IV: {stats['mean']:.4f} ({stats['mean']*100:.2f}%)")
    if stats.get('median') is not None:
        lines.append(f"  Median IV: {stats['median']:.4f} ({stats['median']*100:.2f}%)")
    if stats.get('min') is not None:
        lines.append(f"  Min IV: {stats['min']:.4f} ({stats['min']*100:.2f}%)")
    if stats.get('max') is not None:
        lines.append(f"  Max IV: {stats['max']:.4f} ({stats['max']*100:.2f}%)")
    if stats.get('std') is not None:
        lines.append(f"  Std Dev: {stats['std']:.4f}")
    
    if 'atm_iv' in iv_data:
        atm = iv_data['atm_iv']
        if atm.get('mean') is not None:
            lines.append("")
            lines.append("ATM IV (within 5% of current price):")
            lines.append(f"  Mean: {atm['mean']:.4f} ({atm['mean']*100:.2f}%)")
            lines.append(f"  Count: {atm.get('count', 'N/A')}")
    
    if 'call_iv' in iv_data:
        call = iv_data['call_iv']
        if call.get('mean') is not None:
            lines.append("")
            lines.append("Call Options IV:")
            lines.append(f"  Mean: {call['mean']:.4f} ({call['mean']*100:.2f}%)")
            lines.append(f"  Count: {call.get('count', 'N/A')}")
    
    if 'put_iv' in iv_data:
        put = iv_data['put_iv']
        if put.get('mean') is not None:
            lines.append("")
            lines.append("Put Options IV:")
            lines.append(f"  Mean: {put['mean']:.4f} ({put['mean']*100:.2f}%)")
            lines.append(f"  Count: {put.get('count', 'N/A')}")
    
    return lines


async def process_symbol(
    symbol: str,
    db_instance: Optional[StockDBBase],
    args: argparse.Namespace
) -> Dict[str, Any]:
    """Process a single symbol and gather all information.
    
    Supports both API mode (--api-server) and direct DB mode (--db-path).
    """
    # If --api-server is provided, use API mode
    if args.api_server:
        # Build parameters for API call
        api_params = {
            "latest": args.latest,
            "start_date": args.start_date if not args.latest else None,
            "end_date": args.end_date if not args.latest else None,
            "options_days": args.options_days,
            "force_fetch": args.force_fetch,
            "data_source": args.data_source,
            "timezone": args.timezone,
            "show_price_history": args.show_price_history,
            "options_type": args.options_type,
            "strike_range_percent": args.strike_range_percent,
            "max_options_per_expiry": args.max_options_per_expiry,
            "show_news": args.show_news,
            "show_iv": args.show_iv,
            "no_cache": args.no_cache,
        }
        
        # Remove None values
        api_params = {k: v for k, v in api_params.items() if v is not None}
        
        return await get_stock_info_from_api(symbol, args.api_server, **api_params)
    
    # Direct DB mode - use parallel helper function
    if db_instance is None:
        return {
            "symbol": symbol,
            "error": "No database instance provided and no API server specified",
            "price_info": {"error": "No database instance"},
            "options_info": {"error": "No database instance"},
            "financial_info": {"error": "No database instance"},
            "news_info": {"error": "No database instance"},
            "iv_info": {"error": "No database instance"},
        }
    
    # Use the parallel helper function from fetch_symbol_data
    start_date = None if args.latest else args.start_date
    end_date = None if args.latest else args.end_date
    
    result = await get_stock_info_parallel(
        symbol,
        db_instance,
        start_date=start_date,
        end_date=end_date,
        force_fetch=args.force_fetch,
        data_source=args.data_source,
        timezone_str=args.timezone,
        latest_only=args.latest,
        options_days=args.options_days,
        option_type=args.options_type,
        strike_range_percent=args.strike_range_percent,
        max_options_per_expiry=args.max_options_per_expiry,
        show_news=args.show_news,
        show_iv=args.show_iv,
        enable_cache=not args.no_cache,
        redis_url=os.getenv("REDIS_URL") if not args.no_cache else None
    )
    
    return result


async def main():
    """Main entry point."""
    global args
    args = parse_args()
    
    # Normalize symbols for database storage (I:SPX -> SPX)
    from common.symbol_utils import normalize_symbol_for_db
    args.symbols = [normalize_symbol_for_db(symbol) for symbol in args.symbols]
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Check Polygon availability if needed
    if args.data_source == "polygon" and not POLYGON_AVAILABLE:
        print("Error: Polygon.io data source selected but polygon-api-client is not installed.", file=sys.stderr)
        print("Install with: pip install polygon-api-client", file=sys.stderr)
        sys.exit(1)
    
    # Initialize database instance
    enable_cache = not args.no_cache
    db_instance = None
    
    try:
        if args.db_path and ':' in args.db_path:
            if args.db_path.startswith('questdb://'):
                db_instance = get_stock_db(
                    "questdb",
                    args.db_path,
                    log_level=args.log_level,
                    enable_cache=enable_cache,
                    redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
                )
            elif args.db_path.startswith('postgresql://'):
                db_instance = get_stock_db("postgresql", args.db_path, log_level=args.log_level)
            else:
                db_instance = get_stock_db("remote", args.db_path, log_level=args.log_level)
        else:
            actual_db_path = args.db_path or (get_default_db_path("duckdb") if args.db_type == 'duckdb' else get_default_db_path("db"))
            db_instance = get_stock_db(args.db_type, actual_db_path, log_level=args.log_level)
        
        # Process all symbols
        import time
        overall_start = time.time()
        results = []
        for symbol in args.symbols:
            symbol_start = time.time()
            logger.info(f"[TIMING] Starting data fetch for {symbol}")
            symbol_result = await process_symbol(symbol, db_instance, args)
            symbol_time = (time.time() - symbol_start) * 1000
            logger.info(f"[TIMING] Completed data fetch for {symbol} in {symbol_time:.1f}ms")
            results.append(symbol_result)
        
        overall_time = (time.time() - overall_start) * 1000
        logger.info(f"[TIMING] Total time for all symbols: {overall_time:.1f}ms")
        
        # Display results
        for result in results:
            symbol = result["symbol"]
            
            # Price information
            if result["price_info"]:
                price_lines = format_price_display(result["price_info"], args.timezone, latest_only=args.latest)
                print("\n".join(price_lines))
            
            # Options information
            if result["options_info"]:
                options_lines = format_options_display(result["options_info"], args.max_options_per_expiry)
                print("\n".join(options_lines))
            
            # Financial information
            if result["financial_info"]:
                financial_lines = format_financial_display(result["financial_info"])
                print("\n".join(financial_lines))
            
            # News information
            if result["news_info"]:
                news_lines = format_news_display(result["news_info"])
                print("\n".join(news_lines))
            
            # IV information
            if result["iv_info"]:
                iv_lines = format_iv_display(result["iv_info"])
                print("\n".join(iv_lines))
            
            print()  # Spacing between symbols
        
        # Print cache statistics if available
        if db_instance and hasattr(db_instance, 'get_cache_statistics'):
            try:
                cache_stats = db_instance.get_cache_statistics()
                if cache_stats and (cache_stats.get('hits', 0) > 0 or cache_stats.get('misses', 0) > 0):
                    print("\n" + "=" * 80)
                    print("Cache Statistics")
                    print("=" * 80)
                    print(f"Hits:        {cache_stats.get('hits', 0)}")
                    print(f"Misses:      {cache_stats.get('misses', 0)}")
                    print(f"Sets:        {cache_stats.get('sets', 0)}")
                    print(f"Invalidations: {cache_stats.get('invalidations', 0)}")
                    print(f"Errors:      {cache_stats.get('errors', 0)}")
                    total = cache_stats.get('hits', 0) + cache_stats.get('misses', 0)
                    if total > 0:
                        hit_rate = (cache_stats.get('hits', 0) / total) * 100
                        print(f"Hit Rate:    {hit_rate:.2f}%")
                    print("=" * 80 + "\n")
            except Exception as e:
                if args.log_level == "DEBUG":
                    print(f"Error getting cache statistics: {e}", file=sys.stderr)
    
    finally:
        # Cleanup
        if db_instance and hasattr(db_instance, 'cache') and hasattr(db_instance.cache, 'wait_for_pending_writes'):
            try:
                await db_instance.cache.wait_for_pending_writes(timeout=10.0)
            except Exception:
                pass
        
        if db_instance and hasattr(db_instance, 'close_session') and callable(db_instance.close_session):
            try:
                await db_instance.close_session()
            except Exception as e:
                logger.debug(f"Error closing database session: {e}")


if __name__ == "__main__":
    asyncio.run(main())

