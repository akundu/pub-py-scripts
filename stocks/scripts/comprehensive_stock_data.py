#!/usr/bin/env python3
"""
Comprehensive Stock Data Fetcher using Polygon API

This program fetches comprehensive stock information using the Polygon API including:
- Real-time and Historical Market Data (OHLCV, dividends, splits)
- Options data (quotes, historical data, options chains with Greeks)
- Company Financials (income statements, balance sheets, cash flow)
- Company Details (profile, market cap, shares outstanding, listings)
- News and Events (financial news, earnings calendar, IPO calendar)

Usage:
    python comprehensive_stock_data.py AAPL --api-key YOUR_API_KEY
    python comprehensive_stock_data.py AAPL --sections market,options,financials
"""

import os
import sys
import argparse
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from tabulate import tabulate
import json

try:
    from polygon import RESTClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    print("Error: polygon-api-client not installed. Install with: pip install polygon-api-client")
    sys.exit(1)


@dataclass
class StockDataConfig:
    """Configuration for stock data fetching"""
    api_key: str
    symbol: str
    days_back: int = 180
    start_date: Optional[str] = None  # YYYY-MM-DD format
    end_date: Optional[str] = None    # YYYY-MM-DD format
    include_options: bool = True
    include_financials: bool = True
    include_news: bool = True
    max_news_items: int = 10
    output_format: str = "json"  # json, csv, table
    save_to_file: bool = False
    output_dir: str = "output"


class PolygonStockData:
    """Comprehensive stock data fetcher using Polygon API"""
    
    def __init__(self, config: StockDataConfig):
        self.config = config
        self.client = RESTClient(config.api_key)
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _handle_api_error(self, error: Exception, data_type: str) -> Dict[str, Any]:
        """Handle API errors gracefully"""
        error_msg = f"Error fetching {data_type}: {str(error)}"
        print(f"Warning: {error_msg}")
        return {"error": error_msg, "data": None}

    async def get_market_data(self) -> Dict[str, Any]:
        """Fetch real-time and historical market data"""
        market_data = {}
        
        try:
            # Use configured date range
            start_date = self.config.start_date
            end_date = self.config.end_date
            
            print(f"Fetching historical market data for {self.config.symbol} from {start_date} to {end_date}...")
            
            # Daily bars
            daily_bars = []
            try:
                aggs = self.client.get_aggs(
                    ticker=self.config.symbol,
                    multiplier=1,
                    timespan="day",
                    from_=start_date,
                    to=end_date,
                    adjusted=True,
                    sort="asc",
                    limit=50000
                )
                
                if hasattr(aggs, 'results') and aggs.results:
                    for bar in aggs.results:
                        daily_bars.append({
                            'date': datetime.fromtimestamp(bar.timestamp / 1000).strftime('%Y-%m-%d'),
                            'open': bar.open,
                            'high': bar.high,
                            'low': bar.low,
                            'close': bar.close,
                            'volume': bar.volume,
                            'vwap': getattr(bar, 'vwap', None)
                        })
                        
            except Exception as e:
                print(f"Error fetching daily bars: {e}")
            
            market_data['daily_bars'] = daily_bars
            
            # Get latest quote
            try:
                quote = self.client.get_last_quote(ticker=self.config.symbol)
                if quote:
                    market_data['latest_quote'] = {
                        'bid': getattr(quote, 'bid_price', None),
                        'ask': getattr(quote, 'ask_price', None),
                        'bid_size': getattr(quote, 'bid_size', None),
                        'ask_size': getattr(quote, 'ask_size', None),
                        'timestamp': datetime.fromtimestamp(getattr(quote, 'sip_timestamp', 0) / 1000000000).isoformat() if hasattr(quote, 'sip_timestamp') else None
                    }
            except Exception as e:
                print(f"Error fetching latest quote: {e}")
                
                            # Get latest trade
            try:
                trade = self.client.get_last_trade(ticker=self.config.symbol)
                if trade:
                    market_data['latest_trade'] = {
                        'price': getattr(trade, 'price', None),
                        'size': getattr(trade, 'size', None),
                        'timestamp': datetime.fromtimestamp(getattr(trade, 'sip_timestamp', 0) / 1000000000).isoformat() if hasattr(trade, 'sip_timestamp') else None
                    }
            except Exception as e:
                print(f"Error fetching latest trade: {e}")
                
            # Get dividends
            try:
                dividends = []
                div_data = self.client.list_dividends(
                    ticker=self.config.symbol,
                    limit=50
                )
                if hasattr(div_data, 'results') and div_data.results:
                    for div in div_data.results:
                        dividends.append({
                            'cash_amount': getattr(div, 'cash_amount', None),
                            'currency': getattr(div, 'currency', 'USD'),
                            'dividend_type': getattr(div, 'dividend_type', 'CD'),
                            'ex_dividend_date': getattr(div, 'ex_dividend_date', None),
                            'frequency': getattr(div, 'frequency', None),
                            'pay_date': getattr(div, 'pay_date', None),
                            'record_date': getattr(div, 'record_date', None)
                        })
                market_data['dividends'] = dividends
            except Exception as e:
                print(f"Error fetching dividends: {e}")
                market_data['dividends'] = []
                
            # Get stock splits
            try:
                splits = []
                split_data = self.client.list_splits(
                    ticker=self.config.symbol,
                    limit=50
                )
                if hasattr(split_data, 'results') and split_data.results:
                    for split in split_data.results:
                        split_from = getattr(split, 'split_from', None)
                        split_to = getattr(split, 'split_to', None)
                        splits.append({
                            'execution_date': getattr(split, 'execution_date', None),
                            'split_from': split_from,
                            'split_to': split_to,
                            'split_ratio': split_from / split_to if split_to and split_to != 0 else None
                        })
                market_data['stock_splits'] = splits
            except Exception as e:
                print(f"Error fetching stock splits: {e}")
                market_data['stock_splits'] = []
                
        except Exception as e:
            return self._handle_api_error(e, "market data")
            
        return {"success": True, "data": market_data}

    async def get_options_data(self) -> Dict[str, Any]:
        """Fetch options data including chains and Greeks"""
        if not self.config.include_options:
            return {"success": True, "data": {"message": "Options data skipped per configuration"}}
            
        options_data = {}
        
        try:
            print(f"Fetching options data for {self.config.symbol} (current/live data)...")
            
            # Get options contracts
            try:
                options_contracts = []
                contracts_generator = self.client.list_options_contracts(
                    underlying_ticker=self.config.symbol,
                    limit=50,  # Limit for performance
                    expired=False
                )
                
                # Iterate through the generator to get actual contracts
                contract_count = 0
                for contract in contracts_generator:
                    if contract_count >= 50:  # Limit to first 50 for performance
                        break
                        
                    contract_data = {
                        'ticker': getattr(contract, 'ticker', None),
                        'underlying_ticker': getattr(contract, 'underlying_ticker', None),
                        'contract_type': getattr(contract, 'contract_type', None),
                        'strike_price': getattr(contract, 'strike_price', None),
                        'expiration_date': getattr(contract, 'expiration_date', None),
                        'shares_per_contract': getattr(contract, 'shares_per_contract', 100),
                        'primary_exchange': getattr(contract, 'primary_exchange', None)
                    }
                    
                    # Try to get additional option details if available
                    try:
                        # Note: Some fields might require higher tier access
                        contract_data.update({
                            'exercise_type': getattr(contract, 'exercise_type', None),
                            'additional_underlyings': getattr(contract, 'additional_underlyings', None)
                        })
                    except Exception as e:
                        # Additional fields might not be available
                        pass
                        
                    options_contracts.append(contract_data)
                    contract_count += 1
                        
                options_data['contracts'] = options_contracts
                print(f"Found {len(options_contracts)} options contracts for {self.config.symbol}")
                
                # Try to get options chain snapshot for additional data
                try:
                    chain_snapshot = self.client.list_snapshot_options_chain(
                        underlying_asset=self.config.symbol
                    )
                    
                    chain_data = []
                    if chain_snapshot:
                        snapshot_count = 0
                        for snapshot in chain_snapshot:
                            if snapshot_count >= 20:  # Limit for performance
                                break
                            
                            # Extract day data
                            day_change = None
                            day_change_percent = None
                            if hasattr(snapshot, 'day') and snapshot.day:
                                day_change = getattr(snapshot.day, 'change', None)
                                day_change_percent = getattr(snapshot.day, 'change_percent', None)
                            
                            # Extract quote data
                            last_quote_bid = None
                            last_quote_ask = None
                            if hasattr(snapshot, 'last_quote') and snapshot.last_quote:
                                last_quote_bid = getattr(snapshot.last_quote, 'bid', None)
                                last_quote_ask = getattr(snapshot.last_quote, 'ask', None)
                            
                            # Extract trade data
                            last_trade_price = None
                            if hasattr(snapshot, 'last_trade') and snapshot.last_trade:
                                last_trade_price = getattr(snapshot.last_trade, 'price', None)
                            
                            chain_data.append({
                                'ticker': getattr(snapshot, 'ticker', None),
                                'day_change': day_change,
                                'day_change_percent': day_change_percent,
                                'last_quote_bid': last_quote_bid,
                                'last_quote_ask': last_quote_ask,
                                'last_trade_price': last_trade_price,
                                'open_interest': getattr(snapshot, 'open_interest', None),
                                'implied_volatility': getattr(snapshot, 'implied_volatility', None)
                            })
                            snapshot_count += 1
                    
                    options_data['chain_snapshots'] = chain_data
                    if chain_data:
                        print(f"Found {len(chain_data)} options chain snapshots")
                        
                except Exception as e:
                    print(f"Note: Options chain snapshots not available: {e}")
                    options_data['chain_snapshots'] = []
                
            except Exception as e:
                print(f"Error fetching options contracts: {e}")
                options_data['contracts'] = []
                options_data['chain_snapshots'] = []
                
        except Exception as e:
            return self._handle_api_error(e, "options data")
            
        return {"success": True, "data": options_data}

    async def get_company_financials(self) -> Dict[str, Any]:
        """Fetch company financial statements"""
        if not self.config.include_financials:
            return {"success": True, "data": {"message": "Financial data skipped per configuration"}}
            
        financials_data = {}
        
        try:
            print(f"Financial data for {self.config.symbol} is not available through this Polygon API version...")
            
            # Financial data is not available in the current Polygon API client version
            # This would require a higher tier subscription or different API endpoints
            financials_data['message'] = "Financial statements (income statement, balance sheet, cash flow) are not available through the current Polygon API client. This feature may require a higher tier subscription or different API endpoints."
            financials_data['income_statements'] = []
            financials_data['balance_sheets'] = []
            financials_data['cash_flow_statements'] = []
                
        except Exception as e:
            return self._handle_api_error(e, "financial data")
            
        return {"success": True, "data": financials_data}

    async def get_company_details(self) -> Dict[str, Any]:
        """Fetch company profile and details"""
        company_data = {}
        
        try:
            print(f"Fetching company details for {self.config.symbol}...")
            
            # Get ticker details
            try:
                ticker_details = self.client.get_ticker_details(self.config.symbol)
                if ticker_details:
                    # Handle address object properly
                    address_info = {}
                    if hasattr(ticker_details, 'address') and ticker_details.address:
                        addr = ticker_details.address
                        address_info = {
                            'address1': getattr(addr, 'address1', None),
                            'city': getattr(addr, 'city', None),
                            'state': getattr(addr, 'state', None),
                            'postal_code': getattr(addr, 'postal_code', None),
                        }
                    
                    company_data['profile'] = {
                        'ticker': getattr(ticker_details, 'ticker', None),
                        'name': getattr(ticker_details, 'name', None),
                        'description': getattr(ticker_details, 'description', None),
                        'homepage_url': getattr(ticker_details, 'homepage_url', None),
                        'total_employees': getattr(ticker_details, 'total_employees', None),
                        'list_date': getattr(ticker_details, 'list_date', None),
                        'locale': getattr(ticker_details, 'locale', None),
                        'primary_exchange': getattr(ticker_details, 'primary_exchange', None),
                        'type': getattr(ticker_details, 'type', None),
                        'currency_name': getattr(ticker_details, 'currency_name', None),
                        'cik': getattr(ticker_details, 'cik', None),
                        'composite_figi': getattr(ticker_details, 'composite_figi', None),
                        'share_class_figi': getattr(ticker_details, 'share_class_figi', None),
                        'market_cap': getattr(ticker_details, 'market_cap', None),
                        'phone_number': getattr(ticker_details, 'phone_number', None),
                        'address': address_info,
                        'sic_code': getattr(ticker_details, 'sic_code', None),
                        'sic_description': getattr(ticker_details, 'sic_description', None),
                        'ticker_root': getattr(ticker_details, 'ticker_root', None),
                        'weighted_shares_outstanding': getattr(ticker_details, 'weighted_shares_outstanding', None),
                        'round_lot': getattr(ticker_details, 'round_lot', None)
                    }
                    
                    # Add branding info if available
                    if hasattr(ticker_details, 'branding'):
                        company_data['profile']['branding'] = {
                            'logo_url': getattr(ticker_details.branding, 'logo_url', None),
                            'icon_url': getattr(ticker_details.branding, 'icon_url', None)
                        }
                        
            except Exception as e:
                print(f"Error fetching company details: {e}")
                company_data['profile'] = {}
                
        except Exception as e:
            return self._handle_api_error(e, "company details")
            
        return {"success": True, "data": company_data}

    async def get_news_and_events(self) -> Dict[str, Any]:
        """Fetch news and events data"""
        if not self.config.include_news:
            return {"success": True, "data": {"message": "News data skipped per configuration"}}
            
        news_data = {}
        
        try:
            print(f"Fetching news and events for {self.config.symbol} from {self.config.start_date} to {self.config.end_date}...")
            
            # Get financial news
            try:
                news_articles = []
                # Convert dates to the format expected by Polygon API (ISO format with timezone)
                start_dt = datetime.strptime(self.config.start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(self.config.end_date, '%Y-%m-%d')
                
                # Format for Polygon API (they expect ISO format)
                start_iso = start_dt.strftime('%Y-%m-%d')
                end_iso = (end_dt + timedelta(days=1)).strftime('%Y-%m-%d')  # Add 1 day to include end date
                
                news_generator = self.client.list_ticker_news(
                    ticker=self.config.symbol,
                    published_utc_gte=start_iso,
                    published_utc_lte=end_iso,
                    limit=self.config.max_news_items,
                    order="desc"
                )
                
                # Iterate through the generator to get actual articles
                article_count = 0
                for article in news_generator:
                    if article_count >= self.config.max_news_items:
                        break
                        
                    news_articles.append({
                        'id': getattr(article, 'id', None),
                        'title': getattr(article, 'title', None),
                        'author': getattr(article, 'author', None),
                        'published_utc': getattr(article, 'published_utc', None),
                        'article_url': getattr(article, 'article_url', None),
                        'image_url': getattr(article, 'image_url', None),
                        'description': getattr(article, 'description', None),
                        'keywords': getattr(article, 'keywords', []),
                        'publisher': {
                            'name': getattr(article.publisher, 'name', None),
                            'homepage_url': getattr(article.publisher, 'homepage_url', None),
                            'logo_url': getattr(article.publisher, 'logo_url', None),
                            'favicon_url': getattr(article.publisher, 'favicon_url', None)
                        } if hasattr(article, 'publisher') and article.publisher else None
                    })
                    article_count += 1
                        
                news_data['financial_news'] = news_articles
                print(f"Found {len(news_articles)} news articles for {self.config.symbol}")
                
            except Exception as e:
                print(f"Error fetching news: {e}")
                news_data['financial_news'] = []
                
        except Exception as e:
            return self._handle_api_error(e, "news and events")
            
        return {"success": True, "data": news_data}

    async def get_comprehensive_data(self) -> Dict[str, Any]:
        """Fetch all available data for the stock symbol"""
        print(f"\nFetching comprehensive data for {self.config.symbol}...")
        print("=" * 60)
        
        comprehensive_data = {
            'symbol': self.config.symbol,
            'timestamp': datetime.now().isoformat(),
            'market_data': {},
            'options_data': {},
            'company_financials': {},
            'company_details': {},
            'news_and_events': {}
        }
        
        # Fetch all data sections
        tasks = [
            ('market_data', self.get_market_data()),
            ('company_details', self.get_company_details()),
            ('company_financials', self.get_company_financials()),
            ('options_data', self.get_options_data()),
            ('news_and_events', self.get_news_and_events())
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        # Process results
        for i, (section_name, _) in enumerate(tasks):
            result = results[i]
            if isinstance(result, Exception):
                comprehensive_data[section_name] = self._handle_api_error(result, section_name)
            else:
                comprehensive_data[section_name] = result
                
        return comprehensive_data

    def format_output(self, data: Dict[str, Any], format_type: str = "json") -> str:
        """Format the output data"""
        if format_type == "json":
            return json.dumps(data, indent=2, default=str)
        elif format_type == "table":
            return self._format_as_table(data)
        else:
            return str(data)
            
    def _format_as_table(self, data: Dict[str, Any]) -> str:
        """Format data as readable tables"""
        output = []
        
        # Data range info
        output.append("DATA RANGE")
        output.append("=" * 50)
        
        # Calculate total days
        start_dt = datetime.strptime(self.config.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(self.config.end_date, '%Y-%m-%d')
        total_days = (end_dt - start_dt).days + 1
        
        range_table = [
            ['Start Date', self.config.start_date],
            ['End Date', self.config.end_date],
            ['Total Days', str(total_days)]
        ]
        output.append(tabulate(range_table, headers=['Period', 'Value'], tablefmt='grid'))
        output.append("")
        
        # Company Profile
        if 'company_details' in data and data['company_details'].get('success') and data['company_details']['data'].get('profile'):
            profile = data['company_details']['data']['profile']
            output.append("COMPANY PROFILE")
            output.append("=" * 50)
            profile_table = [
                ['Name', profile.get('name', 'N/A')],
                ['Ticker', profile.get('ticker', 'N/A')],
                ['Exchange', profile.get('primary_exchange', 'N/A')],
                ['Market Cap', f"${profile.get('market_cap', 0):,.0f}" if profile.get('market_cap') else 'N/A'],
                ['Employees', f"{profile.get('total_employees', 0):,}" if profile.get('total_employees') else 'N/A'],
                ['Homepage', profile.get('homepage_url', 'N/A')],
                ['Description', (profile.get('description', 'N/A')[:100] + '...') if profile.get('description') and len(profile.get('description', '')) > 100 else profile.get('description', 'N/A')]
            ]
            output.append(tabulate(profile_table, headers=['Field', 'Value'], tablefmt='grid'))
            output.append("")
        
        # Latest Market Data
        if 'market_data' in data and data['market_data'].get('success'):
            market = data['market_data']['data']
            
            # Latest quote and trade data
            if market.get('latest_trade') or market.get('latest_quote'):
                output.append("LATEST MARKET DATA")
                output.append("=" * 50)
                
                market_table = []
                if market.get('latest_trade'):
                    trade = market['latest_trade']
                    market_table.append(['Last Trade Price', f"${trade.get('price', 'N/A')}"])
                    market_table.append(['Last Trade Size', f"{trade.get('size', 'N/A'):,}" if trade.get('size') else 'N/A'])
                    
                if market.get('latest_quote'):
                    quote = market['latest_quote']
                    market_table.append(['Bid Price', f"${quote.get('bid', 'N/A')}"])
                    market_table.append(['Ask Price', f"${quote.get('ask', 'N/A')}"])
                    market_table.append(['Bid Size', f"{quote.get('bid_size', 'N/A'):,}" if quote.get('bid_size') else 'N/A'])
                    market_table.append(['Ask Size', f"{quote.get('ask_size', 'N/A'):,}" if quote.get('ask_size') else 'N/A'])
                    
                if market_table:
                    output.append(tabulate(market_table, headers=['Metric', 'Value'], tablefmt='grid'))
                    output.append("")
            
            # Historical data summary
            if market.get('daily_bars'):
                output.append("RECENT TRADING HISTORY (Last 5 days)")
                output.append("=" * 50)
                recent_bars = market['daily_bars'][-5:] if market['daily_bars'] and len(market['daily_bars']) > 5 else market['daily_bars']
                if recent_bars:
                    hist_table = []
                    for bar in recent_bars:
                        hist_table.append([
                            bar.get('date', 'N/A'),
                            f"${bar.get('open', 'N/A'):.2f}" if bar.get('open') else 'N/A',
                            f"${bar.get('high', 'N/A'):.2f}" if bar.get('high') else 'N/A',
                            f"${bar.get('low', 'N/A'):.2f}" if bar.get('low') else 'N/A',
                            f"${bar.get('close', 'N/A'):.2f}" if bar.get('close') else 'N/A',
                            f"{bar.get('volume', 'N/A'):,}" if bar.get('volume') else 'N/A'
                        ])
                    output.append(tabulate(hist_table, headers=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], tablefmt='grid'))
                    output.append("")
            
            # Dividends and splits
            if market.get('dividends') or market.get('stock_splits'):
                output.append("CORPORATE ACTIONS")
                output.append("=" * 50)
                
                if market.get('dividends'):
                    recent_divs = market['dividends'][:3]  # Show last 3 dividends
                    if recent_divs:
                        output.append("Recent Dividends:")
                        for div in recent_divs:
                            output.append(f"  • ${div.get('cash_amount', 'N/A')} on {div.get('pay_date', 'N/A')} (Ex-div: {div.get('ex_dividend_date', 'N/A')})")
                        output.append("")
                        
                if market.get('stock_splits'):
                    recent_splits = market['stock_splits'][:3]  # Show last 3 splits
                    if recent_splits:
                        output.append("Recent Stock Splits:")
                        for split in recent_splits:
                            ratio = split.get('split_ratio')
                            ratio_str = f"{ratio:.2f}:1" if ratio else "N/A"
                            output.append(f"  • {ratio_str} split on {split.get('execution_date', 'N/A')}")
                        output.append("")
        
        # Options summary
        if 'options_data' in data and data['options_data'].get('success'):
            options = data['options_data']['data'].get('contracts', [])
            chain_snapshots = data['options_data']['data'].get('chain_snapshots', [])
            
            if options or chain_snapshots:
                output.append("OPTIONS DATA")
                output.append("=" * 50)
                
                # Basic options statistics
                if options:
                    calls = [opt for opt in options if opt.get('contract_type') == 'call']
                    puts = [opt for opt in options if opt.get('contract_type') == 'put']
                    
                    opts_table = [
                        ['Total Options Contracts', f"{len(options):,}"],
                        ['Call Options', f"{len(calls):,}"],
                        ['Put Options', f"{len(puts):,}"]
                    ]
                    
                    # Show expiration dates
                    exp_dates = list(set([opt.get('expiration_date') for opt in options[:20] if opt.get('expiration_date')]))
                    if exp_dates:
                        opts_table.append(['Next Expiration Dates', ', '.join(sorted(exp_dates)[:3])])
                    
                    # Show strike price range
                    strikes = [opt.get('strike_price') for opt in options if opt.get('strike_price')]
                    if strikes:
                        opts_table.append(['Strike Price Range', f"${min(strikes):.2f} - ${max(strikes):.2f}"])
                    
                    output.append(tabulate(opts_table, headers=['Metric', 'Value'], tablefmt='grid'))
                    output.append("")
                
                # Show sample options with live data if available
                if chain_snapshots:
                    output.append("SAMPLE OPTIONS WITH LIVE DATA")
                    output.append("=" * 50)
                    
                    live_options = []
                    for snapshot in chain_snapshots[:10]:  # Show first 10
                        ticker = snapshot.get('ticker', 'N/A')
                        if ticker != 'N/A':
                            # Extract info from ticker (e.g., O:AAPL241231C00200000)
                            try:
                                parts = ticker.split(':')[1] if ':' in ticker else ticker
                                # This is a basic parsing - actual format may vary
                                contract_type = 'Call' if 'C' in parts[-10:] else 'Put' if 'P' in parts[-10:] else 'N/A'
                            except:
                                contract_type = 'N/A'
                        else:
                            contract_type = 'N/A'
                            
                        live_options.append([
                            ticker[:20] + '...' if ticker and len(ticker) > 20 else ticker,
                            contract_type,
                            f"${snapshot.get('last_trade_price', 'N/A')}" if snapshot.get('last_trade_price') else 'N/A',
                            f"${snapshot.get('last_quote_bid', 'N/A')}" if snapshot.get('last_quote_bid') else 'N/A',
                            f"${snapshot.get('last_quote_ask', 'N/A')}" if snapshot.get('last_quote_ask') else 'N/A',
                            f"{snapshot.get('day_change_percent', 'N/A'):.2f}%" if snapshot.get('day_change_percent') else 'N/A'
                        ])
                    
                    if live_options:
                        output.append(tabulate(live_options, headers=['Contract', 'Type', 'Last Price', 'Bid', 'Ask', 'Day Change %'], tablefmt='grid'))
                        output.append("")
        
        # Recent News
        if 'news_and_events' in data and data['news_and_events'].get('success'):
            news = data['news_and_events']['data'].get('financial_news', [])
            if news:
                output.append("RECENT NEWS")
                output.append("=" * 50)
                for i, article in enumerate(news[:5], 1):  # Show top 5 articles
                    title = article.get('title', 'N/A')
                    # Truncate long titles
                    if title and len(title) > 80:
                        title = title[:77] + '...'
                    output.append(f"{i}. {title}")
                    
                    published = article.get('published_utc', 'N/A')
                    if published != 'N/A':
                        try:
                            # Format the date nicely
                            dt = datetime.fromisoformat(published.replace('Z', '+00:00'))
                            published = dt.strftime('%Y-%m-%d %H:%M UTC')
                        except:
                            pass
                    
                    output.append(f"   Published: {published}")
                    
                    publisher = article.get('publisher', {})
                    if publisher and publisher.get('name'):
                        output.append(f"   Source: {publisher.get('name')}")
                    
                    output.append("")
        
        return "\n".join(output)

    async def save_data(self, data: Dict[str, Any]) -> None:
        """Save data to file"""
        if not self.config.save_to_file:
            return
            
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.symbol}_comprehensive_data_{timestamp}.json"
        filepath = os.path.join(self.config.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        print(f"\nData saved to: {filepath}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Fetch comprehensive stock data using Polygon API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using environment variable (recommended) with default 180 days
  export POLYGON_API_KEY=YOUR_API_KEY
  python comprehensive_stock_data.py AAPL
  python comprehensive_stock_data.py MSFT --format table
  
  # Using specific date ranges
  python comprehensive_stock_data.py AAPL --start-date 2024-01-01 --end-date 2024-12-31
  python comprehensive_stock_data.py GOOGL --days-back 60 --format table
  python comprehensive_stock_data.py TSLA --start-date 2024-06-01 --no-options --save
  
  # Using command line argument for API key
  python comprehensive_stock_data.py AAPL --api-key YOUR_API_KEY --start-date 2024-01-01
        """
    )
    
    parser.add_argument('symbol', help='Stock symbol to fetch data for')
    parser.add_argument('--api-key', help='Polygon API key (or set POLYGON_API_KEY environment variable)')
    parser.add_argument('--days-back', type=int, default=180, help='Number of days back for historical data (default: 180)')
    parser.add_argument('--start-date', help='Start date for historical data (YYYY-MM-DD format). Overrides --days-back if provided.')
    parser.add_argument('--end-date', help='End date for historical data (YYYY-MM-DD format). Default is today.')
    parser.add_argument('--format', choices=['json', 'table'], default='json', help='Output format (default: json)')
    parser.add_argument('--no-options', action='store_true', help='Skip options data')
    parser.add_argument('--no-financials', action='store_true', help='Skip financial data')
    parser.add_argument('--no-news', action='store_true', help='Skip news data')
    parser.add_argument('--max-news', type=int, default=10, help='Maximum number of news articles (default: 10)')
    parser.add_argument('--save', action='store_true', help='Save data to JSON file')
    parser.add_argument('--output-dir', default='output', help='Output directory for saved files (default: output)')
    
    args = parser.parse_args()
    
    # Get API key from command line argument or environment variable
    api_key = args.api_key or os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("Error: Polygon API key is required. Either:")
        print("  1. Use --api-key YOUR_API_KEY")
        print("  2. Set environment variable: export POLYGON_API_KEY=YOUR_API_KEY")
        sys.exit(1)
    
    # Handle date arguments
    end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    
    if args.start_date:
        start_date = args.start_date
    else:
        # Use days_back to calculate start_date
        start_dt = datetime.now() - timedelta(days=args.days_back)
        start_date = start_dt.strftime('%Y-%m-%d')
    
    # Validate date formats
    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError as e:
        print(f"Error: Invalid date format. Use YYYY-MM-DD format. {e}")
        sys.exit(1)
    
    # Create configuration
    config = StockDataConfig(
        api_key=api_key,
        symbol=args.symbol.upper(),
        days_back=args.days_back,
        start_date=start_date,
        end_date=end_date,
        include_options=not args.no_options,
        include_financials=not args.no_financials,
        include_news=not args.no_news,
        max_news_items=args.max_news,
        output_format=args.format,
        save_to_file=args.save,
        output_dir=args.output_dir
    )
    
    # Fetch and display data
    async with PolygonStockData(config) as stock_data:
        try:
            comprehensive_data = await stock_data.get_comprehensive_data()
            
            # Format and display output
            if args.format == 'table':
                print("\n" + stock_data.format_output(comprehensive_data, 'table'))
            else:
                print(stock_data.format_output(comprehensive_data, 'json'))
            
            # Save data if requested
            if args.save:
                await stock_data.save_data(comprehensive_data)
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 