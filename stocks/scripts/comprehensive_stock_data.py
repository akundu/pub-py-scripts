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
            
            # For single-day requests on weekends/holidays, expand the search range
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            is_single_day = start_dt == end_dt
            
            # If it's a single day request, expand the range to catch nearby trading days
            if is_single_day:
                search_start = (start_dt - timedelta(days=7)).strftime('%Y-%m-%d')  # Look back 7 days
                search_end = (end_dt + timedelta(days=7)).strftime('%Y-%m-%d')      # Look forward 7 days
                print(f"Fetching historical market data for {self.config.symbol}")
                print(f"Target date: {start_date} (searching {search_start} to {search_end} for nearest trading day)")
            else:
                search_start = start_date
                search_end = end_date
            print(f"Fetching historical market data for {self.config.symbol} from {start_date} to {end_date}...")
            
            # Daily bars
            daily_bars = []
            try:
                aggs = self.client.get_aggs(
                    ticker=self.config.symbol,
                    multiplier=1,
                    timespan="day",
                    from_=search_start,
                    to=search_end,
                    adjusted=True,
                    sort="asc",
                    limit=50000
                )
                
                if hasattr(aggs, 'results') and aggs.results:
                    for bar in aggs.results:
                        bar_date = datetime.fromtimestamp(bar.timestamp / 1000).strftime('%Y-%m-%d')
                        daily_bars.append({
                            'date': bar_date,
                            'open': bar.open,
                            'high': bar.high,
                            'low': bar.low,
                            'close': bar.close,
                            'volume': bar.volume,
                            'vwap': getattr(bar, 'vwap', None)
                        })
                    
                    # If it's a single day request, find the closest trading day
                    if is_single_day and daily_bars:
                        target_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
                        closest_bar = None
                        min_diff = float('inf')
                        
                        for bar in daily_bars:
                            bar_date_dt = datetime.strptime(bar['date'], '%Y-%m-%d')
                            diff = abs((bar_date_dt - target_date_dt).days)
                            if diff < min_diff:
                                min_diff = diff
                                closest_bar = bar
                        
                        if closest_bar:
                            market_data['target_date'] = start_date
                            market_data['closest_trading_date'] = closest_bar['date']
                            market_data['days_difference'] = min_diff
                            print(f"Closest trading day to {start_date}: {closest_bar['date']} ({min_diff} days {'after' if datetime.strptime(closest_bar['date'], '%Y-%m-%d') > target_date_dt else 'before'})")
                        
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
            target_date = self.config.start_date
            is_single_day = self.config.start_date == self.config.end_date
            
            if is_single_day:
                print(f"Fetching options data for {self.config.symbol} for target date: {target_date}")
                print("Note: Searching for options contracts that were active on this date")
            else:
                print(f"Fetching options data for {self.config.symbol} (current/live data)...")
            
            # Get options contracts
            try:
                options_contracts = []
                
                # For historical data, we need to get both expired and non-expired contracts
                # and filter by expiration date
                if is_single_day:
                    # Get expired contracts that were active on the target date
                    try:
                        expired_contracts_generator = self.client.list_options_contracts(
                    underlying_ticker=self.config.symbol,
                            limit=1000,  # Increase limit for historical search
                            expired=True  # Get expired contracts
                        )
                        
                        # Also get current contracts in case some were already active
                        current_contracts_generator = self.client.list_options_contracts(
                            underlying_ticker=self.config.symbol,
                            limit=1000,
                            expired=False
                        )
                
                        # Combine both generators
                        all_contracts = []
                        contract_count = 0
                        
                        # Process expired contracts
                        for contract in expired_contracts_generator:
                            if contract_count >= 500:  # Limit for performance
                                break
                            all_contracts.append(contract)
                            contract_count += 1
                        
                        # Process current contracts
                        for contract in current_contracts_generator:
                            if contract_count >= 1000:  # Total limit
                                break
                            all_contracts.append(contract)
                            contract_count += 1
                        
                        print(f"Found {len(all_contracts)} total options contracts (expired + current)")
                        
                    except Exception as e:
                        print(f"Error fetching expired contracts: {e}")
                        # Fallback to current contracts only
                        all_contracts = list(self.client.list_options_contracts(
                            underlying_ticker=self.config.symbol,
                            limit=100,
                            expired=False
                        ))
                else:
                    # For date ranges, use current contracts
                    all_contracts = list(self.client.list_options_contracts(
                        underlying_ticker=self.config.symbol,
                        limit=50,
                        expired=False
                    ))
                
                # Filter contracts that were active on the target date
                target_date_dt = datetime.strptime(target_date, '%Y-%m-%d')
                active_contracts = []
                
                # Debug: let's see what expiration dates we have
                exp_dates_found = set()
                contracts_by_year = {}
                
                for contract in all_contracts:
                    expiration_date = getattr(contract, 'expiration_date', None)
                    if expiration_date:
                        exp_dates_found.add(expiration_date)
                        year = expiration_date.split('-')[0] if '-' in expiration_date else 'unknown'
                        if year not in contracts_by_year:
                            contracts_by_year[year] = 0
                        contracts_by_year[year] += 1
                        
                        try:
                            exp_date_dt = datetime.strptime(expiration_date, '%Y-%m-%d')
                            # Contract was active if it hadn't expired yet on the target date
                            # Show ALL options that were still active (not expired) on the target date
                            
                            days_to_expiry = (exp_date_dt - target_date_dt).days
                            
                            # Include ALL options that haven't expired yet (days_to_expiry >= 0)
                            # This will show all options that were active on the target date
                            if days_to_expiry >= 0:
                                active_contracts.append(contract)
                        except ValueError:
                            # Skip contracts with invalid date formats
                            continue
                
                # Debug output
                print(f"Debug: Found expiration dates by year: {contracts_by_year}")
                sample_dates = sorted(list(exp_dates_found))[:10]
                print(f"Debug: Sample expiration dates found: {sample_dates}")
                
                if is_single_day:
                    print(f"Found {len(active_contracts)} options contracts that were actively trading around {target_date}")
                    
                    # Sort by expiration date and strike price for better organization
                    active_contracts.sort(key=lambda x: (
                        getattr(x, 'expiration_date', '9999-12-31'),
                        getattr(x, 'contract_type', 'call'),
                        getattr(x, 'strike_price', 999999)
                    ))
                
                # Convert to our format and try to get some pricing data
                print("Processing options contracts...")
                contracts_with_data = 0
                
                for i, contract in enumerate(active_contracts[:100]):  # Show more contracts
                    if i % 20 == 0 and i > 0:  # Progress indicator
                        print(f"  Processed {i} contracts...")
                        
                    contract_data = {
                        'ticker': getattr(contract, 'ticker', None),
                        'underlying_ticker': getattr(contract, 'underlying_ticker', None),
                        'contract_type': getattr(contract, 'contract_type', None),
                        'strike_price': getattr(contract, 'strike_price', None),
                        'expiration_date': getattr(contract, 'expiration_date', None),
                        'shares_per_contract': getattr(contract, 'shares_per_contract', 100),
                        'primary_exchange': getattr(contract, 'primary_exchange', None)
                    }
                    
                    # Try to get snapshot data for Greeks and current pricing
                    if is_single_day:
                        contract_ticker = contract_data.get('ticker')
                        if contract_ticker:
                            try:
                                # Try to get option snapshot for current data (includes Greeks)
                                snapshot = self.client.get_snapshot_option(
                                    underlying_asset=self.config.symbol,
                                    option_contract=contract_ticker
                                )
                                
                                if snapshot:
                                    # Extract pricing data
                                    if hasattr(snapshot, 'fair_market_value') and snapshot.fair_market_value:
                                        contract_data['fair_market_value'] = snapshot.fair_market_value
                                        contract_data['estimated_price'] = snapshot.fair_market_value
                                    if hasattr(snapshot, 'implied_volatility') and snapshot.implied_volatility:
                                        contract_data['implied_volatility'] = snapshot.implied_volatility
                                    
                                    # Extract Greeks
                                    if hasattr(snapshot, 'delta') and snapshot.delta:
                                        contract_data['delta'] = snapshot.delta
                                    if hasattr(snapshot, 'theta') and snapshot.theta:
                                        contract_data['theta'] = snapshot.theta
                                    if hasattr(snapshot, 'gamma') and snapshot.gamma:
                                        contract_data['gamma'] = snapshot.gamma
                                    if hasattr(snapshot, 'vega') and snapshot.vega:
                                        contract_data['vega'] = snapshot.vega
                            
                                    # Extract quote data
                                    if hasattr(snapshot, 'last_quote') and snapshot.last_quote:
                                        contract_data['bid'] = getattr(snapshot.last_quote, 'bid', None)
                                        contract_data['ask'] = getattr(snapshot.last_quote, 'ask', None)
                            
                                    # Extract trade data
                                    if hasattr(snapshot, 'last_trade') and snapshot.last_trade:
                                        contract_data['last_price'] = getattr(snapshot.last_trade, 'price', None)
                            
                                    if any(contract_data.get(key) for key in ['delta', 'estimated_price', 'bid', 'ask']):
                                        contracts_with_data += 1
                        
                            except Exception as e:
                                        # Snapshot might not be available for all contracts
                                pass
                    
                    options_contracts.append(contract_data)
                
                options_data['contracts'] = options_contracts
                
                if is_single_day:
                    historical_prices_count = sum(1 for c in options_contracts if c.get('historical_price'))
                    pricing_data_count = sum(1 for c in options_contracts if c.get('estimated_price') or c.get('bid') or c.get('ask'))
                    greeks_count = sum(1 for c in options_contracts if c.get('delta') is not None)
                    print(f"Successfully found historical prices for {historical_prices_count} options contracts")
                    print(f"Successfully found pricing data for {pricing_data_count} options contracts")
                    print(f"Successfully found Greeks for {greeks_count} options contracts")
                    print(f"Total contracts found: {len(options_contracts)}")
                
            except Exception as e:
                print(f"Error fetching options contracts: {e}")
                options_data['contracts'] = []
                
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
        is_single_day = start_dt == end_dt
        
        range_table = [
            ['Start Date', self.config.start_date],
            ['End Date', self.config.end_date],
            ['Total Days', str(total_days)]
        ]
        
        # Add weekend/holiday information for single day requests
        if is_single_day and 'market_data' in data and data['market_data'].get('success'):
            market = data['market_data']['data']
            if market.get('target_date') and market.get('closest_trading_date'):
                range_table.append(['Target Date', market['target_date']])
                range_table.append(['Closest Trading Day', market['closest_trading_date']])
                if market.get('days_difference', 0) > 0:
                    day_type = 'after' if datetime.strptime(market['closest_trading_date'], '%Y-%m-%d') > start_dt else 'before'
                    range_table.append(['Days Difference', f"{market['days_difference']} days {day_type}"])
        
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
            
            # For single day requests, show the specific date data
            if is_single_day and market.get('daily_bars'):
                target_date = self.config.start_date
                closest_date = market.get('closest_trading_date', target_date)
                
                # Find the bar for the closest trading date
                target_bar = None
                for bar in market['daily_bars']:
                    if bar.get('date') == closest_date:
                        target_bar = bar
                        break
                
                if target_bar:
                    output.append(f"STOCK PRICE DATA FOR {closest_date}")
                    if closest_date != target_date:
                        output.append(f"(Closest trading day to requested date: {target_date})")
                    output.append("=" * 50)
                    
                    price_table = [
                        ['Date', target_bar.get('date', 'N/A')],
                        ['Open', f"${target_bar.get('open', 'N/A'):.2f}" if target_bar.get('open') else 'N/A'],
                        ['High', f"${target_bar.get('high', 'N/A'):.2f}" if target_bar.get('high') else 'N/A'],
                        ['Low', f"${target_bar.get('low', 'N/A'):.2f}" if target_bar.get('low') else 'N/A'],
                        ['Close', f"${target_bar.get('close', 'N/A'):.2f}" if target_bar.get('close') else 'N/A'],
                        ['Volume', f"{target_bar.get('volume', 'N/A'):,}" if target_bar.get('volume') else 'N/A'],
                        ['VWAP', f"${target_bar.get('vwap', 'N/A'):.2f}" if target_bar.get('vwap') else 'N/A']
                    ]
                    output.append(tabulate(price_table, headers=['Metric', 'Value'], tablefmt='grid'))
                    output.append("")
            
            # Show recent trading history for date ranges
            elif market.get('daily_bars') and not is_single_day:
                output.append("RECENT TRADING HISTORY (Last 5 days)")
                output.append("=" * 50)
                recent_bars = market['daily_bars'][-5:] if len(market['daily_bars']) > 5 else market['daily_bars']
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
            
            # Latest quote and trade data (only for date ranges, not single day historical requests)
            if not is_single_day and (market.get('latest_trade') or market.get('latest_quote')):
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
            
            if options:
                if is_single_day:
                    output.append(f"OPTIONS DATA FOR {self.config.start_date}")
                    output.append("=" * 50)
                    output.append("Note: Historical options contracts that were active on this date")
                else:
                    output.append("OPTIONS DATA (Current/Live)")
                    output.append("=" * 50)
                    output.append("Note: Options data shows current/active contracts with live market data")
                    output.append("")
                
                    # Basic options statistics
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
                        opts_table.append(['Expiration Dates (sample)', ', '.join(sorted(exp_dates)[:3])])
                    
                    # Show strike price range
                    strikes = [opt.get('strike_price') for opt in options if opt.get('strike_price')]
                    if strikes:
                        opts_table.append(['Strike Price Range', f"${min(strikes):.2f} - ${max(strikes):.2f}"])
                    
                    output.append(tabulate(opts_table, headers=['Metric', 'Value'], tablefmt='grid'))
                    output.append("")
                
                # Group options by expiration date for better organization
                if is_single_day:
                    # Group options by expiration date
                    options_by_expiry = {}
                    for opt in options:
                        exp_date = opt.get('expiration_date', 'Unknown')
                        if exp_date not in options_by_expiry:
                            options_by_expiry[exp_date] = []
                        options_by_expiry[exp_date].append(opt)
                    
                    # Sort expiration dates
                    sorted_exp_dates = sorted(options_by_expiry.keys())
                    
                    output.append("OPTIONS BY EXPIRATION DATE")
                    output.append("=" * 50)
                    
                    # Show options for each expiration date (limit to first 3 expiration dates)
                    for exp_date in sorted_exp_dates[:3]:
                        exp_options = options_by_expiry[exp_date]
                        calls = [opt for opt in exp_options if opt.get('contract_type') == 'call']
                        puts = [opt for opt in exp_options if opt.get('contract_type') == 'put']
                        
                        output.append(f"\nExpiration Date: {exp_date}")
                        output.append("-" * 30)
                        output.append(f"Calls: {len(calls)}, Puts: {len(puts)}")
                        
                        # Show sample options for this expiration (prioritize those with data)
                        exp_options_with_data = [opt for opt in exp_options if opt.get('historical_price') or opt.get('estimated_price') or opt.get('delta') or opt.get('bid')]
                        display_exp_options = (exp_options_with_data[:12] if exp_options_with_data 
                                             else exp_options[:12])
                        
                        if display_exp_options:
                            options_table = []
                            
                            for contract in display_exp_options:
                                contract_type = contract.get('contract_type', 'N/A').title()
                                strike = contract.get('strike_price', 'N/A')
                                strike_str = f"${strike:.0f}" if isinstance(strike, (int, float)) else str(strike)
                                
                                # Get price information (prioritize different sources)
                                price = 'N/A'
                                if contract.get('historical_price'):
                                    price = f"${contract['historical_price']:.2f}"
                                    if contract.get('exact_date_match'):
                                        price += "*"
                                elif contract.get('last_price'):
                                    price = f"${contract['last_price']:.2f}L"
                                elif contract.get('estimated_price'):
                                    price = f"${contract['estimated_price']:.2f}E"
                                elif contract.get('bid') and contract.get('ask'):
                                    mid_price = (contract['bid'] + contract['ask']) / 2
                                    price = f"${mid_price:.2f}M"
                                
                                # Get bid/ask
                                bid_ask = 'N/A'
                                if contract.get('bid') and contract.get('ask'):
                                    bid_ask = f"${contract['bid']:.2f}/${contract['ask']:.2f}"
                                
                                # Get Greeks
                                delta = f"{contract.get('delta', 'N/A'):.3f}" if contract.get('delta') is not None else 'N/A'
                                theta = f"{contract.get('theta', 'N/A'):.3f}" if contract.get('theta') is not None else 'N/A'
                                
                                # Get volume
                                volume = f"{contract.get('historical_volume', 'N/A'):,}" if contract.get('historical_volume') else 'N/A'
                                
                                # Get implied volatility
                                iv = f"{contract.get('implied_volatility', 'N/A'):.3f}" if contract.get('implied_volatility') else 'N/A'
                                
                                options_table.append([
                                    f"{contract_type} {strike_str}",
                                    price,
                                    bid_ask,
                                    delta,
                                    theta,
                                    iv
                                ])
                            
                            output.append(tabulate(options_table, headers=['Contract', 'Price', 'Bid/Ask', 'Delta', 'Theta', 'IV'], tablefmt='grid'))
                    
                    # Show notes
                    output.append("\nNotes:")
                    output.append("  * = Exact date match for historical price")
                    output.append("  L = Last trade price")
                    output.append("  E = Estimated price (Fair Market Value)")
                    output.append("  M = Mid-price (average of bid/ask)")
                    historical_count = sum(1 for opt in options if opt.get('historical_price'))
                    pricing_count = sum(1 for opt in options if opt.get('estimated_price') or opt.get('bid') or opt.get('last_price'))
                    greeks_count = sum(1 for opt in options if opt.get('delta') is not None)
                    output.append(f"  Historical prices found: {historical_count} contracts")
                    output.append(f"  Current pricing data found: {pricing_count} contracts")
                    output.append(f"  Greeks data found: {greeks_count} contracts")
                    output.append("")
                    
                else:
                    # Original display for date ranges
                    output.append("SAMPLE OPTIONS WITH PRICING DATA")
                    output.append("=" * 50)
                    
                    options_with_prices = [opt for opt in options if opt.get('historical_price') or opt.get('estimated_price')]
                    display_options = (options_with_prices[:10] if options_with_prices 
                                     else options[:10])
                    
                    if display_options:
                        options_table = []
                        
                        for contract in display_options:
                            ticker = contract.get('ticker', 'N/A')
                            contract_type = contract.get('contract_type', 'N/A').title()
                            strike = contract.get('strike_price', 'N/A')
                            expiration = contract.get('expiration_date', 'N/A')
                            
                            strike_str = f"${strike:.0f}" if isinstance(strike, (int, float)) else str(strike)
                            contract_display = f"{contract_type} {strike_str} {expiration}"
                            
                            # Initialize to N/A to avoid unbound local variable errors
                            price = 'N/A'
                            price_date = 'N/A'
                            volume = 'N/A'
                            daily_range = 'N/A'
                            
                            if contract.get('historical_price'):
                                price = f"${contract['historical_price']:.2f}"
                                price_date = contract.get('historical_date', 'N/A')
                                volume = f"{contract.get('historical_volume', 'N/A'):,}" if contract.get('historical_volume') else 'N/A'
                                if contract.get('historical_low') and contract.get('historical_high'):
                                    daily_range = f"${contract['historical_low']:.2f} - ${contract['historical_high']:.2f}"
                            
                            options_table.append([
                                contract_display[:30] + '...' if len(contract_display) > 30 else contract_display,
                                contract_type,
                                price,
                                price_date,
                                volume,
                                daily_range
                            ])
                        
                        headers = ['Contract', 'Type', 'Price', 'Price Date', 'Volume', 'Daily Range']
                        output.append(tabulate(options_table, headers=headers, tablefmt='grid'))
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
