#!/usr/bin/env python3
"""
Streak-Based Trading Strategy Backtester

This program implements a backtesting system that tracks consecutive up/down streaks
and adjusts position sizes based on streak length. The strategy buys during down
streaks and sells during up streaks, with position sizing based on streak multipliers.

Author: AI Assistant
Date: 2025
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import logging

from common.stock_db import StockDBClient
from common.streak_analyzer import StreakAnalyzer


class Action(Enum):
    """Trading action types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Position:
    """Represents a trading position."""
    shares: float = 0.0
    avg_price: float = 0.0
    total_invested: float = 0.0
    
    def add_shares(self, shares: float, price: float, amount: float) -> None:
        """Add shares to position."""
        if shares > 0:
            # Calculate new average price
            total_cost = self.total_invested + amount
            total_shares = self.shares + shares
            self.avg_price = total_cost / total_shares if total_shares > 0 else price
            self.shares = total_shares
            self.total_invested = total_cost
    
    def remove_shares(self, shares: float, price: float) -> float:
        """Remove shares from position and return proceeds."""
        if shares <= 0 or self.shares <= 0:
            return 0.0
        
        shares_to_sell = min(shares, self.shares)
        proceeds = shares_to_sell * price
        
        # Update position
        self.shares -= shares_to_sell
        if self.shares > 0:
            # Recalculate average price (proportional to remaining shares)
            remaining_ratio = self.shares / (self.shares + shares_to_sell)
            self.total_invested *= remaining_ratio
        else:
            # All shares sold
            self.avg_price = 0.0
            self.total_invested = 0.0
        
        return proceeds


@dataclass
class Transaction:
    """Represents a trading transaction."""
    date: datetime
    action: Action
    shares: float
    price: float
    amount: float
    streak_length: int
    streak_type: str
    running_shares: float
    running_cash: float
    
    def __str__(self) -> str:
        return (f"{self.date.strftime('%Y-%m-%d %H:%M')} | {self.action.value:4} | "
                f"{self.shares:8.2f} | ${self.price:8.2f} | ${self.amount:10.2f} | "
                f"{self.streak_length:2d} | {self.streak_type:4} | "
                f"{self.running_shares:8.2f} | ${self.running_cash:12.2f}")


@dataclass
class BacktestResult:
    """Results of the backtest."""
    # Summary statistics
    total_invested: float = 0.0
    total_current_value: float = 0.0
    net_profit_loss: float = 0.0
    net_profit_loss_pct: float = 0.0
    buy_transactions: int = 0
    sell_transactions: int = 0
    final_shares: float = 0.0
    final_cash: float = 0.0
    
    # Performance metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    buy_hold_return: float = 0.0
    buy_hold_return_pct: float = 0.0
    
    # Streak statistics
    total_up_streaks: int = 0
    total_down_streaks: int = 0
    avg_up_streak_length: float = 0.0
    avg_down_streak_length: float = 0.0
    
    # Transaction history
    transactions: List[Transaction] = field(default_factory=list)
    
    def calculate_metrics(self, initial_price: float, final_price: float, 
                         price_history: List[float]) -> None:
        """Calculate performance metrics."""
        # Buy and hold comparison
        if initial_price > 0:
            self.buy_hold_return = final_price - initial_price
            self.buy_hold_return_pct = (self.buy_hold_return / initial_price) * 100
        
        # Max drawdown calculation
        if price_history:
            peak = price_history[0]
            max_dd = 0.0
            
            for price in price_history:
                if price > peak:
                    peak = price
                drawdown = (peak - price) / peak if peak > 0 else 0
                max_dd = max(max_dd, drawdown)
            
            self.max_drawdown = max_dd
            self.max_drawdown_pct = max_dd * 100
        
        # Sharpe ratio (simplified - assuming 0% risk-free rate)
        if self.transactions:
            returns = []
            for i in range(1, len(self.transactions)):
                prev_value = self.transactions[i-1].running_cash + (
                    self.transactions[i-1].running_shares * self.transactions[i-1].price
                )
                curr_value = self.transactions[i].running_cash + (
                    self.transactions[i].running_shares * self.transactions[i].price
                )
                if prev_value > 0:
                    returns.append((curr_value - prev_value) / prev_value)
            
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                self.sharpe_ratio = avg_return / std_return if std_return > 0 else 0.0


class StreakBacktester:
    """
    Backtesting engine for streak-based trading strategy.
    
    This class implements a trading strategy that:
    1. Buys during down streaks with increasing position sizes
    2. Sells during up streaks with increasing position sizes
    3. Uses streak multipliers to adjust position sizing
    """
    
    def __init__(self, 
                 stock_symbol: str,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 lookback_days: int = 90,
                 down_threshold_percent: float = 2.0,
                 up_threshold_percent: float = 2.0,
                 base_investment_amount: float = 1000.0,
                 max_total_investment: float = 10000.0,
                 streak_multiplier: float = 1.5,
                 debug: bool = False,
                 port: int = 9100):
        """
        Initialize the backtester.
        
        Args:
            stock_symbol: Stock ticker symbol
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
            lookback_days: Days to go back from today if start_date not provided
            down_threshold_percent: Minimum percentage drop to trigger buy
            up_threshold_percent: Minimum percentage gain to trigger sell
            base_investment_amount: Base dollar amount to invest
            max_total_investment: Maximum total investment allowed
            streak_multiplier: Multiplier for each consecutive streak day
            debug: Enable debug output
            port: Port for StockDBClient
        """
        self.stock_symbol = stock_symbol.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.lookback_days = lookback_days
        self.down_threshold_percent = down_threshold_percent
        self.up_threshold_percent = up_threshold_percent
        self.base_investment_amount = base_investment_amount
        self.max_total_investment = max_total_investment
        self.streak_multiplier = streak_multiplier
        self.debug = debug
        self.port = port
        
        # Initialize components
        self.streak_analyzer = StreakAnalyzer()
        self.client = None
        
        # Trading state
        self.position = Position()
        self.current_cash = max_total_investment
        self.total_invested = 0.0
        
        # Streak tracking
        self.current_streak_type = None
        self.current_streak_length = 0
        
        # Results
        self.result = BacktestResult()
        self.transactions = []
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'streak_backtest_{self.stock_symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def _initialize_client(self) -> None:
        """Initialize the database client."""
        if not self.port:
            # Prompt user for port if not provided
            try:
                self.port = int(input("Enter database server port (default: 9100): ") or "9100")
            except (ValueError, KeyboardInterrupt):
                self.port = 9100
                print(f"Using default port: {self.port}")
        
        server_addr = f"localhost:{self.port}"
        print(f"Connecting to database server at {server_addr}...")
        
        self.client = StockDBClient(server_addr)
        # Note: StockDBClient doesn't have _ensure_tables_exist method, so we skip that
    
    async def _get_data_dates(self) -> Tuple[str, str]:
        """Determine start and end dates for data retrieval."""
        if self.end_date:
            end_date = self.end_date
        else:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if self.start_date:
            start_date = self.start_date
        else:
            # Calculate start date from lookback days
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=self.lookback_days)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        return start_date, end_date
    
    async def _fetch_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch daily and hourly data for the stock."""
        start_date, end_date = await self._get_data_dates()
        
        self.logger.info(f"Fetching data for {self.stock_symbol} from {start_date} to {end_date}")
        
        # Fetch daily data
        daily_df = await self.client.get_stock_data(
            self.stock_symbol,
            start_date=start_date,
            end_date=end_date,
            interval='daily'
        )
        
        if daily_df.empty:
            raise ValueError(f"No daily data found for {self.stock_symbol}")
        
        # Fetch hourly data for more granular streak detection
        hourly_df = await self.client.get_stock_data(
            self.stock_symbol,
            start_date=start_date,
            end_date=end_date,
            interval='hourly'
        )
        
        self.logger.info(f"Retrieved {len(daily_df)} daily records and {len(hourly_df)} hourly records")
        
        return daily_df, hourly_df
    
    def _calculate_investment_amount(self, streak_length: int) -> float:
        """Calculate investment amount based on streak length."""
        if streak_length <= 0:
            return self.base_investment_amount
        
        multiplier = self.streak_multiplier ** (streak_length - 1)
        amount = self.base_investment_amount * multiplier
        
        # Ensure we don't exceed available cash or max investment
        max_amount = min(self.current_cash, self.max_total_investment - self.total_invested)
        return min(amount, max_amount)
    
    def _detect_streak_change(self, current_return: float) -> Tuple[str, int]:
        """
        Detect if there's a streak change based on current return.
        
        Returns:
            Tuple of (streak_type, streak_length)
        """
        if current_return < -self.down_threshold_percent:
            # Down streak
            if self.current_streak_type == 'down':
                self.current_streak_length += 1
            else:
                # Streak direction changed
                self.current_streak_type = 'down'
                self.current_streak_length = 1
        elif current_return > self.up_threshold_percent:
            # Up streak
            if self.current_streak_type == 'up':
                self.current_streak_length += 1
            else:
                # Streak direction changed
                self.current_streak_type = 'up'
                self.current_streak_length = 1
        else:
            # No significant change - streak continues but doesn't increment
            pass
        
        return self.current_streak_type, self.current_streak_length
    
    def _execute_buy(self, date: datetime, price: float, streak_length: int) -> None:
        """Execute a buy transaction."""
        investment_amount = self._calculate_investment_amount(streak_length)
        
        if investment_amount <= 0 or self.current_cash < investment_amount:
            return
        
        shares = investment_amount / price
        self.position.add_shares(shares, price, investment_amount)
        self.current_cash -= investment_amount
        self.total_invested += investment_amount
        
        # Record transaction
        transaction = Transaction(
            date=date,
            action=Action.BUY,
            shares=shares,
            price=price,
            amount=investment_amount,
            streak_length=streak_length,
            streak_type='down',
            running_shares=self.position.shares,
            running_cash=self.current_cash
        )
        
        self.transactions.append(transaction)
        self.result.buy_transactions += 1
        
        if self.debug:
            self.logger.info(f"BUY: {shares:.2f} shares at ${price:.2f} "
                           f"(streak: {streak_length}, amount: ${investment_amount:.2f})")
    
    def _execute_sell(self, date: datetime, price: float, streak_length: int) -> None:
        """Execute a sell transaction."""
        if self.position.shares <= 0:
            return
        
        # Calculate sell amount based on streak
        base_sell_amount = self._calculate_investment_amount(streak_length)
        shares_to_sell = min(base_sell_amount / price, self.position.shares)
        
        if shares_to_sell <= 0:
            return
        
        proceeds = self.position.remove_shares(shares_to_sell, price)
        self.current_cash += proceeds
        
        # Record transaction
        transaction = Transaction(
            date=date,
            action=Action.SELL,
            shares=shares_to_sell,
            price=price,
            amount=proceeds,
            streak_length=streak_length,
            streak_type='up',
            running_shares=self.position.shares,
            running_cash=self.current_cash
        )
        
        self.transactions.append(transaction)
        self.result.sell_transactions += 1
        
        if self.debug:
            self.logger.info(f"SELL: {shares_to_sell:.2f} shares at ${price:.2f} "
                           f"(streak: {streak_length}, proceeds: ${proceeds:.2f})")
    
    async def _run_daily_backtest(self, daily_df: pd.DataFrame) -> None:
        """Run the main backtest using daily data."""
        self.logger.info("Starting daily backtest...")
        
        # Sort data by date
        daily_df = daily_df.sort_index()
        
        # Calculate daily returns
        daily_df['return'] = daily_df['close'].pct_change() * 100
        
        # Initialize tracking variables
        initial_price = daily_df['close'].iloc[0]
        price_history = [initial_price]
        
        for date, row in daily_df.iterrows():
            if pd.isna(row['return']):
                continue
            
            current_return = row['return']
            current_price = row['close']
            
            # Detect streak change
            streak_type, streak_length = self._detect_streak_change(current_return)
            
            # Execute trades based on streak
            if streak_type == 'down' and streak_length >= 1:
                self._execute_buy(date, current_price, streak_length)
            elif streak_type == 'up' and streak_length >= 1:
                self._execute_sell(date, current_price, streak_length)
            
            # Track price history for drawdown calculation
            price_history.append(current_price)
        
        # Final calculations
        final_price = daily_df['close'].iloc[-1]
        self.result.final_shares = self.position.shares
        self.result.final_cash = self.current_cash
        self.result.total_invested = self.total_invested
        self.result.total_current_value = (self.position.shares * final_price) + self.current_cash
        self.result.net_profit_loss = self.result.total_current_value - self.max_total_investment
        self.result.net_profit_loss_pct = (self.result.net_profit_loss / self.max_total_investment) * 100
        
        # Calculate performance metrics
        self.result.calculate_metrics(initial_price, final_price, price_history)
        
        # Calculate streak statistics
        up_streaks, down_streaks = self.streak_analyzer.compute_streaks(daily_df)
        self.result.total_up_streaks = len(up_streaks)
        self.result.total_down_streaks = len(down_streaks)
        
        if up_streaks:
            self.result.avg_up_streak_length = sum(s['length'] for s in up_streaks) / len(up_streaks)
        if down_streaks:
            self.result.avg_down_streak_length = sum(s['length'] for s in down_streaks) / len(down_streaks)
        
        # Copy transactions to result
        self.result.transactions = self.transactions.copy()
    
    def _print_summary(self) -> None:
        """Print summary of backtest results."""
        print("\n" + "="*80)
        print(f"STREAK BACKTEST RESULTS FOR {self.stock_symbol}")
        print("="*80)
        
        print(f"\nSUMMARY STATISTICS:")
        print(f"  Total Invested:                    ${self.result.total_invested:>12,.2f}")
        print(f"  Total Current Value:               ${self.result.total_current_value:>12,.2f}")
        print(f"  Net Profit/Loss:                   ${self.result.net_profit_loss:>12,.2f}")
        print(f"  Net Profit/Loss %:                 {self.result.net_profit_loss_pct:>12.2f}%")
        print(f"  Buy Transactions:                  {self.result.buy_transactions:>12}")
        print(f"  Sell Transactions:                 {self.result.sell_transactions:>12}")
        print(f"  Final Shares:                      {self.result.final_shares:>12.2f}")
        print(f"  Final Cash:                        ${self.result.final_cash:>12,.2f}")
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"  Max Drawdown:                      {self.result.max_drawdown_pct:>12.2f}%")
        print(f"  Sharpe Ratio:                      {self.result.sharpe_ratio:>12.2f}")
        print(f"  Buy & Hold Return:                 ${self.result.buy_hold_return:>12,.2f}")
        print(f"  Buy & Hold Return %:               {self.result.buy_hold_return_pct:>12.2f}%")
        
        print(f"\nSTREAK STATISTICS:")
        print(f"  Total Up Streaks:                  {self.result.total_up_streaks:>12}")
        print(f"  Total Down Streaks:                {self.result.total_down_streaks:>12}")
        print(f"  Average Up Streak Length:          {self.result.avg_up_streak_length:>12.2f}")
        print(f"  Average Down Streak Length:        {self.result.avg_down_streak_length:>12.2f}")
        
        print(f"\nSTRATEGY PARAMETERS:")
        print(f"  Down Threshold:                    {self.down_threshold_percent:>12.2f}%")
        print(f"  Up Threshold:                      {self.up_threshold_percent:>12.2f}%")
        print(f"  Base Investment:                   ${self.base_investment_amount:>12,.2f}")
        print(f"  Max Total Investment:              ${self.max_total_investment:>12,.2f}")
        print(f"  Streak Multiplier:                 {self.streak_multiplier:>12.2f}")
    
    def _print_transaction_log(self) -> None:
        """Print detailed transaction log if debug is enabled."""
        if not self.debug or not self.transactions:
            return
        
        print(f"\nTRANSACTION LOG:")
        print("-" * 120)
        print(f"{'Date':<20} {'Action':<6} {'Shares':<10} {'Price':<10} {'Amount':<12} {'Streak':<8} {'Type':<6} {'Shares':<10} {'Cash':<14}")
        print("-" * 120)
        
        for transaction in self.transactions:
            print(transaction)
        
        print("-" * 120)
    
    async def run_backtest(self) -> BacktestResult:
        """Run the complete backtest."""
        try:
            # Initialize client
            await self._initialize_client()
            
            # Fetch data
            daily_df, hourly_df = await self._fetch_data()
            
            # Run backtest
            await self._run_daily_backtest(daily_df)
            
            # Print results
            self._print_summary()
            self._print_transaction_log()
            
            return self.result
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}", exc_info=True)
            raise
        finally:
            if self.client:
                await self.client.close_session()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Streak-based trading strategy backtester',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic backtest with default parameters
  python scripts/streak_backtester.py AAPL
  
  # Custom backtest with specific dates and parameters
  python scripts/streak_backtester.py AAPL --start-date 2024-01-01 --end-date 2024-12-31 --down-threshold 1.5 --up-threshold 1.5
  
  # Backtest with aggressive streak multiplier
  python scripts/streak_backtester.py AAPL --streak-multiplier 2.0 --base-investment 2000 --max-investment 20000
  
  # Debug mode with transaction details
  python scripts/streak_backtester.py AAPL --debug
        """
    )
    
    parser.add_argument("stock_symbol", help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--start-date", help="Start date for backtesting (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for backtesting (YYYY-MM-DD, default: today)")
    parser.add_argument("--lookback-days", type=int, default=90, 
                       help="Days to go back from today if start_date not provided (default: 90)")
    parser.add_argument("--down-threshold", type=float, default=2.0, metavar="PERCENT",
                       help="Minimum percentage drop to trigger buy (default: 2.0)")
    parser.add_argument("--up-threshold", type=float, default=2.0, metavar="PERCENT",
                       help="Minimum percentage gain to trigger sell (default: 2.0)")
    parser.add_argument("--base-investment", type=float, default=1000.0, metavar="DOLLARS",
                       help="Base dollar amount to invest (default: 1000)")
    parser.add_argument("--max-investment", type=float, default=10000.0, metavar="DOLLARS",
                       help="Maximum total investment allowed (default: 10000)")
    parser.add_argument("--streak-multiplier", type=float, default=1.5, metavar="MULTIPLIER",
                       help="Multiplier for each consecutive streak day (default: 1.5)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output and transaction log")
    parser.add_argument("--port", type=int, default=9100, help="Port for database server (default: 9100)")
    
    return parser.parse_args()


async def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Create and run backtester
        backtester = StreakBacktester(
            stock_symbol=args.stock_symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            lookback_days=args.lookback_days,
            down_threshold_percent=args.down_threshold,
            up_threshold_percent=args.up_threshold,
            base_investment_amount=args.base_investment,
            max_total_investment=args.max_investment,
            streak_multiplier=args.streak_multiplier,
            debug=args.debug,
            port=args.port
        )
        
        result = await backtester.run_backtest()
        
        print(f"\nBacktest completed successfully!")
        return result
        
    except KeyboardInterrupt:
        print("\nBacktest interrupted by user.")
        return None
    except Exception as e:
        print(f"\nBacktest failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(main())
