"""
Backtesting Engine

Event-driven backtesting engine that executes strategies and tracks performance.
Handles data processing, signal generation, trade execution, and performance calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from .config import BacktestConfig, ProcessingConfig
from .portfolio import Portfolio
from .metrics import MetricsCalculator, PerformanceMetrics
from ..strategies.base import AbstractStrategy, Signal, Direction, SignalResult, PositionSizeResult, RiskParams
from ..data.fetcher import DataFetcher


class BacktestEngine:
    """
    Event-driven backtesting engine.
    
    Features:
    - Event-driven architecture
    - Realistic trade execution with slippage and commissions
    - Portfolio state tracking
    - Performance metrics calculation
    - Multi-strategy support
    - Parallel processing capabilities
    """
    
    def __init__(
        self, 
        config: BacktestConfig,
        processing_config: Optional[ProcessingConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.processing_config = processing_config or ProcessingConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        # DataFetcher defaults to database (StockDBBase) implementation
        self.data_fetcher = DataFetcher(logger=self.logger, default_db_type="sqlite")
        self.metrics_calculator = MetricsCalculator(config.risk_free_rate, self.logger)
        
        # State
        self.portfolio: Optional[Portfolio] = None
        self.strategies: Dict[str, AbstractStrategy] = {}
        self.benchmark_data: Optional[pd.DataFrame] = None
        
    def add_strategy(self, strategy: AbstractStrategy) -> None:
        """Add a strategy to the backtesting engine."""
        if not strategy.initialized:
            raise ValueError(f"Strategy {strategy.get_strategy_name()} not initialized")
        
        self.strategies[strategy.get_strategy_name()] = strategy
        self.logger.info(f"Added strategy: {strategy.get_strategy_name()}")
    
    def remove_strategy(self, strategy_name: str) -> None:
        """Remove a strategy from the backtesting engine."""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            self.logger.info(f"Removed strategy: {strategy_name}")
    
    async def run_backtest(
        self,
        ticker: str,
        data_source: str = "database",
        db_config: Optional[str] = None,
        benchmark_ticker: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run backtest for a single ticker with separate training and prediction phases.
        
        Args:
            ticker: Stock ticker symbol
            data_source: Data source ("database", "yfinance", "alpaca")
            db_config: Database configuration
            benchmark_ticker: Benchmark ticker for comparison
            
        Returns:
            Dictionary with backtest results
        """
        if not self.strategies:
            raise ValueError("No strategies added to backtesting engine")
        
        self.logger.debug(f"Starting backtest for {ticker}")
        
        # Store the ticker being backtested
        self.current_ticker = ticker
        
        # Initialize portfolio
        self.portfolio = Portfolio(self.config, self.logger)
        
        # Get training and prediction date ranges
        training_start = self.config.training_start_date
        training_end = self.config.training_end_date
        prediction_start = self.config.start_date
        prediction_end = self.config.end_date
        
        self.logger.info(f"Training period: {training_start.strftime('%Y-%m-%d')} to {training_end.strftime('%Y-%m-%d')}")
        self.logger.info(f"Prediction period: {prediction_start.strftime('%Y-%m-%d')} to {prediction_end.strftime('%Y-%m-%d')}")
        
        # Phase 1: Train strategies on training data
        self.logger.info("Phase 1: Training strategies on training data")
        training_data = await self._fetch_data_for_period(
            ticker, data_source, db_config, training_start, training_end
        )
        
        if not training_data.empty:
            for strategy_name, strategy in self.strategies.items():
                try:
                    self.logger.info(f"Training strategy: {strategy_name}")
                    strategy.train(training_data)
                except Exception as e:
                    self.logger.warning(f"Failed to train strategy {strategy_name}: {str(e)}")
        else:
            raise ValueError(f"No training data available for {ticker} in the specified training period")
        
        # Phase 2: Fetch prediction data and run backtest
        self.logger.info("Phase 2: Running backtest on prediction data")
        data = await self._fetch_data_for_period(
            ticker, data_source, db_config, prediction_start, prediction_end
        )
        
        if data.empty:
            raise ValueError(f"No prediction data available for {ticker}")
        
        # Fetch benchmark data for prediction period
        benchmark_ticker = benchmark_ticker or self.config.benchmark_ticker
        if benchmark_ticker:
            self.benchmark_data = await self._fetch_data_for_period(
                benchmark_ticker, data_source, db_config, prediction_start, prediction_end
            )
        
        # Run backtest on prediction data
        await self._execute_backtest(data, ticker)
        
        # Close all open positions at end of backtest period
        await self._close_all_positions(data, ticker, prediction_end)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Prepare results
        results = {
            'ticker': ticker,
            'benchmark_ticker': benchmark_ticker,
            'config': self.config.to_dict(),
            'metrics': metrics.to_dict(),
            'portfolio_summary': self.portfolio.get_portfolio_summary(),
            'equity_curve': self.portfolio.get_equity_curve().to_dict(),
            'trades': [self._trade_to_dict(trade) for trade in self.portfolio.trades],
            'strategies': list(self.strategies.keys()),
            'training_period': {
                'start': training_start.isoformat(),
                'end': training_end.isoformat()
            },
            'prediction_period': {
                'start': prediction_start.isoformat(),
                'end': prediction_end.isoformat()
            }
        }
        
        self.logger.info(f"Backtest completed for {ticker}")
        return results
    
    async def run_multi_stock_backtest(
        self,
        tickers: List[str],
        data_source: str = "database",
        db_config: Optional[str] = None,
        benchmark_ticker: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run backtest for multiple tickers in parallel.
        
        Args:
            tickers: List of stock ticker symbols
            data_source: Data source
            db_config: Database configuration
            benchmark_ticker: Benchmark ticker for comparison
            
        Returns:
            Dictionary with aggregated backtest results
        """
        self.logger.info(f"Starting multi-stock backtest for {len(tickers)} tickers")
        
        results = {}
        failed_tickers = []
        
        if self.processing_config.use_multiprocessing:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.processing_config.max_workers) as executor:
                futures = {
                    executor.submit(self._run_single_backtest, ticker, data_source, db_config, benchmark_ticker): ticker
                    for ticker in tickers
                }
                
                for future in as_completed(futures, timeout=self.processing_config.timeout):
                    ticker = futures[future]
                    try:
                        result = future.result()
                        results[ticker] = result
                        self.logger.info(f"Completed backtest for {ticker}")
                    except Exception as e:
                        self.logger.error(f"Backtest failed for {ticker}: {str(e)}")
                        failed_tickers.append(ticker)
        else:
            # Sequential execution
            for ticker in tickers:
                try:
                    result = await self.run_backtest(ticker, data_source, db_config, benchmark_ticker)
                    results[ticker] = result
                except Exception as e:
                    self.logger.error(f"Backtest failed for {ticker}: {str(e)}")
                    failed_tickers.append(ticker)
        
        # Aggregate results
        aggregated_results = self._aggregate_results(results)
        aggregated_results['failed_tickers'] = failed_tickers
        
        self.logger.info(f"Multi-stock backtest completed. Success: {len(results)}, Failed: {len(failed_tickers)}")
        return aggregated_results
    
    async def _fetch_data_for_period(
        self,
        ticker: str,
        data_source: str,
        db_config: Optional[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch data for a ticker for a specific date range.
        
        Args:
            ticker: Stock ticker symbol
            data_source: Data source to use
            db_config: Database configuration
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Get required lookback from strategies
            required_lookback_days = 0
            for strategy in self.strategies.values():
                if hasattr(strategy, 'get_required_lookback'):
                    lookback = strategy.get_required_lookback()
                    required_lookback_days = max(required_lookback_days, lookback)
            
            # Extend start_date to include lookback period
            actual_start_date = start_date
            if required_lookback_days > 0:
                # Add extra days for lookback (use business days approximation: 1.4x calendar days)
                calendar_days = int(required_lookback_days * 1.4)
                actual_start_date = start_date - timedelta(days=calendar_days)
                self.logger.debug(f"Strategy requires {required_lookback_days} days lookback, fetching from {actual_start_date}")
            
            self.logger.debug(f"Fetching data for {ticker} from {data_source}")
            self.logger.debug(f"Date range: {actual_start_date} to {end_date}")
            self.logger.debug(f"DB config: {db_config[:50] if db_config else 'None'}..." if db_config else "DB config: None")
            
            if data_source == "database":
                data = await self.data_fetcher.fetch_from_database(
                    ticker, 
                    actual_start_date, 
                    end_date,
                    db_config
                )
            elif data_source == "yfinance":
                data = await self.data_fetcher.fetch_from_yfinance(
                    ticker,
                    actual_start_date,
                    end_date
                )
            elif data_source == "alpaca":
                data = await self.data_fetcher.fetch_from_alpaca(
                    ticker,
                    actual_start_date,
                    end_date
                )
            else:
                raise ValueError(f"Unsupported data source: {data_source}")
            
            self.logger.debug(f"Fetched data for {ticker}: {len(data)} rows")
            if not data.empty:
                self.logger.debug(f"Data columns: {list(data.columns)}")
                self.logger.debug(f"Date range in data: {data.index.min()} to {data.index.max()}")
                self.logger.debug(f"Sample data:\n{data.head(5)}")
            else:
                self.logger.warning(f"No data fetched for {ticker}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {ticker}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    async def _fetch_data(
        self, 
        ticker: str, 
        data_source: str, 
        db_config: Optional[str]
    ) -> pd.DataFrame:
        """
        Fetch data for a ticker using config dates.
        
        This is a wrapper around _fetch_data_for_period for backward compatibility.
        """
        return await self._fetch_data_for_period(
            ticker, 
            data_source, 
            db_config, 
            self.config.start_date, 
            self.config.end_date
        )
    
    async def _execute_backtest(self, data: pd.DataFrame, ticker: str) -> None:
        """Execute the backtesting logic."""
        if self.portfolio is None:
            raise ValueError("Portfolio not initialized")
        
        self.logger.debug(f"Starting backtest execution with {len(data)} rows of data")
        if data.empty:
            self.logger.warning("No data to process in backtest")
            return
        
        self.logger.debug(f"Processing backtest for date range: {data.index.min()} to {data.index.max()}")
        
        processed_count = 0
        # Process data day by day
        for date, row in data.iterrows():
            processed_count += 1
            if processed_count <= 5:
                self.logger.debug(f"Processing row {processed_count}/{len(data)} for date {date}, price={row['close']}")
            elif processed_count % 10 == 0:
                self.logger.debug(f"Processing row {processed_count}/{len(data)} for date {date}")
            current_price = row['close']
            current_prices = {ticker: current_price}
            
            # Update portfolio prices
            self.portfolio.update_position_prices(current_prices, date)
            
            # Check stop loss and take profit
            self.portfolio.check_stop_loss_take_profit(date)
            
            # Generate signals for each strategy
            for strategy_name, strategy in self.strategies.items():
                try:
                    # Get historical data up to current date
                    historical_data = data.loc[:date]
                    
                    # Debug historical data
                    if processed_count <= 5:
                        self.logger.debug(f"Strategy {strategy_name} on {date}: historical_data length={len(historical_data)}")
                    
                    # Generate signal
                    signal = strategy.generate_signal(historical_data)
                    
                    # Debug signal generation
                    if processed_count <= 5 or signal.signal != 'HOLD':
                        self.logger.debug(f"Strategy {strategy_name} on {date}: signal={signal.signal}, confidence={signal.confidence}")
                    
                    # Only process non-HOLD signals
                    if signal.signal != Signal.HOLD:
                        # Calculate position size
                        position_size = strategy.calculate_position_size(
                            self.portfolio.cash,
                            signal,
                            self.portfolio.risk_params,
                            current_price
                        )
                        
                        # Debug position sizing
                        if processed_count <= 5:
                            self.logger.debug(f"Strategy {strategy_name} on {date}: position_size={position_size.size}, dollar_amount={position_size.dollar_amount}, cash={self.portfolio.cash}")
                        
                        # Execute trade
                        trade_executed = self.portfolio.execute_trade(
                            ticker=ticker,
                            signal=signal,
                            position_size=position_size,
                            current_price=current_price,
                            current_date=date,
                            strategy_name=strategy_name
                        )
                        
                        if trade_executed and processed_count <= 5:
                            self.logger.info(f"Trade executed by {strategy_name} on {date}")
                        elif not trade_executed and processed_count <= 5:
                            self.logger.warning(f"Trade NOT executed by {strategy_name} on {date} even though signal={signal.signal}")
                    
                except Exception as e:
                    self.logger.warning(f"Error processing strategy {strategy_name} on {date}: {str(e)}")
                    import traceback
                    self.logger.warning(traceback.format_exc())
                    continue
        
        self.logger.info(f"Backtest execution completed. Processed {processed_count} rows. Total trades: {len(self.portfolio.trades)}")
    
    async def _close_all_positions(self, data: pd.DataFrame, main_ticker: str, end_date: datetime) -> None:
        """Close all open positions at the end of the backtest period."""
        if not self.portfolio.positions:
            return
        
        self.logger.info(f"Closing {len(self.portfolio.positions)} open positions at end of backtest")
        
        # Get the final price for each ticker
        final_prices = {}
        for ticker in self.portfolio.positions.keys():
            # Use the last close price from the data
            if not data.empty:
                final_prices[ticker] = data['close'].iloc[-1]
        
        # Close each position
        for ticker_symbol, position in list(self.portfolio.positions.items()):
            if ticker_symbol not in final_prices:
                self.logger.warning(f"No final price available for {ticker_symbol}, using current position price")
                final_price = position.current_price
            else:
                final_price = final_prices[ticker_symbol]
            
            # Create a sell signal
            sell_signal = SignalResult(
                signal=Signal.SELL,
                direction=Direction.LONG,
                confidence=100.0,
                expected_movement=0.0,
                expected_movement_pct=0.0,
                expected_price=final_price,
                probability_distribution={},
                reasoning="Closing position at end of backtest",
                metadata={}
            )
            
            # Create position size to sell entire position
            sell_size = PositionSizeResult(
                size=position.size,
                size_pct=100.0,
                dollar_amount=position.size * final_price,
                risk_amount=0.0,
                stop_loss=None,
                take_profit=None
            )
            
            # Execute sell
            commission = self.config.commission_per_trade
            slippage_cost = position.size * final_price * self.config.slippage_pct
            actual_price = final_price * (1 - self.config.slippage_pct)
            
            self.logger.info(f"Closing position: {ticker_symbol}, {position.size:.2f} shares at ${actual_price:.2f}")
            self.portfolio._execute_sell(
                ticker=ticker_symbol,
                signal=sell_signal,
                position_size=sell_size,
                price=actual_price,
                date=end_date,
                strategy_name="BacktestEnd",
                commission=commission,
                slippage_cost=slippage_cost
            )
        
        self.logger.info(f"Closed all positions. Total trades: {len(self.portfolio.trades)}")
    
    def _calculate_metrics(self) -> PerformanceMetrics:
        """Calculate performance metrics."""
        if self.portfolio is None:
            raise ValueError("Portfolio not initialized")
        
        benchmark_returns = None
        if self.benchmark_data is not None and not self.benchmark_data.empty:
            benchmark_returns = self.benchmark_data['close'].pct_change().dropna()
        
        return self.metrics_calculator.calculate_metrics(
            self.portfolio,
            benchmark_returns,
            self.config.start_date,
            self.config.end_date
        )
    
    def _aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from multiple tickers."""
        if not results:
            return {}
        
        # Calculate aggregate metrics
        total_returns = [r['metrics']['total_return'] for r in results.values()]
        sharpe_ratios = [r['metrics']['sharpe_ratio'] for r in results.values()]
        max_drawdowns = [r['metrics']['max_drawdown'] for r in results.values()]
        
        aggregated_metrics = {
            'avg_total_return': np.mean(total_returns),
            'median_total_return': np.median(total_returns),
            'std_total_return': np.std(total_returns),
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'median_sharpe_ratio': np.median(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'median_max_drawdown': np.median(max_drawdowns),
            'num_successful_backtests': len(results),
            'win_rate': len([r for r in total_returns if r > 0]) / len(total_returns) * 100
        }
        
        return {
            'aggregated_metrics': aggregated_metrics,
            'individual_results': results,
            'config': self.config.to_dict()
        }
    
    def _trade_to_dict(self, trade) -> Dict[str, Any]:
        """Convert trade object to dictionary."""
        return {
            'ticker': trade.ticker,
            'entry_date': trade.entry_date.isoformat(),
            'exit_date': trade.exit_date.isoformat(),
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'size': trade.size,
            'direction': trade.direction.value,
            'pnl': trade.pnl,
            'pnl_pct': trade.pnl_pct,
            'commission': trade.commission,
            'slippage': trade.slippage,
            'strategy_name': trade.strategy_name,
            'signal_reasoning': trade.signal_reasoning,
            'duration_days': trade.duration_days
        }
    
    def _run_single_backtest(
        self, 
        ticker: str, 
        data_source: str, 
        db_config: Optional[str], 
        benchmark_ticker: Optional[str]
    ) -> Dict[str, Any]:
        """Run backtest for a single ticker (for parallel execution)."""
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(
                self.run_backtest(ticker, data_source, db_config, benchmark_ticker)
            )
        finally:
            loop.close()
    
    def get_strategy_performance(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a specific strategy."""
        if strategy_name not in self.strategies:
            return None
        
        # Filter trades by strategy
        strategy_trades = [t for t in self.portfolio.trades if t.strategy_name == strategy_name]
        
        if not strategy_trades:
            return None
        
        # Calculate strategy-specific metrics
        total_pnl = sum(t.pnl for t in strategy_trades)
        winning_trades = [t for t in strategy_trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(strategy_trades) * 100
        
        return {
            'strategy_name': strategy_name,
            'num_trades': len(strategy_trades),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_trade_pnl': total_pnl / len(strategy_trades),
            'trades': [self._trade_to_dict(trade) for trade in strategy_trades]
        }
    
    def reset(self) -> None:
        """Reset the backtesting engine state."""
        self.portfolio = None
        self.benchmark_data = None
        
        # Reset strategies if they have reset methods
        for strategy in self.strategies.values():
            if hasattr(strategy, 'reset'):
                strategy.reset()
