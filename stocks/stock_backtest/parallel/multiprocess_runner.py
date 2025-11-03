"""
Parallel Processing System

Handles parallel execution of backtests across multiple stocks and strategies
with configurable worker pools, progress tracking, and error handling.
"""

import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Callable, Union
import logging
import time
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
import queue
import threading
from tqdm import tqdm

from ..backtesting.engine import BacktestEngine
from ..backtesting.config import BacktestConfig, ProcessingConfig
from ..strategies.base import AbstractStrategy
from ..data.fetcher import DataFetcher


@dataclass
class ProcessingResult:
    """Result from parallel processing."""
    ticker: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float = 0.0


class MultiProcessRunner:
    """
    Parallel processing runner for multi-stock backtesting.
    
    Features:
    - Configurable worker pools (processes/threads)
    - Progress tracking with tqdm
    - Error handling and recovery
    - Result aggregation
    - Memory management
    """
    
    def __init__(
        self,
        processing_config: ProcessingConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.processing_config = processing_config
        self.logger = logger or logging.getLogger(__name__)
        
        # Progress tracking
        self.progress_bar: Optional[tqdm] = None
        self.completed_count = 0
        self.total_count = 0
        
    async def run_parallel_backtests(
        self,
        tickers: List[str],
        strategies: List[AbstractStrategy],
        backtest_config: BacktestConfig,
        data_source: str = "database",
        db_config: Optional[str] = None,
        benchmark_ticker: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run backtests for multiple tickers in parallel.
        
        Args:
            tickers: List of ticker symbols
            strategies: List of strategies to test
            backtest_config: Backtesting configuration
            data_source: Data source to use
            db_config: Database configuration
            benchmark_ticker: Benchmark ticker for comparison
            
        Returns:
            Aggregated results dictionary
        """
        self.logger.info(f"Starting parallel backtests for {len(tickers)} tickers with {len(strategies)} strategies")
        
        self.total_count = len(tickers)
        self.completed_count = 0
        
        # Initialize progress bar
        self.progress_bar = tqdm(total=len(tickers), desc="Processing tickers", unit="ticker")
        
        results = {}
        failed_tickers = []
        
        try:
            if self.processing_config.use_multiprocessing:
                # Use process pool for CPU-intensive tasks
                results = await self._run_with_process_pool(
                    tickers, strategies, backtest_config, data_source, db_config, benchmark_ticker
                )
            else:
                # Use thread pool for I/O-intensive tasks
                results = await self._run_with_thread_pool(
                    tickers, strategies, backtest_config, data_source, db_config, benchmark_ticker
                )
            
            # Separate successful and failed results
            successful_results = {k: v for k, v in results.items() if v.success}
            failed_tickers = [k for k, v in results.items() if not v.success]
            
            # Aggregate successful results
            aggregated_results = self._aggregate_results(successful_results)
            aggregated_results['failed_tickers'] = failed_tickers
            
            self.logger.info(f"Parallel backtests completed. Success: {len(successful_results)}, Failed: {len(failed_tickers)}")
            
            return aggregated_results
            
        finally:
            if self.progress_bar:
                self.progress_bar.close()
    
    async def _run_with_process_pool(
        self,
        tickers: List[str],
        strategies: List[AbstractStrategy],
        backtest_config: BacktestConfig,
        data_source: str,
        db_config: Optional[str],
        benchmark_ticker: Optional[str]
    ) -> Dict[str, ProcessingResult]:
        """Run backtests using process pool."""
        
        # Prepare arguments for each ticker
        args_list = []
        for ticker in tickers:
            args = (
                ticker,
                strategies,
                backtest_config,
                data_source,
                db_config,
                benchmark_ticker
            )
            args_list.append(args)
        
        # Use ThreadPoolExecutor for async compatibility
        with ThreadPoolExecutor(max_workers=self.processing_config.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self._run_single_backtest_process, *args): args[0]
                for args in args_list
            }
            
            results = {}
            
            # Process completed futures
            for future in as_completed(futures, timeout=self.processing_config.timeout):
                ticker = futures[future]
                try:
                    result = future.result()
                    results[ticker] = result
                    self._update_progress(ticker, result.success)
                    
                except Exception as e:
                    error_result = ProcessingResult(
                        ticker=ticker,
                        success=False,
                        error=str(e),
                        processing_time=0.0
                    )
                    results[ticker] = error_result
                    self._update_progress(ticker, False)
                    self.logger.error(f"Process pool error for {ticker}: {str(e)}")
        
        return results
    
    async def _run_with_thread_pool(
        self,
        tickers: List[str],
        strategies: List[AbstractStrategy],
        backtest_config: BacktestConfig,
        data_source: str,
        db_config: Optional[str],
        benchmark_ticker: Optional[str]
    ) -> Dict[str, ProcessingResult]:
        """Run backtests using thread pool."""
        
        results = {}
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.processing_config.max_workers)
        
        # Create tasks
        tasks = []
        for ticker in tickers:
            task = self._run_single_backtest_async(
                ticker, strategies, backtest_config, data_source, db_config, benchmark_ticker, semaphore
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(task_results):
            ticker = tickers[i]
            if isinstance(result, Exception):
                error_result = ProcessingResult(
                    ticker=ticker,
                    success=False,
                    error=str(result),
                    processing_time=0.0
                )
                results[ticker] = error_result
                self._update_progress(ticker, False)
            else:
                results[ticker] = result
                self._update_progress(ticker, result.success)
        
        return results
    
    async def _run_single_backtest_async(
        self,
        ticker: str,
        strategies: List[AbstractStrategy],
        backtest_config: BacktestConfig,
        data_source: str,
        db_config: Optional[str],
        benchmark_ticker: Optional[str],
        semaphore: asyncio.Semaphore
    ) -> ProcessingResult:
        """Run single backtest asynchronously."""
        
        async with semaphore:
            start_time = time.time()
            
            try:
                # Create backtest engine
                engine = BacktestEngine(backtest_config, self.processing_config, self.logger)
                
                # Add strategies
                for strategy in strategies:
                    engine.add_strategy(strategy)
                
                # Run backtest
                result = await engine.run_backtest(ticker, data_source, db_config, benchmark_ticker)
                
                processing_time = time.time() - start_time
                
                return ProcessingResult(
                    ticker=ticker,
                    success=True,
                    result=result,
                    processing_time=processing_time
                )
                
            except Exception as e:
                processing_time = time.time() - start_time
                self.logger.error(f"Backtest failed for {ticker}: {str(e)}")
                
                return ProcessingResult(
                    ticker=ticker,
                    success=False,
                    error=str(e),
                    processing_time=processing_time
                )
    
    def _run_single_backtest_process(
        self,
        ticker: str,
        strategies: List[AbstractStrategy],
        backtest_config: BacktestConfig,
        data_source: str,
        db_config: Optional[str],
        benchmark_ticker: Optional[str]
    ) -> ProcessingResult:
        """Run single backtest in separate process."""
        
        start_time = time.time()
        
        try:
            # Create new event loop for this process
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Create backtest engine
                engine = BacktestEngine(backtest_config, self.processing_config, self.logger)
                
                # Add strategies
                for strategy in strategies:
                    engine.add_strategy(strategy)
                
                # Run backtest
                result = loop.run_until_complete(
                    engine.run_backtest(ticker, data_source, db_config, benchmark_ticker)
                )
                
                processing_time = time.time() - start_time
                
                return ProcessingResult(
                    ticker=ticker,
                    success=True,
                    result=result,
                    processing_time=processing_time
                )
                
            finally:
                loop.close()
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Process backtest failed for {ticker}: {str(e)}")
            
            return ProcessingResult(
                ticker=ticker,
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    def _update_progress(self, ticker: str, success: bool) -> None:
        """Update progress bar."""
        self.completed_count += 1
        
        if self.progress_bar:
            status = "✓" if success else "✗"
            self.progress_bar.set_postfix({
                'ticker': ticker,
                'status': status,
                'completed': f"{self.completed_count}/{self.total_count}"
            })
            self.progress_bar.update(1)
    
    def _aggregate_results(self, results: Dict[str, ProcessingResult]) -> Dict[str, Any]:
        """Aggregate results from multiple tickers."""
        
        if not results:
            return {}
        
        # Extract successful results
        successful_results = {ticker: result.result for ticker, result in results.items() if result.success}
        
        if not successful_results:
            return {'error': 'No successful backtests'}
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(successful_results)
        
        # Calculate portfolio-level metrics
        portfolio_metrics = self._calculate_portfolio_metrics(successful_results)
        
        # Calculate performance statistics
        performance_stats = self._calculate_performance_statistics(successful_results)
        
        return {
            'aggregate_metrics': aggregate_metrics,
            'portfolio_metrics': portfolio_metrics,
            'performance_statistics': performance_stats,
            'individual_results': successful_results,
            'processing_summary': {
                'total_tickers': len(results),
                'successful_tickers': len(successful_results),
                'failed_tickers': len(results) - len(successful_results),
                'avg_processing_time': np.mean([r.processing_time for r in results.values()]),
                'total_processing_time': sum([r.processing_time for r in results.values()])
            }
        }
    
    def _calculate_aggregate_metrics(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate metrics across all tickers."""
        
        metrics_to_aggregate = [
            'total_return', 'annualized_return', 'sharpe_ratio', 
            'max_drawdown', 'win_rate', 'profit_factor'
        ]
        
        aggregate_metrics = {}
        
        for metric in metrics_to_aggregate:
            values = []
            for result in results.values():
                if 'metrics' in result and metric in result['metrics']:
                    value = result['metrics'][metric]
                    if not np.isnan(value) and not np.isinf(value):
                        values.append(value)
            
            if values:
                aggregate_metrics[metric] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        return aggregate_metrics
    
    def _calculate_portfolio_metrics(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate portfolio-level metrics."""
        
        # Equal-weighted portfolio
        equal_weighted_returns = []
        equal_weighted_sharpe = []
        
        for result in results.values():
            if 'metrics' in result:
                total_return = result['metrics'].get('total_return', 0)
                sharpe_ratio = result['metrics'].get('sharpe_ratio', 0)
                
                if not np.isnan(total_return) and not np.isinf(total_return):
                    equal_weighted_returns.append(total_return)
                
                if not np.isnan(sharpe_ratio) and not np.isinf(sharpe_ratio):
                    equal_weighted_sharpe.append(sharpe_ratio)
        
        portfolio_metrics = {
            'equal_weighted_return': np.mean(equal_weighted_returns) if equal_weighted_returns else 0,
            'equal_weighted_sharpe': np.mean(equal_weighted_sharpe) if equal_weighted_sharpe else 0,
            'portfolio_diversification': len(results),
            'positive_return_rate': len([r for r in equal_weighted_returns if r > 0]) / len(equal_weighted_returns) * 100 if equal_weighted_returns else 0
        }
        
        return portfolio_metrics
    
    def _calculate_performance_statistics(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance statistics."""
        
        # Extract key metrics
        total_returns = []
        sharpe_ratios = []
        max_drawdowns = []
        win_rates = []
        
        for result in results.values():
            if 'metrics' in result:
                metrics = result['metrics']
                
                total_return = metrics.get('total_return', 0)
                sharpe_ratio = metrics.get('sharpe_ratio', 0)
                max_drawdown = metrics.get('max_drawdown', 0)
                win_rate = metrics.get('win_rate', 0)
                
                if not np.isnan(total_return) and not np.isinf(total_return):
                    total_returns.append(total_return)
                
                if not np.isnan(sharpe_ratio) and not np.isinf(sharpe_ratio):
                    sharpe_ratios.append(sharpe_ratio)
                
                if not np.isnan(max_drawdown) and not np.isinf(max_drawdown):
                    max_drawdowns.append(max_drawdown)
                
                if not np.isnan(win_rate) and not np.isinf(win_rate):
                    win_rates.append(win_rate)
        
        performance_stats = {
            'return_statistics': {
                'mean': np.mean(total_returns) if total_returns else 0,
                'median': np.median(total_returns) if total_returns else 0,
                'std': np.std(total_returns) if total_returns else 0,
                'min': np.min(total_returns) if total_returns else 0,
                'max': np.max(total_returns) if total_returns else 0
            },
            'sharpe_statistics': {
                'mean': np.mean(sharpe_ratios) if sharpe_ratios else 0,
                'median': np.median(sharpe_ratios) if sharpe_ratios else 0,
                'std': np.std(sharpe_ratios) if sharpe_ratios else 0
            },
            'drawdown_statistics': {
                'mean': np.mean(max_drawdowns) if max_drawdowns else 0,
                'median': np.median(max_drawdowns) if max_drawdowns else 0,
                'max': np.max(max_drawdowns) if max_drawdowns else 0
            },
            'win_rate_statistics': {
                'mean': np.mean(win_rates) if win_rates else 0,
                'median': np.median(win_rates) if win_rates else 0,
                'std': np.std(win_rates) if win_rates else 0
            }
        }
        
        return performance_stats
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get processing summary statistics."""
        return {
            'total_processed': self.completed_count,
            'total_requested': self.total_count,
            'success_rate': self.completed_count / self.total_count * 100 if self.total_count > 0 else 0,
            'processing_config': self.processing_config.to_dict()
        }
