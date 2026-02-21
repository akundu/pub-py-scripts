"""
Performance Metrics Calculation

Comprehensive performance metrics for backtesting results including
risk-adjusted returns, drawdown analysis, and statistical measures.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from .portfolio import Portfolio, Trade


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    # Basic metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    
    # Risk metrics
    max_drawdown: float
    calmar_ratio: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional Value at Risk 95%
    
    # Trade metrics
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Additional metrics
    alpha: float
    beta: float
    information_ratio: float
    time_in_market: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'num_trades': self.num_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'alpha': self.alpha,
            'beta': self.beta,
            'information_ratio': self.information_ratio,
            'time_in_market': self.time_in_market
        }


class MetricsCalculator:
    """
    Calculator for comprehensive performance metrics.
    """
    
    def __init__(self, risk_free_rate: float = 0.02, logger: Optional[logging.Logger] = None):
        self.risk_free_rate = risk_free_rate
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_metrics(
        self,
        portfolio: Portfolio,
        benchmark_returns: Optional[pd.Series] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            portfolio: Portfolio object with trades and equity history
            benchmark_returns: Benchmark returns for comparison
            start_date: Start date for calculation
            end_date: End date for calculation
            
        Returns:
            PerformanceMetrics object
        """
        # Get equity curve
        equity_curve = portfolio.get_equity_curve()
        if equity_curve.empty:
            return self._empty_metrics()
        
        # Filter by date range if specified
        if start_date:
            equity_curve = equity_curve[equity_curve.index >= start_date]
        if end_date:
            equity_curve = equity_curve[equity_curve.index <= end_date]
        
        if equity_curve.empty:
            return self._empty_metrics()
        
        # Calculate returns
        returns = equity_curve['equity'].pct_change().dropna()
        
        # Basic metrics
        total_return = self._calculate_total_return(equity_curve)
        annualized_return = self._calculate_annualized_return(returns)
        volatility = self._calculate_volatility(returns)
        
        # Risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        calmar_ratio = self._calculate_calmar_ratio(annualized_return, max_drawdown)
        var_95, cvar_95 = self._calculate_var_cvar(returns)
        
        # Trade metrics
        trade_metrics = self._calculate_trade_metrics(portfolio.trades)
        
        # Benchmark comparison
        alpha, beta, information_ratio = self._calculate_benchmark_metrics(
            returns, benchmark_returns
        )
        
        # Time in market
        time_in_market = self._calculate_time_in_market(portfolio.trades, equity_curve)
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            num_trades=trade_metrics['num_trades'],
            win_rate=trade_metrics['win_rate'],
            profit_factor=trade_metrics['profit_factor'],
            avg_win=trade_metrics['avg_win'],
            avg_loss=trade_metrics['avg_loss'],
            largest_win=trade_metrics['largest_win'],
            largest_loss=trade_metrics['largest_loss'],
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio,
            time_in_market=time_in_market
        )
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics for edge cases."""
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            var_95=0.0,
            cvar_95=0.0,
            num_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            alpha=0.0,
            beta=0.0,
            information_ratio=0.0,
            time_in_market=0.0
        )
    
    def _calculate_total_return(self, equity_curve: pd.DataFrame) -> float:
        """Calculate total return percentage."""
        if len(equity_curve) < 2:
            return 0.0
        
        initial_equity = equity_curve['equity'].iloc[0]
        final_equity = equity_curve['equity'].iloc[-1]
        return (final_equity - initial_equity) / initial_equity * 100
    
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return."""
        if len(returns) < 2:
            return 0.0
        
        # Calculate compound annual growth rate
        total_return = (1 + returns).prod() - 1
        years = len(returns) / 252  # Assuming 252 trading days per year
        
        if years <= 0:
            return 0.0
        
        annualized_return = (1 + total_return) ** (1 / years) - 1
        return annualized_return * 100
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        if len(returns) < 2:
            return 0.0
        
        return returns.std() * np.sqrt(252) * 100  # Annualized
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        return sharpe
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(252)
        return sortino
    
    def _calculate_max_drawdown(self, equity_curve: pd.DataFrame) -> float:
        """Calculate maximum drawdown."""
        if len(equity_curve) < 2:
            return 0.0
        
        equity = equity_curve['equity']
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak * 100
        
        return abs(drawdown.min())
    
    def _calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return 0.0
        
        return annualized_return / max_drawdown
    
    def _calculate_var_cvar(self, returns: pd.Series) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional Value at Risk."""
        if len(returns) < 2:
            return 0.0, 0.0
        
        var_95 = np.percentile(returns, 5) * 100  # 5th percentile
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        
        return var_95, cvar_95
    
    def _calculate_trade_metrics(self, trades: List[Trade]) -> Dict[str, float]:
        """Calculate trade-based metrics."""
        if not trades:
            return {
                'num_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        num_trades = len(trades)
        win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0.0
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        avg_win = total_wins / len(winning_trades) if winning_trades else 0.0
        avg_loss = total_losses / len(losing_trades) if losing_trades else 0.0
        
        largest_win = max((t.pnl for t in trades), default=0.0)
        largest_loss = min((t.pnl for t in trades), default=0.0)
        
        return {
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }
    
    def _calculate_benchmark_metrics(
        self, 
        returns: pd.Series, 
        benchmark_returns: Optional[pd.Series]
    ) -> Tuple[float, float, float]:
        """Calculate alpha, beta, and information ratio."""
        if benchmark_returns is None or len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0, 0.0, 0.0
        
        # Align returns and benchmark returns
        common_index = returns.index.intersection(benchmark_returns.index)
        if len(common_index) < 2:
            return 0.0, 0.0, 0.0
        
        aligned_returns = returns.loc[common_index]
        aligned_benchmark = benchmark_returns.loc[common_index]
        
        # Calculate beta
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        
        if benchmark_variance == 0:
            beta = 0.0
        else:
            beta = covariance / benchmark_variance
        
        # Calculate alpha
        alpha = (aligned_returns.mean() - self.risk_free_rate / 252) - beta * (aligned_benchmark.mean() - self.risk_free_rate / 252)
        alpha = alpha * 252 * 100  # Annualized percentage
        
        # Calculate information ratio
        excess_returns = aligned_returns - aligned_benchmark
        tracking_error = excess_returns.std()
        
        if tracking_error == 0:
            information_ratio = 0.0
        else:
            information_ratio = excess_returns.mean() / tracking_error * np.sqrt(252)
        
        return alpha, beta, information_ratio
    
    def _calculate_time_in_market(self, trades: List[Trade], equity_curve: pd.DataFrame) -> float:
        """Calculate percentage of time in market."""
        if not trades or equity_curve.empty:
            return 0.0
        
        total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if total_days == 0:
            return 0.0
        
        # Calculate total days with positions
        position_days = 0
        for trade in trades:
            position_days += trade.duration_days
        
        return (position_days / total_days) * 100
    
    def calculate_rolling_metrics(
        self, 
        equity_curve: pd.DataFrame, 
        window: int = 252
    ) -> pd.DataFrame:
        """Calculate rolling performance metrics."""
        if equity_curve.empty:
            return pd.DataFrame()
        
        returns = equity_curve['equity'].pct_change().dropna()
        
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        # Rolling returns
        rolling_metrics['rolling_return'] = returns.rolling(window).apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        
        # Rolling volatility
        rolling_metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252) * 100
        
        # Rolling Sharpe ratio
        rolling_metrics['rolling_sharpe'] = (
            returns.rolling(window).mean() - self.risk_free_rate / 252
        ) / returns.rolling(window).std() * np.sqrt(252)
        
        # Rolling max drawdown
        rolling_peak = equity_curve['equity'].rolling(window).max()
        rolling_drawdown = (equity_curve['equity'] - rolling_peak) / rolling_peak * 100
        rolling_metrics['rolling_drawdown'] = rolling_drawdown.rolling(window).min()
        
        return rolling_metrics.dropna()
