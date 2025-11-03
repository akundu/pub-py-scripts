"""
Portfolio Management System

Handles portfolio state, position tracking, and trade execution
for the backtesting framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from .config import BacktestConfig, PositionSizingMethod
from ..strategies.base import Signal, Direction, SignalResult, PositionSizeResult, RiskParams


@dataclass
class Position:
    """Represents a position in a single asset."""
    ticker: str
    size: float  # Number of shares/units
    entry_price: float
    entry_date: datetime
    current_price: float
    direction: Direction
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.size * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        if self.direction == Direction.LONG:
            return self.size * (self.current_price - self.entry_price)
        else:  # SHORT
            return self.size * (self.entry_price - self.current_price)
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized profit/loss percentage."""
        if self.direction == Direction.LONG:
            return (self.current_price - self.entry_price) / self.entry_price * 100
        else:  # SHORT
            return (self.entry_price - self.current_price) / self.entry_price * 100


@dataclass
class Trade:
    """Represents a completed trade."""
    ticker: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    size: float
    direction: Direction
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    strategy_name: str
    signal_reasoning: str
    
    @property
    def duration_days(self) -> int:
        """Trade duration in days."""
        return (self.exit_date - self.entry_date).days


class Portfolio:
    """
    Portfolio management system for backtesting.
    
    Handles:
    - Position tracking
    - Trade execution
    - Cash management
    - Risk management
    - Performance tracking
    """
    
    def __init__(self, config: BacktestConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Portfolio state
        self.cash = config.initial_capital
        self.initial_capital = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        
        # Risk management
        self.risk_params = RiskParams(
            max_position_size=config.max_position_size,
            max_portfolio_risk=config.max_portfolio_risk,
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
            position_sizing_method=config.position_sizing.value
        )
        
        # Performance tracking
        self.total_commission_paid = 0.0
        self.total_slippage_cost = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = config.initial_capital
        
    @property
    def total_equity(self) -> float:
        """Total portfolio equity (cash + positions)."""
        position_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + position_value
    
    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized profit/loss."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def total_realized_pnl(self) -> float:
        """Total realized profit/loss from closed trades."""
        return sum(trade.pnl for trade in self.trades)
    
    @property
    def total_pnl(self) -> float:
        """Total profit/loss (realized + unrealized)."""
        return self.total_realized_pnl + self.total_unrealized_pnl
    
    @property
    def total_return_pct(self) -> float:
        """Total return percentage."""
        return (self.total_equity - self.initial_capital) / self.initial_capital * 100
    
    def update_position_prices(self, prices: Dict[str, float], current_date: datetime) -> None:
        """Update current prices for all positions."""
        for ticker, price in prices.items():
            if ticker in self.positions:
                self.positions[ticker].current_price = price
        
        # Update equity history
        self.equity_history.append((current_date, self.total_equity))
        
        # Update peak equity and drawdown
        if self.total_equity > self.peak_equity:
            self.peak_equity = self.total_equity
        
        current_drawdown = (self.peak_equity - self.total_equity) / self.peak_equity * 100
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def execute_trade(
        self,
        ticker: str,
        signal: SignalResult,
        position_size: PositionSizeResult,
        current_price: float,
        current_date: datetime,
        strategy_name: str
    ) -> bool:
        """
        Execute a trade based on signal and position size.
        
        Returns:
            True if trade was executed, False otherwise
        """
        if signal.signal == Signal.HOLD:
            return False
        
        # Check if we have enough capital
        required_capital = position_size.dollar_amount
        if signal.signal == Signal.BUY and required_capital > self.cash:
            self.logger.warning(f"Insufficient cash for {ticker} trade: {required_capital:.2f} > {self.cash:.2f}")
            return False
        
        # Check minimum trade amount
        if required_capital < self.config.min_trade_amount:
            self.logger.warning(f"Trade amount too small for {ticker}: {required_capital:.2f} < min_trade_amount={self.config.min_trade_amount:.2f}")
            return False
        
        # Apply slippage
        slippage_cost = required_capital * self.config.slippage_pct
        actual_price = current_price
        if signal.signal == Signal.BUY:
            actual_price = current_price * (1 + self.config.slippage_pct)
        elif signal.signal == Signal.SELL:
            actual_price = current_price * (1 - self.config.slippage_pct)
        
        # Calculate commission
        commission = self.config.commission_per_trade
        
        # Execute the trade
        if signal.signal == Signal.BUY:
            return self._execute_buy(ticker, signal, position_size, actual_price, current_date, strategy_name, commission, slippage_cost)
        elif signal.signal == Signal.SELL:
            return self._execute_sell(ticker, signal, position_size, actual_price, current_date, strategy_name, commission, slippage_cost)
        
        return False
    
    def _execute_buy(
        self,
        ticker: str,
        signal: SignalResult,
        position_size: PositionSizeResult,
        price: float,
        date: datetime,
        strategy_name: str,
        commission: float,
        slippage_cost: float
    ) -> bool:
        """Execute a buy order."""
        # Adjust position size if total cost exceeds available cash
        total_cost = position_size.dollar_amount + commission + slippage_cost
        
        self.logger.info(f"Attempting to buy {ticker}: dollar_amount={position_size.dollar_amount:.2f}, commission={commission:.2f}, slippage={slippage_cost:.2f}, total_cost={total_cost:.2f}, cash={self.cash:.2f}")
        
        if total_cost > self.cash:
            # Adjust the position size to fit within available cash
            available_for_trade = self.cash - commission - slippage_cost
            if available_for_trade <= 0:
                self.logger.warning(f"Insufficient cash after commissions/slippage for {ticker}: available={available_for_trade:.2f}")
                return False
            
            # Recalculate position size based on available cash
            adjusted_size = available_for_trade / price
            position_size = PositionSizeResult(
                size=adjusted_size,
                size_pct=(available_for_trade / self.cash) * 100,
                dollar_amount=available_for_trade,
                risk_amount=position_size.risk_amount,
                stop_loss=position_size.stop_loss,
                take_profit=position_size.take_profit
            )
            self.logger.info(f"Adjusted position size for {ticker} to fit available cash: new_size={adjusted_size:.2f} shares, dollar_amount={available_for_trade:.2f}")
            
            # Recalculate total cost with adjusted position
            total_cost = position_size.dollar_amount + commission + slippage_cost
        
        # Update cash
        self.cash -= total_cost
        
        # Create or update position
        if ticker in self.positions:
            # Add to existing position
            existing_pos = self.positions[ticker]
            total_size = existing_pos.size + position_size.size
            avg_price = (existing_pos.size * existing_pos.entry_price + position_size.size * price) / total_size
            
            self.positions[ticker] = Position(
                ticker=ticker,
                size=total_size,
                entry_price=avg_price,
                entry_date=existing_pos.entry_date,
                current_price=price,
                direction=Direction.LONG,
                stop_loss=position_size.stop_loss,
                take_profit=position_size.take_profit
            )
        else:
            # Create new position
            self.positions[ticker] = Position(
                ticker=ticker,
                size=position_size.size,
                entry_price=price,
                entry_date=date,
                current_price=price,
                direction=Direction.LONG,
                stop_loss=position_size.stop_loss,
                take_profit=position_size.take_profit
            )
        
        # Update tracking
        self.total_commission_paid += commission
        self.total_slippage_cost += slippage_cost
        
        self.logger.info(f"Bought {position_size.size:.2f} shares of {ticker} at ${price:.2f}")
        return True
    
    def _execute_sell(
        self,
        ticker: str,
        signal: SignalResult,
        position_size: PositionSizeResult,
        price: float,
        date: datetime,
        strategy_name: str,
        commission: float,
        slippage_cost: float
    ) -> bool:
        """Execute a sell order."""
        if ticker not in self.positions:
            return False
        
        position = self.positions[ticker]
        
        # Determine sell size
        sell_size = min(position_size.size, position.size)
        if sell_size <= 0:
            return False
        
        # Calculate proceeds
        proceeds = sell_size * price - commission - slippage_cost
        
        # Update cash
        self.cash += proceeds
        
        # Calculate P&L
        if position.direction == Direction.LONG:
            pnl = sell_size * (price - position.entry_price)
        else:  # SHORT
            pnl = sell_size * (position.entry_price - price)
        
        pnl_pct = pnl / (sell_size * position.entry_price) * 100
        
        # Create trade record
        trade = Trade(
            ticker=ticker,
            entry_date=position.entry_date,
            exit_date=date,
            entry_price=position.entry_price,
            exit_price=price,
            size=sell_size,
            direction=position.direction,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission,
            slippage=slippage_cost,
            strategy_name=strategy_name,
            signal_reasoning=signal.reasoning
        )
        self.trades.append(trade)
        
        # Update position
        remaining_size = position.size - sell_size
        if remaining_size <= 0:
            # Close position completely
            del self.positions[ticker]
        else:
            # Partial close
            self.positions[ticker].size = remaining_size
        
        # Update tracking
        self.total_commission_paid += commission
        self.total_slippage_cost += slippage_cost
        
        self.logger.info(f"Sold {sell_size:.2f} shares of {ticker} at ${price:.2f}, P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
        return True
    
    def check_stop_loss_take_profit(self, current_date: datetime) -> List[Trade]:
        """Check and execute stop loss/take profit orders."""
        executed_trades = []
        
        for ticker, position in list(self.positions.items()):
            should_close = False
            reason = ""
            
            # Check stop loss
            if position.stop_loss is not None:
                if position.direction == Direction.LONG and position.current_price <= position.stop_loss:
                    should_close = True
                    reason = "Stop loss triggered"
                elif position.direction == Direction.SHORT and position.current_price >= position.stop_loss:
                    should_close = True
                    reason = "Stop loss triggered"
            
            # Check take profit
            if position.take_profit is not None:
                if position.direction == Direction.LONG and position.current_price >= position.take_profit:
                    should_close = True
                    reason = "Take profit triggered"
                elif position.direction == Direction.SHORT and position.current_price <= position.take_profit:
                    should_close = True
                    reason = "Take profit triggered"
            
            if should_close:
                # Close position
                signal = SignalResult(
                    signal=Signal.SELL,
                    direction=position.direction,
                    confidence=100.0,
                    expected_movement=0.0,
                    expected_movement_pct=0.0,
                    expected_price=position.current_price,
                    probability_distribution={},
                    reasoning=reason,
                    metadata={}
                )
                
                position_size = PositionSizeResult(
                    size=position.size,
                    size_pct=0.0,
                    dollar_amount=position.size * position.current_price,
                    risk_amount=0.0
                )
                
                if self.execute_trade(ticker, signal, position_size, position.current_price, current_date, "Risk Management"):
                    executed_trades.append(self.trades[-1])  # Last trade is the one we just executed
        
        return executed_trades
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary statistics."""
        return {
            'total_equity': self.total_equity,
            'cash': self.cash,
            'position_value': sum(pos.market_value for pos in self.positions.values()),
            'total_pnl': self.total_pnl,
            'total_return_pct': self.total_return_pct,
            'max_drawdown': self.max_drawdown,
            'num_positions': len(self.positions),
            'num_trades': len(self.trades),
            'total_commission_paid': self.total_commission_paid,
            'total_slippage_cost': self.total_slippage_cost,
            'positions': {ticker: {
                'size': pos.size,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'unrealized_pnl_pct': pos.unrealized_pnl_pct
            } for ticker, pos in self.positions.items()}
        }
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_history, columns=['date', 'equity'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
