"""
Exit strategy management for credit spreads.

This module manages exit timing (same-day vs overnight) and calculates
actual ROI using real option prices.
"""

from datetime import datetime, time
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ExitStrategyManager:
    """Manage exit timing and overnight hold decisions."""

    # Exit time constants (Pacific Time)
    FORCE_EXIT_TIME = time(15, 0)  # 3:00 PM PT (force close for 0 DTE)
    PREFERRED_EOD_EXIT = time(14, 0)  # 2:00 PM PT (prefer exit if profitable)
    MARKET_CLOSE = time(13, 0)  # 1:00 PM PT (4:00 PM ET)

    def __init__(self, max_loss_per_position: float = 30000.0):
        """
        Initialize exit strategy manager.

        Args:
            max_loss_per_position: Maximum allowed loss per position
        """
        self.max_loss_per_position = max_loss_per_position

    def should_exit_eod(
        self,
        dte: int,
        current_pnl: float,
        profit_target_pct: float,
        current_time: datetime,
        entry_credit: float,
        max_loss: float
    ) -> Tuple[bool, str]:
        """
        Determine if position should exit at end of day.

        Exit Rules:
        1. 0 DTE: ALWAYS exit at EOD (force close at 3:00 PM PT)
        2. Profit target hit: EXIT immediately
        3. Non-0 DTE + positive P&L + after 2:00 PM: EXIT same day
        4. Non-0 DTE + negative P&L: HOLD overnight (give time to recover)
        5. Loss exceeds max: EXIT immediately (stop loss)

        Args:
            dte: Days to expiration
            current_pnl: Current profit/loss in dollars
            profit_target_pct: Profit target percentage (e.g., 0.5 for 50%)
            current_time: Current timestamp
            entry_credit: Entry credit received
            max_loss: Maximum loss for this position

        Returns:
            Tuple of (should_exit, reason)
        """
        current_hour_min = current_time.time()

        # Rule 5: Stop loss - exit if loss exceeds maximum
        if current_pnl < 0 and abs(current_pnl) > self.max_loss_per_position:
            logger.warning(
                f"STOP LOSS: P&L={current_pnl:.2f} exceeds max "
                f"loss={self.max_loss_per_position:.2f}"
            )
            return (True, "stop_loss")

        # Rule 2: Profit target hit
        profit_target = entry_credit * profit_target_pct
        if current_pnl >= profit_target:
            logger.info(
                f"PROFIT TARGET HIT: P&L={current_pnl:.2f} >= "
                f"target={profit_target:.2f} ({profit_target_pct*100:.0f}%)"
            )
            return (True, "profit_target")

        # Rule 1: 0 DTE must exit at EOD
        if dte == 0:
            if current_hour_min >= self.FORCE_EXIT_TIME:
                logger.info(f"0 DTE: Forcing EOD exit at {current_time}")
                return (True, "0dte_eod_forced")

        # Rule 3: Non-0 DTE + profitable + late in day → exit
        if dte > 0 and current_pnl > 0 and current_hour_min >= self.PREFERRED_EOD_EXIT:
            logger.info(
                f"Non-0 DTE: Exiting profitable position ({current_pnl:.2f}) "
                f"at EOD to avoid overnight risk"
            )
            return (True, "profitable_eod")

        # Rule 4: Non-0 DTE + negative → hold overnight
        if dte > 0 and current_pnl < 0:
            logger.debug(
                f"Non-0 DTE: Holding overnight (P&L={current_pnl:.2f}, "
                f"DTE={dte}) to allow theta decay"
            )
            return (False, "hold_overnight")

        # Default: continue monitoring
        return (False, "continue")

    def calculate_exit_roi(
        self,
        entry_credit: float,
        exit_price: float,
        max_loss: float,
        num_contracts: int = 1,
        spread_type: str = "2-leg"
    ) -> Dict[str, float]:
        """
        Calculate ROI at exit using actual prices.

        For credit spreads:
        - Entry credit: What we collected
        - Exit cost: What we pay to buy back
        - P&L: Entry credit - Exit cost
        - ROI: P&L / max_loss

        Args:
            entry_credit: Total credit received at entry
            exit_price: Total cost to close position
            max_loss: Maximum potential loss
            num_contracts: Number of contracts
            spread_type: "2-leg" (vertical spread) or "4-leg" (iron condor)

        Returns:
            Dictionary with:
                - entry_credit: Credit received
                - exit_cost: Cost to close
                - pnl: Profit/loss
                - roi_pct: Return on investment percentage
                - entry_roi: ROI at entry (theoretical max)
                - exit_roi: ROI at exit (actual)
        """
        if max_loss <= 0:
            logger.warning("Max loss is zero, cannot calculate ROI")
            return {
                'entry_credit': entry_credit,
                'exit_cost': exit_price,
                'pnl': 0.0,
                'roi_pct': 0.0,
                'entry_roi': 0.0,
                'exit_roi': 0.0
            }

        # Calculate P&L
        # For credit spreads: P&L = credit received - cost to close
        pnl = entry_credit - exit_price

        # Calculate ROI
        # Entry ROI: If held to expiration and expired worthless (max profit)
        entry_roi = (entry_credit / max_loss) * 100

        # Exit ROI: Actual realized return
        exit_roi = (pnl / max_loss) * 100

        logger.debug(
            f"Exit ROI: credit={entry_credit:.2f}, cost={exit_price:.2f}, "
            f"pnl={pnl:.2f}, roi={exit_roi:.1f}% "
            f"(entry_roi={entry_roi:.1f}%)"
        )

        return {
            'entry_credit': entry_credit,
            'exit_cost': exit_price,
            'pnl': pnl,
            'roi_pct': exit_roi,
            'entry_roi': entry_roi,
            'exit_roi': exit_roi
        }

    def should_exit_now(
        self,
        dte: int,
        current_pnl: float,
        profit_target_pct: float,
        current_time: datetime,
        entry_credit: float,
        max_loss: float,
        underlying_price: float,
        short_strike: float,
        option_type: str,
        breach_threshold_pct: float = 0.02
    ) -> Tuple[bool, str]:
        """
        Comprehensive exit decision (includes breach detection).

        Args:
            (standard exit params...)
            underlying_price: Current underlying price
            short_strike: Short strike price
            option_type: 'call' or 'put'
            breach_threshold_pct: % threshold for breach (default 2%)

        Returns:
            Tuple of (should_exit, reason)
        """
        # First check standard exit rules
        should_exit, reason = self.should_exit_eod(
            dte, current_pnl, profit_target_pct, current_time,
            entry_credit, max_loss
        )

        if should_exit:
            return (should_exit, reason)

        # Check for strike breach
        is_breached = self.is_strike_breached(
            underlying_price, short_strike, option_type, breach_threshold_pct
        )

        if is_breached:
            logger.warning(
                f"STRIKE BREACH: {option_type} strike {short_strike:.2f} "
                f"breached by underlying={underlying_price:.2f}"
            )
            return (True, "strike_breach")

        return (False, reason)

    def is_strike_breached(
        self,
        underlying_price: float,
        short_strike: float,
        option_type: str,
        threshold_pct: float = 0.02
    ) -> bool:
        """
        Check if short strike has been breached.

        Args:
            underlying_price: Current underlying price
            short_strike: Short strike price
            option_type: 'call' or 'put'
            threshold_pct: % threshold (e.g., 0.02 = 2%)

        Returns:
            True if breached
        """
        threshold_amount = short_strike * threshold_pct

        if option_type.lower() == 'call':
            # Call breached if price above strike
            breached = underlying_price > (short_strike + threshold_amount)
        else:  # put
            # Put breached if price below strike
            breached = underlying_price < (short_strike - threshold_amount)

        return breached

    def get_hold_duration_hours(
        self,
        entry_time: datetime,
        exit_time: datetime
    ) -> float:
        """
        Calculate hold duration in hours.

        Args:
            entry_time: Entry timestamp
            exit_time: Exit timestamp

        Returns:
            Hours held
        """
        duration = (exit_time - entry_time).total_seconds() / 3600
        return duration

    def categorize_exit_reason(self, reason: str) -> str:
        """
        Categorize exit reason for reporting.

        Returns:
            Category: 'profit_target', 'eod_exit', 'breach', 'stop_loss', 'other'
        """
        if reason == 'profit_target':
            return 'profit_target'
        elif reason in ['0dte_eod_forced', 'profitable_eod']:
            return 'eod_exit'
        elif reason == 'strike_breach':
            return 'breach'
        elif reason == 'stop_loss':
            return 'stop_loss'
        else:
            return 'other'


# Convenience function
def should_exit_position(
    dte: int,
    current_pnl: float,
    profit_target_pct: float,
    current_time: datetime,
    entry_credit: float,
    max_loss: float
) -> bool:
    """
    Simple function to check if position should exit.

    Returns:
        True if should exit
    """
    manager = ExitStrategyManager()
    should_exit, _ = manager.should_exit_eod(
        dte, current_pnl, profit_target_pct, current_time,
        entry_credit, max_loss
    )
    return should_exit
