#!/usr/bin/env python3
"""
Alert Manager for Continuous Mode

Sends notifications via console, file, and optionally email.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from enum import Enum

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.continuous.opportunity_scanner import TradeOpportunity
from scripts.continuous.config import CONFIG


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    OPPORTUNITY = "OPPORTUNITY"
    EXIT = "EXIT"
    ERROR = "ERROR"


class AlertManager:
    """Manages alerts for continuous trading mode."""

    def __init__(self, config=None):
        """Initialize alert manager."""
        self.config = config or CONFIG

        # Ensure log directory exists
        if self.config.alert_to_file:
            self.config.alerts_log.parent.mkdir(parents=True, exist_ok=True)

    def _format_timestamp(self) -> str:
        """Get formatted timestamp."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _write_to_log(self, message: str):
        """Write alert to log file."""
        if not self.config.alert_to_file:
            return

        try:
            with open(self.config.alerts_log, 'a') as f:
                f.write(f"{message}\n")
        except Exception as e:
            print(f"Error writing to log: {e}")

    def _send_email(self, subject: str, body: str):
        """Send email alert (placeholder)."""
        if not self.config.alert_to_email or not self.config.email_recipient:
            return

        # TODO: Implement email sending (SMTP, SendGrid, etc.)
        print(f"[Email Alert] {subject}")

    def send_alert(
        self,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        send_email: bool = False
    ):
        """
        Send a general alert.

        Args:
            message: Alert message
            level: Alert severity level
            send_email: Force email notification
        """
        timestamp = self._format_timestamp()
        formatted = f"[{timestamp}] [{level.value}] {message}"

        # Console
        if self.config.alert_to_console:
            # Color coding
            if level == AlertLevel.OPPORTUNITY:
                print(f"\033[92m{formatted}\033[0m")  # Green
            elif level == AlertLevel.WARNING:
                print(f"\033[93m{formatted}\033[0m")  # Yellow
            elif level == AlertLevel.ERROR:
                print(f"\033[91m{formatted}\033[0m")  # Red
            elif level == AlertLevel.EXIT:
                print(f"\033[96m{formatted}\033[0m")  # Cyan
            else:
                print(formatted)

        # Log file
        self._write_to_log(formatted)

        # Email
        if send_email:
            self._send_email(f"Trading Alert: {level.value}", message)

    def alert_regime_change(self, old_regime: str, new_regime: str, vix: float):
        """Alert on regime change."""
        message = f"Regime change: {old_regime.upper()} → {new_regime.upper()} (VIX {vix:.2f})"
        self.send_alert(message, AlertLevel.WARNING, send_email=True)

    def alert_opportunities(self, opportunities: List[TradeOpportunity]):
        """Alert on new opportunities."""
        if not opportunities:
            return

        count = len(opportunities)
        message = f"Found {count} trade opportunity(ies)"
        self.send_alert(message, AlertLevel.OPPORTUNITY)

        # Show top 3
        for i, opp in enumerate(opportunities[:3], 1):
            detail = (
                f"  #{i}: {opp.dte}DTE {opp.band} {opp.spread_type.upper()} "
                f"@ {opp.entry_time_pst} | "
                f"Win:{opp.expected_win_pct:.1f}% ROI:{opp.expected_roi_pct:.1f}% | "
                f"Credit:${opp.estimated_credit:.0f} Risk:${opp.estimated_max_risk:.0f}"
            )
            self.send_alert(detail, AlertLevel.INFO)

    def alert_position_exit(
        self,
        position_id: str,
        reason: str,
        current_pnl: float,
        credit: float
    ):
        """Alert on position exit signal."""
        pnl_pct = (current_pnl / credit * 100) if credit > 0 else 0
        message = (
            f"EXIT SIGNAL - Position {position_id} | "
            f"Reason: {reason} | "
            f"P&L: ${current_pnl:.2f} ({pnl_pct:+.1f}%)"
        )
        self.send_alert(message, AlertLevel.EXIT, send_email=True)

    def alert_risk_limit(self, limit_type: str, current: float, max_allowed: float):
        """Alert on risk limit breach."""
        message = (
            f"RISK LIMIT BREACH - {limit_type}: "
            f"${current:,.0f} exceeds ${max_allowed:,.0f}"
        )
        self.send_alert(message, AlertLevel.ERROR, send_email=True)

    def alert_market_hours(self, is_open: bool):
        """Alert on market open/close."""
        status = "OPEN" if is_open else "CLOSED"
        message = f"Market hours {status}"
        self.send_alert(message, AlertLevel.INFO)


# Global alert manager instance
ALERTS = AlertManager()


if __name__ == '__main__':
    """Test alert manager."""
    print("Testing alert manager...")

    alerts = AlertManager()

    # Test different alert types
    alerts.send_alert("System initialized", AlertLevel.INFO)
    alerts.send_alert("Low volume detected", AlertLevel.WARNING)
    alerts.send_alert("High-quality opportunity found", AlertLevel.OPPORTUNITY)
    alerts.send_alert("Position approaching stop loss", AlertLevel.EXIT)
    alerts.send_alert("Database connection failed", AlertLevel.ERROR)

    # Test regime change
    alerts.alert_regime_change('low', 'medium', 18.5)

    # Test opportunities
    from scripts.continuous.opportunity_scanner import TradeOpportunity

    test_opps = [
        TradeOpportunity(
            timestamp=datetime.now().isoformat(),
            config_rank=1,
            dte=3,
            band='P98',
            spread_type='iron_condor',
            flow_mode='with_flow',
            entry_time_pst='07:30',
            expected_win_pct=95.5,
            expected_roi_pct=346.2,
            sharpe=0.45,
            trade_score=125.8,
            estimated_credit=285.0,
            estimated_max_risk=1715.0,
            is_in_entry_window=True,
            meets_quality_threshold=True,
        )
    ]

    alerts.alert_opportunities(test_opps)

    # Test exit signal
    alerts.alert_position_exit('pos_001', 'Profit target hit', 142.50, 285.0)

    # Test risk limit
    alerts.alert_risk_limit('Total Capital at Risk', 55000, 50000)

    print(f"\nAlerts logged to: {alerts.config.alerts_log}")
