#!/usr/bin/env python3
"""
Continuous Trading Mode - Alert-Only System

Monitors market conditions, detects opportunities, and sends alerts.
NO automated trading - all alerts require manual execution.

Usage:
    python scripts/continuous/continuous_mode.py
    python scripts/continuous/continuous_mode.py --ticker NDX --trend sideways
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.continuous.config import CONFIG
from scripts.continuous.market_data import get_current_market_context, MarketContext
from scripts.continuous.opportunity_scanner import scan_opportunities, filter_actionable_opportunities
from scripts.continuous.position_tracker import PositionTracker
from scripts.continuous.alert_manager import ALERTS, AlertLevel
from scripts.continuous.dashboard import DashboardData


class ContinuousMode:
    """Main continuous trading mode orchestrator."""

    def __init__(
        self,
        ticker: str = 'NDX',
        trend: str = 'sideways',
        config=None
    ):
        """Initialize continuous mode."""
        self.ticker = ticker
        self.trend = trend
        self.config = config or CONFIG

        self.tracker = PositionTracker()
        self.dashboard_data = DashboardData()

        self.last_regime = None
        self.last_scan_time = None
        self.last_market_data_time = None
        self.last_position_check_time = None

        self.current_market_context: Optional[MarketContext] = None
        self.is_running = False

    def update_market_data(self):
        """Update market context."""
        try:
            self.current_market_context = get_current_market_context(self.ticker, self.trend)
            self.last_market_data_time = time.time()

            # Check for regime change
            current_regime = self.current_market_context.vix_regime

            if self.last_regime is None:
                ALERTS.send_alert(
                    f"System initialized | VIX: {self.current_market_context.vix_level:.2f} "
                    f"| Regime: {current_regime.upper()}",
                    AlertLevel.INFO
                )
                self.last_regime = current_regime

            elif current_regime != self.last_regime:
                ALERTS.alert_regime_change(
                    self.last_regime,
                    current_regime,
                    self.current_market_context.vix_level
                )
                self.last_regime = current_regime

        except Exception as e:
            ALERTS.send_alert(f"Error updating market data: {e}", AlertLevel.ERROR)

    def scan_for_opportunities(self):
        """Scan for trade opportunities."""
        if self.current_market_context is None:
            return

        try:
            # Scan opportunities
            opportunities = scan_opportunities(
                self.current_market_context,
                top_n=self.config.regime_top_n_configs
            )

            # Filter to actionable
            actionable = filter_actionable_opportunities(
                opportunities,
                require_entry_window=True,
                require_quality=True,
                top_n=5
            )

            if actionable:
                ALERTS.alert_opportunities(actionable)

            self.last_scan_time = time.time()

        except Exception as e:
            ALERTS.send_alert(f"Error scanning opportunities: {e}", AlertLevel.ERROR)

    def check_positions(self):
        """Check open positions for exit signals."""
        try:
            # Check exit conditions
            self.tracker.check_exit_signals()

            # Check risk limits
            total_risk = self.tracker.get_total_risk()
            if total_risk > self.config.max_total_risk:
                ALERTS.alert_risk_limit(
                    'Total Capital at Risk',
                    total_risk,
                    self.config.max_total_risk
                )

            open_count = len(self.tracker.get_open_positions())
            if open_count > self.config.max_positions:
                ALERTS.send_alert(
                    f"Position limit exceeded: {open_count} > {self.config.max_positions}",
                    AlertLevel.WARNING
                )

            self.last_position_check_time = time.time()

        except Exception as e:
            ALERTS.send_alert(f"Error checking positions: {e}", AlertLevel.ERROR)

    def update_dashboard(self):
        """Update dashboard data file."""
        try:
            self.dashboard_data.save_to_file()
        except Exception as e:
            ALERTS.send_alert(f"Error updating dashboard: {e}", AlertLevel.ERROR)

    def run_cycle(self):
        """Run one monitoring cycle."""
        now = time.time()

        # Update market data (every 1 minute)
        if (self.last_market_data_time is None or
            now - self.last_market_data_time >= self.config.market_data_interval_seconds):
            self.update_market_data()

        # Scan for opportunities (every 5 minutes, only during market hours)
        if self.current_market_context and self.current_market_context.is_market_hours:
            if (self.last_scan_time is None or
                now - self.last_scan_time >= self.config.scan_interval_seconds):
                self.scan_for_opportunities()

        # Check positions (every 1 minute)
        if (self.last_position_check_time is None or
            now - self.last_position_check_time >= 60):
            self.check_positions()

        # Update dashboard (every cycle)
        if self.config.dashboard_enabled:
            self.update_dashboard()

    def run(self):
        """Run continuous mode."""
        self.is_running = True

        ALERTS.send_alert("=" * 80, AlertLevel.INFO)
        ALERTS.send_alert("CONTINUOUS TRADING MODE - ALERT-ONLY", AlertLevel.INFO)
        ALERTS.send_alert("=" * 80, AlertLevel.INFO)
        ALERTS.send_alert(f"Ticker: {self.ticker}", AlertLevel.INFO)
        ALERTS.send_alert(f"Trend: {self.trend.upper()}", AlertLevel.INFO)
        ALERTS.send_alert(f"Scan Interval: {self.config.scan_interval_seconds}s", AlertLevel.INFO)
        ALERTS.send_alert(f"Trading Hours: {self.config.trading_start_hour}:00 - {self.config.trading_end_hour}:00 PST", AlertLevel.INFO)

        if self.config.dashboard_enabled:
            ALERTS.send_alert(
                f"Dashboard: http://localhost:{self.config.dashboard_port}",
                AlertLevel.INFO
            )

        ALERTS.send_alert("=" * 80, AlertLevel.INFO)
        ALERTS.send_alert("Press Ctrl+C to stop", AlertLevel.INFO)
        ALERTS.send_alert("=" * 80, AlertLevel.INFO)

        try:
            while self.is_running:
                self.run_cycle()
                time.sleep(10)  # Sleep 10 seconds between cycles

        except KeyboardInterrupt:
            ALERTS.send_alert("\nShutting down...", AlertLevel.INFO)
            self.is_running = False

        except Exception as e:
            ALERTS.send_alert(f"Fatal error: {e}", AlertLevel.ERROR)
            self.is_running = False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Continuous trading mode - alert-only system')
    parser.add_argument('--ticker', type=str, default='NDX',
                        help='Ticker symbol (default: NDX)')
    parser.add_argument('--trend', type=str, default='sideways',
                        choices=['up', 'down', 'sideways'],
                        help='Market trend (default: sideways)')
    parser.add_argument('--scan-interval', type=int, default=300,
                        help='Opportunity scan interval in seconds (default: 300)')
    parser.add_argument('--no-dashboard', action='store_true',
                        help='Disable dashboard')
    args = parser.parse_args()

    # Update config
    CONFIG.scan_interval_seconds = args.scan_interval
    CONFIG.default_ticker = args.ticker
    CONFIG.default_trend = args.trend

    if args.no_dashboard:
        CONFIG.dashboard_enabled = False

    # Start continuous mode
    mode = ContinuousMode(ticker=args.ticker, trend=args.trend)
    mode.run()


if __name__ == '__main__':
    main()
