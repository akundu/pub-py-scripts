#!/usr/bin/env python3
"""
Configuration for Continuous Trading Mode
"""

from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ContinuousConfig:
    """Configuration for continuous trading mode."""

    # Paths
    grid_file: Path = Path('results/backtest_tight/grid_analysis_tight.csv')
    positions_file: Path = Path('data/continuous/positions.json')
    alerts_log: Path = Path('logs/continuous/alerts.log')
    dashboard_data: Path = Path('data/continuous/dashboard_data.json')

    # Scanning
    scan_interval_seconds: int = 300  # 5 minutes
    market_data_interval_seconds: int = 60  # 1 minute

    # Regime Detection
    default_ticker: str = 'NDX'
    default_trend: str = 'sideways'  # up/down/sideways
    regime_top_n_configs: int = 50

    # Opportunity Filtering
    min_composite_score: float = 50.0  # Minimum trade score
    min_win_rate: float = 90.0  # Minimum 90% win rate
    min_roi: float = 5.0  # Minimum 5% actual ROI (avg_pnl / max_risk)
    min_sharpe: float = 0.30  # Minimum Sharpe ratio

    # Trading Hours (PST)
    trading_start_hour: int = 6  # 06:00 PST (09:00 EST)
    trading_end_hour: int = 13  # 13:00 PST (16:00 EST)

    # Entry Time Windows (PST)
    preferred_entry_hours: List[int] = None  # None = any time in trading hours

    # Position Limits (for monitoring manual trades)
    max_positions: int = 5
    max_total_risk: float = 50000.0  # $50k max total risk

    # Exit Thresholds (for alerting)
    profit_target_pct: float = 0.50  # Exit at 50% of max profit
    stop_loss_mult: float = 2.0  # Exit at 2x credit loss
    time_exit_dte: int = 1  # Alert to exit 1 day before expiration

    # Alerts
    alert_to_console: bool = True
    alert_to_file: bool = True
    alert_to_email: bool = False  # Disabled by default
    email_recipient: Optional[str] = None

    # Dashboard
    dashboard_enabled: bool = True
    dashboard_port: int = 5001
    dashboard_refresh_seconds: int = 30

    def __post_init__(self):
        """Set default values that require instance creation."""
        if self.preferred_entry_hours is None:
            self.preferred_entry_hours = [7, 8]  # 07:00-08:59 PST

        # Ensure directories exist
        self.positions_file.parent.mkdir(parents=True, exist_ok=True)
        self.alerts_log.parent.mkdir(parents=True, exist_ok=True)
        self.dashboard_data.parent.mkdir(parents=True, exist_ok=True)


# Global config instance
CONFIG = ContinuousConfig()
