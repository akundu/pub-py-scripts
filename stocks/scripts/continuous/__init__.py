"""
Continuous Trading Mode - Alert-Only System

Components:
- config: Configuration settings
- market_data: Real-time market context
- opportunity_scanner: Regime-based opportunity detection
- position_tracker: Manual position management
- alert_manager: Alert notifications
- dashboard: Web dashboard
- continuous_mode: Main orchestrator

Usage:
    python scripts/continuous/continuous_mode.py --ticker NDX --trend sideways
"""

__version__ = '1.0.0'
