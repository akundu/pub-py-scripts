"""
Utility modules for analyze_credit_spread_intervals.py

This package contains refactored utility functions organized by responsibility:
- timezone_utils: Timezone handling and conversion
- price_utils: Price fetching from database
- capital_utils: Capital lifecycle management
- arg_parser: Command line argument parsing
- rate_limiter: Sliding window rate limiting for transaction throttling
- dynamic_width_utils: Dynamic spread width calculation based on strike distance
- scale_in_utils: Scale-in on breach strategy for layered entries
- delta_utils: Delta calculation and filtering for probability-based entry criteria
- tiered_investment_utils: Tiered investment strategy with per-tier N contracts and M width
- spread_builder: Spread building and option pricing
- backtest_engine: P&L calculation and profit target checking
- interval_analyzer: Core interval analysis
- metrics: Trading metrics computation and statistics reporting
- data_loader: CSV loading and binary caching
- grid_search: Grid search optimization engine
- output_formatter: Display and printing utilities
- continuous_runner: Continuous analysis mode
- time_allocated_tiered_utils: Time-allocated tiered investment strategy with hourly windows
- strategies: Strategy framework (base, registry, single_entry, scale_in, tiered, time_allocated_tiered)
- price_movement_utils: Close-to-close and time-to-close price movement analysis
- max_move_utils: Intraday extreme movement tables by 30-min time slots
- risk_gradient_utils: Risk gradient analysis with QuestDB + grid config generation
- close_predictor_gate: Close predictor risk gate for filtering/annotating unsafe spreads
- predictor_tier_adapter: Adapter between close predictor confidence and tier deployment thresholds
"""

__all__ = [
    'timezone_utils',
    'price_utils',
    'capital_utils',
    'arg_parser',
    'rate_limiter',
    'dynamic_width_utils',
    'scale_in_utils',
    'delta_utils',
    'tiered_investment_utils',
    'spread_builder',
    'backtest_engine',
    'interval_analyzer',
    'metrics',
    'data_loader',
    'grid_search',
    'output_formatter',
    'continuous_runner',
    'strategies',
    'price_movement_utils',
    'max_move_utils',
    'risk_gradient_utils',
    'close_predictor_gate',
    'time_allocated_tiered_utils',
    'predictor_tier_adapter',
]
