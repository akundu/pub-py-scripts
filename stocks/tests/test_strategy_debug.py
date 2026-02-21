#!/usr/bin/env python3
"""Debug script to test time-allocated tiered strategy."""

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = CURRENT_DIR / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from credit_spread_utils.time_allocated_tiered_utils import (
    load_time_allocated_tiered_config,
)
from credit_spread_utils.max_move_utils import load_csv_data

# Load config
config_path = "scripts/json/time_allocated_tiered_config_ndx.json"
config = load_time_allocated_tiered_config(config_path)

print(f"Config loaded successfully:")
print(f"  Ticker: {config.ticker}")
print(f"  Equities Dir: {config.equities_dir}")
print(f"  Total Capital: ${config.total_capital:,.2f}")
print(f"  Hourly Windows: {len(config.hourly_windows)}")
for w in config.hourly_windows:
    print(f"    {w.label}: {w.start_hour_pst}:00-{w.end_hour_pst}:{w.end_minute_pst:02d} PST, Budget: {w.budget_pct*100:.1f}%")

print(f"\n  Slope Config:")
print(f"    Skip Slope: {config.slope_config.skip_slope}")
print(f"    Lookback Bars: {config.slope_config.lookback_bars}")
print(f"    Flatten Ratio: {config.slope_config.flatten_ratio_threshold}")

# Test loading intraday data
test_date = "2026-02-04"
print(f"\n  Testing intraday data load for {test_date}...")
intraday_df = load_csv_data(
    config.ticker, test_date, Path(config.equities_dir)
)

if intraday_df is not None:
    print(f"    ✓ Loaded {len(intraday_df)} bars")
    print(f"    Date range: {intraday_df['timestamp'].min()} to {intraday_df['timestamp'].max()}")
    print(f"    Columns: {list(intraday_df.columns)}")
else:
    print(f"    ✗ Failed to load intraday data")
    print(f"    Looking for file: equities_output/{config.ticker}/{config.ticker}_equities_{test_date}.csv")
