"""Shared tier definitions for tiered portfolio v2.

Used by both run_tiered_backtest_v2.py and the live advisor so that
tier definitions, risk limits, and theta params are a single source of truth.
"""

# ---------------------------------------------------------------------------
# Risk limits
# ---------------------------------------------------------------------------
MAX_RISK_PER_TRADE = 50_000
DAILY_BUDGET = 500_000

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
MAX_TRADES_PER_WINDOW = 2       # max trades per rolling window
TRADE_WINDOW_MINUTES = 10       # rolling window size in minutes

# ---------------------------------------------------------------------------
# Tier definitions (priority-ordered)
# ---------------------------------------------------------------------------
# Priority order: 0DTE first (highest ROI/risk), then ascending DTE, EOD last
# Rule: DTE < 3 requires >= P90 (except rolls which can target any DTE).
TIERS = [
    # Intraday tiers
    {"dte": 0,  "percentile": 95, "spread_width": 50, "directional": "pursuit",     "eod_threshold": None, "label": "dte0_p95",      "entry_start": "14:30", "entry_end": "17:30", "priority": 1},
    {"dte": 1,  "percentile": 90, "spread_width": 50, "directional": "pursuit",     "eod_threshold": None, "label": "dte1_p90",      "entry_start": "14:30", "entry_end": "17:30", "priority": 2},
    {"dte": 2,  "percentile": 90, "spread_width": 50, "directional": "pursuit",     "eod_threshold": None, "label": "dte2_p90",      "entry_start": "14:30", "entry_end": "17:30", "priority": 3},
    {"dte": 3,  "percentile": 80, "spread_width": 30, "directional": "pursuit",     "eod_threshold": None, "label": "dte3_p80",      "entry_start": "14:30", "entry_end": "17:30", "priority": 4},
    {"dte": 5,  "percentile": 75, "spread_width": 30, "directional": "pursuit",     "eod_threshold": None, "label": "dte5_p75",      "entry_start": "14:30", "entry_end": "17:30", "priority": 5},
    {"dte": 10, "percentile": 90, "spread_width": 50, "directional": "pursuit",     "eod_threshold": None, "label": "dte10_p90",     "entry_start": "14:30", "entry_end": "17:30", "priority": 6},
    # EOD tiers (all >= P90, evaluated at 3:45 PM ET)
    {"dte": 1,  "percentile": 90, "spread_width": 50, "directional": "pursuit_eod", "eod_threshold": 0.01, "label": "dte1_p90_eod",  "entry_start": "14:30", "entry_end": "20:00", "priority": 7},
    {"dte": 2,  "percentile": 90, "spread_width": 50, "directional": "pursuit_eod", "eod_threshold": 0.01, "label": "dte2_p90_eod",  "entry_start": "14:30", "entry_end": "20:00", "priority": 8},
    {"dte": 3,  "percentile": 90, "spread_width": 50, "directional": "pursuit_eod", "eod_threshold": 0.01, "label": "dte3_p90_eod",  "entry_start": "14:30", "entry_end": "20:00", "priority": 9},
]

# ---------------------------------------------------------------------------
# Theta decay exit params (by DTE)
# ---------------------------------------------------------------------------
THETA_PARAMS_0DTE = {
    "ahead": 0.35,
    "min_decay": 0.60,
    "cut_behind": 0.50,
    "cut_min_time": 0.70,
}

THETA_PARAMS_MULTI_DAY = {
    "ahead": 0.25,
    "min_decay": 0.50,
    "cut_behind": 0.40,
    "cut_min_time": 0.60,
}

# ---------------------------------------------------------------------------
# Strategy params shared between backtest and live advisor
# ---------------------------------------------------------------------------
STRATEGY_DEFAULTS = {
    "lookback": 120,
    "option_types": ["put", "call"],
    "interval_minutes": 10,
    "num_contracts": 1,
    "profit_target_0dte": 0.75,
    "profit_target_multiday": 0.50,
    "min_roi_per_day": 0.025,
    "min_credit": 0.75,
    "min_total_credit": 0,
    "min_credit_per_point": 0,
    "max_contracts": 0,
    "use_mid": False,
    "min_volume": 2,
    "stop_loss_multiplier": 0,
    "roll_enabled": True,
    "max_rolls": 2,
    "roll_check_start_utc": "18:00",
    "roll_proximity_pct": 0.005,
    "contract_sizing": "max_budget",
}

# Unique DTE and percentile values across all tiers
ALL_DTES = sorted(set(t["dte"] for t in TIERS))
ALL_PERCENTILES = sorted(set(t["percentile"] for t in TIERS))

BACKTEST_ID = "tiered_portfolio_v2"
