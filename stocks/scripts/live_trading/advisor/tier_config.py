"""Shared tier definitions for tiered portfolio v2.

All constants are derived from the canonical YAML profile
(profiles/tiered_v2.yaml) so that the backtest runner, live advisor,
and HTML report all use a single source of truth.
"""

from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Load canonical profile YAML
# ---------------------------------------------------------------------------
_PROFILE_PATH = Path(__file__).resolve().parent / "profiles" / "tiered_v2.yaml"
with open(_PROFILE_PATH, "r") as _f:
    _PROFILE = yaml.safe_load(_f)

# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------
TICKERS = _PROFILE.get("tickers", [_PROFILE.get("default_ticker", "NDX")])
DEFAULT_TICKER = _PROFILE.get("default_ticker", TICKERS[0])

# Per-ticker parameters (spread widths, max_move_cap, backtest dates, etc.)
TICKER_PARAMS = _PROFILE.get("ticker_params", {})

# ---------------------------------------------------------------------------
# Risk limits
# ---------------------------------------------------------------------------
_risk = _PROFILE.get("risk", {})
MAX_RISK_PER_TRADE = _risk.get("max_risk_per_trade", 50_000)
DAILY_BUDGET = _risk.get("daily_budget", 500_000)

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
MAX_TRADES_PER_WINDOW = _risk.get("max_trades_per_window", 2)
TRADE_WINDOW_MINUTES = _risk.get("trade_window_minutes", 10)

# ---------------------------------------------------------------------------
# Tier definitions (priority-ordered)
# ---------------------------------------------------------------------------
TIERS = []
for _t in _PROFILE.get("tiers", []):
    TIERS.append({
        "dte": _t["dte"],
        "percentile": _t["percentile"],
        "spread_width": _t["spread_width"],
        "directional": _t["directional"],
        "eod_threshold": _t.get("eod_threshold"),
        "label": _t["label"],
        "entry_start": _t.get("entry_start", "14:30"),
        "entry_end": _t.get("entry_end", "17:30"),
        "priority": _t["priority"],
    })

# ---------------------------------------------------------------------------
# Theta decay exit params (by DTE)
# ---------------------------------------------------------------------------
_theta = _PROFILE.get("theta_params", {})
_theta_0dte = _theta.get("0dte", {})
_theta_multi = _theta.get("multi_day", {})

THETA_PARAMS_0DTE = {
    "ahead": _theta_0dte.get("ahead", 0.35),
    "min_decay": _theta_0dte.get("min_decay", 0.60),
    "cut_behind": _theta_0dte.get("cut_behind", 0.50),
    "cut_min_time": _theta_0dte.get("cut_min_time", 0.70),
}

THETA_PARAMS_MULTI_DAY = {
    "ahead": _theta_multi.get("ahead", 0.25),
    "min_decay": _theta_multi.get("min_decay", 0.50),
    "cut_behind": _theta_multi.get("cut_behind", 0.40),
    "cut_min_time": _theta_multi.get("cut_min_time", 0.60),
}

# ---------------------------------------------------------------------------
# Strategy params shared between backtest and live advisor
# ---------------------------------------------------------------------------
STRATEGY_DEFAULTS = dict(_PROFILE.get("strategy_defaults", {}))

# ---------------------------------------------------------------------------
# Exit rules
# ---------------------------------------------------------------------------
EXIT_RULES = dict(_PROFILE.get("exit_rules", {}))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_spread_width(tier: dict, ticker: str) -> int:
    """Resolve the actual spread width for a tier + ticker combination.

    The tier's spread_width is defined in NDX-native units (50pt, 30pt).
    TICKER_PARAMS[ticker].spread_width_map translates to ticker-appropriate values.
    """
    tp = TICKER_PARAMS.get(ticker, {})
    width_map = tp.get("spread_width_map", {})
    # YAML loads dict keys as ints, so try both int and str
    native_width = tier["spread_width"]
    return width_map.get(native_width, width_map.get(str(native_width), native_width))


def get_ticker_param(ticker: str, key: str, default=None):
    """Get a per-ticker parameter, falling back to default."""
    return TICKER_PARAMS.get(ticker, {}).get(key, default)


# ---------------------------------------------------------------------------
# Backtest info
# ---------------------------------------------------------------------------
BACKTEST_ID = _PROFILE.get("name", "tiered_portfolio_v2")

# Unique DTE and percentile values across all tiers
ALL_DTES = sorted(set(t["dte"] for t in TIERS))
ALL_PERCENTILES = sorted(set(t["percentile"] for t in TIERS))
