"""Load and validate advisor profile YAML files.

An AdvisorProfile captures everything the evaluator needs to run any
backtest configuration as a live advisor — ticker, risk limits, tiers,
exit rules, strategy defaults, etc.

Profiles live in scripts/live_trading/advisor/profiles/<name>.yaml
or can be loaded from an absolute path.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

PROFILES_DIR = Path(__file__).resolve().parent / "profiles"


@dataclass
class RiskConfig:
    max_risk_per_trade: float = 50_000
    daily_budget: float = 500_000
    max_trades_per_window: int = 2
    trade_window_minutes: int = 10


@dataclass
class ProviderConfig:
    equity_csv_dir: str = "equities_output"
    options_csv_dir: str = "csv_exports/options"
    options_fallback_csv_dir: str = "options_csv_output_full"
    dte_buckets: List[int] = field(default_factory=lambda: list(range(0, 12)))


@dataclass
class SignalConfig:
    name: str = "percentile_range"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExitRuleConfig:
    roll_enabled: bool = True
    max_rolls: int = 2
    roll_check_start_utc: str = "18:00"
    roll_proximity_pct: float = 0.005
    early_itm_check_utc: str = "14:00"
    max_move_cap: float = 150.0
    zero_dte_proximity_warn: float = 0.005
    profit_target_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    time_exit_utc: Optional[str] = None


@dataclass
class TierDef:
    label: str
    priority: int
    directional: str  # "pursuit", "pursuit_eod", "orb", "consecutive", "gap_fade"
    entry_start: str = "14:30"
    entry_end: str = "17:30"
    dte: Optional[int] = None
    percentile: Optional[int] = None
    spread_width: Optional[float] = None
    eod_threshold: Optional[float] = None
    percent_beyond: Optional[str] = None
    min_width: Optional[float] = None
    max_width: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdvisorProfile:
    name: str
    ticker: str  # active ticker (default_ticker or --ticker override)
    risk: RiskConfig
    providers: ProviderConfig
    signal: SignalConfig
    instrument: str
    tiers: List[TierDef]
    exit_rules: ExitRuleConfig
    strategy_defaults: Dict[str, Any]
    tickers: List[str] = field(default_factory=list)
    ticker_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def all_dtes(self) -> List[int]:
        """Unique DTEs across all tiers, sorted."""
        return sorted(set(t.dte for t in self.tiers if t.dte is not None))

    @property
    def all_percentiles(self) -> List[int]:
        """Unique percentiles across all tiers, sorted."""
        return sorted(set(t.percentile for t in self.tiers if t.percentile is not None))


def _parse_risk(raw: Dict) -> RiskConfig:
    return RiskConfig(
        max_risk_per_trade=raw.get("max_risk_per_trade", 50_000),
        daily_budget=raw.get("daily_budget", 500_000),
        max_trades_per_window=raw.get("max_trades_per_window", 2),
        trade_window_minutes=raw.get("trade_window_minutes", 10),
    )


def _parse_providers(raw: Dict) -> ProviderConfig:
    equity = raw.get("equity", {})
    options = raw.get("options", {})
    return ProviderConfig(
        equity_csv_dir=equity.get("csv_dir", "equities_output"),
        options_csv_dir=options.get("csv_dir", "csv_exports/options"),
        options_fallback_csv_dir=options.get("fallback_csv_dir", "options_csv_output_full"),
        dte_buckets=options.get("dte_buckets", list(range(0, 12))),
    )


def _parse_signal(raw: Dict) -> SignalConfig:
    return SignalConfig(
        name=raw.get("name", "percentile_range"),
        params=raw.get("params", {}),
    )


def _parse_exit_rules(raw: Dict) -> ExitRuleConfig:
    return ExitRuleConfig(
        roll_enabled=raw.get("roll_enabled", True),
        max_rolls=raw.get("max_rolls", 2),
        roll_check_start_utc=raw.get("roll_check_start_utc", "18:00"),
        roll_proximity_pct=raw.get("roll_proximity_pct", 0.005),
        early_itm_check_utc=raw.get("early_itm_check_utc", "14:00"),
        max_move_cap=raw.get("max_move_cap", 150.0),
        zero_dte_proximity_warn=raw.get("zero_dte_proximity_warn", 0.005),
        profit_target_pct=raw.get("profit_target_pct"),
        stop_loss_pct=raw.get("stop_loss_pct"),
        time_exit_utc=raw.get("time_exit_utc"),
    )


def _parse_tier(raw: Dict) -> TierDef:
    known_keys = {
        "label", "priority", "directional", "entry_start", "entry_end",
        "dte", "percentile", "spread_width", "eod_threshold",
        "percent_beyond", "min_width", "max_width",
    }
    extra = {k: v for k, v in raw.items() if k not in known_keys}
    return TierDef(
        label=raw["label"],
        priority=raw["priority"],
        directional=raw["directional"],
        entry_start=raw.get("entry_start", "14:30"),
        entry_end=raw.get("entry_end", "17:30"),
        dte=raw.get("dte"),
        percentile=raw.get("percentile"),
        spread_width=raw.get("spread_width"),
        eod_threshold=raw.get("eod_threshold"),
        percent_beyond=raw.get("percent_beyond"),
        min_width=raw.get("min_width"),
        max_width=raw.get("max_width"),
        extra=extra,
    )


def load_profile(name_or_path: str) -> AdvisorProfile:
    """Load an advisor profile from a YAML file.

    Args:
        name_or_path: Either a profile name (looked up in profiles/ dir)
                      or an absolute/relative path to a YAML file.

    Returns:
        AdvisorProfile with all fields populated.

    Raises:
        FileNotFoundError: If the profile YAML doesn't exist.
        ValueError: If required fields are missing.
    """
    path = Path(name_or_path)
    if not path.suffix:
        # Treat as a profile name — look up in profiles dir
        path = PROFILES_DIR / f"{name_or_path}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError(f"Empty profile: {path}")

    # Required fields
    name = raw.get("name")
    if not name:
        raise ValueError(f"Profile missing 'name': {path}")

    # Support both single 'ticker' and multi-ticker 'tickers'/'default_ticker'
    tickers_list = raw.get("tickers", [])
    default_ticker = raw.get("default_ticker")
    single_ticker = raw.get("ticker")

    if tickers_list:
        ticker = default_ticker or tickers_list[0]
    elif single_ticker:
        ticker = single_ticker
        tickers_list = [single_ticker]
    else:
        raise ValueError(f"Profile missing 'ticker' or 'tickers': {path}")

    ticker_params = raw.get("ticker_params", {})

    tiers_raw = raw.get("tiers")
    if not tiers_raw:
        raise ValueError(f"Profile missing 'tiers': {path}")

    # Parse sections
    risk = _parse_risk(raw.get("risk", {}))
    providers = _parse_providers(raw.get("providers", {}))
    signal = _parse_signal(raw.get("signal", {}))
    exit_rules = _parse_exit_rules(raw.get("exit_rules", {}))
    tiers = [_parse_tier(t) for t in tiers_raw]
    instrument = raw.get("instrument", "credit_spread")
    strategy_defaults = raw.get("strategy_defaults", {})

    profile = AdvisorProfile(
        name=name,
        ticker=ticker,
        risk=risk,
        providers=providers,
        signal=signal,
        instrument=instrument,
        tiers=tiers,
        exit_rules=exit_rules,
        strategy_defaults=strategy_defaults,
        tickers=tickers_list,
        ticker_params=ticker_params,
    )

    logger.info(f"Loaded profile '{name}': {ticker} (tickers: {tickers_list}), {len(tiers)} tiers")
    return profile


def list_profiles() -> List[str]:
    """List available profile names in the profiles directory."""
    if not PROFILES_DIR.exists():
        return []
    return sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))


def from_tier_config() -> AdvisorProfile:
    """Build an AdvisorProfile from the legacy tier_config.py constants.

    Used for backwards compatibility when no YAML profile exists.
    """
    from .tier_config import (
        ALL_DTES,
        ALL_PERCENTILES,
        DAILY_BUDGET,
        MAX_RISK_PER_TRADE,
        MAX_TRADES_PER_WINDOW,
        STRATEGY_DEFAULTS,
        TIERS,
        TRADE_WINDOW_MINUTES,
    )

    risk = RiskConfig(
        max_risk_per_trade=MAX_RISK_PER_TRADE,
        daily_budget=DAILY_BUDGET,
        max_trades_per_window=MAX_TRADES_PER_WINDOW,
        trade_window_minutes=TRADE_WINDOW_MINUTES,
    )

    providers = ProviderConfig(
        equity_csv_dir="equities_output",
        options_csv_dir="csv_exports/options",
        options_fallback_csv_dir="options_csv_output_full",
        dte_buckets=list(range(0, 12)),
    )

    signal = SignalConfig(
        name="percentile_range",
        params={
            "lookback": STRATEGY_DEFAULTS["lookback"],
            "percentiles": ALL_PERCENTILES,
            "dte_windows": ALL_DTES,
        },
    )

    exit_rules = ExitRuleConfig(
        roll_enabled=STRATEGY_DEFAULTS.get("roll_enabled", True),
        max_rolls=STRATEGY_DEFAULTS.get("max_rolls", 2),
        roll_check_start_utc=STRATEGY_DEFAULTS.get("roll_check_start_utc", "18:00"),
        roll_proximity_pct=STRATEGY_DEFAULTS.get("roll_proximity_pct", 0.005),
    )

    tiers = []
    for t in TIERS:
        tiers.append(TierDef(
            label=t["label"],
            priority=t["priority"],
            directional=t["directional"],
            entry_start=t["entry_start"],
            entry_end=t["entry_end"],
            dte=t["dte"],
            percentile=t["percentile"],
            spread_width=t["spread_width"],
            eod_threshold=t["eod_threshold"],
        ))

    from .tier_config import TICKERS as _TICKERS, DEFAULT_TICKER, TICKER_PARAMS as _TP

    return AdvisorProfile(
        name="tiered_portfolio_v2",
        ticker=DEFAULT_TICKER,
        risk=risk,
        providers=providers,
        signal=signal,
        instrument="credit_spread",
        tiers=tiers,
        exit_rules=exit_rules,
        strategy_defaults=STRATEGY_DEFAULTS,
        tickers=_TICKERS,
        ticker_params=_TP,
    )
