"""Streaming configuration — YAML-based config for IBKR market data streaming.

Specifies which symbols to stream, persistence targets (Redis, QuestDB),
polling intervals, and market hours behavior.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# ── IBKR limits with 50% safety buffer ────────────────────────────────────────
# IBKR standard: ~100 market data lines. We cap at 50 (50% buffer).
# IBKR API: 50 msg/sec soft limit. We use 22 msg/sec (56% buffer).
# reqMktData: each subscription = 1 line. We batch tick processing.
MAX_SUBSCRIPTIONS = 50       # 50% of ~100 IBKR market data lines
RATE_LIMIT_MSG_SEC = 22.0    # 50% of 45 msg/sec (which itself has headroom to 50)
TICK_BATCH_INTERVAL = 0.5    # Seconds between tick batch processing


@dataclass
class StreamingSymbolConfig:
    """A single symbol to stream."""
    symbol: str
    sec_type: str = "IND"      # IND, STK, OPT, FOP
    exchange: str = ""          # CBOE, NASDAQ, SMART, etc. (auto-resolved if empty)


@dataclass
class StreamingConfig:
    """Full streaming configuration loaded from YAML."""

    # Symbols to subscribe
    symbols: list[StreamingSymbolConfig] = field(default_factory=list)

    # Persistence
    redis_url: str = "redis://localhost:6379/0"
    questdb_url: str = ""  # questdb://user:pass@host:port/db (empty = disabled)

    # Timing
    tick_batch_interval: float = TICK_BATCH_INTERVAL  # How often to flush ticks to persistence
    market_hours_only: bool = True  # Only stream during US market hours (9:30-16:00 ET)

    # Limits (enforced with 50% buffer)
    max_subscriptions: int = MAX_SUBSCRIPTIONS
    rate_limit_msg_sec: float = RATE_LIMIT_MSG_SEC

    # Redis channel prefix (matches polygon_realtime_streamer format)
    redis_channel_prefix: str = "realtime"

    # Enable/disable persistence targets
    redis_enabled: bool = True
    questdb_enabled: bool = False

    # WebSocket broadcast to /ws/quotes clients
    ws_broadcast_enabled: bool = True

    # Minimum seconds between Redis publishes per symbol (throttle)
    redis_publish_interval: float = 1.0

    # Hard price gate: reject any tick more than this % from previous close (0.35 = ±35%)
    close_band_pct: float = 0.35

    # Streaming mode for tick data: "auto" (ib_insync if available, else CPG websocket),
    # "websocket" (force CPG WebSocket), "polling" (force CPG snapshot polling)
    streaming_mode: str = "auto"
    cpg_poll_interval: float = 1.5  # seconds between polls (polling mode only)

    # Option quote streaming (background pre-fetch)
    option_quotes_enabled: bool = False
    # Loop-tick interval — how often the cycle wakes to (1) re-read CSV and
    # (2) check whether the IBKR overlay gate is due.  Should be ≪ greeks_interval
    # so the gate can fire close to its scheduled time.  Default 5s gives the
    # 25s greeks_interval ±2.5s of jitter at most.
    option_quotes_poll_interval: float = 5.0
    option_quotes_strike_range_pct: float = 15.0   # Legacy — used as fallback if
                                                   # ibkr_/csv_ specific ranges aren't set.
    option_quotes_num_expirations: int = 12        # Resolve enough expirations to cover up to 10 DTE

    # CSV exports as primary fast source for option quotes
    option_quotes_csv_primary: bool = True         # Use CSV exports as primary (instant bid/ask)
    option_quotes_csv_dir: str = ""                # Empty = auto-resolve ../../csv_exports/options

    # IBKR overlay cadence.  Realistic floor depends on per-call latency
    # (5s p50 / 10s p95 from utp.py latency-probe) and ibkr_max_parallel.
    # 25s + parallel=6 fits the typical p50; tail outliers may still time
    # out at 0.9 × interval and retry on the next cycle.
    option_quotes_greeks_interval: float = 25.0

    # Max concurrent IBKR option-quote fetches.  Higher = shorter cycle but
    # more pressure on the broker.  Defaults to 6 — comfortably under CPG's
    # ~5-parallel choke point with one in flight, well within TWS limits.
    # Tuning guidance:
    #   CPG (--ibkr-api rest):   6  (default; safe)
    #   TWS (default):           8–12 (more headroom)
    # If you raise this above 8, watch /market/streaming/latency for an uptick
    # in p95/p99 — that's a sign IBKR is starting to queue at the wire.
    option_quotes_ibkr_max_parallel: int = 6

    # ── Tiered fetch ─────────────────────────────────────────────────────────
    # Split what we ask from IBKR (hot, frequent, narrow) vs CSV (warm, slower,
    # broad).  IBKR option quote latency is ~5s p50 / ~10s p95 per call, so a
    # realistic IBKR cycle floor is ~60s for 3 symbols × 3 DTEs × 2 types at
    # concurrency 3.  CSV is file I/O — cheap and fast.
    option_quotes_ibkr_strike_range_pct: float = 3.0   # ±% of spot, IBKR tier
    option_quotes_csv_strike_range_pct: float = 15.0   # ±% of spot, CSV tier
    # Which DTEs to fetch from IBKR.  None = all expirations (legacy behavior).
    # Default [0, 1, 2] keeps IBKR focused on near-term where freshness matters
    # most; longer DTEs are served from CSV.
    option_quotes_ibkr_dte_list: Optional[list[int]] = None
    # Max DTE for CSV tier.  CSV loads expirations up to this many trading days
    # out (or whatever is available).  IBKR is independently capped by ibkr_dte_list.
    option_quotes_csv_dte_max: int = 10

    # ── Read-time merge ──────────────────────────────────────────────────────
    # IBKR data is preferred for any strike where the IBKR cache is fresher
    # than this threshold.  Outside that window — or on strikes IBKR doesn't
    # cover — the CSV cache fills in (subject to its own staleness gate).
    # Stale IBKR is used as last resort when CSV doesn't have the strike.
    option_quotes_ibkr_max_age_sec: float = 90.0

    # CSV staleness gate (applies during market hours only).  CSV exports
    # update on a delay; if the latest snapshot is older than this we suppress
    # the row so callers don't trade on minutes-old data.  Outside market
    # hours, CSV is served regardless of age (live data isn't being produced
    # anyway).  Set to 0 to disable the gate.
    option_quotes_csv_max_age_market_sec: float = 900.0   # 15 min

    # ── Pre/post-market window ───────────────────────────────────────────────
    # Extend the "market is active" window so the streamer fetches IBKR data
    # before/after regular hours.  Defaults give 09:20–16:10 ET — 10 min on
    # each side of regular hours (09:30–16:00 ET).
    option_quotes_premarket_minutes: int = 10
    option_quotes_postmarket_minutes: int = 10

    # ── Trade defaults (daemon-wide) ─────────────────────────────────────────
    # When set, these are applied to the global `settings` at daemon startup
    # and surfaced via GET /trade/defaults. Every LIMIT caller (CLI trade,
    # playbook, scanner trade handler) that doesn't pin its own value picks
    # these up. None = leave the current env-var default in place.
    default_order_type: Optional[str] = None            # "MARKET" | "LIMIT"
    limit_slippage_pct: Optional[float] = None          # 0..100
    limit_quote_max_age_sec: Optional[float] = None     # seconds

    def validate(self) -> list[str]:
        """Validate config. Returns list of errors (empty = valid)."""
        errors = []
        if not self.symbols:
            errors.append("No symbols configured for streaming")
        if len(self.symbols) > self.max_subscriptions:
            errors.append(
                f"Too many symbols ({len(self.symbols)}) — max {self.max_subscriptions} "
                f"(IBKR limit with 50% safety buffer)"
            )
        if self.tick_batch_interval < 0.1:
            errors.append("tick_batch_interval must be >= 0.1 seconds")
        if self.redis_enabled and not self.redis_url:
            errors.append("redis_enabled=true but no redis_url specified")
        if self.questdb_enabled and not self.questdb_url:
            errors.append("questdb_enabled=true but no questdb_url specified")
        return errors


# ── Index exchange mapping (same as ibkr.py) ─────────────────────────────────
_INDEX_EXCHANGES = {
    "SPX": "CBOE",
    "NDX": "NASDAQ",
    "RUT": "RUSSELL",
    "DJX": "CBOE",
    "VIX": "CBOE",
}


def _resolve_symbol(raw: dict | str) -> StreamingSymbolConfig:
    """Convert a raw YAML entry to StreamingSymbolConfig."""
    if isinstance(raw, str):
        symbol = raw.upper()
        # Auto-detect index vs stock
        if symbol in _INDEX_EXCHANGES:
            return StreamingSymbolConfig(
                symbol=symbol, sec_type="IND", exchange=_INDEX_EXCHANGES[symbol]
            )
        return StreamingSymbolConfig(symbol=symbol, sec_type="STK", exchange="SMART")

    # Dict form: {symbol: SPX, sec_type: IND, exchange: CBOE}
    symbol = raw.get("symbol", "").upper()
    sec_type = raw.get("sec_type", "").upper()
    exchange = raw.get("exchange", "")

    if not sec_type:
        if symbol in _INDEX_EXCHANGES:
            sec_type = "IND"
            exchange = exchange or _INDEX_EXCHANGES[symbol]
        else:
            sec_type = "STK"
            exchange = exchange or "SMART"

    return StreamingSymbolConfig(symbol=symbol, sec_type=sec_type, exchange=exchange)


def _get_trade_default(raw: dict, key: str, caster):
    """Look up `key` in either `raw["trade_defaults"][key]` or `raw[key]`.

    Returns None if neither is present. Nested form wins when both are set
    (it's the more-explicit layout).
    """
    nested = raw.get("trade_defaults") or {}
    if isinstance(nested, dict) and key in nested and nested[key] is not None:
        return caster(nested[key])
    if key in raw and raw[key] is not None:
        return caster(raw[key])
    return None


def load_streaming_config(path: str | Path) -> StreamingConfig:
    """Load streaming config from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Streaming config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    # Parse symbols
    symbols = []
    for entry in raw.get("symbols", []):
        symbols.append(_resolve_symbol(entry))

    config = StreamingConfig(
        symbols=symbols,
        redis_url=os.environ.get("REDIS_URL") or raw.get("redis_url", "redis://localhost:6379/0"),
        questdb_url=raw.get("questdb_url", ""),
        tick_batch_interval=float(raw.get("tick_batch_interval", TICK_BATCH_INTERVAL)),
        market_hours_only=raw.get("market_hours_only", True),
        max_subscriptions=int(raw.get("max_subscriptions", MAX_SUBSCRIPTIONS)),
        rate_limit_msg_sec=float(raw.get("rate_limit_msg_sec", RATE_LIMIT_MSG_SEC)),
        redis_channel_prefix=raw.get("redis_channel_prefix", "realtime"),
        redis_enabled=raw.get("redis_enabled", True),
        questdb_enabled=raw.get("questdb_enabled", bool(raw.get("questdb_url"))),
        ws_broadcast_enabled=raw.get("ws_broadcast_enabled", True),
        redis_publish_interval=float(raw.get("redis_publish_interval", 1.0)),
        close_band_pct=float(raw.get("close_band_pct", 0.35)),
        streaming_mode=raw.get("streaming_mode", "auto"),
        cpg_poll_interval=float(raw.get("cpg_poll_interval", 1.5)),
        option_quotes_enabled=raw.get("option_quotes_enabled", False),
        option_quotes_poll_interval=float(raw.get("option_quotes_poll_interval", 5.0)),
        option_quotes_strike_range_pct=float(raw.get("option_quotes_strike_range_pct", 3.0)),
        option_quotes_num_expirations=int(raw.get("option_quotes_num_expirations", 3)),
        option_quotes_csv_primary=raw.get("option_quotes_csv_primary", True),
        option_quotes_csv_dir=raw.get("option_quotes_csv_dir", ""),
        option_quotes_greeks_interval=float(raw.get("option_quotes_greeks_interval", 25.0)),
        option_quotes_ibkr_max_parallel=int(raw.get("option_quotes_ibkr_max_parallel", 6)),
        # Strike ranges — both default to 5%.  Legacy field used as fallback.
        option_quotes_ibkr_strike_range_pct=float(
            raw.get(
                "option_quotes_ibkr_strike_range_pct",
                float(raw.get("option_quotes_strike_range_pct", 3.0)),
            )
        ),
        option_quotes_csv_strike_range_pct=float(
            raw.get(
                "option_quotes_csv_strike_range_pct",
                float(raw.get("option_quotes_strike_range_pct", 15.0)),
            )
        ),
        option_quotes_ibkr_dte_list=raw.get("option_quotes_ibkr_dte_list", None),
        option_quotes_csv_dte_max=int(raw.get("option_quotes_csv_dte_max", 10)),
        option_quotes_ibkr_max_age_sec=float(
            raw.get("option_quotes_ibkr_max_age_sec", 90.0)
        ),
        option_quotes_csv_max_age_market_sec=float(
            raw.get("option_quotes_csv_max_age_market_sec", 900.0)
        ),
        option_quotes_premarket_minutes=int(
            raw.get("option_quotes_premarket_minutes", 10)
        ),
        option_quotes_postmarket_minutes=int(
            raw.get("option_quotes_postmarket_minutes", 10)
        ),
        # Trade defaults: accept either a nested `trade_defaults:` block or
        # top-level keys. Nested form is preferred (clearer scoping); top-level
        # kept for simpler configs.
        default_order_type=_get_trade_default(raw, "default_order_type", str),
        limit_slippage_pct=_get_trade_default(raw, "limit_slippage_pct", float),
        limit_quote_max_age_sec=_get_trade_default(raw, "limit_quote_max_age_sec", float),
    )

    errors = config.validate()
    if errors:
        for err in errors:
            logger.error("Streaming config error: %s", err)
        raise ValueError(f"Invalid streaming config: {'; '.join(errors)}")

    logger.info(
        "Streaming config loaded: %d symbols, redis=%s, questdb=%s, batch=%.1fs",
        len(config.symbols),
        config.redis_enabled,
        config.questdb_enabled,
        config.tick_batch_interval,
    )
    return config
