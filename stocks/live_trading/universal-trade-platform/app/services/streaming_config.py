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
    option_quotes_poll_interval: float = 15.0      # Seconds between CSV read cycles
    option_quotes_strike_range_pct: float = 3.0
    option_quotes_num_expirations: int = 6         # Cover DTE 0 through 5

    # CSV exports as primary fast source for option quotes
    option_quotes_csv_primary: bool = True         # Use CSV exports as primary (instant bid/ask)
    option_quotes_csv_dir: str = ""                # Empty = auto-resolve ../../csv_exports/options
    option_quotes_greeks_interval: float = 60.0    # Seconds between IBKR fetches (prices + greeks)

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
        option_quotes_poll_interval=float(raw.get("option_quotes_poll_interval", 2.0)),
        option_quotes_strike_range_pct=float(raw.get("option_quotes_strike_range_pct", 3.0)),
        option_quotes_num_expirations=int(raw.get("option_quotes_num_expirations", 3)),
        option_quotes_csv_primary=raw.get("option_quotes_csv_primary", True),
        option_quotes_csv_dir=raw.get("option_quotes_csv_dir", ""),
        option_quotes_greeks_interval=float(raw.get("option_quotes_greeks_interval", 60.0)),  # IBKR overlay interval
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
