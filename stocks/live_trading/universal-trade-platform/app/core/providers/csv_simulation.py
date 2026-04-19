"""CSV-backed simulation provider — replays historical equity + options data.

Registered as ``Broker.IBKR`` so every existing UTP code path works unchanged.
Data is loaded entirely into memory at ``connect()`` time (files are typically
5-20 MB total) and then served by snapping to the nearest timestamp at or
before ``SimulationClock.sim_time``.
"""

from __future__ import annotations

import csv
import logging
import uuid
from bisect import bisect_right
from datetime import date, datetime, timezone
from pathlib import Path
from typing import ClassVar

from app.core.provider import BrokerProvider
from app.models import (
    AccountBalances,
    Broker,
    EquityOrder,
    MultiLegOrder,
    OptionAction,
    OrderResult,
    OrderStatus,
    Position,
    Quote,
)
from app.services.simulation_clock import get_sim_clock

logger = logging.getLogger(__name__)


def _parse_ts(raw: str) -> datetime:
    """Parse a CSV timestamp string into a tz-aware UTC datetime."""
    raw = raw.strip()
    # Try ISO format first (handles both 'T' and space separators)
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S.%f%z",
    ):
        try:
            return datetime.strptime(raw, fmt).astimezone(timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {raw!r}")


def _safe_float(val: str | None, default: float = 0.0) -> float:
    """Convert a CSV cell to float, returning *default* for empty/None."""
    if val is None or val.strip() == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val: str | None, default: int = 0) -> int:
    if val is None or val.strip() == "":
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


class CSVSimulationProvider(BrokerProvider):
    """Replay historical CSV data as if it were a live broker feed.

    Registered as ``Broker.IBKR`` so the rest of UTP sees no difference.
    """

    broker: ClassVar[Broker] = Broker.IBKR

    def __init__(
        self,
        sim_date: date,
        tickers: list[str],
        equities_dir: Path,
        options_dir: Path,
        csv_exports_dir: Path | None = None,
    ) -> None:
        self.sim_date = sim_date
        self.tickers = tickers
        self.equities_dir = Path(equities_dir)
        self.options_dir = Path(options_dir)
        self.csv_exports_dir = Path(csv_exports_dir) if csv_exports_dir else None

        # equity data: ticker -> sorted list of (datetime, row_dict)
        self._equity_bars: dict[str, list[tuple[datetime, dict]]] = {}
        # pre-extracted sorted timestamps per ticker (for bisect)
        self._equity_timestamps: dict[str, list[datetime]] = {}

        # options data: ticker -> sorted list of (datetime, list[row_dict])
        # grouped by timestamp so we can snap once and get all rows for that ts
        self._option_snapshots: dict[str, list[tuple[datetime, list[dict]]]] = {}
        self._option_timestamps: dict[str, list[datetime]] = {}

        # track filled order ids
        self._filled_orders: dict[str, OrderResult] = {}

    # ------------------------------------------------------------------
    # Directory / file resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_ticker_dir(base_dir: Path, ticker: str) -> Path | None:
        """Try ``TICKER``, ``I:TICKER`` directory names (symlink handling)."""
        for candidate in (ticker, f"I:{ticker}"):
            d = base_dir / candidate
            if d.is_dir():
                return d
        return None

    def _equity_csv_path(self, ticker: str) -> Path | None:
        d = self._resolve_ticker_dir(self.equities_dir, ticker)
        if d is None:
            return None
        date_str = self.sim_date.isoformat()
        # files may use the directory-name ticker (e.g. I:SPX) or
        # the resolved name when accessed through a symlink (SPX → I:SPX)
        resolved_name = d.resolve().name
        for name_ticker in (d.name, ticker, resolved_name):
            p = d / f"{name_ticker}_equities_{date_str}.csv"
            if p.is_file():
                return p
        return None

    def _options_csv_path(self, ticker: str) -> Path | None:
        d = self._resolve_ticker_dir(self.options_dir, ticker)
        if d is None:
            return None
        date_str = self.sim_date.isoformat()
        resolved_name = d.resolve().name
        for name_ticker in (d.name, ticker, resolved_name):
            p = d / f"{name_ticker}_options_{date_str}.csv"
            if p.is_file():
                return p
        return None

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------

    def _load_equity_csv(self, ticker: str) -> None:
        path = self._equity_csv_path(ticker)
        if path is None:
            logger.warning("No equity CSV for %s on %s", ticker, self.sim_date)
            self._equity_bars[ticker] = []
            self._equity_timestamps[ticker] = []
            return

        rows: list[tuple[datetime, dict]] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = _parse_ts(row["timestamp"])
                except (ValueError, KeyError):
                    continue
                rows.append((ts, row))

        rows.sort(key=lambda x: x[0])
        self._equity_bars[ticker] = rows
        self._equity_timestamps[ticker] = [r[0] for r in rows]
        logger.info("Loaded %d equity bars for %s from %s", len(rows), ticker, path.name)

    def _load_options_csv(self, ticker: str) -> None:
        path = self._options_csv_path(ticker)
        if path is None:
            logger.warning("No options CSV for %s on %s", ticker, self.sim_date)
            self._option_snapshots[ticker] = []
            self._option_timestamps[ticker] = []
            return

        # Group rows by timestamp
        ts_groups: dict[datetime, list[dict]] = {}
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = _parse_ts(row["timestamp"])
                except (ValueError, KeyError):
                    continue
                ts_groups.setdefault(ts, []).append(row)

        snapshots = sorted(ts_groups.items(), key=lambda x: x[0])
        self._option_snapshots[ticker] = snapshots
        self._option_timestamps[ticker] = [s[0] for s in snapshots]
        total_rows = sum(len(rows) for _, rows in snapshots)
        logger.info(
            "Loaded %d option rows (%d timestamps) for %s from %s",
            total_rows,
            len(snapshots),
            ticker,
            path.name,
        )

    # ------------------------------------------------------------------
    # Timestamp snapping helpers
    # ------------------------------------------------------------------

    def _find_equity_bar(self, ticker: str, ts: datetime) -> dict | None:
        """Return the equity bar at the largest timestamp <= *ts*."""
        timestamps = self._equity_timestamps.get(ticker, [])
        if not timestamps:
            return None
        idx = bisect_right(timestamps, ts) - 1
        if idx < 0:
            return None
        return self._equity_bars[ticker][idx][1]

    def _find_option_rows(self, ticker: str, ts: datetime) -> list[dict]:
        """Return all option rows at the nearest timestamp <= *ts*."""
        timestamps = self._option_timestamps.get(ticker, [])
        if not timestamps:
            return []
        idx = bisect_right(timestamps, ts) - 1
        if idx < 0:
            return []
        return self._option_snapshots[ticker][idx][1]

    # ------------------------------------------------------------------
    # BrokerProvider ABC implementation
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Load all CSV data for the configured tickers and sim_date."""
        for ticker in self.tickers:
            self._load_equity_csv(ticker)
            self._load_options_csv(ticker)
        logger.info(
            "CSVSimulationProvider connected: date=%s, tickers=%s",
            self.sim_date,
            self.tickers,
        )

    async def disconnect(self) -> None:
        """No-op."""

    async def get_quote(self, symbol: str) -> Quote:
        clock = get_sim_clock()
        ts = clock.sim_time if clock else datetime.now(timezone.utc)

        bar = self._find_equity_bar(symbol, ts)
        if bar is None:
            return Quote(
                symbol=symbol,
                bid=0.0,
                ask=0.0,
                last=0.0,
                volume=0,
                timestamp=ts,
                source="simulation",
                quote_age_seconds=0.0,
                quote_source="simulation",
            )

        return Quote(
            symbol=symbol,
            bid=_safe_float(bar.get("low")),
            ask=_safe_float(bar.get("high")),
            last=_safe_float(bar.get("close")),
            volume=_safe_int(bar.get("volume")),
            timestamp=ts,
            source="simulation",
            quote_age_seconds=0.0,
            quote_source="simulation",
        )

    async def get_option_quotes(
        self,
        symbol: str,
        expiration: str | None = None,
        option_type: str | None = None,
        strike_min: float | None = None,
        strike_max: float | None = None,
    ) -> list[dict]:
        """Return option quotes for *symbol* filtered by the given criteria.

        Not part of the BrokerProvider ABC but used via ``hasattr`` by the
        market_data module.
        """
        clock = get_sim_clock()
        ts = clock.sim_time if clock else datetime.now(timezone.utc)

        rows = self._find_option_rows(symbol, ts)
        if not rows:
            return []

        results: list[dict] = []
        for row in rows:
            row_type = row.get("type", "").lower()  # "put" or "call"
            row_strike = _safe_float(row.get("strike"))

            # Filter by option type
            if option_type is not None:
                if row_type != option_type.lower():
                    continue

            # Filter by expiration
            if expiration is not None:
                row_exp = row.get("expiration", "")
                if row_exp != expiration:
                    continue

            # Filter by strike range
            if strike_min is not None and row_strike < strike_min:
                continue
            if strike_max is not None and row_strike > strike_max:
                continue

            bid = _safe_float(row.get("bid"))
            ask = _safe_float(row.get("ask"))
            last = (bid + ask) / 2 if (bid + ask) > 0 else _safe_float(row.get("day_close"))

            greeks: dict = {}
            for key in ("delta", "gamma", "theta", "vega", "implied_volatility"):
                val = row.get(key, "")
                if val and val.strip():
                    greeks[key if key != "implied_volatility" else "iv"] = _safe_float(val)

            results.append(
                {
                    "symbol": row.get("ticker", ""),
                    "strike": row_strike,
                    "bid": bid,
                    "ask": ask,
                    "last": last,
                    "volume": _safe_int(row.get("volume")),
                    "open_interest": 0,
                    "greeks": greeks,
                    "type": row_type,
                    "expiration": row.get("expiration", ""),
                }
            )

        return results

    async def get_option_chain(self, symbol: str) -> dict:
        """Scan loaded options to find unique expirations and strikes."""
        clock = get_sim_clock()
        ts = clock.sim_time if clock else datetime.now(timezone.utc)

        rows = self._find_option_rows(symbol, ts)
        expirations: set[str] = set()
        strikes: set[float] = set()
        for row in rows:
            exp = row.get("expiration", "")
            if exp:
                expirations.add(exp)
            strike = _safe_float(row.get("strike"))
            if strike > 0:
                strikes.add(strike)

        return {
            "expirations": sorted(expirations),
            "strikes": sorted(strikes),
        }

    async def execute_multi_leg_order(self, order: MultiLegOrder) -> OrderResult:
        """Instant fill using bid/ask from the option CSV.

        SELL legs fill at bid, BUY legs fill at ask. Net credit is positive
        when the spread is a credit spread.
        """
        clock = get_sim_clock()
        ts = clock.sim_time if clock else datetime.now(timezone.utc)

        net_price = 0.0
        leg_details: list[dict] = []

        for leg in order.legs:
            ticker = leg.symbol
            rows = self._find_option_rows(ticker, ts)

            # Find the matching option row by strike + type
            leg_price = 0.0
            for row in rows:
                row_strike = _safe_float(row.get("strike"))
                row_type = row.get("type", "").lower()
                if (
                    abs(row_strike - leg.strike) < 0.01
                    and row_type == leg.option_type.value.lower()
                ):
                    bid = _safe_float(row.get("bid"))
                    ask = _safe_float(row.get("ask"))
                    if leg.action in (OptionAction.SELL_TO_OPEN, OptionAction.SELL_TO_CLOSE):
                        leg_price = bid
                    else:
                        leg_price = ask
                    break

            if leg.action in (OptionAction.SELL_TO_OPEN, OptionAction.SELL_TO_CLOSE):
                net_price += leg_price
            else:
                net_price -= leg_price

            leg_details.append(
                {
                    "symbol": leg.symbol,
                    "strike": leg.strike,
                    "action": leg.action.value,
                    "price": leg_price,
                }
            )

        order_id = str(uuid.uuid4())
        result = OrderResult(
            order_id=order_id,
            broker=Broker.IBKR,
            status=OrderStatus.FILLED,
            message="Simulation fill",
            filled_price=round(net_price, 4),
            filled_quantity=order.quantity,
            extra={"legs": leg_details, "source": "csv_simulation"},
        )
        self._filled_orders[order_id] = result
        return result

    async def execute_equity_order(self, order: EquityOrder) -> OrderResult:
        """Instant fill at the current last price."""
        quote = await self.get_quote(order.symbol)
        order_id = str(uuid.uuid4())
        result = OrderResult(
            order_id=order_id,
            broker=Broker.IBKR,
            status=OrderStatus.FILLED,
            message=f"Simulation fill: {order.side.value} {order.quantity} {order.symbol}",
            filled_price=quote.last,
            filled_quantity=order.quantity,
            extra={"source": "csv_simulation"},
        )
        self._filled_orders[order_id] = result
        return result

    async def check_margin(self, order: MultiLegOrder) -> dict:
        """Simple width-based margin estimate."""
        strikes = [leg.strike for leg in order.legs]
        width = max(strikes) - min(strikes) if len(strikes) >= 2 else 0
        margin = width * 100 * order.quantity
        return {
            "init_margin": margin,
            "maint_margin": margin,
            "commission": 0.0,
        }

    async def get_positions(self) -> list[Position]:
        """No persistent positions in simulation mode."""
        return []

    async def get_order_status(self, order_id: str) -> OrderResult:
        """Return FILLED for any known order, FAILED otherwise."""
        if order_id in self._filled_orders:
            return self._filled_orders[order_id]
        return OrderResult(
            order_id=order_id,
            broker=Broker.IBKR,
            status=OrderStatus.FAILED,
            message="Unknown order",
        )

    async def get_account_balances(self) -> AccountBalances:
        return AccountBalances(
            cash=1_000_000.0,
            net_liquidation=1_000_000.0,
            buying_power=1_000_000.0,
            maint_margin_req=0.0,
            available_funds=1_000_000.0,
            broker="ibkr",
        )

    # ------------------------------------------------------------------
    # Convenience: all equity timestamps (useful for SimulationClock init)
    # ------------------------------------------------------------------

    def get_all_equity_timestamps(self) -> list[datetime]:
        """Return a merged + deduplicated sorted list of all equity timestamps."""
        all_ts: set[datetime] = set()
        for ts_list in self._equity_timestamps.values():
            all_ts.update(ts_list)
        return sorted(all_ts)
