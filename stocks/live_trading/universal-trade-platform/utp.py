#!/usr/bin/env python3
"""
UTP — Universal Trade Platform CLI

One command for everything: portfolio, quotes, margin checks, trading,
playbooks, status, reconciliation, readiness testing, and the API server.

Subcommands:
  portfolio     Show current positions, P&L, and account summary
  quote         Get a real-time quote for one or more symbols
  options       Show option chain strikes and quotes (calls/puts)
  margin        Check margin/cost for a hypothetical trade (no execution)
  trade         Execute a trade (equity, option, credit-spread, debit-spread, iron-condor)
  playbook      Execute or validate a YAML trade playbook
  status        Dashboard of active positions, pending orders, recent closes
  orders        Show working (open) orders at the broker
  cancel        Cancel a working order
  trades        Today's trades with order/position IDs and details
  reconcile     Compare system vs broker positions
  readiness     Test IBKR connectivity and trade-type support
  server        Start the REST API server
  repl          Interactive REPL connected to daemon
  journal       View trade history and ledger entries
  performance   Show performance metrics (win rate, Sharpe, drawdown)

Execution modes (for trade/playbook/margin/quote/reconcile/readiness):
  --dry-run     Use stub providers, no broker connection (default)
  --paper       Connect to IBKR paper account (port 7497)
  --live        Connect to IBKR live account (port 7496) — requires confirmation

Usage:
  python utp.py portfolio
  python utp.py quote SPY AAPL
  python utp.py margin credit-spread --symbol RUT --short-strike 2550 \\
    --long-strike 2575 --option-type CALL --expiration 2026-03-16
  python utp.py trade equity --symbol SPY --side BUY --quantity 1
  python utp.py trade credit-spread --symbol SPX --short-strike 5500 \\
    --long-strike 5475 --option-type PUT --expiration 2026-03-20 \\
    --quantity 1 --net-price 3.50 --paper
  python utp.py trade --validate-all
  python utp.py playbook execute playbooks/example_mixed.yaml
  python utp.py playbook validate playbooks/example_mixed.yaml
  python utp.py status
  python utp.py options RUT --type CALL --expiration 2026-03-16 --live
  python utp.py options SPX --list-expirations --paper
  python utp.py reconcile
  python utp.py reconcile --flush --show --live
  python utp.py readiness --symbol SPX
  python utp.py server --server-port 8000
  python utp.py daemon --paper                     # Long-running daemon (IBKR + HTTP + bg loops)
  python utp.py daemon --server-port 9000 --paper   # Custom API port
  python utp.py journal --days 7
  python utp.py performance --days 30
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

# Python 3.14 compat
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# ── Display helpers ──────────────────────────────────────────────────────────

def _color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def _print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def _print_section(title: str) -> None:
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def _print_step(title: str, status: str, detail: str = "") -> None:
    icon = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP", "warn": "WARN"}[status]
    color_code = {"pass": "92", "fail": "91", "skip": "93", "warn": "93"}[status]
    print(f"  [{_color(icon, color_code)}] {title}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"         {line}")


def _next_trading_day() -> str:
    """Return the next weekday as YYYY-MM-DD."""
    d = date.today() + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d.strftime("%Y-%m-%d")


# Default spread widths per symbol
_DEFAULT_WIDTHS = {"SPX": 25, "NDX": 50, "RUT": 5, "DJX": 25, "TQQQ": 1, "SPY": 1, "QQQ": 1}

# Index -> equity proxy (indices have no equity contracts)
_EQUITY_PROXY = {"SPX": "SPY", "NDX": "QQQ", "RUT": "IWM"}


# ── HTTP Client Library ─────────────────────────────────────────────────────


class TradingClient:
    """Async HTTP client for UTP daemon server.

    Usage::

        async with TradingClient("http://localhost:8000") as client:
            positions = await client.get_positions()
            result = await client.trade_credit_spread(
                symbol="SPX", short_strike=5500, long_strike=5475,
                option_type="PUT", expiration="2026-03-20",
                quantity=1, net_price=3.50,
            )

    Works from any machine on the LAN without auth (if trust_local_network is enabled).
    For external access, pass api_key to the constructor.
    """

    def __init__(self, server_url: str = "http://localhost:8000", api_key: str | None = None) -> None:
        self._url = server_url
        self._api_key = api_key
        self._client = None

    async def connect(self) -> None:
        import httpx
        headers = {}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        self._client = httpx.AsyncClient(base_url=self._url, headers=headers, timeout=30.0)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()

    def _ensure_connected(self):
        if not self._client:
            raise RuntimeError("Not connected. Use 'async with TradingClient(...)' or call connect() first.")

    async def _get(self, path: str, **params) -> dict | list:
        self._ensure_connected()
        resp = await self._client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    async def _post(self, path: str, **json_data) -> dict:
        self._ensure_connected()
        resp = await self._client.post(path, json=json_data)
        resp.raise_for_status()
        return resp.json()

    # ── Health ──
    async def health(self) -> dict:
        """Check daemon health."""
        return await self._get("/health")

    # ── Trading ──
    async def trade_credit_spread(
        self,
        symbol: str,
        short_strike: float,
        long_strike: float,
        option_type: str,
        expiration: str,
        quantity: int = 1,
        net_price: float | None = None,
    ) -> dict:
        """Execute a credit spread trade. Returns order result."""
        legs = [
            {
                "symbol": symbol,
                "expiration": expiration,
                "strike": short_strike,
                "option_type": option_type.upper(),
                "action": "SELL_TO_OPEN",
                "quantity": 1,
            },
            {
                "symbol": symbol,
                "expiration": expiration,
                "strike": long_strike,
                "option_type": option_type.upper(),
                "action": "BUY_TO_OPEN",
                "quantity": 1,
            },
        ]
        payload = {
            "multi_leg_order": {
                "broker": "ibkr",
                "legs": legs,
                "order_type": "LIMIT" if net_price else "MARKET",
                "net_price": net_price,
                "quantity": quantity,
            }
        }
        self._ensure_connected()
        resp = await self._client.post("/trade/execute", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def trade_equity(
        self,
        symbol: str,
        side: str,
        quantity: int,
        limit_price: float | None = None,
    ) -> dict:
        """Execute an equity trade. Returns order result."""
        payload = {
            "equity_order": {
                "broker": "ibkr",
                "symbol": symbol,
                "side": side.upper(),
                "quantity": quantity,
                "order_type": "LIMIT" if limit_price else "MARKET",
                "limit_price": limit_price,
            }
        }
        self._ensure_connected()
        resp = await self._client.post("/trade/execute", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def close_position(
        self,
        position_id: str,
        quantity: int | None = None,
        net_price: float = 0.05,
    ) -> dict:
        """Close a position by ID."""
        payload = {"position_id": position_id, "net_price": net_price}
        if quantity is not None:
            payload["quantity"] = quantity
        return await self._post("/trade/close", **payload)

    # ── Queries ──
    async def get_positions(self) -> list:
        """Get all open positions."""
        data = await self._get("/dashboard/summary")
        return data.get("active_positions", [])

    async def get_quote(self, symbol: str) -> dict:
        """Get a real-time quote."""
        return await self._get(f"/market/quote/{symbol.upper()}")

    async def get_portfolio_summary(self) -> dict:
        """Get full portfolio summary."""
        return await self._get("/dashboard/summary")

    async def get_trades(self, days: int = 0, include_all: bool = False) -> list:
        """Get trade history."""
        params = {}
        if days:
            params["days"] = days
        if include_all:
            params["include_all"] = "true"
        return await self._get("/account/trades", **params)

    async def get_orders(self) -> list:
        """Get open/working orders."""
        return await self._get("/account/orders")

    async def cancel_order(self, order_id: str) -> dict:
        """Cancel a working order."""
        return await self._post("/account/cancel", order_id=order_id)

    async def get_performance(self) -> dict:
        """Get performance metrics."""
        return await self._get("/dashboard/performance")

    async def get_options(self, symbol: str, **kwargs) -> dict:
        """Get option chain data."""
        return await self._get(f"/market/options/{symbol.upper()}", **kwargs)

    # ── Advisor ──
    async def get_advisor_recommendations(self) -> dict:
        """Get current advisor entry/exit recommendations."""
        return await self._get("/dashboard/advisor/recommendations")

    async def get_advisor_status(self) -> dict:
        """Get advisor status (active, profile, counts)."""
        return await self._get("/dashboard/advisor/status")

    async def confirm_advisor_trade(self, priority: int) -> dict:
        """Confirm an advisor trade recommendation by priority number."""
        self._ensure_connected()
        resp = await self._client.post("/trade/advisor/confirm", params={"priority": priority})
        resp.raise_for_status()
        return resp.json()


class TradingClientSync:
    """Synchronous wrapper around TradingClient.

    Usage::

        client = TradingClientSync("http://localhost:8000")
        client.connect()
        positions = client.get_positions()
        client.disconnect()
    """

    def __init__(self, server_url: str = "http://localhost:8000", api_key: str | None = None) -> None:
        self._async_client = TradingClient(server_url, api_key)
        self._loop = None

    def _get_loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def connect(self) -> None:
        self._get_loop().run_until_complete(self._async_client.connect())

    def disconnect(self) -> None:
        self._get_loop().run_until_complete(self._async_client.disconnect())

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
        if self._loop and not self._loop.is_closed():
            self._loop.close()

    def health(self) -> dict:
        return self._get_loop().run_until_complete(self._async_client.health())

    def trade_credit_spread(self, **kwargs) -> dict:
        return self._get_loop().run_until_complete(self._async_client.trade_credit_spread(**kwargs))

    def trade_equity(self, **kwargs) -> dict:
        return self._get_loop().run_until_complete(self._async_client.trade_equity(**kwargs))

    def close_position(self, position_id: str, **kwargs) -> dict:
        return self._get_loop().run_until_complete(self._async_client.close_position(position_id, **kwargs))

    def get_positions(self) -> list:
        return self._get_loop().run_until_complete(self._async_client.get_positions())

    def get_quote(self, symbol: str) -> dict:
        return self._get_loop().run_until_complete(self._async_client.get_quote(symbol))

    def get_portfolio_summary(self) -> dict:
        return self._get_loop().run_until_complete(self._async_client.get_portfolio_summary())

    def get_trades(self, **kwargs) -> list:
        return self._get_loop().run_until_complete(self._async_client.get_trades(**kwargs))

    def get_orders(self) -> list:
        return self._get_loop().run_until_complete(self._async_client.get_orders())

    def cancel_order(self, order_id: str) -> dict:
        return self._get_loop().run_until_complete(self._async_client.cancel_order(order_id))

    def get_performance(self) -> dict:
        return self._get_loop().run_until_complete(self._async_client.get_performance())

    def get_options(self, symbol: str, **kwargs) -> dict:
        return self._get_loop().run_until_complete(self._async_client.get_options(symbol, **kwargs))


# Daemon shared state — populated by background tasks, read by HTTP endpoints
_daemon_state: dict = {
    "advisor_entries": [],
    "advisor_exits": [],
    "advisor_profile": None,
    "advisor_last_eval": None,
}


# ── Mode / service initialization ────────────────────────────────────────────

def _get_mode(args) -> str:
    """Return execution mode string.

    Checks explicit flags first (--live, --paper, --dry-run), then
    falls back to the subcommand's _default_mode (set by _add_connection_args).
    """
    if getattr(args, "live", False):
        return "live"
    if getattr(args, "paper", False):
        return "paper"
    if getattr(args, "dry_run", False):
        return "dry-run"
    return getattr(args, "_default_mode", "dry-run")


def _mode_label(mode: str) -> str:
    color = {"live": "91", "paper": "93", "dry-run": "92"}[mode]
    return _color(mode.upper(), color)


def _resolve_data_dir(base_dir: str, mode: str) -> Path:
    """Return a mode-specific data directory.

    Live and paper IBKR accounts are completely separate, so their
    positions, ledger entries, and caches must not be mixed.

    - ``live``    → ``<base>/live/``
    - ``paper``   → ``<base>/paper/``
    - ``dry-run`` → ``<base>/`` (unchanged)
    """
    base = Path(base_dir)
    if mode in ("live", "paper"):
        return base / mode
    return base


def _detect_server(args) -> str | None:
    """Return server URL if --server given or daemon is running, else None."""
    server = getattr(args, "server", None)
    if server:
        return server
    # Auto-detect: try default daemon URL
    import urllib.request
    try:
        port = getattr(args, "server_port", 8000)
        url = f"http://localhost:{port}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=1) as resp:
            if resp.status == 200:
                return f"http://localhost:{port}"
    except Exception:
        pass
    return None


async def _try_daemon(server: str, http_func, args) -> int | None:
    """Try routing through daemon HTTP. Returns exit code on success, None to fall back.

    Only falls back to direct IBKR when --allow-fallback is set (default: off).
    Without it, connection errors report the failure and return exit code 1
    to avoid client-id conflicts with a potentially running daemon.
    """
    allow_fallback = getattr(args, "allow_fallback", False)
    try:
        return await http_func(args, server)
    except Exception as e:
        err_name = type(e).__name__
        connection_errors = ("ConnectError", "ConnectTimeout", "ConnectionRefusedError")
        is_conn_error = err_name in connection_errors or "refused" in str(e).lower()

        if is_conn_error and allow_fallback:
            print(f"  Daemon unreachable ({err_name}), falling back to direct mode...")
            return None

        if is_conn_error:
            print(f"  Daemon unreachable ({err_name}).")
            print(f"  Start the daemon or use --allow-fallback to connect directly.")
            return 1

        if "timeout" in err_name.lower():
            print(f"  Daemon request timed out ({err_name}). The daemon may be busy.")
            print(f"  Retry the command or check daemon logs.")
            return 1
        raise


def _init_read_only_services(data_dir: str = "data/utp", mode: str = "dry-run") -> None:
    """Initialize ledger + position store for read-only subcommands."""
    from app.services.ledger import init_ledger
    from app.services.position_store import init_position_store

    d = _resolve_data_dir(data_dir, mode)
    init_ledger(d)
    init_position_store(d)


async def _init_services(args) -> Optional[object]:
    """Initialize providers, ledger, and position store based on mode.

    Returns the live provider instance (if any) for later disconnect.
    """
    from app.services.ledger import init_ledger
    from app.services.position_store import init_position_store
    from app.core.provider import ProviderRegistry

    mode = _get_mode(args)
    data_dir = _resolve_data_dir(getattr(args, "data_dir", "data/utp"), mode)
    init_ledger(data_dir)
    init_position_store(data_dir)

    live_provider = None

    if mode == "dry-run":
        from app.core.providers.robinhood import RobinhoodProvider
        from app.core.providers.etrade import EtradeProvider
        from app.core.providers.ibkr import IBKRProvider

        for p in [RobinhoodProvider(), EtradeProvider(), IBKRProvider()]:
            ProviderRegistry.register(p)
            await p.connect()
    else:
        import app.config
        from app.config import Settings

        port = getattr(args, "port", None)
        if port is None:
            port = 7497 if mode == "paper" else 7496

        os.environ["IBKR_HOST"] = getattr(args, "host", "127.0.0.1")
        os.environ["IBKR_PORT"] = str(port)
        os.environ["IBKR_CLIENT_ID"] = str(getattr(args, "client_id", 10))
        os.environ["IBKR_READONLY"] = "false"
        exchange = getattr(args, "exchange", None)
        if exchange:
            os.environ["IBKR_EXCHANGE"] = exchange
        app.config.settings = Settings()

        from app.core.providers.ibkr import IBKRLiveProvider
        live_provider = IBKRLiveProvider(exchange=exchange or None)
        ProviderRegistry.register(live_provider)
        await live_provider.connect()

    return live_provider


async def _init_ibkr_readonly(args) -> object:
    """Initialize IBKR in read-only mode for margin/quote/reconcile."""
    import app.config
    from app.config import Settings
    from app.core.provider import ProviderRegistry
    from app.core.providers.ibkr import IBKRLiveProvider
    from app.services.ledger import init_ledger
    from app.services.position_store import init_position_store

    mode = _get_mode(args)
    port = getattr(args, "port", None)
    if port is None:
        port = 7497 if mode == "paper" else 7496

    os.environ["IBKR_HOST"] = getattr(args, "host", "127.0.0.1")
    os.environ["IBKR_PORT"] = str(port)
    os.environ["IBKR_CLIENT_ID"] = str(getattr(args, "client_id", 10))
    os.environ["IBKR_READONLY"] = "true"
    exchange = getattr(args, "exchange", None)
    if exchange:
        os.environ["IBKR_EXCHANGE"] = exchange
    app.config.settings = Settings()

    data_dir = _resolve_data_dir(getattr(args, "data_dir", "data/utp"), mode)
    init_ledger(data_dir)
    init_position_store(data_dir)

    host = getattr(args, "host", "127.0.0.1")
    provider = IBKRLiveProvider(exchange=exchange or None)
    ProviderRegistry.register(provider)

    try:
        await provider.connect()
    except (ConnectionRefusedError, OSError) as e:
        # Try the other port before giving up
        alt_port = 7496 if port == 7497 else 7497
        alt_label = "live" if alt_port == 7496 else "paper"
        print(f"  Connection to {host}:{port} failed.")
        print(f"  Trying {alt_label} port {alt_port}...")
        os.environ["IBKR_PORT"] = str(alt_port)
        app.config.settings = Settings()
        try:
            await provider.connect()
            print(f"  Connected to {host}:{alt_port} ({alt_label})\n")
        except (ConnectionRefusedError, OSError):
            print(f"  Connection to {host}:{alt_port} also failed.")
            print(f"\n  Make sure TWS or IB Gateway is running with API enabled.")
            print(f"  Tried ports: {port}, {alt_port}")
            raise SystemExit(1)

    return provider


async def _disconnect(provider) -> None:
    """Disconnect a live provider and clear registry."""
    from app.core.provider import ProviderRegistry
    if provider:
        await provider.disconnect()
    ProviderRegistry.clear()


# ── Trade instruction building & execution ──────────────────────────────────

def _build_instruction_from_args(subcommand: str, args) -> dict:
    """Map CLI subcommand args to a PlaybookInstruction-compatible dict."""
    if subcommand == "equity":
        return {
            "id": f"cli_equity_{args.symbol}",
            "type": "equity",
            "params": {
                "symbol": args.symbol,
                "action": args.side.upper(),
                "quantity": args.quantity,
                "order_type": args.order_type.upper(),
                "limit_price": args.limit_price,
            },
        }
    elif subcommand == "option":
        return {
            "id": f"cli_option_{args.symbol}",
            "type": "single_option",
            "params": {
                "symbol": args.symbol,
                "expiration": args.expiration,
                "strike": args.strike,
                "option_type": args.option_type.upper(),
                "action": args.action.upper(),
                "quantity": args.quantity,
                "order_type": args.order_type.upper(),
                "limit_price": args.limit_price,
            },
        }
    elif subcommand == "credit-spread":
        close = getattr(args, "close", False)
        return {
            "id": f"cli_cs_{args.symbol}",
            "type": "credit_spread" if not close else "credit_spread_close",
            "params": {
                "symbol": args.symbol,
                "expiration": args.expiration,
                "short_strike": args.short_strike,
                "long_strike": args.long_strike,
                "option_type": args.option_type.upper(),
                "quantity": args.quantity,
                "net_price": args.net_price,
            },
        }
    elif subcommand == "debit-spread":
        close = getattr(args, "close", False)
        return {
            "id": f"cli_ds_{args.symbol}",
            "type": "debit_spread" if not close else "debit_spread_close",
            "params": {
                "symbol": args.symbol,
                "expiration": args.expiration,
                "long_strike": args.long_strike,
                "short_strike": args.short_strike,
                "option_type": args.option_type.upper(),
                "quantity": args.quantity,
                "net_price": args.net_price,
            },
        }
    elif subcommand == "iron-condor":
        close = getattr(args, "close", False)
        return {
            "id": f"cli_ic_{args.symbol}",
            "type": "iron_condor" if not close else "iron_condor_close",
            "params": {
                "symbol": args.symbol,
                "expiration": args.expiration,
                "put_short": args.put_short,
                "put_long": args.put_long,
                "call_short": args.call_short,
                "call_long": args.call_long,
                "quantity": args.quantity,
                "net_price": args.net_price,
            },
        }
    else:
        raise ValueError(f"Unknown subcommand: {subcommand}")


def _get_symbol_from_instruction(instr: dict) -> str:
    """Extract the primary symbol from an instruction dict."""
    params = instr.get("params", {})
    return params.get("symbol", "") or instr.get("symbol", "")


async def _execute_single_order(
    instruction_dict: dict,
    broker_str: str,
    mode: str,
    poll_timeout: float,
    poll_interval: float,
    label: str = "",
    closing_position_id: str | None = None,
    closing_quantity: int | None = None,
) -> dict:
    """Execute one trade, track fill, verify position and ledger.

    Returns a dict with: order_result, position_id, passed (bool), checks (list).
    """
    from app.models import Broker, OrderStatus
    from app.services.playbook_service import PlaybookService, PlaybookInstruction
    from app.services.trade_service import execute_trade, await_order_fill, TERMINAL_STATUSES
    from app.services.position_store import get_position_store
    from app.services.ledger import get_ledger

    service = PlaybookService()
    broker = Broker(broker_str)
    checks = []

    # Build trade request
    # instruction_dict may have params as a nested dict (from _build_instruction_from_args)
    # or as flat keys (from legacy callers)
    if "params" in instruction_dict and isinstance(instruction_dict["params"], dict):
        raw_params = {k: v for k, v in instruction_dict["params"].items() if v is not None}
    else:
        raw_params = {k: v for k, v in instruction_dict.items() if k not in ("id", "type") and v is not None}
    instr = PlaybookInstruction(
        id=instruction_dict["id"],
        type=instruction_dict["type"],
        params=raw_params,
    )

    try:
        trade_request = service.instruction_to_trade_request(instr, broker)
    except Exception as e:
        checks.append(("Build TradeRequest", "fail", str(e)))
        return {"order_result": None, "position_id": None, "passed": False, "checks": checks}

    checks.append(("Build TradeRequest", "pass", f"type={instruction_dict['type']}"))

    # If closing an existing position, tag the request so trade_service closes it
    if closing_position_id:
        trade_request.closing_position_id = closing_position_id
        trade_request.closing_quantity = closing_quantity

    # Execute
    dry_run = mode == "dry-run"
    try:
        order_result = await execute_trade(trade_request, dry_run=dry_run)
    except Exception as e:
        checks.append(("Submit Order", "fail", str(e)))
        return {"order_result": None, "position_id": None, "passed": False, "checks": checks}

    order_id = order_result.order_id
    short_id = order_id[:8]
    checks.append(("Submit Order", "pass", f"order_id={short_id}... status={order_result.status.value}"))

    # Track fill (not for dry-run, as stubs already create the position in execute_trade)
    if not dry_run and order_result.status not in TERMINAL_STATUSES:
        print(f"         Tracking {short_id}... ", end="", flush=True)

        async def _on_update(result, elapsed):
            print(".", end="", flush=True)

        order_result = await await_order_fill(
            broker=broker,
            order_id=order_id,
            poll_interval=poll_interval,
            timeout=poll_timeout,
            on_status_update=_on_update,
        )
        print()

        if order_result.status == OrderStatus.FILLED:
            price_str = f"${order_result.filled_price:.2f}" if order_result.filled_price else "market"
            checks.append(("Fill Tracking", "pass", f"FILLED at {price_str}"))
        elif order_result.status in TERMINAL_STATUSES:
            checks.append(("Fill Tracking", "pass", f"{order_result.status.value}: {order_result.message}"))
        else:
            checks.append(("Fill Tracking", "warn",
                          f"Still {order_result.status.value} after {poll_timeout:.0f}s"))
    elif dry_run:
        checks.append(("Fill Tracking", "pass", "dry-run — instant simulated fill"))

    # Verify position in store
    store = get_position_store()
    position_id = None
    if store:
        if closing_position_id:
            # For close trades, verify the original position was updated
            all_positions = {**{p.get("position_id"): p for p in store.get_open_positions()},
                            **{p.get("position_id"): p for p in store.get_closed_positions()}}
            affected_pos = all_positions.get(closing_position_id)
            if affected_pos:
                position_id = closing_position_id
                if affected_pos.get("status") == "closed":
                    checks.append(("Position Store", "pass",
                                  f"CLOSED position_id={closing_position_id[:8]}..."))
                elif closing_quantity:
                    remaining = int(affected_pos.get("quantity", 0))
                    checks.append(("Position Store", "pass",
                                  f"PARTIAL CLOSE position_id={closing_position_id[:8]}... "
                                  f"remaining={remaining}"))
                else:
                    checks.append(("Position Store", "warn",
                                  f"Position {closing_position_id[:8]}... still open"))
            elif order_result and order_result.status == OrderStatus.FILLED:
                checks.append(("Position Store", "fail", "Original position not found after fill"))
            else:
                checks.append(("Position Store", "skip", "Order not filled — position not updated"))
        else:
            open_positions = [p for p in store.get_open_positions()]
            # Find position matching this specific order (by order_id)
            matching = [p for p in open_positions if p.get("order_id") == order_id]
            if not matching:
                # Fallback: match by symbol for legacy positions without order_id
                symbol = _get_symbol_from_instruction(instruction_dict)
                matching = [p for p in open_positions if p.get("symbol") == symbol]
            if matching:
                position_id = matching[-1].get("position_id")
                checks.append(("Position Store", "pass",
                              f"position_id={position_id[:8]}... symbol={matching[-1].get('symbol')}"))
            else:
                # For live orders that didn't fill, position won't exist
                if order_result and order_result.status == OrderStatus.FILLED:
                    checks.append(("Position Store", "fail", "Position not found after fill"))
                elif dry_run:
                    checks.append(("Position Store", "fail", "Position not found after dry-run"))
                else:
                    checks.append(("Position Store", "skip", "Order not filled — no position expected"))
    else:
        checks.append(("Position Store", "skip", "Store not initialized"))

    # Verify ledger entries
    ledger = get_ledger()
    if ledger:
        recent = await ledger.get_recent(20)
        order_entries = [e for e in recent if e.order_id == order_id]
        if order_entries:
            event_types = [e.event_type.value if hasattr(e.event_type, 'value') else str(e.event_type)
                          for e in order_entries]
            checks.append(("Ledger Entries", "pass", f"events: {', '.join(event_types)}"))
        else:
            checks.append(("Ledger Entries", "warn", "No ledger entries found for this order"))
    else:
        checks.append(("Ledger Entries", "skip", "Ledger not initialized"))

    passed = all(c[1] in ("pass", "skip", "warn") for c in checks)
    return {
        "order_result": order_result,
        "position_id": position_id,
        "passed": passed,
        "checks": checks,
    }


def _generate_safe_defaults(
    symbol: str,
    expiration: str,
    chain_data: Optional[dict],
    mode: str,
) -> list[dict]:
    """Generate safe validation orders for all 5 trade types.

    For paper/live: uses far-OTM strikes and $0.01 limit prices to avoid real fills.
    For dry-run: uses reasonable strikes (stubs fill everything).
    """
    eq_symbol = _EQUITY_PROXY.get(symbol, symbol)
    width = _DEFAULT_WIDTHS.get(symbol, 5)

    instructions = []

    # 1. Equity — buy 1 share
    eq_instr = {
        "id": "validate_equity",
        "type": "equity",
        "symbol": eq_symbol,
        "action": "BUY",
        "quantity": 1,
        "order_type": "MARKET" if mode == "dry-run" else "LIMIT",
    }
    if mode != "dry-run":
        eq_instr["limit_price"] = 0.01  # Won't fill — for validation only
    instructions.append(eq_instr)

    # Pick strikes from chain
    strikes = sorted(chain_data.get("strikes", [])) if chain_data else []
    if not strikes:
        # Use dummy strikes for dry-run
        strikes = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]

    # Far OTM: bottom 10% for puts, top 10% for calls
    low_idx = max(0, len(strikes) // 10)
    high_idx = min(len(strikes) - 1, len(strikes) * 9 // 10)

    put_strike = strikes[low_idx + 1] if low_idx + 1 < len(strikes) else strikes[0]
    call_strike = strikes[high_idx] if high_idx < len(strikes) else strikes[-1]

    # 2. Single option — buy 1 far-OTM put
    instructions.append({
        "id": "validate_single_option",
        "type": "single_option",
        "symbol": symbol,
        "expiration": expiration,
        "strike": put_strike,
        "option_type": "PUT",
        "action": "BUY_TO_OPEN",
        "quantity": 1,
        "order_type": "LIMIT",
        "limit_price": 0.01,
    })

    # 3. Credit spread — far-OTM put credit spread
    cs_short = strikes[low_idx + 2] if low_idx + 2 < len(strikes) else put_strike
    cs_long_candidates = [s for s in strikes if s <= cs_short - width]
    cs_long = cs_long_candidates[-1] if cs_long_candidates else strikes[0]
    instructions.append({
        "id": "validate_credit_spread",
        "type": "credit_spread",
        "symbol": symbol,
        "expiration": expiration,
        "short_strike": cs_short,
        "long_strike": cs_long,
        "option_type": "PUT",
        "quantity": 1,
        "net_price": 0.01,
    })

    # 4. Debit spread — far-OTM call debit spread
    ds_long = strikes[high_idx - 1] if high_idx - 1 >= 0 else call_strike
    ds_short_candidates = [s for s in strikes if s >= ds_long + width]
    ds_short = ds_short_candidates[0] if ds_short_candidates else strikes[-1]
    instructions.append({
        "id": "validate_debit_spread",
        "type": "debit_spread",
        "symbol": symbol,
        "expiration": expiration,
        "long_strike": ds_long,
        "short_strike": ds_short,
        "option_type": "CALL",
        "quantity": 1,
        "net_price": 0.01,
    })

    # 5. Iron condor — far-OTM on both sides
    ic_put_short = strikes[low_idx + 3] if low_idx + 3 < len(strikes) else cs_short
    ic_put_long_candidates = [s for s in strikes if s <= ic_put_short - width]
    ic_put_long = ic_put_long_candidates[-1] if ic_put_long_candidates else strikes[0]
    ic_call_short = strikes[high_idx - 2] if high_idx - 2 >= 0 else call_strike
    ic_call_long_candidates = [s for s in strikes if s >= ic_call_short + width]
    ic_call_long = ic_call_long_candidates[0] if ic_call_long_candidates else strikes[-1]
    instructions.append({
        "id": "validate_iron_condor",
        "type": "iron_condor",
        "symbol": symbol,
        "expiration": expiration,
        "put_short": ic_put_short,
        "put_long": ic_put_long,
        "call_short": ic_call_short,
        "call_long": ic_call_long,
        "quantity": 1,
        "net_price": 0.01,
    })

    return instructions


async def _run_validate_all(args) -> int:
    """Run all 5 trade types as a comprehensive validation."""
    mode = _get_mode(args)
    symbol = args.symbol
    expiration = args.expiration or _next_trading_day()
    broker = getattr(args, "broker", "ibkr")

    _print_header("Trade Validation — All 5 Types")
    print(f"  Mode:        {_mode_label(mode)}")
    print(f"  Broker:      {broker}")
    print(f"  Symbol:      {symbol}")
    print(f"  Expiration:  {expiration}")
    print(f"  Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Live confirmation
    if mode == "live":
        print(f"\n  {_color('WARNING: You are about to place LIVE orders!', '91')}")
        print(f"  These are validation orders with far-OTM strikes and $0.01 limits.")
        print(f"  They should NOT fill, but check your positions after.")
        confirm = input(f"\n  Type 'yes' to proceed: ")
        if confirm.strip().lower() != "yes":
            print("  Aborted.")
            return 1

    # Init services
    live_provider = await _init_services(args)

    # Fetch option chain (for paper/live to get real strikes)
    chain_data = None
    if mode != "dry-run" and broker == "ibkr" and live_provider:
        _print_section("OPTION CHAIN")
        try:
            chain_data = await live_provider.get_option_chain(symbol)
            n_exp = len(chain_data.get("expirations", []))
            n_str = len(chain_data.get("strikes", []))
            _print_step("Fetch Option Chain", "pass", f"{n_exp} expirations, {n_str} strikes")
        except Exception as e:
            _print_step("Fetch Option Chain", "fail", str(e))

    # Generate safe defaults
    instructions = _generate_safe_defaults(symbol, expiration, chain_data, mode)

    type_labels = {
        "equity": "EQUITY",
        "single_option": "SINGLE OPTION",
        "credit_spread": "CREDIT SPREAD (Bull Put)",
        "debit_spread": "DEBIT SPREAD (Call)",
        "iron_condor": "IRON CONDOR",
    }

    results = []
    position_ids = []

    for i, instr in enumerate(instructions, 1):
        instr_type = instr["type"]
        _print_section(f"TRADE TYPE {i}: {type_labels.get(instr_type, instr_type.upper())}")

        # Show order details
        detail_parts = [f"{k}={v}" for k, v in instr.items() if k not in ("id", "type") and v is not None]
        print(f"  Order: {', '.join(detail_parts)}")
        print()

        result = await _execute_single_order(
            instruction_dict=instr,
            broker_str=broker,
            mode=mode,
            poll_timeout=args.poll_timeout,
            poll_interval=args.poll_interval,
        )

        for title, status, detail in result["checks"]:
            _print_step(title, status, detail)

        if result["position_id"]:
            position_ids.append(result["position_id"])

        results.append(result)

    # Cleanup
    if args.cleanup and position_ids:
        _print_section("CLEANUP")
        await _cleanup_positions(position_ids, mode)

    # Summary
    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    print(f"\n{'=' * 70}")
    if failed == 0:
        print(f"  {_color('ALL 5 TRADE TYPES VALIDATED', '92')} ({passed} passed, {failed} failed)")
    else:
        print(f"  {_color(f'{failed} TRADE TYPE(S) FAILED', '91')} ({passed} passed, {failed} failed)")
    print(f"{'=' * 70}\n")

    # Disconnect
    await _disconnect(live_provider)
    return 1 if failed > 0 else 0


async def _cleanup_positions(position_ids: list[str], mode: str) -> None:
    """Close positions created during validation."""
    from app.services.position_store import get_position_store

    store = get_position_store()
    if not store:
        _print_step("Cleanup", "skip", "Position store not initialized")
        return

    closed = 0
    for pos_id in position_ids:
        try:
            store.close_position(pos_id, exit_price=0.0, reason="validation_cleanup")
            closed += 1
        except Exception as e:
            _print_step(f"Close {pos_id[:8]}...", "fail", str(e))

    _print_step("Cleanup", "pass", f"Closed {closed}/{len(position_ids)} positions")


# ── IBKR Readiness test functions ───────────────────────────────────────────

_READINESS_EQUITY_MAP = {"SPX": "SPY", "NDX": "QQQ", "RUT": "IWM", "DJX": "DJX"}
_READINESS_EQUITY_SYMBOLS = {"TQQQ", "SPY", "QQQ", "AAPL", "DJX"}


async def _readiness_test_equity(provider, symbol: str, step_start: int) -> list[tuple]:
    """Test equity contract qualification. Returns list of (step, title, status, detail)."""
    results = []
    eq_symbol = _READINESS_EQUITY_MAP.get(symbol, symbol)

    try:
        from ib_insync import Stock
        stock = Stock(eq_symbol, "SMART", "USD")
        qualified = await provider._qualify_contract_cached(stock)
        if qualified:
            c = qualified[0]
            results.append((
                step_start, f"Equity Qualify ({eq_symbol})", "pass",
                f"conId={c.conId}, secType={c.secType}, exchange={c.exchange}"
            ))
        else:
            results.append((step_start, f"Equity Qualify ({eq_symbol})", "fail",
                            "qualifyContracts returned empty"))
    except Exception as e:
        results.append((step_start, f"Equity Qualify ({eq_symbol})", "fail", str(e)))

    return results


async def _readiness_test_single_option(
    provider, symbol: str, expiration: str, chain_data: Optional[dict],
    step_start: int, exchange: str
) -> list[tuple]:
    """Test single option contract qualification."""
    results = []
    exp_yyyymmdd = expiration.replace("-", "")

    strike = None
    if chain_data:
        strikes = chain_data.get("strikes", [])
        if strikes:
            strike = strikes[len(strikes) // 2]

    if strike is None:
        results.append((step_start, "Single Option Qualify", "skip",
                        "No strikes available from chain"))
        return results

    try:
        from ib_insync import Option
        opt = Option(symbol, exp_yyyymmdd, strike, "P", exchange)
        qualified = await provider._qualify_contract_cached(opt)
        if qualified:
            c = qualified[0]
            results.append((step_start, f"Single Option ({strike}P)", "pass",
                            f"conId={c.conId}, localSymbol={c.localSymbol}"))
        else:
            results.append((step_start, f"Single Option ({strike}P)", "fail",
                            "qualifyContracts returned empty"))
    except Exception as e:
        results.append((step_start, f"Single Option ({strike}P)", "fail", str(e)))

    return results


async def _readiness_test_credit_spread(
    provider, symbol: str, expiration: str, chain_data: Optional[dict],
    spread_width: int, step_start: int, exchange: str,
    skip_margin: bool, quantity: int
) -> list[tuple]:
    """Test credit spread (2-leg BAG combo)."""
    results = []
    exp_yyyymmdd = expiration.replace("-", "")

    strikes = chain_data.get("strikes", []) if chain_data else []
    short_strike = None
    long_strike = None

    if strikes:
        mid_idx = len(strikes) // 2
        short_strike = strikes[mid_idx]
        target_long = short_strike - spread_width
        candidates = [s for s in strikes if s <= target_long]
        if candidates:
            long_strike = candidates[-1]

    if short_strike is None or long_strike is None:
        results.append((step_start, "Credit Spread Build", "skip",
                        "Could not determine strikes from chain"))
        return results

    short_qualified = None
    long_qualified = None

    try:
        from ib_insync import Option
        short_opt = Option(symbol, exp_yyyymmdd, short_strike, "P", exchange)
        short_qualified = await provider._qualify_contract_cached(short_opt)
        if short_qualified:
            results.append((step_start, f"Credit Spread Short ({short_strike}P)", "pass",
                            f"conId={short_qualified[0].conId}"))
        else:
            results.append((step_start, f"Credit Spread Short ({short_strike}P)", "fail",
                            "qualifyContracts returned empty"))
    except Exception as e:
        results.append((step_start, f"Credit Spread Short ({short_strike}P)", "fail", str(e)))

    try:
        from ib_insync import Option
        long_opt = Option(symbol, exp_yyyymmdd, long_strike, "P", exchange)
        long_qualified = await provider._qualify_contract_cached(long_opt)
        if long_qualified:
            results.append((step_start + 1, f"Credit Spread Long ({long_strike}P)", "pass",
                            f"conId={long_qualified[0].conId}"))
        else:
            results.append((step_start + 1, f"Credit Spread Long ({long_strike}P)", "fail",
                            "qualifyContracts returned empty"))
    except Exception as e:
        results.append((step_start + 1, f"Credit Spread Long ({long_strike}P)", "fail", str(e)))

    if short_qualified and long_qualified:
        try:
            from ib_insync import ComboLeg, Contract
            combo_legs = [
                ComboLeg(conId=short_qualified[0].conId, ratio=1, action="SELL", exchange=exchange),
                ComboLeg(conId=long_qualified[0].conId, ratio=1, action="BUY", exchange=exchange),
            ]
            Contract(symbol=symbol, secType="BAG", exchange=exchange,
                     currency="USD", comboLegs=combo_legs)
            results.append((step_start + 2, "Credit Spread BAG Combo", "pass",
                            f"secType=BAG, 2 legs, width={short_strike - long_strike}"))
        except Exception as e:
            results.append((step_start + 2, "Credit Spread BAG Combo", "fail", str(e)))

        if not skip_margin:
            try:
                from app.models import (
                    Broker, MultiLegOrder, OptionAction, OptionLeg, OptionType, OrderType,
                )
                margin_order = MultiLegOrder(
                    broker=Broker.IBKR,
                    legs=[
                        OptionLeg(symbol=symbol, expiration=expiration, strike=short_strike,
                                  option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN, quantity=1),
                        OptionLeg(symbol=symbol, expiration=expiration, strike=long_strike,
                                  option_type=OptionType.PUT, action=OptionAction.BUY_TO_OPEN, quantity=1),
                    ],
                    order_type=OrderType.LIMIT, net_price=1.00, quantity=quantity,
                )
                margin = await asyncio.wait_for(provider.check_margin(margin_order), timeout=5.0)
                if "error" in margin:
                    results.append((step_start + 3, "Credit Spread Margin", "fail", margin["error"]))
                else:
                    detail = (
                        f"Init margin: ${margin.get('init_margin', 0):,.2f}, "
                        f"Commission: ${margin.get('commission', 0):,.2f}"
                    )
                    results.append((step_start + 3, "Credit Spread Margin", "pass", detail))
            except asyncio.TimeoutError:
                results.append((step_start + 3, "Credit Spread Margin", "skip",
                                "Timed out (expected outside market hours for combos)"))
            except Exception as e:
                results.append((step_start + 3, "Credit Spread Margin", "fail", str(e)))
        else:
            results.append((step_start + 3, "Credit Spread Margin", "skip", "Skipped via --skip-margin"))
    else:
        results.append((step_start + 2, "Credit Spread BAG Combo", "skip", "Legs failed to qualify"))
        results.append((step_start + 3, "Credit Spread Margin", "skip", "Legs failed to qualify"))

    return results


async def _readiness_test_debit_spread(
    provider, symbol: str, expiration: str, chain_data: Optional[dict],
    spread_width: int, step_start: int, exchange: str
) -> list[tuple]:
    """Test debit spread (2-leg BAG combo)."""
    results = []
    exp_yyyymmdd = expiration.replace("-", "")

    strikes = chain_data.get("strikes", []) if chain_data else []
    long_strike = None
    short_strike = None

    if strikes:
        mid_idx = len(strikes) // 2
        long_strike = strikes[mid_idx]
        target_short = long_strike + spread_width
        candidates = [s for s in strikes if s >= target_short]
        if candidates:
            short_strike = candidates[0]

    if long_strike is None or short_strike is None:
        results.append((step_start, "Debit Spread Build", "skip",
                        "Could not determine strikes from chain"))
        return results

    try:
        from ib_insync import Option, ComboLeg, Contract
        long_opt = Option(symbol, exp_yyyymmdd, long_strike, "C", exchange)
        long_qualified = await provider._qualify_contract_cached(long_opt)
        short_opt = Option(symbol, exp_yyyymmdd, short_strike, "C", exchange)
        short_qualified = await provider._qualify_contract_cached(short_opt)

        if long_qualified and short_qualified:
            combo_legs = [
                ComboLeg(conId=long_qualified[0].conId, ratio=1, action="BUY", exchange=exchange),
                ComboLeg(conId=short_qualified[0].conId, ratio=1, action="SELL", exchange=exchange),
            ]
            Contract(symbol=symbol, secType="BAG", exchange=exchange,
                     currency="USD", comboLegs=combo_legs)
            results.append((step_start, f"Debit Spread ({long_strike}C/{short_strike}C)", "pass",
                            f"secType=BAG, 2 legs (BUY {long_strike}C, SELL {short_strike}C)"))
        else:
            results.append((step_start, "Debit Spread", "fail",
                            "One or both legs failed to qualify"))
    except Exception as e:
        results.append((step_start, "Debit Spread", "fail", str(e)))

    return results


async def _readiness_test_iron_condor(
    provider, symbol: str, expiration: str, chain_data: Optional[dict],
    spread_width: int, step_start: int, exchange: str
) -> list[tuple]:
    """Test iron condor (4-leg BAG combo)."""
    results = []
    exp_yyyymmdd = expiration.replace("-", "")

    strikes = chain_data.get("strikes", []) if chain_data else []
    if len(strikes) < 10:
        results.append((step_start, "Iron Condor Build", "skip",
                        f"Not enough strikes ({len(strikes)}) for iron condor"))
        return results

    mid_idx = len(strikes) // 2
    put_short = strikes[mid_idx - 2]
    call_short = strikes[mid_idx + 2]

    put_long_candidates = [s for s in strikes if s <= put_short - spread_width]
    call_long_candidates = [s for s in strikes if s >= call_short + spread_width]

    if not put_long_candidates or not call_long_candidates:
        results.append((step_start, "Iron Condor Build", "skip",
                        "Could not find strikes for all 4 legs with given width"))
        return results

    put_long = put_long_candidates[-1]
    call_long = call_long_candidates[0]

    try:
        from ib_insync import Option, ComboLeg, Contract

        legs_spec = [
            (put_short, "P", "SELL"),
            (put_long, "P", "BUY"),
            (call_short, "C", "SELL"),
            (call_long, "C", "BUY"),
        ]

        qualified_legs = []
        all_qualified = True
        for strike, right, action in legs_spec:
            opt = Option(symbol, exp_yyyymmdd, strike, right, exchange)
            q = await provider._qualify_contract_cached(opt)
            if q:
                qualified_legs.append((q[0], action))
            else:
                all_qualified = False
                break

        if all_qualified:
            combo_legs = [
                ComboLeg(conId=c.conId, ratio=1, action=action, exchange=exchange)
                for c, action in qualified_legs
            ]
            Contract(symbol=symbol, secType="BAG", exchange=exchange,
                     currency="USD", comboLegs=combo_legs)

            detail = (
                f"secType=BAG, 4 legs\n"
                f"  Put spread:  SELL {put_short}P / BUY {put_long}P\n"
                f"  Call spread: SELL {call_short}C / BUY {call_long}C"
            )
            results.append((step_start, "Iron Condor BAG Combo", "pass", detail))
        else:
            results.append((step_start, "Iron Condor BAG Combo", "fail",
                            "One or more legs failed to qualify"))
    except Exception as e:
        results.append((step_start, "Iron Condor BAG Combo", "fail", str(e)))

    return results


async def _run_readiness_test(args) -> int:
    """Run all readiness steps for 5 trade types. Returns number of failures."""

    os.environ["IBKR_HOST"] = args.host
    os.environ["IBKR_PORT"] = str(args.port)
    os.environ["IBKR_CLIENT_ID"] = str(args.client_id)
    os.environ["IBKR_MARKET_DATA_TYPE"] = str(args.market_data_type)
    os.environ["IBKR_CONNECT_TIMEOUT"] = str(args.timeout)
    os.environ["IBKR_READONLY"] = "true"
    if args.exchange:
        os.environ["IBKR_EXCHANGE"] = args.exchange

    from app.config import Settings
    import app.config
    app.config.settings = Settings()

    from app.core.providers.ibkr import IBKRLiveProvider

    symbol = args.symbol.upper()
    expiration = args.expiration or _next_trading_day()
    spread_width = args.spread_width or _DEFAULT_WIDTHS.get(symbol, 25)
    exchange = args.exchange or "SMART"

    print(f"\n{'=' * 70}")
    print(f"  IBKR Comprehensive Readiness Test — All Trade Types")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")
    print(f"\n  Symbol:      {symbol}")
    print(f"  Expiration:  {expiration}")
    print(f"  Width:       {spread_width}")
    print(f"  Exchange:    {exchange}")
    print(f"  Port:        {args.port}")
    print(f"  Client ID:   {args.client_id}")
    print()

    failures = 0
    skips = 0
    provider = IBKRLiveProvider(exchange=args.exchange or None)

    # Connect
    print(f"{'─' * 70}")
    print(f"  CONNECTIVITY")
    print(f"{'─' * 70}")
    try:
        await provider.connect()
        _print_step(f"Connect to TWS", "pass",
                    f"Connected to {args.host}:{args.port} (clientId={args.client_id})")
    except Exception as e:
        _print_step("Connect to TWS", "fail", str(e))
        print(f"\n  Cannot proceed without connection. Aborting.")
        return 1

    # Option Chain
    chain_data = None
    try:
        chain_data = await provider.get_option_chain(symbol)
        n_exp = len(chain_data.get("expirations", []))
        n_str = len(chain_data.get("strikes", []))
        exp_yyyymmdd = expiration.replace("-", "")
        exp_available = exp_yyyymmdd in chain_data.get("expirations", [])
        detail = (
            f"{n_exp} expirations, {n_str} strikes\n"
            f"Target exp {expiration} {'FOUND' if exp_available else 'NOT FOUND'} in chain"
        )
        _print_step("Option Chain", "pass", detail)
    except Exception as e:
        _print_step("Option Chain", "fail", str(e))
        failures += 1

    # Trade Type 1: Equity
    print(f"\n{'─' * 70}")
    print(f"  TRADE TYPE 1: EQUITY")
    print(f"{'─' * 70}")
    eq_results = await _readiness_test_equity(provider, symbol, step_start=3)
    for step, title, status, detail in eq_results:
        _print_step(f"Step {step}: {title}", status, detail)
        if status == "fail":
            failures += 1
        elif status == "skip":
            skips += 1

    # Trade Type 2: Single Option
    print(f"\n{'─' * 70}")
    print(f"  TRADE TYPE 2: SINGLE OPTION")
    print(f"{'─' * 70}")
    so_results = await _readiness_test_single_option(
        provider, symbol, expiration, chain_data, step_start=4, exchange=exchange
    )
    for step, title, status, detail in so_results:
        _print_step(f"Step {step}: {title}", status, detail)
        if status == "fail":
            failures += 1
        elif status == "skip":
            skips += 1

    # Trade Type 3: Credit Spread
    print(f"\n{'─' * 70}")
    print(f"  TRADE TYPE 3: CREDIT SPREAD (Bull Put)")
    print(f"{'─' * 70}")
    cs_results = await _readiness_test_credit_spread(
        provider, symbol, expiration, chain_data, spread_width,
        step_start=5, exchange=exchange,
        skip_margin=args.skip_margin, quantity=args.quantity,
    )
    for step, title, status, detail in cs_results:
        _print_step(f"Step {step}: {title}", status, detail)
        if status == "fail":
            failures += 1
        elif status == "skip":
            skips += 1

    # Trade Type 4: Debit Spread
    print(f"\n{'─' * 70}")
    print(f"  TRADE TYPE 4: DEBIT SPREAD (Call)")
    print(f"{'─' * 70}")
    ds_results = await _readiness_test_debit_spread(
        provider, symbol, expiration, chain_data, spread_width,
        step_start=9, exchange=exchange,
    )
    for step, title, status, detail in ds_results:
        _print_step(f"Step {step}: {title}", status, detail)
        if status == "fail":
            failures += 1
        elif status == "skip":
            skips += 1

    # Trade Type 5: Iron Condor
    print(f"\n{'─' * 70}")
    print(f"  TRADE TYPE 5: IRON CONDOR")
    print(f"{'─' * 70}")
    ic_results = await _readiness_test_iron_condor(
        provider, symbol, expiration, chain_data, spread_width,
        step_start=10, exchange=exchange,
    )
    for step, title, status, detail in ic_results:
        _print_step(f"Step {step}: {title}", status, detail)
        if status == "fail":
            failures += 1
        elif status == "skip":
            skips += 1

    # Cache Stats
    print(f"\n{'─' * 70}")
    print(f"  CACHE STATS")
    print(f"{'─' * 70}")
    stats = provider.cache_stats
    for cache_name, cache_stats in stats.items():
        print(f"  {cache_name}: {cache_stats}")

    await provider.disconnect()

    # Summary
    all_results = eq_results + so_results + cs_results + ds_results + ic_results
    passed = sum(1 for _, _, s, _ in all_results if s == "pass") + (2 - failures)
    skipped = sum(1 for _, _, s, _ in all_results if s == "skip")

    print(f"\n{'=' * 70}")
    if failures == 0:
        print(f"  {_color('ALL CHECKS PASSED', '92')} ({passed} passed, {skipped} skipped, {failures} failed)")
        print(f"  Ready to trade {symbol} — all 5 trade types validated!")
    else:
        print(f"  {_color(f'{failures} CHECK(S) FAILED', '91')} ({passed} passed, {skipped} skipped)")
        print(f"  Review failures above before trading.")
    print(f"{'=' * 70}\n")

    return failures


# ── HTTP client variants ──────────────────────────────────────────────────────

async def _cmd_portfolio_http(args, server: str) -> int:
    """Portfolio view via HTTP."""
    import httpx
    async with httpx.AsyncClient(base_url=server, timeout=30.0) as client:
        recent_n = getattr(args, "recent", 0)
        params = {}
        if recent_n > 0:
            params["recent_count"] = recent_n

        resp = await client.get("/dashboard/portfolio", params=params)
        if resp.status_code != 200:
            print(f"  Error: {resp.status_code} {resp.text}")
            return 1
        data = resp.json()

        _print_header(f"Portfolio (via {server})")

        # Account balances
        balances = data.get("balances", {})
        if balances.get("net_liquidation", 0) > 0:
            print(f"  Net Liquidation:  ${balances['net_liquidation']:>14,.2f}")
            print(f"  Cash:             ${balances.get('cash', 0):>14,.2f}")
            print(f"  Buying Power:     ${balances.get('buying_power', 0):>14,.2f}")
            avail = balances.get("available_funds", 0)
            if avail > 0:
                print(f"  Available Funds:  ${avail:>14,.2f}")
            margin = balances.get("maint_margin_req", 0)
            if margin > 0:
                print(f"  Margin Req:       ${margin:>14,.2f}")
            print()

        # P&L summary
        unrealized = data.get("unrealized_pnl", 0)
        realized = data.get("realized_pnl", 0)
        total = data.get("total_pnl", 0)
        pnl_color = "92" if total >= 0 else "91"
        print(f"  Unrealized P&L:   {_color(f'${unrealized:>+14,.2f}', '92' if unrealized >= 0 else '91')}")
        print(f"  Realized P&L:     {_color(f'${realized:>+14,.2f}', '92' if realized >= 0 else '91')}")
        print(f"  Total P&L:        {_color(f'${total:>+14,.2f}', pnl_color)}")

        by_source = data.get("positions_by_source", {})
        if by_source:
            print(f"\n  Positions by source:")
            for src, count in by_source.items():
                print(f"    {src:>15}: {count}")

        # Positions
        positions = data.get("positions", [])
        _print_section("Active Positions")
        if not positions:
            print("    (no open positions)")
        else:
            # Check if we have broker-enriched data
            has_marks = any(p.get("avg_cost") is not None for p in positions)

            if has_marks:
                print(f"  {'ID':<6} {'Symbol':>6} {'Type':>12} {'Strikes':>16} {'Qty':>5} {'AvgCost':>10} {'Mark':>10} {'MktVal':>10} {'P&L':>12} {'Exp':>12}")
                print(f"  {'─'*6} {'─'*6} {'─'*12} {'─'*16} {'─'*5} {'─'*10} {'─'*10} {'─'*10} {'─'*12} {'─'*12}")
                total_upnl = 0.0
                for p in positions:
                    sym = p.get("symbol", "?")
                    otype = p.get("order_type", "?")
                    qty = p.get("quantity", 0)
                    exp = p.get("expiration") or "---"
                    pid = p.get("position_id", "")[:6]
                    legs_summary = p.get("legs_summary", "")

                    # Build strikes display from legs or legs_summary
                    strikes_s = ""
                    if legs_summary:
                        strikes_s = legs_summary
                    elif isinstance(p.get("legs"), list) and p["legs"]:
                        parts = []
                        for leg in p["legs"]:
                            if isinstance(leg, dict):
                                s = leg.get("strike", "")
                                ot = leg.get("option_type", "")
                                r = "P" if ot == "PUT" else "C" if ot == "CALL" else ""
                                parts.append(f"{r}{s}")
                        strikes_s = "/".join(parts)

                    if p.get("avg_cost") is not None:
                        avg_cost = p["avg_cost"]
                        mark = p.get("market_price", 0) or 0
                        mv = p.get("market_value", 0) or 0
                        upnl = p.get("broker_unrealized_pnl", 0) or 0
                        total_upnl += upnl
                        pnl_c = "92" if upnl >= 0 else "91"
                        mark_s = f"${mark:>9.4f}" if mark else f"{'---':>10}"
                        mv_s = f"${mv:>9.2f}" if mv else f"{'---':>10}"
                        upnl_s = _color(f"${upnl:>+10,.2f}", pnl_c) if upnl else f"{'---':>12}"
                        print(f"  {pid:<6} {sym:>6} {otype:>12} {strikes_s:>16} {qty:>5.0f} "
                              f"${avg_cost:>9.4f} {mark_s} {mv_s} "
                              f"{upnl_s:>20} {exp:>12}")
                    else:
                        entry = p.get("entry_price", 0)
                        print(f"  {pid:<6} {sym:>6} {otype:>12} {strikes_s:>16} {qty:>5.0f} "
                              f"${abs(entry):>9.4f} {'---':>10} {'---':>10} {'---':>12} {exp:>12}")

                upnl_c = "92" if total_upnl >= 0 else "91"
                print(f"  {'':>6} {'':>6} {'':>12} {'':>16} {'':>5} {'':>10} {'':>10} {'TOTAL':>10} "
                      f"{_color(f'${total_upnl:>+10,.2f}', upnl_c):>20}")
            else:
                print(f"  {'Symbol':<10} {'Type':<10} {'Qty':>6} {'Entry':>10} {'P&L':>10} {'Status':<8}")
                print(f"  {'---':<10} {'---':<10} {'---':>6} {'---':>10} {'---':>10} {'---':<8}")
                for p in positions:
                    pnl = p.get("unrealized_pnl") or 0
                    pnl_color = "92" if pnl >= 0 else "91"
                    print(f"  {p.get('symbol',''):<10} {p.get('order_type',''):<10} "
                          f"{p.get('quantity',0):>6} {p.get('entry_price',0):>10.2f} "
                          f"{_color(f'${pnl:>8.2f}', pnl_color)} {p.get('status',''):<8}")

        # Recent closed trades
        recent_n = getattr(args, "recent", 0)
        recent = data.get("recent_closed", [])
        if recent_n > 0:
            # User requested N recent trades
            show_recent = recent[:recent_n]
        elif recent:
            # Default: show up to 5 recent
            show_recent = recent[:5]
        else:
            show_recent = []

        if show_recent:
            _print_section(f"Recent Closed (last {len(show_recent)})")
            for pos in show_recent:
                pnl = pos.get("pnl", 0)
                pc = "92" if pnl >= 0 else "91"
                sym = pos.get("symbol", "?")
                reason = pos.get("exit_reason", "?")
                exp = pos.get("expiration", "")
                exit_time = (pos.get("exit_time") or "")[:16]
                print(f"  {sym:>8} | P&L={_color(f'${pnl:+,.2f}', pc):>20} | {reason:<15} | {exp} | {exit_time}")

        print()
    return 0


async def _cmd_quote_http(args, server: str) -> int:
    """Quote via HTTP — fetches all symbols in parallel."""
    import asyncio
    import httpx
    symbols = args.symbols

    async with httpx.AsyncClient(base_url=server, timeout=30.0) as client:
        # Fire all quote requests concurrently
        async def _fetch_one(sym):
            try:
                resp = await client.get(f"/market/quote/{sym.upper()}")
                return sym, resp
            except Exception as e:
                return sym, e

        results = await asyncio.gather(*[_fetch_one(s) for s in symbols])

        # Print in original order
        from datetime import datetime as _dt, timezone as _tz
        now = _dt.now(_tz.utc)
        for sym, result in results:
            if isinstance(result, Exception):
                print(f"  {sym}: error ({type(result).__name__})")
            elif result.status_code == 200:
                q = result.json()
                # Compute age of quote
                ts_str = q.get("timestamp", "")
                source = q.get("source", "")
                age_str = ""
                if ts_str:
                    try:
                        ts = _dt.fromisoformat(ts_str)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=_tz.utc)
                        age = (now - ts).total_seconds()
                        if age < 2:
                            age_str = "just now"
                        elif age < 60:
                            age_str = f"{age:.0f}s ago"
                        elif age < 3600:
                            age_str = f"{age/60:.0f}m ago"
                        else:
                            age_str = f"{age/3600:.1f}h ago"
                    except Exception:
                        pass
                src_label = f" [{source}]" if source else ""
                age_label = f" ({age_str})" if age_str else ""
                print(f"  {q['symbol']:<8} Bid: {q['bid']:>10.2f}  Ask: {q['ask']:>10.2f}  "
                      f"Last: {q['last']:>10.2f}  Vol: {q.get('volume',0):>10,}"
                      f"  {age_label}{src_label}")
            else:
                print(f"  {sym}: error {result.status_code}")
    return 0


async def _cmd_status_http(args, server: str) -> int:
    """Status via HTTP."""
    import httpx
    async with httpx.AsyncClient(base_url=server, timeout=30.0) as client:
        resp = await client.get("/dashboard/status")
        if resp.status_code != 200:
            print(f"  Error: {resp.status_code}")
            return 1
        data = resp.json()
        _print_header(f"Status (via {server})")
        positions = data.get("active_positions", [])
        print(f"  Active positions: {len(positions)}")
        orders = data.get("in_transit_orders", [])
        print(f"  In-transit orders: {len(orders)}")
        recent = data.get("recent_closed", [])
        print(f"  Recent closed: {len(recent)}")
    return 0


async def _cmd_orders_http(args, server: str) -> int:
    """Orders via HTTP."""
    import httpx
    async with httpx.AsyncClient(base_url=server, timeout=30.0) as client:
        resp = await client.get("/account/orders")
        if resp.status_code != 200:
            print(f"  Error: {resp.status_code}")
            return 1
        orders = resp.json()
        if not orders:
            print("  No open orders.")
            return 0
        _print_header("Open Orders")
        for o in orders:
            print(f"  [{o.get('order_id','?')[:8]}] {o.get('status','')} — {o.get('message','')}")
    return 0


async def _cmd_cancel_http(args, server: str) -> int:
    """Cancel orders via HTTP (through daemon)."""
    import httpx
    order_id = getattr(args, "order_id", None)
    cancel_all = getattr(args, "all", False)

    if not order_id and not cancel_all:
        print("  Specify --order-id <ID> or --all")
        return 1

    async with httpx.AsyncClient(base_url=server, timeout=30.0) as client:
        if cancel_all:
            # Get all orders then cancel each
            resp = await client.get("/account/orders")
            if resp.status_code != 200:
                print(f"  Error fetching orders: {resp.status_code}")
                return 1
            orders = resp.json()
            if not orders:
                print("  No open orders to cancel")
                return 0
            print(f"  Cancelling {len(orders)} order(s)...")
            for o in orders:
                oid = o.get("order_id", "?")
                cr = await client.post("/account/cancel", params={"order_id": oid})
                if cr.status_code == 200:
                    d = cr.json()
                    print(f"    Order #{oid}: {d.get('status', '?')} — {d.get('message', '')}")
                else:
                    print(f"    Order #{oid}: Error {cr.status_code}")
        else:
            resp = await client.post("/account/cancel", params={"order_id": order_id})
            if resp.status_code == 200:
                d = resp.json()
                status_color = "92" if d.get("status") == "CANCELLED" else "91"
                print(f"  Status: {_color(d.get('status', '?'), status_color)}")
                print(f"  {d.get('message', '')}")
            else:
                print(f"  Error: {resp.status_code} {resp.text}")
                return 1
    return 0


async def _cmd_playbook_http(args, server: str) -> int:
    """Playbook via HTTP (through daemon)."""
    import httpx
    from pathlib import Path as P

    action = getattr(args, "playbook_action", None)
    if not action:
        print("  Error: specify an action (execute or validate)")
        return 1

    playbook_file = getattr(args, "playbook_file", None)
    if not playbook_file:
        print("  Error: specify a playbook YAML file")
        return 1

    path = P(playbook_file)
    if not path.exists():
        print(f"  Error: Playbook file not found: {path}")
        return 1

    # Read the YAML and send to server
    yaml_content = path.read_text()

    mode = _get_mode(args)
    headers = {}
    if mode == "dry-run":
        headers["X-Dry-Run"] = "true"

    endpoint = "/playbook/execute" if action == "execute" else "/playbook/validate"

    async with httpx.AsyncClient(base_url=server, timeout=120.0) as client:
        resp = await client.post(endpoint, json={"yaml_content": yaml_content}, headers=headers)
        if resp.status_code != 200:
            print(f"  Error: {resp.status_code} {resp.text}")
            return 1
        data = resp.json()

        _print_header(f"Playbook {'Execution' if action == 'execute' else 'Validation'} (via {server})")
        if action == "validate":
            print(f"  Status: {_color('VALID', '92')}")
            print(f"  Name: {data.get('name', '?')}")
            print(f"  Instructions: {data.get('instruction_count', '?')}")
        else:
            results = data.get("results", [])
            passed = sum(1 for r in results if r.get("status") == "success")
            failed = len(results) - passed
            color = "92" if failed == 0 else "91"
            print(f"  Results: {_color(f'{passed} passed, {failed} failed', color)}")
            for r in results:
                status_icon = _color("OK", "92") if r.get("status") == "success" else _color("FAIL", "91")
                print(f"    [{status_icon:>12}] {r.get('instruction_id', '?')}: {r.get('message', '')}")
    return 0


async def _cmd_trades_http(args, server: str) -> int:
    """Trades via HTTP."""
    import httpx
    params = {}
    if getattr(args, "days", 0):
        params["days"] = args.days
    if getattr(args, "show_all", False):
        params["include_all"] = "true"
    async with httpx.AsyncClient(base_url=server, timeout=30.0) as client:
        resp = await client.get("/account/trades", params=params)
        if resp.status_code != 200:
            print(f"  Error: {resp.status_code}")
            return 1
        trades = resp.json()
        if not trades:
            print("  No trades found.")
            return 0
        _print_header("Trades")
        for t in trades:
            pnl = t.get("pnl") or 0
            pnl_color = "92" if pnl >= 0 else "91"
            print(f"  {t.get('symbol',''):<10} {t.get('status',''):<8} "
                  f"P&L: {_color(f'${pnl:.2f}', pnl_color)}")
    return 0


async def _cmd_close_http(args, server: str) -> int:
    """Close position via HTTP."""
    import httpx
    position_id = args.position_id
    simulate = getattr(args, "simulate", False)
    qty = getattr(args, "quantity", None)
    net_price = getattr(args, "net_price", None)

    async with httpx.AsyncClient(base_url=server, timeout=30.0) as client:
        if simulate:
            # Look up position details first, then run margin check
            resp = await client.get("/dashboard/portfolio")
            if resp.status_code != 200:
                print(f"  Error fetching portfolio: {resp.status_code}")
                return 1
            positions = resp.json().get("positions", [])
            match = None
            for p in positions:
                if p.get("position_id", "").startswith(position_id):
                    match = p
                    break
            if not match:
                print(f"  Position {position_id} not found")
                return 1

            # Build closing order for margin check
            legs = match.get("legs") or []
            if not legs and match.get("order_type") == "multi_leg":
                print(f"  Cannot simulate: position has no leg details")
                return 1

            close_qty = qty or int(abs(match.get("quantity", 1)))
            exp = match.get("expiration", "")

            _print_header(f"Simulated Close (margin check only — NOT executed)")
            print(f"  Position:   {position_id[:8]}")
            print(f"  Symbol:     {match.get('symbol')}")
            print(f"  Type:       {match.get('order_type')}")
            print(f"  Qty:        {close_qty} of {int(abs(match.get('quantity', 0)))}")

            # Show what price will be used
            mark = match.get("market_price", 0)
            mark_abs = abs(mark) if mark else 0
            if net_price is not None:
                print(f"  Net price:  ${net_price:.2f} (user-specified)")
            elif mark_abs > 0:
                print(f"  Net price:  ${mark_abs:.2f} (current mark — no --net-price specified)")
            else:
                print(f"  Net price:  $0.05 (fallback — mark unavailable)")

            if legs:
                print(f"  Closing legs:")
                for leg in legs:
                    action = leg.get("action", "")
                    close_action = "BUY_TO_CLOSE" if "SELL" in action else "SELL_TO_CLOSE"
                    ot = leg.get("option_type", "PUT")
                    strike = leg.get("strike", 0)
                    con_id = leg.get("con_id", "?")
                    print(f"    {close_action:>18} {ot} {strike} x{close_qty} (conId={con_id})")

                # Show current mark value to help set net_price
                mark = match.get("market_price", 0)
                if mark:
                    mark_per = abs(mark)
                    print(f"\n  Current mark:   ${mark_per:.4f} per spread")
                    print(f"  Cost to close {close_qty}:  ~${mark_per * close_qty * 100:.2f}")
                    if (net_price or 0.05) < mark_per * 0.8:
                        print(f"  {_color('Warning:', '93')} --net-price ${net_price or 0.05:.2f} "
                              f"is well below the mark (${mark_per:.2f}). Order will likely be cancelled.")
                        print(f"  Suggest: --net-price {mark_per:.2f} or higher")

                # Try margin check (may fail for some symbols)
                from app.services.trade_service import build_closing_trade_request
                trade_req = build_closing_trade_request(match, close_qty, net_price or 0.05)
                if trade_req.multi_leg_order:
                    try:
                        margin_resp = await client.post("/market/margin", json={
                            "order": trade_req.multi_leg_order.model_dump(), "timeout": 15.0,
                        })
                        if margin_resp.status_code == 200:
                            md = margin_resp.json()
                            if not md.get("error"):
                                print(f"\n  Margin Requirements:")
                                print(f"    Initial margin:     ${md.get('init_margin', 0):>12,.2f}")
                                print(f"    Maintenance margin: ${md.get('maint_margin', 0):>12,.2f}")
                                print(f"    Commission:         ${md.get('commission', 0):>12,.2f}")
                    except Exception:
                        pass  # Margin check is optional — conId close works without it
            else:
                print(f"\n  Equity close: {match.get('symbol')} x{close_qty}")

            print(f"\n  {_color('NOT EXECUTED', '93')} — remove --simulate to close")
            return 0

        # Actual close — submits order to IBKR
        payload = {"position_id": position_id}
        if qty:
            payload["quantity"] = qty
        if net_price is not None:
            payload["net_price"] = net_price

        # Show order summary before executing
        confirm = getattr(args, "confirm", False)
        price_label = f"${net_price:.2f}" if net_price else "current mark"

        _print_header("Close Order Summary")
        print(f"  Position:   {position_id[:8]}")
        print(f"  Quantity:   {qty or 'all'}")
        print(f"  Net price:  {price_label}")
        if net_price:
            cost = abs(net_price) * (qty or 1) * 100
            print(f"  Est. cost:  ${cost:,.2f} (debit)")

        if not confirm:
            print(f"\n  {_color('NOT EXECUTED', '93')} — add --confirm to close the position")
            return 0

        print(f"\n  Submitting closing order to IBKR @ {price_label}...")
        resp = await client.post("/trade/close", json=payload)
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            print(f"  Error: {resp.status_code} — {detail}")
            return 1
        data = resp.json()
        order_result = data.get("order_result", {})
        pos = data.get("position", {})
        symbol = pos.get("symbol", "?")
        order_status = order_result.get("status", "?")
        fill_price = order_result.get("filled_price")

        if data.get("status") == "order_not_filled":
            msg = order_result.get("message", "")
            print(f"  {_color('NOT FILLED', '91')} — {order_status}")
            if msg:
                print(f"  Detail: {msg}")
            if data.get("message"):
                print(f"  {data['message']}")
            print(f"  Local position NOT modified.")
            return 1

        # Order filled
        pos_status = pos.get("status", "?")
        if pos_status == "closed":
            pnl = pos.get("pnl")
            pnl_str = f"${pnl:+,.2f}" if pnl is not None else "N/A"
            pnl_color = "92" if (pnl or 0) >= 0 else "91"
            fill_str = f" @ ${fill_price:.2f}" if fill_price else ""
            print(f"  {_color('FILLED & CLOSED', '92')} {symbol}{fill_str}")
            print(f"  P&L: {_color(pnl_str, pnl_color)}")
        else:
            remaining = int(abs(pos.get("quantity", 0)))
            closed_qty = qty or "all"
            fill_str = f" @ ${fill_price:.2f}" if fill_price else ""
            print(f"  {_color('FILLED', '92')} {symbol} — closed {closed_qty}{fill_str}, {remaining} remaining")
    return 0


async def _cmd_performance_http(args, server: str) -> int:
    """Performance via HTTP."""
    import httpx
    async with httpx.AsyncClient(base_url=server, timeout=30.0) as client:
        resp = await client.get("/dashboard/performance")
        if resp.status_code != 200:
            print(f"  Error: {resp.status_code}")
            return 1
        data = resp.json()
        _print_header(f"Performance (via {server})")
        print(f"  Total trades: {data.get('total_trades', 0)}")
        print(f"  Win rate:     {data.get('win_rate', 0):.1%}")
        print(f"  Net P&L:      ${data.get('net_pnl', 0):.2f}")
        print(f"  Sharpe:       {data.get('sharpe', 0):.2f}")
        print(f"  Max Drawdown: ${data.get('max_drawdown', 0):.2f}")
        print(f"  Profit Factor:{data.get('profit_factor', 0):.2f}")
    return 0


async def _cmd_journal_http(args, server: str) -> int:
    """Journal via HTTP."""
    import httpx
    async with httpx.AsyncClient(base_url=server, timeout=30.0) as client:
        resp = await client.get("/ledger/entries/recent", params={"n": getattr(args, "limit", 20)})
        if resp.status_code != 200:
            print(f"  Error: {resp.status_code}")
            return 1
        entries = resp.json()
        if not entries:
            print("  No ledger entries.")
            return 0
        _print_header("Journal")
        for e in entries:
            print(f"  [{e.get('sequence_number',0):>4}] {e.get('event_type',''):<20} "
                  f"{e.get('timestamp','')[:19]}")
    return 0


async def _cmd_executions_http(args, server: str) -> int:
    """Executions via HTTP — routes through running daemon."""
    import httpx

    params = {}
    if getattr(args, "flush", False):
        params["flush"] = "true"
    symbol_filter = getattr(args, "symbol", None)
    if symbol_filter:
        params["symbol"] = symbol_filter

    async with httpx.AsyncClient(base_url=server, timeout=30.0) as client:
        resp = await client.get("/account/executions", params=params)
        if resp.status_code != 200:
            print(f"  Error: {resp.status_code} {resp.text}")
            return 1
        data = resp.json()

    fetched = data.get("fetched", 0)
    new = data.get("new", 0)
    total = data.get("total_cached", 0)
    groups = data.get("orders", [])

    if fetched:
        print(f"  Fetched {fetched} executions from IBKR, {new} new (total: {total})")
    else:
        print(f"  Cached: {total} executions")

    if not groups:
        print("  No executions found")
        return 0

    _print_header(f"Execution History — {len(groups)} Orders ({total} fills)")

    for g in groups:
        otype = g.get("order_type", "?")
        sym = g.get("symbol", "?")
        exp = g.get("expiration", "")
        time_str = (g.get("time") or "?")[:19]
        net = g.get("net_amount", 0)
        comm = g.get("total_commission", 0)
        perm = g.get("perm_id", "?")
        legs = g.get("legs", [])

        if "spread" in otype or "condor" in otype:
            type_color = "95"
        elif otype == "single_option":
            type_color = "96"
        else:
            type_color = "97"

        net_color = "92" if net >= 0 else "91"
        net_label = "credit" if net > 0 else "debit"

        print(f"\n  {_color(otype.upper().replace('_', ' '), type_color):>28} | "
              f"{sym:>6} | {time_str} | "
              f"{_color(f'${net:>+10,.2f}', net_color)} ({net_label}) | "
              f"comm=${comm:.2f} | perm={perm}")

        if exp:
            print(f"  {'':>28}   exp: {exp}")

        for leg in legs:
            side = leg.get("side", "?")
            side_color = "91" if side == "SLD" else "92"
            sec = leg.get("sec_type", "")

            if sec in ("OPT", "FOP"):
                right = leg.get("right", "?")
                strike = leg.get("strike", 0)
                desc = f"{right}{strike:.0f}"
                print(f"  {'':>8} {_color(side, side_color):>12} "
                      f"{desc:>10} x{leg.get('shares', 0):.0f} @ ${leg.get('price', 0):.2f} "
                      f"(conId={leg.get('con_id', '?')})")
            else:
                print(f"  {'':>8} {_color(side, side_color):>12} "
                      f"{leg.get('symbol', '?'):>10} x{leg.get('shares', 0):.0f} @ ${leg.get('price', 0):.2f}")

    print()
    return 0


async def _cmd_reconcile_http(args, server: str) -> int:
    """Reconcile via HTTP."""
    import httpx

    async with httpx.AsyncClient(base_url=server, timeout=30.0) as client:
        # Hard reset: clear daemon's in-memory store + re-sync
        if getattr(args, "hard_reset", False):
            _print_header("HARD RESET (via daemon)")
            resp = await client.post("/account/hard-reset")
            if resp.status_code != 200:
                print(f"  Error: {resp.status_code} {resp.text}")
                return 1
            d = resp.json()
            print(f"  Cleared: {d.get('cleared', 0)} positions + {d.get('executions_cleared', 0)} executions")
            print(f"  Re-synced: {d.get('synced_new', 0)} new, {d.get('synced_updated', 0)} updated")
            print(f"  Open positions: {d.get('open_positions', 0)}")
            print()

        # Flush: clear open positions in daemon memory, preserve closed, re-sync
        elif getattr(args, "flush", False):
            _print_header("Flush (via daemon — preserving closed positions)")
            resp = await client.post("/account/flush")
            if resp.status_code != 200:
                print(f"  Error: {resp.status_code} {resp.text}")
                return 1
            d = resp.json()
            print(f"  Open cleared:      {d.get('open_cleared', 0)}")
            print(f"  Closed preserved:  {d.get('closed_preserved', 0)}")
            print(f"  Re-synced:         {d.get('synced_new', 0)} new, {d.get('synced_updated', 0)} updated")
            print(f"  Open positions:    {d.get('open_positions', 0)}")
            print()

        broker = getattr(args, "broker", "ibkr")
        resp = await client.get("/account/reconciliation", params={"broker": broker})
        if resp.status_code != 200:
            print(f"  Error: {resp.status_code} {resp.text}")
            return 1
        data = resp.json()

        _print_header(f"Reconciliation (via {server})")
        print(f"  Broker:             {data.get('broker', 'ibkr')}")
        print(f"  System positions:   {data.get('total_system_positions', 0)}")
        print(f"  Broker positions:   {data.get('total_broker_positions', 0)}")
        print(f"  Matched:            {data.get('matched', 0)}")

        discrepancies = data.get("discrepancies", [])
        if discrepancies:
            _print_section("Discrepancies")
            for d in discrepancies:
                dtype = d.get("discrepancy_type", "?")
                sym = d.get("symbol", "?")
                details = d.get("details", "")
                color = "92" if dtype == "matched" else "91"
                print(f"  [{_color(dtype, color)}] {sym}: {details}")
        else:
            print(f"\n  {_color('No discrepancies', '92')}")

        # Show synced positions if --show
        if getattr(args, "show", False):
            resp2 = await client.get("/dashboard/summary")
            if resp2.status_code == 200:
                positions = resp2.json().get("active_positions", [])
                if positions:
                    _print_section("Synced Positions")
                    for p in positions:
                        sym = p.get("symbol", "?")
                        qty = p.get("quantity", 0)
                        print(f"  {sym:<10} qty={qty}")
    return 0


async def _cmd_options_http(args, server: str) -> int:
    """Options chain via HTTP."""
    import httpx
    import math
    from datetime import date as _date

    symbol = args.symbol.upper() if hasattr(args, 'symbol') else 'SPX'

    params = {}
    if getattr(args, "list_expirations", False):
        params["list_expirations"] = "true"

    exp = getattr(args, "expiration", None)
    # Default expiration to today (same as direct mode)
    if not exp and not getattr(args, "list_expirations", False):
        exp = _date.today().isoformat()
    if exp:
        params["expiration"] = exp

    otype = getattr(args, "type", None)
    if otype and otype != "BOTH":
        params["option_type"] = otype

    strike_min = getattr(args, "strike_min", None)
    strike_max = getattr(args, "strike_max", None)

    async with httpx.AsyncClient(base_url=server, timeout=30.0) as client:
        # If no explicit strike bounds, use --strike-range % around current price
        if strike_min is None and strike_max is None and exp:
            strike_range = getattr(args, "strike_range", 15)
            pct = strike_range / 100.0
            try:
                qr = await client.get(f"/market/quote/{symbol}")
                if qr.status_code == 200:
                    qdata = qr.json()
                    price = qdata.get("last") or qdata.get("bid") or qdata.get("ask") or 0
                    if price and price > 0:
                        strike_min = round(price * (1 - pct), 2)
                        strike_max = round(price * (1 + pct), 2)
            except Exception:
                pass

        if strike_min is not None:
            params["strike_min"] = strike_min
        if strike_max is not None:
            params["strike_max"] = strike_max

        resp = await client.get(f"/market/options/{symbol}", params=params)
        if resp.status_code != 200:
            print(f"  Error: {resp.status_code} {resp.text}")
            return 1
        data = resp.json()

        if getattr(args, "list_expirations", False):
            _print_header(f"Expirations for {symbol}")
            for exp_item in data.get("expirations", []):
                print(f"  {exp_item}")
            return 0

        _print_header(f"Option Chain: {symbol} (via {server})")
        chain = data.get("chain", {})
        print(f"  Expirations: {len(chain.get('expirations', []))}")
        print(f"  Strikes:     {len(chain.get('strikes', []))}")
        if exp:
            print(f"  Expiration:  {exp}")
        if strike_min is not None and strike_max is not None:
            range_pct = getattr(args, "strike_range", 15)
            print(f"  Strike range: ${strike_min:,.0f} — ${strike_max:,.0f} (±{range_pct}%)")

        quotes = data.get("quotes", {})
        for ot_key in ("call", "put"):
            ot_data = quotes.get(ot_key)
            if isinstance(ot_data, dict) and "error" in ot_data:
                _print_section(f"{ot_key.upper()}S")
                err_msg = ot_data["error"]
                print(f"  {_color(f'Error: {err_msg}', '91')}")
                continue
            if ot_data and isinstance(ot_data, list) and ot_data:
                # Filter out junk rows (bid=-1, nan, no data)
                valid = []
                for q in ot_data:
                    bid, ask, last = q.get("bid", 0), q.get("ask", 0), q.get("last", 0)
                    vol = q.get("volume", 0)
                    has_bid = isinstance(bid, (int, float)) and bid > 0 and not (isinstance(bid, float) and math.isnan(bid))
                    has_ask = isinstance(ask, (int, float)) and ask > 0 and not (isinstance(ask, float) and math.isnan(ask))
                    has_last = isinstance(last, (int, float)) and last > 0 and not (isinstance(last, float) and math.isnan(last))
                    has_vol = isinstance(vol, (int, float)) and vol > 0
                    if has_bid or has_ask or has_last or has_vol:
                        valid.append(q)

                _print_section(f"{ot_key.upper()}S ({len(valid)} strikes)")
                print(f"  {'Strike':>10} {'Bid':>8} {'Ask':>8} {'Last':>8} {'Volume':>8}")
                print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
                for q in valid:
                    def _fp(v):
                        if v is None or (isinstance(v, float) and (math.isnan(v) or v <= 0)):
                            return "    —"
                        return f"{v:>8.2f}"
                    print(f"  {q.get('strike',0):>10.1f} {_fp(q.get('bid'))}"
                          f" {_fp(q.get('ask'))} {_fp(q.get('last'))}"
                          f" {q.get('volume',0):>8}")
        if not quotes:
            print(f"\n  No quotes returned — expiration {exp} may not be available.")
            print(f"  Use --list-expirations to see available dates.")
    return 0


async def _cmd_margin_http(args, server: str) -> int:
    """Margin check via HTTP."""
    import httpx

    subcommand = getattr(args, "margin_type", None)
    if not subcommand:
        print("  Error: specify a trade type (credit-spread, debit-spread, iron-condor, option)")
        return 1

    try:
        order = _build_margin_order(subcommand, args)
    except Exception as e:
        print(f"  {_color('ERROR', '91')}: {e}")
        return 1

    async with httpx.AsyncClient(base_url=server, timeout=30.0) as client:
        resp = await client.post("/market/margin", json={
            "order": order.model_dump(),
            "timeout": 10.0,
        })
        if resp.status_code != 200:
            print(f"  Error: {resp.status_code} {resp.text}")
            return 1
        data = resp.json()

        _print_header(f"Margin Check (via {server})")
        if data.get("error"):
            print(f"  Error: {data['error']}")
        else:
            print(f"  Initial margin:     ${data.get('init_margin', 0):>12,.2f}")
            print(f"  Maintenance margin: ${data.get('maint_margin', 0):>12,.2f}")
            print(f"  Commission:         ${data.get('commission', 0):>12,.2f}")
    return 0


async def _cmd_trade_http(args, server: str) -> int:
    """Execute trade via HTTP."""
    import httpx

    subcommand = getattr(args, "subcommand", None)
    if not subcommand:
        # Check for --validate-all
        if getattr(args, "validate_all", False):
            print("  --validate-all requires direct IBKR connection, not daemon")
            return 1
        print("  Error: specify a trade type")
        return 1

    instr = _build_instruction_from_args(subcommand, args)

    from app.models import PlaybookInstruction
    from app.services.playbook_service import PlaybookService
    trade_request = PlaybookService().instruction_to_trade_request(
        PlaybookInstruction(**instr), getattr(args, "broker", "ibkr")
    )

    payload = {}
    if trade_request.equity_order:
        payload["equity_order"] = trade_request.equity_order.model_dump()
    elif trade_request.multi_leg_order:
        payload["multi_leg_order"] = trade_request.multi_leg_order.model_dump()

    mode = _get_mode(args)
    simulate = getattr(args, "simulate", False)

    async with httpx.AsyncClient(base_url=server, timeout=60.0) as client:
        if simulate:
            # --simulate: use margin check (whatIfOrder) — qualifies contracts,
            # checks margin, shows costs, but does NOT place the order.
            if trade_request.multi_leg_order:
                margin_payload = {"order": trade_request.multi_leg_order.model_dump(), "timeout": 15.0}
            else:
                print(f"  {_color('--simulate only supported for multi-leg trades (spreads, condors)', '93')}")
                print(f"  For equity, use: python utp.py margin single-option ...")
                return 1

            _print_header("Simulated Trade (margin check only — NOT executed)")
            # Show what would be traded
            order = trade_request.multi_leg_order
            print(f"  Symbol:     {order.legs[0].symbol if order.legs else '?'}")
            print(f"  Type:       {subcommand}")
            print(f"  Quantity:   {order.quantity}")
            if order.net_price:
                print(f"  Net price:  ${order.net_price:.2f}")
            print(f"  Legs:")
            for leg in (order.legs or []):
                print(f"    {leg.action.value:>15} {leg.option_type.value:>4} "
                      f"strike={leg.strike} exp={leg.expiration} qty={leg.quantity}")

            resp = await client.post("/market/margin", json=margin_payload)
            if resp.status_code != 200:
                print(f"\n  {_color(f'Margin check failed: {resp.status_code} {resp.text}', '91')}")
                return 1
            data = resp.json()
            if data.get("error"):
                err_detail = data.get("error", "unknown")
                print(f"\n  {_color(f'Margin error: {err_detail}', '91')}")
                return 1
            print(f"\n  Margin Requirements:")
            print(f"    Initial margin:     ${data.get('init_margin', 0):>12,.2f}")
            print(f"    Maintenance margin: ${data.get('maint_margin', 0):>12,.2f}")
            print(f"    Commission:         ${data.get('commission', 0):>12,.2f}")
            if data.get("equity_with_loan"):
                print(f"    Equity w/ loan:     ${data['equity_with_loan']:>12,.2f}")
            print(f"\n  {_color('NOT EXECUTED', '93')} — remove --simulate to place the order")
            return 0

        # Show order summary before executing
        confirm = getattr(args, "confirm", False)

        _print_header("Trade Order Summary")
        if trade_request.equity_order:
            eo = trade_request.equity_order
            print(f"  Type:       equity")
            print(f"  Symbol:     {eo.symbol}")
            print(f"  Side:       {eo.side.value}")
            print(f"  Quantity:   {eo.quantity}")
            if eo.limit_price:
                print(f"  Price:      ${eo.limit_price:.2f}")
                total = eo.limit_price * eo.quantity
                action = "spend" if eo.side.value == "BUY" else "receive"
                print(f"  Est. total: ${total:,.2f} ({action})")
            else:
                print(f"  Price:      MARKET")
        elif trade_request.multi_leg_order:
            order = trade_request.multi_leg_order
            print(f"  Type:       {subcommand}")
            print(f"  Symbol:     {order.legs[0].symbol if order.legs else '?'}")
            print(f"  Quantity:   {order.quantity}")
            print(f"  Exchange:   SMART (best execution)")
            if order.net_price:
                # Determine credit vs debit
                is_credit = order.legs[0].action.value in ("SELL_TO_OPEN", "SELL_TO_CLOSE")
                if is_credit:
                    print(f"  Net credit: ${order.net_price:.2f} per spread")
                    total = order.net_price * order.quantity * 100
                    print(f"  You receive: ~${total:,.2f}")
                else:
                    print(f"  Net debit:  ${order.net_price:.2f} per spread")
                    total = order.net_price * order.quantity * 100
                    print(f"  You spend:  ~${total:,.2f}")
            print(f"  Legs:")
            for leg in (order.legs or []):
                print(f"    {leg.action.value:>15} {leg.option_type.value:>4} "
                      f"strike={leg.strike} exp={leg.expiration} qty={leg.quantity}")

        if not confirm and mode != "dry-run":
            print(f"\n  {_color('NOT EXECUTED', '93')} — add --confirm to place the order")
            return 0

        headers = {}
        if mode == "dry-run":
            headers["X-Dry-Run"] = "true"

        print(f"\n  Submitting order...")
        resp = await client.post("/trade/execute", json=payload, headers=headers)
        if resp.status_code != 200:
            print(f"  Error: {resp.status_code} {resp.text}")
            return 1
        data = resp.json()

        status = data.get("status", "?")
        color = "92" if status in ("FILLED", "SUBMITTED") else "91"
        print(f"  Order: {_color(status, color)}")
        print(f"  ID:    {data.get('order_id', '?')}")
        if data.get("filled_price"):
            print(f"  Fill:  ${data['filled_price']:.2f}")
        if data.get("message"):
            print(f"  Msg:   {data['message']}")
    return 0


# ── REPL ──────────────────────────────────────────────────────────────────────

async def _cmd_repl(args) -> int:
    """Interactive REPL that sends commands to the daemon."""
    server = getattr(args, "server", None) or f"http://localhost:{getattr(args, 'server_port', 8000)}"

    # Check server is reachable
    import httpx
    try:
        async with httpx.AsyncClient(base_url=server, timeout=2.0) as client:
            resp = await client.get("/health")
            if resp.status_code != 200:
                print(f"  Cannot reach daemon at {server}")
                return 1
            health = resp.json()
    except Exception as e:
        print(f"  Cannot reach daemon at {server}: {e}")
        return 1

    mode = "LIVE" if health.get("ibkr_connected") else "PAPER"
    prompt_str = f"[{mode}] utp> "

    try:
        import readline  # noqa: F401
    except ImportError:
        pass

    print(f"  Connected to UTP daemon at {server}")
    print(f"  Type 'help' for commands, 'quit' to exit.\n")

    while True:
        try:
            line = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in ("quit", "exit", "q"):
            break
        if line == "help":
            print("  Commands: portfolio, quote <SYM>, status, orders, trades,")
            print("           close <ID>, performance, journal, advisor, y <N>, health, quit")
            continue

        parts = line.split()
        cmd = parts[0]

        try:
            if cmd in ("portfolio", "port", "pos"):
                ns = argparse.Namespace(server=server, server_port=8000)
                await _cmd_portfolio_http(ns, server)
            elif cmd in ("quote",) and len(parts) > 1:
                ns = argparse.Namespace(symbols=parts[1:], server=server)
                await _cmd_quote_http(ns, server)
            elif cmd in ("status", "st"):
                ns = argparse.Namespace(server=server)
                await _cmd_status_http(ns, server)
            elif cmd in ("orders", "oo"):
                ns = argparse.Namespace(server=server)
                await _cmd_orders_http(ns, server)
            elif cmd in ("trades", "activity"):
                ns = argparse.Namespace(days=0, show_all=False, server=server)
                await _cmd_trades_http(ns, server)
            elif cmd in ("close", "cl") and len(parts) > 1:
                ns = argparse.Namespace(position_id=parts[1], quantity=None, net_price=0.05, server=server)
                await _cmd_close_http(ns, server)
            elif cmd in ("performance", "perf"):
                ns = argparse.Namespace(server=server)
                await _cmd_performance_http(ns, server)
            elif cmd in ("journal", "log"):
                ns = argparse.Namespace(limit=20, server=server)
                await _cmd_journal_http(ns, server)
            elif cmd == "advisor":
                async with httpx.AsyncClient(base_url=server, timeout=5.0) as hc:
                    resp = await hc.get("/dashboard/advisor/recommendations")
                    if resp.status_code == 200:
                        data = resp.json()
                        entries = data.get("entries", [])
                        exits = data.get("exits", [])
                        if not entries and not exits:
                            print("  No recommendations.")
                        if entries:
                            print("  Entry Recommendations:")
                            for e in entries:
                                print(f"    [{e.get('priority')}] {e.get('tier_label')} "
                                      f"— {e.get('direction')} {e.get('short_strike')}/{e.get('long_strike')} "
                                      f"DTE={e.get('dte')} x{e.get('num_contracts', 1)}")
                        if exits:
                            print("  Exit Recommendations:")
                            for e in exits:
                                print(f"    [{e.get('priority')}] {e.get('tier_label')} — {e.get('action')}")
                    else:
                        print(f"  Error: {resp.status_code}")
            elif cmd == "y" and len(parts) > 1:
                async with httpx.AsyncClient(base_url=server, timeout=30.0) as hc:
                    for p_str in parts[1:]:
                        try:
                            p_int = int(p_str)
                            resp = await hc.post("/trade/advisor/confirm", params={"priority": p_int})
                            if resp.status_code == 200:
                                data = resp.json()
                                print(f"  Confirmed: {data.get('message')}")
                            else:
                                print(f"  Error: {resp.json().get('detail', resp.text)}")
                        except ValueError:
                            print(f"  Invalid priority: {p_str}")
            elif cmd == "health":
                async with httpx.AsyncClient(base_url=server, timeout=5.0) as client:
                    resp = await client.get("/health")
                    print(f"  {resp.json()}")
            else:
                print(f"  Unknown command: {cmd}")
        except Exception as e:
            print(f"  Error: {e}")

    return 0


# ── portfolio ────────────────────────────────────────────────────────────────

async def _cmd_portfolio(args) -> int:
    """Show current positions, P&L, and account summary."""
    server = _detect_server(args)
    if server:
        rc = await _try_daemon(server, _cmd_portfolio_http, args)
        if rc is not None:
            return rc

    from app.services.position_store import get_position_store
    from app.services.dashboard_service import DashboardService

    mode = _get_mode(args)
    live_provider = None
    balances = None

    if mode in ("paper", "live"):
        # Connect to IBKR and sync positions before displaying
        live_provider = await _init_ibkr_readonly(args)
        store = get_position_store()
        if store:
            from app.services.position_sync import PositionSyncService
            from app.services.ledger import get_ledger
            ledger = get_ledger()
            if ledger:
                sync_svc = PositionSyncService(store, ledger)
                result = await sync_svc.sync_all_brokers()
                total = result.new_positions + result.updated_positions
                if total:
                    print(f"  Synced {total} position(s) from IBKR")
        # Fetch account balances from broker
        if live_provider:
            try:
                balances = await live_provider.get_account_balances()
            except Exception as e:
                logger.debug("Failed to fetch account balances: %s", e)
        print(f"  Mode: {_mode_label(mode)}")
    else:
        _init_read_only_services(args.data_dir, mode)

    store = get_position_store()
    if not store:
        print("  Position store not initialized")
        if live_provider:
            await _disconnect(live_provider)
        return 1

    svc = DashboardService(store)
    summary = svc.get_summary()

    # Populate summary with broker balances if available
    if balances:
        summary.cash_available = balances.cash
        summary.net_liquidation = balances.net_liquidation
        summary.buying_power = balances.buying_power
        summary.maint_margin_req = balances.maint_margin_req
        summary.available_funds = balances.available_funds

    _print_header("Portfolio Summary")

    # Fetch IBKR portfolio items once — used for both summary P&L and per-position display
    portfolio_items = []
    if live_provider and mode in ("paper", "live"):
        try:
            portfolio_items = await live_provider.get_portfolio_items()
        except Exception as e:
            logger.debug("Failed to fetch IBKR portfolio items: %s", e)

    # Account balances (from broker)
    if balances and balances.net_liquidation > 0:
        print(f"  Net Liquidation:  ${summary.net_liquidation:>14,.2f}")
        print(f"  Cash:             ${summary.cash_available:>14,.2f}")
        print(f"  Buying Power:     ${summary.buying_power:>14,.2f}")
        print(f"  Available Funds:  ${summary.available_funds:>14,.2f}")
        if summary.maint_margin_req > 0:
            print(f"  Margin Req:       ${summary.maint_margin_req:>14,.2f}")
        print()

    # Position P&L — prefer broker-reported unrealized when available
    broker_total_upnl = None
    if portfolio_items:
        broker_total_upnl = sum(item["unrealized_pnl"] for item in portfolio_items)

    unrealized = broker_total_upnl if broker_total_upnl is not None else summary.unrealized_pnl
    total_pnl = unrealized + summary.realized_pnl
    pnl_color = "92" if total_pnl >= 0 else "91"
    print(f"  Cash Deployed:    ${summary.cash_deployed:>14,.2f}")
    print(f"  Unrealized P&L:   {_color(f'${unrealized:>+14,.2f}', pnl_color)}")
    print(f"  Realized P&L:     {_color(f'${summary.realized_pnl:>+14,.2f}', pnl_color)}")
    print(f"  Total P&L:        {_color(f'${total_pnl:>+14,.2f}', pnl_color)}")

    if summary.positions_by_source:
        print(f"\n  Positions by source:")
        for src, count in summary.positions_by_source.items():
            print(f"    {src:>15}: {count}")

    _print_section("Active Positions")
    if summary.active_positions:
        # Match authoritative P&L from IBKR's portfolio data to our positions
        broker_pnl = {}
        if portfolio_items:
            try:
                # Convert TrackedPosition objects to dicts for the shared helper
                pos_dicts = []
                for pos in summary.active_positions:
                    pd = {
                        "position_id": pos.position_id,
                        "order_type": pos.order_type,
                        "symbol": pos.symbol,
                        "quantity": pos.quantity,
                        "expiration": pos.expiration,
                        "legs": pos.legs if isinstance(pos.legs, list) else [],
                    }
                    pos_dicts.append(pd)
                broker_pnl = _match_broker_pnl(portfolio_items, pos_dicts)
            except Exception as e:
                logger.debug("Failed to match IBKR portfolio items: %s", e)

        # Header
        has_marks = bool(broker_pnl)
        if has_marks:
            print(f"  {'Symbol':>8} {'Type':>14} {'Qty':>5} {'AvgCost':>10} {'Mark':>10} {'MktVal':>10} {'P&L':>12} {'Exp':>12} {'ID':>14}")
            print(f"  {'─' * 8} {'─' * 14} {'─' * 5} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 12} {'─' * 12} {'─' * 14}")
        else:
            print(f"  {'Symbol':>8} {'Type':>14} {'Qty':>5} {'Entry':>10} {'Expiration':>12} {'Source':>12} {'ID':>14}")
            print(f"  {'─' * 8} {'─' * 14} {'─' * 5} {'─' * 10} {'─' * 12} {'─' * 12} {'─' * 14}")

        total_broker_upnl = 0.0
        for pos in summary.active_positions:
            exp = pos.expiration or "—"
            src = pos.source.value if hasattr(pos.source, 'value') else str(pos.source)
            short_id = pos.position_id[:12] if pos.position_id else "—"

            if has_marks and pos.position_id in broker_pnl:
                pnl_data = broker_pnl[pos.position_id]
                upnl = pnl_data["unrealized_pnl"]
                mv = pnl_data["market_value"]
                avg_cost = pnl_data["avg_cost"]
                mark = pnl_data["market_price"]
                total_broker_upnl += upnl
                pnl_color = "92" if upnl >= 0 else "91"
                print(f"  {pos.symbol:>8} {pos.order_type:>14} {pos.quantity:>5.0f} "
                      f"${avg_cost:>9.4f} ${mark:>9.4f} ${mv:>9.2f} "
                      f"{_color(f'${upnl:>+10,.2f}', pnl_color):>20} {exp:>12} {short_id:>14}")
            elif has_marks:
                print(f"  {pos.symbol:>8} {pos.order_type:>14} {pos.quantity:>5.0f} "
                      f"${abs(pos.entry_price):>9.4f} {'—':>10} {'—':>10} {'—':>12} {exp:>12} {short_id:>14}")
            else:
                print(f"  {pos.symbol:>8} {pos.order_type:>14} {pos.quantity:>5.0f} "
                      f"${pos.entry_price:>9.2f} {exp:>12} {src:>12} {short_id:>14}")

        if has_marks:
            upnl_color = "92" if total_broker_upnl >= 0 else "91"
            print(f"  {'':>8} {'':>14} {'':>5} {'':>10} {'':>10} {'TOTAL':>10} "
                  f"{_color(f'${total_broker_upnl:>+10,.2f}', upnl_color):>20}")
    else:
        print("    (no open positions)")

    # Closed positions summary
    closed = store.get_closed_positions()
    if closed:
        recent = sorted(closed, key=lambda p: p.get("exit_time", ""), reverse=True)[:5]
        _print_section("Recent Closed (last 5)")
        for pos in recent:
            pnl = pos.get("pnl", 0)
            pc = "92" if pnl >= 0 else "91"
            sym = pos.get("symbol", "?")
            reason = pos.get("exit_reason", "?")
            print(f"  {sym:>8} | P&L={_color(f'${pnl:+,.2f}', pc):>20} | {reason}")

    print()
    if live_provider:
        await _disconnect(live_provider)
    return 0


# ── quote ────────────────────────────────────────────────────────────────────

async def _cmd_quote(args) -> int:
    """Get real-time quotes for one or more symbols."""
    server = _detect_server(args)
    if server:
        rc = await _try_daemon(server, _cmd_quote_http, args)
        if rc is not None:
            return rc

    mode = _get_mode(args)
    symbols = args.symbols

    if mode == "dry-run":
        # Stub provider gives fake data
        provider = await _init_services(args)
        from app.core.provider import ProviderRegistry
        from app.models import Broker
        ibkr = ProviderRegistry.get(Broker.IBKR)
    else:
        ibkr = await _init_ibkr_readonly(args)

    _print_header("Quotes")

    print(f"  {'Symbol':>8} {'Bid':>10} {'Ask':>10} {'Last':>10} {'Spread':>8} {'Volume':>12}  {'Updated':>12} {'Source'}")
    print(f"  {'─' * 8} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 8} {'─' * 12}  {'─' * 12} {'─' * 16}")

    import asyncio as _aio
    from datetime import datetime as _dt, timezone as _tz

    async def _fetch_quote(sym):
        try:
            return sym, await ibkr.get_quote(sym), None
        except Exception as e:
            return sym, None, e

    results = await _aio.gather(*[_fetch_quote(s) for s in symbols])
    now = _dt.now(_tz.utc)

    for symbol, q, err in results:
        if err:
            print(f"  {symbol:>8} {_color(f'ERROR: {err}', '91')}")
        else:
            spread = q.ask - q.bid if q.ask and q.bid else 0
            age = (now - q.timestamp).total_seconds() if q.timestamp else 0
            if age < 2:
                age_str = "just now"
            elif age < 60:
                age_str = f"{age:.0f}s ago"
            elif age < 3600:
                age_str = f"{age/60:.0f}m ago"
            else:
                age_str = f"{age/3600:.1f}h ago"
            print(f"  {q.symbol:>8} ${q.bid:>9.2f} ${q.ask:>9.2f} ${q.last:>9.2f} "
                  f"${spread:>6.2f} {q.volume:>12,}  {age_str:>12} {q.source}")

    print()

    if mode != "dry-run":
        await _disconnect(ibkr)
    else:
        await _disconnect(provider)
    return 0


# ── options ──────────────────────────────────────────────────────────────────

async def _cmd_options(args) -> int:
    """Show available option strikes and quotes for a symbol."""
    server = _detect_server(args)
    if server:
        rc = await _try_daemon(server, _cmd_options_http, args)
        if rc is not None:
            return rc

    from datetime import date as _date

    mode = _get_mode(args)
    symbol = args.symbol.upper()
    option_type = getattr(args, "type", "CALL").upper()
    expiration = getattr(args, "expiration", None)
    strike_min = getattr(args, "strike_min", None)
    strike_max = getattr(args, "strike_max", None)
    strike_range = getattr(args, "strike_range", 15)
    list_expirations = getattr(args, "list_expirations", False)

    # Default expiration to today
    if not expiration and not list_expirations:
        expiration = _date.today().isoformat()

    if mode == "dry-run":
        provider = await _init_services(args)
        from app.core.provider import ProviderRegistry
        from app.models import Broker
        ibkr = ProviderRegistry.get(Broker.IBKR)
    else:
        ibkr = await _init_ibkr_readonly(args)

    try:
        chain = await ibkr.get_option_chain(symbol)
    except Exception as e:
        print(f"  {_color(f'ERROR: {e}', '91')}")
        if mode != "dry-run":
            await _disconnect(ibkr)
        return 1

    expirations = chain.get("expirations", [])
    strikes = sorted(chain.get("strikes", []))

    if list_expirations:
        _print_header(f"Option Expirations — {symbol}")
        print(f"  {len(expirations)} expiration(s) available:")
        for exp in expirations[:50]:
            print(f"    {exp}")
        if len(expirations) > 50:
            print(f"    ... and {len(expirations) - 50} more")
        print()
        if mode != "dry-run":
            await _disconnect(ibkr)
        else:
            await _disconnect(provider)
        return 0

    # Check if expiration is available
    exp_normalized = expiration.replace("-", "")
    exp_match = None
    for e in expirations:
        if e.replace("-", "") == exp_normalized:
            exp_match = e
            break

    if not exp_match:
        print(f"  {_color(f'Expiration {expiration} not available for {symbol}', '91')}")
        # Show nearest expirations
        print(f"  Available expirations (nearest 10):")
        for e in expirations[:10]:
            print(f"    {e}")
        print()
        if mode != "dry-run":
            await _disconnect(ibkr)
        else:
            await _disconnect(provider)
        return 1

    # Auto-range: use --strike-range % (default 15%) around current price
    # to avoid qualifying hundreds of strikes. Explicit --strike-min/--strike-max
    # override this.
    import math
    auto_range = False
    range_source = ""
    pct = strike_range / 100.0
    if strike_min is None and strike_max is None:
        price = None
        # Try to get a live quote for the underlying
        if mode != "dry-run":
            try:
                q = await ibkr.get_quote(symbol)
                p = q.last or q.bid or q.ask
                if p and not math.isnan(p) and p > 0:
                    price = p
                    range_source = "current price"
            except Exception:
                pass
        # Fallback: use median of available strikes as proxy
        if price is None and strikes:
            price = strikes[len(strikes) // 2]
            range_source = "median strike"
        if price and price > 0:
            strike_min = round(price * (1 - pct), 2)
            strike_max = round(price * (1 + pct), 2)
            auto_range = True

    # Determine which types to fetch
    types_to_fetch = ["CALL", "PUT"] if option_type == "BOTH" else [option_type]

    _print_header(f"Options — {symbol} exp {exp_match}")
    print(f"  Total strikes available: {len(strikes)}")
    if auto_range:
        print(f"  Auto-range: ${strike_min:,.0f} - ${strike_max:,.0f} "
              f"(±{strike_range}% of {range_source}, use --strike-range/--strike-min/--strike-max to override)")

    def _fmt_price(v):
        if v is None or (isinstance(v, float) and (math.isnan(v) or v <= 0)):
            return "—"
        return f"${v:.2f}"

    # Fetch all option types concurrently — this is the main speedup when
    # fetching BOTH calls and puts, since each is an independent IBKR round-trip.
    import asyncio as _aio

    async def _fetch_one(otype):
        try:
            return otype, await ibkr.get_option_quotes(
                symbol, exp_match, otype,
                strike_min=strike_min, strike_max=strike_max,
            ), None
        except Exception as e:
            return otype, [], e

    fetch_results = await _aio.gather(*[_fetch_one(ot) for ot in types_to_fetch])

    for otype, quotes, err in fetch_results:
        if err is not None:
            print(f"  {_color(f'ERROR fetching {otype} quotes: {err}', '91')}")
            continue

        # Filter out junk rows: bid=-1 means no market, nan means no data
        valid_quotes = []
        for q in quotes:
            bid, ask, last = q["bid"], q["ask"], q["last"]
            has_bid = bid > 0 and not math.isnan(bid)
            has_ask = ask > 0 and not math.isnan(ask)
            has_last = last > 0 and not math.isnan(last)
            has_volume = q["volume"] > 0
            if has_bid or has_ask or has_last or has_volume:
                valid_quotes.append(q)

        filtered = len(quotes) - len(valid_quotes)
        _print_section(f"{otype}S ({len(valid_quotes)} strikes"
                       f"{f', {filtered} empty filtered' if filtered else ''})")

        if valid_quotes:
            # Determine max symbol width for alignment
            sym_width = max(len(q.get("symbol", "")) for q in valid_quotes)
            sym_width = max(sym_width, 6)  # minimum "Symbol" header width
            print(f"  {'Symbol':<{sym_width}} {'Strike':>10} {'Bid':>10} {'Ask':>10} {'Last':>10} {'Volume':>10} {'OI':>10}")
            print(f"  {'─' * sym_width} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10}")
            for q in valid_quotes:
                sym = q.get("symbol", "")
                print(f"  {sym:<{sym_width}} {q['strike']:>10.1f} {_fmt_price(q['bid']):>10} {_fmt_price(q['ask']):>10} "
                      f"{_fmt_price(q['last']):>10} {q['volume']:>10,} {q['open_interest']:>10,}")
        else:
            print("  (no quotes with data — market may be closed)")

    print()
    if mode != "dry-run":
        await _disconnect(ibkr)
    else:
        await _disconnect(provider)
    return 0


# ── margin ───────────────────────────────────────────────────────────────────

def _build_margin_order(subcommand: str, args):
    """Build a MultiLegOrder from margin subcommand args for margin checking."""
    from app.models import Broker, MultiLegOrder, OptionLeg, OptionAction, OptionType, OrderType

    valid_types = ("credit-spread", "debit-spread", "iron-condor", "option")
    if subcommand not in valid_types:
        raise ValueError(f"Unknown margin subcommand: {subcommand}")

    broker = Broker(getattr(args, "broker", "ibkr"))
    expiration = args.expiration
    quantity = getattr(args, "quantity", 1)
    net_price = getattr(args, "net_price", 1.00) or 1.00

    if subcommand == "credit-spread":
        option_type = OptionType(args.option_type.upper())
        legs = [
            OptionLeg(symbol=args.symbol, expiration=expiration, strike=args.short_strike,
                      option_type=option_type, action=OptionAction.SELL_TO_OPEN, quantity=1),
            OptionLeg(symbol=args.symbol, expiration=expiration, strike=args.long_strike,
                      option_type=option_type, action=OptionAction.BUY_TO_OPEN, quantity=1),
        ]
    elif subcommand == "debit-spread":
        option_type = OptionType(args.option_type.upper())
        legs = [
            OptionLeg(symbol=args.symbol, expiration=expiration, strike=args.long_strike,
                      option_type=option_type, action=OptionAction.BUY_TO_OPEN, quantity=1),
            OptionLeg(symbol=args.symbol, expiration=expiration, strike=args.short_strike,
                      option_type=option_type, action=OptionAction.SELL_TO_OPEN, quantity=1),
        ]
    elif subcommand == "iron-condor":
        legs = [
            OptionLeg(symbol=args.symbol, expiration=expiration, strike=args.put_short,
                      option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN, quantity=1),
            OptionLeg(symbol=args.symbol, expiration=expiration, strike=args.put_long,
                      option_type=OptionType.PUT, action=OptionAction.BUY_TO_OPEN, quantity=1),
            OptionLeg(symbol=args.symbol, expiration=expiration, strike=args.call_short,
                      option_type=OptionType.CALL, action=OptionAction.SELL_TO_OPEN, quantity=1),
            OptionLeg(symbol=args.symbol, expiration=expiration, strike=args.call_long,
                      option_type=OptionType.CALL, action=OptionAction.BUY_TO_OPEN, quantity=1),
        ]
    elif subcommand == "option":
        option_type = OptionType(args.option_type.upper())
        action = OptionAction(getattr(args, "action", "BUY_TO_OPEN").upper())
        legs = [
            OptionLeg(symbol=args.symbol, expiration=expiration, strike=args.strike,
                      option_type=option_type, action=action, quantity=1),
        ]
    else:
        raise ValueError(f"Unknown margin subcommand: {subcommand}")

    return MultiLegOrder(broker=broker, legs=legs, order_type=OrderType.LIMIT,
                         net_price=net_price, quantity=quantity)


async def _cmd_margin(args) -> int:
    """Check margin/cost for a hypothetical trade without executing."""
    server = _detect_server(args)
    if server:
        rc = await _try_daemon(server, _cmd_margin_http, args)
        if rc is not None:
            return rc

    subcommand = getattr(args, "margin_type", None)
    if not subcommand:
        print("  Error: specify a trade type (credit-spread, debit-spread, iron-condor, option)")
        return 1

    mode = _get_mode(args)

    try:
        order = _build_margin_order(subcommand, args)
    except Exception as e:
        print(f"  {_color('ERROR', '91')}: {e}")
        return 1

    _print_header(f"Margin Check: {subcommand.replace('-', ' ').title()}")
    print(f"  Mode:       {_mode_label(mode)}")
    print(f"  Symbol:     {args.symbol}")
    print(f"  Expiration: {args.expiration}")
    print(f"  Quantity:   {order.quantity}")

    # Show legs
    print(f"\n  Legs:")
    for i, leg in enumerate(order.legs, 1):
        right = leg.option_type.value[0]
        print(f"    {i}. {leg.action.value:15} {leg.symbol} {leg.strike}{right} {leg.expiration}")
    print()

    # Connect and check margin
    if mode == "dry-run":
        provider = await _init_services(args)
        from app.core.provider import ProviderRegistry
        from app.models import Broker
        ibkr = ProviderRegistry.get(Broker.IBKR)
    else:
        ibkr = await _init_ibkr_readonly(args)
        provider = ibkr

    try:
        import asyncio as _asyncio
        try:
            margin = await _asyncio.wait_for(ibkr.check_margin(order), timeout=30.0)
        except _asyncio.TimeoutError:
            margin = {"error": "Margin check timed out (10s) — may work during market hours",
                      "init_margin": 0.0, "maint_margin": 0.0, "commission": 0.0}

        _print_section("MARGIN REQUIREMENTS")

        if margin.get("error"):
            _print_step("Margin Check", "warn", margin["error"])
        else:
            _print_step("Margin Check", "pass", "Margin data received")

        init_margin = margin.get("init_margin", 0)
        maint_margin = margin.get("maint_margin", 0)
        commission = margin.get("commission", 0)

        print(f"\n  Initial Margin:      ${init_margin:>12,.2f}")
        print(f"  Maintenance Margin:  ${maint_margin:>12,.2f}")
        print(f"  Commission:          ${commission:>12,.2f}")

        # Additional margin details if available
        if margin.get("equity_with_loan"):
            print(f"  Equity w/ Loan:      ${margin['equity_with_loan']:>12,.2f}")
        if margin.get("init_margin_before"):
            print(f"  Init Margin Before:  ${margin['init_margin_before']:>12,.2f}")
            print(f"  Init Margin After:   ${margin.get('init_margin_after', 0):>12,.2f}")

        # Max loss calculation for spreads
        if subcommand == "credit-spread":
            width = abs(order.legs[0].strike - order.legs[1].strike)
            max_loss = width * 100 * order.quantity
            net_credit = (order.net_price or 0) * 100 * order.quantity
            print(f"\n  {'─' * 40}")
            print(f"  Spread Width:        ${width:>12,.2f}")
            print(f"  Max Loss (per qty):  ${max_loss:>12,.2f}")
            if net_credit > 0:
                print(f"  Net Credit:          ${net_credit:>12,.2f}")
                print(f"  Max Risk:            ${max_loss - net_credit:>12,.2f}")

        elif subcommand == "iron-condor":
            put_width = abs(order.legs[0].strike - order.legs[1].strike)
            call_width = abs(order.legs[2].strike - order.legs[3].strike)
            wider = max(put_width, call_width)
            max_loss = wider * 100 * order.quantity
            print(f"\n  {'─' * 40}")
            print(f"  Put Spread Width:    ${put_width:>12,.2f}")
            print(f"  Call Spread Width:   ${call_width:>12,.2f}")
            print(f"  Max Loss (wider):    ${max_loss:>12,.2f}")

    except Exception as e:
        _print_step("Margin Check", "fail", str(e))

    print()
    await _disconnect(provider)
    return 0


# ── auto-price helper ─────────────────────────────────────────────────────────

async def _auto_price_spread(provider, symbol: str, expiration: str,
                             strikes: list[float], option_type: str,
                             spread_type: str,
                             use_mid: bool = False) -> float | None:
    """Fetch live quotes for spread legs and compute the net price.

    For credit spreads (you sell short, buy long):
      market price = short_bid - long_ask  (immediate fill — you sell at bid, buy at ask)
      mid          = average of market and best-case

    For debit spreads (you buy long, sell short):
      market price = long_ask - short_bid  (immediate fill — you buy at ask, sell at bid)
      mid          = average of market and best-case

    Args:
        use_mid: If True, return mid-point between market and best-case.
                 If False (default), return the market price.

    Returns the selected price, or None if quotes unavailable.
    """
    if not hasattr(provider, "get_option_quotes"):
        return None

    strike_min = min(strikes) - 1
    strike_max = max(strikes) + 1

    quotes = await provider.get_option_quotes(
        symbol, expiration, option_type,
        strike_min=strike_min, strike_max=strike_max,
    )

    by_strike = {q["strike"]: q for q in quotes}
    missing = [s for s in strikes if s not in by_strike]
    if missing:
        return None

    if spread_type == "credit_spread":
        short_strike, long_strike = strikes[0], strikes[1]
        short_q = by_strike[short_strike]
        long_q = by_strike[long_strike]
        # Market: sell short at bid, buy long at ask
        market = round(short_q["bid"] - long_q["ask"], 2)
        # Best-case: sell short at ask, buy long at bid (unlikely to fill)
        best = round(short_q["ask"] - long_q["bid"], 2)
        mid = round((market + best) / 2, 2)
        print(f"  Short {short_strike} bid/ask: ${short_q['bid']:.2f} / ${short_q['ask']:.2f}")
        print(f"  Long  {long_strike} bid/ask: ${long_q['bid']:.2f} / ${long_q['ask']:.2f}")
        print(f"  Market: ${market:.2f}  |  Mid: ${mid:.2f}  |  Best: ${best:.2f}")
        return mid if use_mid else market
    elif spread_type == "debit_spread":
        long_strike, short_strike = strikes[0], strikes[1]
        long_q = by_strike[long_strike]
        short_q = by_strike[short_strike]
        # Market: buy long at ask, sell short at bid
        market = round(long_q["ask"] - short_q["bid"], 2)
        # Best-case: buy long at bid, sell short at ask (unlikely to fill)
        best = round(long_q["bid"] - short_q["ask"], 2)
        mid = round((market + best) / 2, 2)
        print(f"  Long  {long_strike} bid/ask: ${long_q['bid']:.2f} / ${long_q['ask']:.2f}")
        print(f"  Short {short_strike} bid/ask: ${short_q['bid']:.2f} / ${short_q['ask']:.2f}")
        print(f"  Market: ${market:.2f}  |  Mid: ${mid:.2f}  |  Best: ${best:.2f}")
        return mid if use_mid else market
    return None


async def _auto_price_iron_condor(provider, symbol: str, expiration: str,
                                  put_short: float, put_long: float,
                                  call_short: float, call_long: float,
                                  use_mid: bool = False) -> float | None:
    """Fetch live quotes for all 4 iron condor legs and compute net credit.

    Market price: sum of (sell at bid, buy at ask) for each spread wing.
    Mid: average of market and best-case.

    Args:
        use_mid: If True, return mid-point. If False (default), return market price.
    """
    if not hasattr(provider, "get_option_quotes"):
        return None

    all_strikes = [put_short, put_long, call_short, call_long]
    strike_min = min(all_strikes) - 1
    strike_max = max(all_strikes) + 1

    put_quotes = await provider.get_option_quotes(
        symbol, expiration, "PUT", strike_min=strike_min, strike_max=strike_max)
    call_quotes = await provider.get_option_quotes(
        symbol, expiration, "CALL", strike_min=strike_min, strike_max=strike_max)

    puts_by_strike = {q["strike"]: q for q in put_quotes}
    calls_by_strike = {q["strike"]: q for q in call_quotes}

    if put_short not in puts_by_strike or put_long not in puts_by_strike:
        return None
    if call_short not in calls_by_strike or call_long not in calls_by_strike:
        return None

    ps = puts_by_strike[put_short]
    pl = puts_by_strike[put_long]
    cs = calls_by_strike[call_short]
    cl = calls_by_strike[call_long]

    # Market: sell shorts at bid, buy longs at ask
    market = round((ps["bid"] - pl["ask"]) + (cs["bid"] - cl["ask"]), 2)
    # Best-case: sell shorts at ask, buy longs at bid
    best = round((ps["ask"] - pl["bid"]) + (cs["ask"] - cl["bid"]), 2)
    mid = round((market + best) / 2, 2)

    print(f"  Put  short {put_short} bid/ask: ${ps['bid']:.2f} / ${ps['ask']:.2f}"
          f"    long {put_long} bid/ask: ${pl['bid']:.2f} / ${pl['ask']:.2f}")
    print(f"  Call short {call_short} bid/ask: ${cs['bid']:.2f} / ${cs['ask']:.2f}"
          f"    long {call_long} bid/ask: ${cl['bid']:.2f} / ${cl['ask']:.2f}")
    print(f"  Market: ${market:.2f}  |  Mid: ${mid:.2f}  |  Best: ${best:.2f}")
    return mid if use_mid else market


# ── trade ────────────────────────────────────────────────────────────────────

async def _cmd_trade(args) -> int:
    """Execute a trade (equity, option, credit-spread, debit-spread, iron-condor)."""
    server = _detect_server(args)
    if server and not getattr(args, "validate_all", False):
        rc = await _try_daemon(server, _cmd_trade_http, args)
        if rc is not None:
            return rc

    mode = _get_mode(args)
    broker = getattr(args, "broker", "ibkr")

    if getattr(args, "validate_all", False):
        return await _run_validate_all(args)

    subcommand = getattr(args, "subcommand", None)
    if not subcommand:
        print("  Error: specify a trade type subcommand or --validate-all")
        print("  Trade types: equity, option, credit-spread, debit-spread, iron-condor")
        return 1

    instr = _build_instruction_from_args(subcommand, args)

    is_close = getattr(args, "close", False)
    close_label = " (CLOSE)" if is_close else ""
    _print_header(f"Trade: {subcommand.replace('-', ' ').title()}{close_label}")
    print(f"  Mode:      {_mode_label(mode)}")
    print(f"  Broker:    {broker}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ── Auto-price: fetch live quotes and compute net price ──
    auto_price = getattr(args, "auto_price", False)
    use_mid = getattr(args, "mid", False)
    needs_price = subcommand in ("credit-spread", "debit-spread", "iron-condor")
    no_price_given = instr.get("net_price") is None

    if needs_price and (auto_price or no_price_given) and mode != "dry-run":
        # Need a broker connection to fetch quotes — connect early
        live_provider = await _init_services(args)
        from app.core.provider import ProviderRegistry
        from app.models import Broker as _Broker
        ibkr = ProviderRegistry.get(_Broker.IBKR)

        close = getattr(args, "close", False)
        computed_price = None
        try:
            if subcommand == "credit-spread":
                # Closing a credit spread = buying it back = debit spread pricing
                spread_type = "debit_spread" if close else "credit_spread"
                computed_price = await _auto_price_spread(
                    ibkr, args.symbol, args.expiration,
                    [args.short_strike, args.long_strike],
                    args.option_type.upper(), spread_type,
                    use_mid=use_mid)
            elif subcommand == "debit-spread":
                spread_type = "credit_spread" if close else "debit_spread"
                computed_price = await _auto_price_spread(
                    ibkr, args.symbol, args.expiration,
                    [args.long_strike, args.short_strike],
                    args.option_type.upper(), spread_type,
                    use_mid=use_mid)
            elif subcommand == "iron-condor":
                computed_price = await _auto_price_iron_condor(
                    ibkr, args.symbol, args.expiration,
                    args.put_short, args.put_long,
                    args.call_short, args.call_long,
                    use_mid=use_mid)
        except Exception as e:
            print(f"  {_color('Auto-price failed', '93')}: {e}")

        if computed_price is not None and no_price_given:
            instr["net_price"] = computed_price
            price_label = "mid-point" if use_mid else "market"
            print(f"  {_color(f'Using {price_label} price', '92')}: ${computed_price:.2f}")
        elif computed_price is None and no_price_given:
            print(f"  {_color('WARNING', '93')}: Could not auto-price. Specify --net-price or quotes may be unavailable.")
            await _disconnect(live_provider)
            return 1
        print()
    else:
        live_provider = None

    detail_parts = [f"{k}={v}" for k, v in instr.items() if k not in ("id", "type") and v is not None]
    print(f"  Order: {', '.join(detail_parts)}")

    if mode == "live":
        print(f"\n  {_color('WARNING: This is a LIVE order!', '91')}")
        confirm = input(f"  Type 'yes' to proceed: ")
        if confirm.strip().lower() != "yes":
            print("  Aborted.")
            await _disconnect(live_provider)
            return 1

    if live_provider is None:
        live_provider = await _init_services(args)

    _print_section("EXECUTION")

    result = await _execute_single_order(
        instruction_dict=instr,
        broker_str=broker,
        mode=mode,
        poll_timeout=args.poll_timeout,
        poll_interval=args.poll_interval,
        closing_position_id=getattr(args, "closing_position_id", None),
        closing_quantity=getattr(args, "closing_quantity", None),
    )

    for title, status, detail in result["checks"]:
        _print_step(title, status, detail)

    print()
    if result["passed"]:
        print(f"  {_color('ORDER COMPLETE', '92')}")
    else:
        print(f"  {_color('ORDER FAILED', '91')}")

    if result["order_result"]:
        r = result["order_result"]
        print(f"  Order ID:   {r.order_id}")
        print(f"  Status:     {r.status.value}")
        if r.filled_price is not None:
            fill = r.filled_price
            # IBKR returns negative prices for credits — display as positive credit
            if fill < 0 and subcommand in ("credit-spread", "iron-condor"):
                print(f"  Credit:     ${abs(fill):.2f} per contract")
            elif fill > 0 and subcommand in ("debit-spread",):
                print(f"  Debit:      ${fill:.2f} per contract")
            else:
                print(f"  Fill Price: ${fill:.2f}")
    if result["position_id"]:
        print(f"  Position:   {result['position_id']}")

    # ── Verbose: P&L breakdown ──
    verbose = getattr(args, "verbose", False)
    if verbose and result["order_result"] and subcommand in ("credit-spread", "debit-spread", "iron-condor"):
        r = result["order_result"]
        fill = r.filled_price
        quantity = getattr(args, "quantity", 1)
        close = getattr(args, "close", False)

        _print_section("Trade Details")

        if subcommand == "credit-spread":
            short_strike = args.short_strike
            long_strike = args.long_strike
            spread_width = abs(long_strike - short_strike)

            if close:
                if fill is not None:
                    debit_paid = abs(fill)
                    print(f"  Action:       CLOSE (buy back spread)")
                    print(f"  Debit Paid:   ${debit_paid:.2f} x {quantity} x 100 = ${debit_paid * quantity * 100:,.2f}")
            else:
                credit = abs(fill) if fill is not None else (instr.get("net_price") or 0)
                max_profit = credit * quantity * 100
                max_loss = (spread_width - credit) * quantity * 100
                option_type = args.option_type.upper()

                if option_type == "CALL":
                    breakeven = short_strike + credit
                    win_condition = f"{args.symbol} stays below {short_strike}"
                else:
                    breakeven = short_strike - credit
                    win_condition = f"{args.symbol} stays above {short_strike}"

                print(f"  Credit:       ${credit:.2f} x {quantity} x 100 = ${max_profit:,.2f}")
                print(f"  Spread Width: ${spread_width:.2f} ({short_strike} / {long_strike})")
                print(f"  Max Profit:   {_color(f'${max_profit:,.2f}', '92')}")
                print(f"  Max Loss:     {_color(f'${max_loss:,.2f}', '91')}")
                print(f"  Breakeven:    {args.symbol} @ {breakeven:.2f}")
                print(f"  Win If:       {win_condition}")
                print(f"  Expiration:   {args.expiration}")

        elif subcommand == "debit-spread":
            long_strike = args.long_strike
            short_strike = args.short_strike
            spread_width = abs(short_strike - long_strike)

            if close:
                if fill is not None:
                    credit_recv = abs(fill)
                    print(f"  Action:       CLOSE (sell back spread)")
                    print(f"  Credit Recv:  ${credit_recv:.2f} x {quantity} x 100 = ${credit_recv * quantity * 100:,.2f}")
            else:
                debit = abs(fill) if fill is not None else (instr.get("net_price") or 0)
                max_loss_total = debit * quantity * 100
                max_profit_total = (spread_width - debit) * quantity * 100

                print(f"  Debit:        ${debit:.2f} x {quantity} x 100 = ${max_loss_total:,.2f}")
                print(f"  Spread Width: ${spread_width:.2f}")
                print(f"  Max Profit:   {_color(f'${max_profit_total:,.2f}', '92')}")
                print(f"  Max Loss:     {_color(f'${max_loss_total:,.2f}', '91')}")

        elif subcommand == "iron-condor":
            put_width = abs(args.put_short - args.put_long)
            call_width = abs(args.call_long - args.call_short)
            max_width = max(put_width, call_width)

            if close:
                if fill is not None:
                    debit_paid = abs(fill)
                    print(f"  Action:       CLOSE (buy back condor)")
                    print(f"  Debit Paid:   ${debit_paid:.2f} x {quantity} x 100 = ${debit_paid * quantity * 100:,.2f}")
            else:
                credit = abs(fill) if fill is not None else (instr.get("net_price") or 0)
                max_profit = credit * quantity * 100
                max_loss = (max_width - credit) * quantity * 100

                print(f"  Credit:       ${credit:.2f} x {quantity} x 100 = ${max_profit:,.2f}")
                print(f"  Put Spread:   {args.put_short} / {args.put_long} (${put_width:.2f} wide)")
                print(f"  Call Spread:  {args.call_short} / {args.call_long} (${call_width:.2f} wide)")
                print(f"  Max Profit:   {_color(f'${max_profit:,.2f}', '92')}")
                print(f"  Max Loss:     {_color(f'${max_loss:,.2f}', '91')}")
                print(f"  Win If:       {args.symbol} stays between {args.put_short} and {args.call_short}")

    print()

    await _disconnect(live_provider)
    return 0 if result["passed"] else 1


# ── playbook ─────────────────────────────────────────────────────────────────

async def _cmd_playbook(args) -> int:
    """Execute or validate a YAML playbook."""
    server = _detect_server(args)
    if server:
        rc = await _try_daemon(server, _cmd_playbook_http, args)
        if rc is not None:
            return rc
    action = getattr(args, "playbook_action", None)
    if not action:
        print("  Error: specify an action (execute or validate)")
        return 1

    playbook_file = args.playbook_file
    if not playbook_file:
        print("  Error: specify a playbook YAML file")
        return 1

    from pathlib import Path as P
    from app.services.playbook_service import PlaybookService, PlaybookValidationError

    path = P(playbook_file)
    if not path.exists():
        print(f"  Error: Playbook file not found: {path}")
        return 1

    service = PlaybookService()
    try:
        playbook = service.load(path)
    except PlaybookValidationError as e:
        print(f"  {_color('VALIDATION ERROR', '91')}: {e}")
        return 1

    _print_header(f"Playbook: {playbook.name}")
    print(f"  Description:  {playbook.description}")
    print(f"  Broker:       {playbook.broker.value}")
    print(f"  Instructions: {len(playbook.instructions)}")
    print()
    for instr in playbook.instructions:
        print(f"  [{instr.type:>15}] {instr.id}")
    print()

    if action == "validate":
        print(f"{'─' * 70}")
        print(f"  VALIDATION RESULTS")
        print(f"{'─' * 70}")
        validations = await service.validate(playbook)
        all_valid = True
        for v in validations:
            if v["valid"]:
                print(f"  [{_color('PASS', '92')}] {v['instruction_id']}")
            else:
                print(f"  [{_color('FAIL', '91')}] {v['instruction_id']}: {v['error']}")
                all_valid = False
        print()
        return 0 if all_valid else 1

    # Execute
    mode = _get_mode(args)
    dry_run = mode == "dry-run"

    live_provider = await _init_services(args)

    # Build fill-tracking hook for live orders
    post_submit_hook = None
    if not dry_run:
        from app.models import OrderResult, OrderStatus, TradeRequest
        from app.services.trade_service import await_order_fill, TERMINAL_STATUSES

        poll_timeout = args.poll_timeout
        poll_interval = args.poll_interval

        async def _fill_hook(instruction_id, trade_request, initial_result):
            order_id = initial_result.order_id
            broker = initial_result.broker
            short_id = order_id[:8]
            if initial_result.status in TERMINAL_STATUSES:
                return initial_result
            print(f"         Tracking order {short_id}... ", end="", flush=True)
            last_status = initial_result.status

            async def _on_update(result, elapsed):
                nonlocal last_status
                if result.status != last_status:
                    print(f"\n         [{result.status.value}] after {elapsed:.0f}s", end="", flush=True)
                    last_status = result.status
                else:
                    print(".", end="", flush=True)

            final = await await_order_fill(broker=broker, order_id=order_id,
                                           poll_interval=poll_interval, timeout=poll_timeout,
                                           on_status_update=_on_update)
            if final.status == OrderStatus.FILLED:
                price_str = f"${final.filled_price:.2f}" if final.filled_price else "market"
                print(f"\n         {_color('FILLED', '92')} at {price_str}")
            elif final.status in TERMINAL_STATUSES:
                print(f"\n         {_color(final.status.value, '91')}: {final.message}")
            else:
                print(f"\n         {_color('TIMEOUT', '93')}: still {final.status.value} after {poll_timeout:.0f}s")
            return final

        post_submit_hook = _fill_hook

    print(f"{'─' * 70}")
    print(f"  EXECUTION ({_mode_label(mode)})")
    print(f"{'─' * 70}")

    result = await service.execute(playbook, dry_run=dry_run, post_submit_hook=post_submit_hook)

    from app.models import OrderStatus
    print()
    for ir in result.results:
        if ir.status in ("success", "dry_run"):
            status_str = _color("PASS", "92")
            detail = ""
            if ir.order_result:
                parts = [f"order_id={ir.order_result.order_id[:8]}..."]
                if ir.order_result.filled_price is not None:
                    parts.append(f"fill=${ir.order_result.filled_price:.2f}")
                if ir.order_result.status == OrderStatus.FILLED:
                    parts.append("FILLED")
                elif ir.order_result.status == OrderStatus.SUBMITTED:
                    parts.append("SUBMITTED (may still fill)")
                detail = ", ".join(parts)
        elif ir.status == "failed":
            status_str = _color("FAIL", "91")
            detail = ir.error or ""
        else:
            status_str = _color("SKIP", "93")
            detail = ""

        print(f"  [{status_str}] {ir.instruction_id}")
        if detail:
            print(f"         {detail}")

    print(f"\n{'─' * 70}")
    print(f"  Total: {result.total}  |  "
          f"Succeeded: {_color(str(result.succeeded), '92')}  |  "
          f"Failed: {_color(str(result.failed), '91')}")
    print(f"{'─' * 70}\n")

    await _disconnect(live_provider)
    return 1 if result.failed > 0 else 0


# ── status ───────────────────────────────────────────────────────────────────

async def _cmd_status(args) -> int:
    """Dashboard: active positions, pending orders, recent closes."""
    server = _detect_server(args)
    if server:
        rc = await _try_daemon(server, _cmd_status_http, args)
        if rc is not None:
            return rc

    from app.services.position_store import get_position_store
    from app.services.dashboard_service import DashboardService

    mode = _get_mode(args)
    live_provider = None

    if mode in ("paper", "live"):
        live_provider = await _init_ibkr_readonly(args)
        store = get_position_store()
        if store:
            from app.services.position_sync import PositionSyncService
            from app.services.ledger import get_ledger
            ledger = get_ledger()
            if ledger:
                sync_svc = PositionSyncService(store, ledger)
                await sync_svc.sync_all_brokers()
        print(f"  Mode: {_mode_label(mode)}")
    else:
        _init_read_only_services(args.data_dir, mode)

    store = get_position_store()
    if not store:
        print("  Position store not initialized")
        if live_provider:
            await _disconnect(live_provider)
        return 1

    svc = DashboardService(store)
    status = svc.get_status()

    _print_header("System Status")

    # Active positions
    print(f"  Active Positions: {len(status.active_positions)}")
    print(f"{'─' * 70}")
    if status.active_positions:
        for pos in status.active_positions:
            exp_str = f" exp={pos.expiration}" if pos.expiration else ""
            src = pos.source.value if hasattr(pos.source, 'value') else str(pos.source)
            print(f"    {pos.symbol:>8} | {pos.order_type:>14} | "
                  f"qty={pos.quantity:.0f} | entry=${pos.entry_price:.2f}{exp_str} | {src}")
    else:
        print("    (none)")

    # In-transit orders
    print(f"\n  In-Transit Orders: {len(status.in_transit_orders)}")
    print(f"{'─' * 70}")
    if status.in_transit_orders:
        for order in status.in_transit_orders:
            print(f"    {order.get('symbol', '?'):>8} | {order.get('type', '?'):>10} | "
                  f"id={order.get('order_id', '?')[:8]}...")
    else:
        print("    (none)")

    # Recent closed
    print(f"\n  Recent Closed Positions: {len(status.recent_closed)}")
    print(f"{'─' * 70}")
    if status.recent_closed:
        for pos in status.recent_closed[:10]:
            pnl = pos.get("pnl", 0)
            pnl_color = "92" if pnl >= 0 else "91"
            print(f"    {pos.get('symbol', '?'):>8} | "
                  f"P&L={_color(f'${pnl:+,.2f}', pnl_color)} | "
                  f"reason={pos.get('exit_reason', '?')}")
    else:
        print("    (none)")

    # Connection status
    if status.connection_status:
        print(f"\n  Broker Connections:")
        print(f"{'─' * 70}")
        for broker_name, info in status.connection_status.items():
            connected = info.get("connected", False)
            status_str = _color("CONNECTED", "92") if connected else _color("DISCONNECTED", "91")
            print(f"    {broker_name:>10}: {status_str}")

    # Cache stats
    if status.cache_stats:
        print(f"\n  Cache Stats:")
        print(f"{'─' * 70}")
        for key, val in status.cache_stats.items():
            print(f"    {key:>25}: {val}")

    print()
    if live_provider:
        await _disconnect(live_provider)
    return 0


# ── flush ─────────────────────────────────────────────────────────────────────

async def _cmd_flush(args) -> int:
    """Flush local position store and/or ledger."""
    server = _detect_server(args)
    if server:
        # Route through daemon's flush endpoint (clears in-memory + disk)
        import httpx
        async with httpx.AsyncClient(base_url=server, timeout=30.0) as client:
            resp = await client.post("/account/flush")
            if resp.status_code != 200:
                print(f"  Error: {resp.status_code} {resp.text}")
                return 1
            d = resp.json()
            print(f"  Flushed via daemon:")
            print(f"    Open cleared:     {d.get('open_cleared', 0)}")
            print(f"    Closed preserved: {d.get('closed_preserved', 0)}")
            print(f"    Re-synced:        {d.get('synced_new', 0)} new")
        return 0
    import json

    data_dir = _resolve_data_dir(args.data_dir, _get_mode(args))
    positions_file = data_dir / "positions.json"
    ledger_file = data_dir / "ledger" / "ledger.jsonl"
    snapshots_dir = data_dir / "ledger" / "snapshots"

    flushed = []

    if args.what in ("positions", "all"):
        if positions_file.exists():
            # Count before flush
            try:
                with open(positions_file) as f:
                    count = len(json.load(f))
            except Exception:
                count = 0
            positions_file.write_text("{}")
            flushed.append(f"positions ({count} entries cleared)")
        else:
            flushed.append("positions (already empty)")

    if args.what in ("ledger", "all"):
        if ledger_file.exists():
            count = sum(1 for _ in open(ledger_file))
            ledger_file.write_text("")
            flushed.append(f"ledger ({count} entries cleared)")
        else:
            flushed.append("ledger (already empty)")
        # Clear snapshots too
        if snapshots_dir.exists():
            import shutil
            snap_count = len(list(snapshots_dir.glob("*.json")))
            shutil.rmtree(snapshots_dir)
            snapshots_dir.mkdir(parents=True, exist_ok=True)
            if snap_count:
                flushed.append(f"snapshots ({snap_count} cleared)")

    _print_header("Flush Complete")
    for item in flushed:
        print(f"  {_color('✓', '92')} {item}")

    print(f"\n  Data dir: {data_dir}")
    print()
    return 0


# ── reconcile ────────────────────────────────────────────────────────────────

async def _cmd_reconcile(args) -> int:
    """Compare system vs broker positions, optionally flushing local state first."""
    server = _detect_server(args)
    if server:
        rc = await _try_daemon(server, _cmd_reconcile_http, args)
        if rc is not None:
            return rc

    import json as _json
    from app.services.position_store import get_position_store, init_position_store
    from app.services.position_sync import PositionSyncService
    from app.services.ledger import get_ledger, init_ledger
    from app.models import Broker

    mode = _get_mode(args)
    data_dir = _resolve_data_dir(getattr(args, "data_dir", "data/utp"), mode)

    # ── Hard reset: clear EVERYTHING ──
    if getattr(args, "hard_reset", False):
        _print_header("HARD RESET — Clearing ALL Local Data")

        positions_file = data_dir / "positions.json"
        ledger_file = data_dir / "ledger" / "ledger.jsonl"
        snapshots_dir = data_dir / "ledger" / "snapshots"
        exec_file = data_dir / "executions.json"

        for f, label in [(positions_file, "Positions"), (ledger_file, "Ledger"), (exec_file, "Executions")]:
            if f.exists():
                count = 0
                try:
                    if f.suffix == ".json":
                        count = len(_json.load(open(f)))
                    else:
                        count = sum(1 for _ in open(f))
                except Exception:
                    pass
                if f.suffix == ".json":
                    f.write_text("{}")
                else:
                    f.write_text("")
                print(f"  {_color('✓', '92')} {label} cleared ({count} entries)")
            else:
                print(f"  {_color('✓', '92')} {label} (already empty)")

        if snapshots_dir.exists():
            import shutil
            snap_count = len(list(snapshots_dir.glob("*.json")))
            shutil.rmtree(snapshots_dir)
            snapshots_dir.mkdir(parents=True, exist_ok=True)
            if snap_count:
                print(f"  {_color('✓', '92')} Snapshots cleared ({snap_count})")

        print(f"\n  {_color('All local state cleared.', '93')} Rebuilding from broker...\n")
        # Fall through to connect and sync

    # ── Flush local state before connecting ──
    elif getattr(args, "flush", False):
        positions_file = data_dir / "positions.json"
        ledger_file = data_dir / "ledger" / "ledger.jsonl"
        snapshots_dir = data_dir / "ledger" / "snapshots"

        _print_header("Flushing Local State (preserving closed positions)")

        # Preserve closed positions — they contain P&L history that IBKR
        # doesn't provide via API. Only flush open positions.
        closed_preserved = 0
        open_flushed = 0
        if positions_file.exists():
            try:
                with open(positions_file) as f:
                    all_positions = _json.load(f)
                # Keep closed positions, remove open ones
                preserved = {}
                for pid, pos in all_positions.items():
                    if pos.get("status") == "closed":
                        preserved[pid] = pos
                        closed_preserved += 1
                    else:
                        open_flushed += 1
                positions_file.write_text(_json.dumps(preserved, indent=2))
                print(f"  {_color('✓', '92')} Open positions cleared ({open_flushed}), "
                      f"closed preserved ({closed_preserved})")
            except Exception:
                positions_file.write_text("{}")
                print(f"  {_color('✓', '92')} Positions cleared (parse error)")
        else:
            print(f"  {_color('✓', '92')} Positions (already empty)")

        if ledger_file.exists():
            count = sum(1 for _ in open(ledger_file))
            ledger_file.write_text("")
            print(f"  {_color('✓', '92')} Ledger cleared ({count} entries)")
        else:
            print(f"  {_color('✓', '92')} Ledger (already empty)")

        if snapshots_dir.exists():
            import shutil
            snap_count = len(list(snapshots_dir.glob("*.json")))
            shutil.rmtree(snapshots_dir)
            snapshots_dir.mkdir(parents=True, exist_ok=True)
            if snap_count:
                print(f"  {_color('✓', '92')} Snapshots cleared ({snap_count})")

        print(f"  {_color('Note:', '93')} Closed positions preserved for P&L history. "
              f"Execution cache (executions.json) untouched.")

        print()

    # ── Connect and reconcile ──
    if mode == "dry-run":
        provider = await _init_services(args)
    else:
        provider = await _init_ibkr_readonly(args)

    store = get_position_store()
    ledger = get_ledger()
    if not store or not ledger:
        print("  Services not initialized")
        await _disconnect(provider)
        return 1

    # After flush/hard-reset, sync broker positions into the now-empty store
    if (getattr(args, "flush", False) or getattr(args, "hard_reset", False)) and mode != "dry-run":
        sync_service = PositionSyncService(store, ledger)
        result = await sync_service.sync_all_brokers()
        total = result.new_positions + result.updated_positions
        if total:
            print(f"  Synced {total} position(s) from broker into clean store\n")

    # Fetch and merge IBKR executions (last ~7 days) for trade grouping
    if mode != "dry-run" and provider and hasattr(provider, "get_executions"):
        try:
            from app.services.execution_store import get_execution_store, init_execution_store
            exec_store = get_execution_store()
            if not exec_store:
                exec_store = init_execution_store(data_dir)
            raw_execs = await provider.get_executions()
            if raw_execs:
                new_count = exec_store.merge_executions(raw_execs)
                if new_count:
                    print(f"  Fetched {len(raw_execs)} executions from IBKR, {new_count} new (total: {exec_store.count})")
                else:
                    print(f"  Executions up to date ({exec_store.count} stored)")
        except Exception as e:
            logger.debug("Failed to fetch IBKR executions: %s", e)

    sync_service = PositionSyncService(store, ledger)
    report = await sync_service.reconcile(Broker.IBKR)

    _print_header("Reconciliation Report")
    print(f"  Mode:               {_mode_label(mode)}")
    print(f"  Broker:             {report.broker}")
    print(f"  System Positions:   {report.total_system_positions}")
    print(f"  Broker Positions:   {report.total_broker_positions}")
    print(f"  Matched:            {report.matched}")
    discrepancy_count = len([d for d in report.discrepancies if d.discrepancy_type != "matched"])
    print(f"  Discrepancies:      {discrepancy_count}")

    if report.discrepancies:
        print(f"\n{'─' * 70}")
        for entry in report.discrepancies:
            if entry.discrepancy_type == "matched":
                icon = _color("OK", "92")
            elif entry.discrepancy_type == "quantity_mismatch":
                icon = _color("MISMATCH", "93")
            elif entry.discrepancy_type == "missing_in_system":
                icon = _color("MISSING SYS", "91")
            else:
                icon = _color("MISSING BRK", "91")
            print(f"  [{icon:>20}] {entry.symbol}: {entry.details}")

        # Explain mismatches
        mismatches = [d for d in report.discrepancies if d.discrepancy_type == "quantity_mismatch"]
        if mismatches:
            print(f"\n  {_color('Note:', '93')} Quantity mismatches can occur when:")
            print(f"    - Broker reports negative qty for short positions (e.g. -1 = short)")
            print(f"    - Partial fills changed position size outside the system")
            print(f"    - Manual trades were placed directly through the broker")

    # ── Show synced positions ──
    if getattr(args, "show", False) or getattr(args, "portfolio", False):
        open_positions = store.get_open_positions()
        if open_positions:
            _print_section("Synced Positions")
            print(f"  {'Symbol':>8} {'Type':>14} {'Qty':>6} {'Entry':>10} {'Source':>15}")
            print(f"  {'─' * 8} {'─' * 14} {'─' * 6} {'─' * 10} {'─' * 15}")
            for pos in open_positions:
                sym = pos.get("symbol", "?")
                otype = pos.get("order_type", "?")
                qty = pos.get("quantity", 0)
                entry = pos.get("entry_price", 0)
                src = pos.get("source", "?")
                print(f"  {sym:>8} {otype:>14} {qty:>6.1f} ${entry:>9.2f} {src:>15}")
        else:
            _print_section("Synced Positions")
            print("    (no open positions)")

    if getattr(args, "portfolio", False):
        from app.services.dashboard_service import DashboardService
        svc = DashboardService(store)
        summary = svc.get_summary()

        # Fetch account balances if connected to broker
        if mode != "dry-run" and provider:
            try:
                balances = await provider.get_account_balances()
                if balances and balances.net_liquidation > 0:
                    _print_section("Account Balances")
                    print(f"  Net Liquidation:  ${balances.net_liquidation:>14,.2f}")
                    print(f"  Cash:             ${balances.cash:>14,.2f}")
                    print(f"  Buying Power:     ${balances.buying_power:>14,.2f}")
                    print(f"  Available Funds:  ${balances.available_funds:>14,.2f}")
                    if balances.maint_margin_req > 0:
                        print(f"  Margin Req:       ${balances.maint_margin_req:>14,.2f}")
            except Exception:
                pass

        _print_section("Portfolio Summary")
        pnl_color = "92" if summary.total_pnl >= 0 else "91"
        print(f"  Cash Deployed:    ${summary.cash_deployed:>14,.2f}")
        print(f"  Unrealized P&L:   {_color(f'${summary.unrealized_pnl:>+14,.2f}', pnl_color)}")
        print(f"  Realized P&L:     {_color(f'${summary.realized_pnl:>+14,.2f}', pnl_color)}")
        print(f"  Total P&L:        {_color(f'${summary.total_pnl:>+14,.2f}', pnl_color)}")
        if summary.positions_by_source:
            print(f"\n  Positions by source:")
            for src, count in summary.positions_by_source.items():
                print(f"    {src:>15}: {count}")

    print()
    await _disconnect(provider)
    return 0


# ── readiness ────────────────────────────────────────────────────────────────

async def _cmd_readiness(args) -> int:
    """Test IBKR connectivity and trade-type support."""
    server = _detect_server(args)
    if server:
        # Readiness needs direct IBKR — can't go through daemon (tests order submission)
        print(f"  {_color('Note:', '93')} Daemon detected at {server}, but readiness test needs a")
        print(f"  dedicated IBKR connection. Use a different --client-id:")
        print(f"  python utp.py readiness --symbol SPX --paper --client-id 11")
        return 1

    mode = _get_mode(args)

    if mode == "dry-run":
        print("  Readiness testing requires a broker connection.")
        print("  Use --paper (port 7497) or --live (port 7496).")
        print(f"\n  Example: python utp.py readiness --symbol SPX --paper --port 7497")
        return 1

    port = getattr(args, "port", None)
    if port is None:
        port = 7497 if mode == "paper" else 7496

    ready_args = argparse.Namespace(
        symbol=args.symbol,
        expiration=getattr(args, "expiration", None),
        host=getattr(args, "host", "127.0.0.1"),
        port=port,
        client_id=getattr(args, "client_id", 10),
        exchange=getattr(args, "exchange", None) or "SMART",
        skip_margin=getattr(args, "skip_margin", False),
        market_data_type=getattr(args, "market_data_type", 4),
        timeout=15,
        spread_width=None,
        quantity=1,
    )

    return await _run_readiness_test(ready_args)


# ── server ───────────────────────────────────────────────────────────────────

def _cmd_server(args) -> int:
    """Start the REST API server."""
    import uvicorn

    host = getattr(args, "server_host", "0.0.0.0")
    port = getattr(args, "server_port", 8000)
    reload = getattr(args, "reload", False)

    print(f"  Starting UTP API server on {host}:{port}")
    if reload:
        print(f"  Auto-reload enabled")
    print()

    uvicorn.run("app.main:app", host=host, port=port, reload=reload)
    return 0


# ── executions ────────────────────────────────────────────────────────────────

async def _cmd_executions(args) -> int:
    """Show IBKR executions grouped by order — identifies multi-leg trades."""
    server = _detect_server(args)
    if server:
        rc = await _try_daemon(server, _cmd_executions_http, args)
        if rc is not None:
            return rc

    from app.services.execution_store import ExecutionStore, init_execution_store

    mode = _get_mode(args)
    if mode == "dry-run":
        print("  --live or --paper required for execution history")
        return 1

    data_dir = _resolve_data_dir(getattr(args, "data_dir", "data/utp"), mode)
    exec_store = init_execution_store(data_dir)

    # Flush if requested
    if getattr(args, "flush", False):
        count = exec_store.flush()
        print(f"  Cleared {count} cached executions")

    # Fetch from IBKR
    provider = await _init_ibkr_readonly(args)
    try:
        if hasattr(provider, "get_executions"):
            raw = await provider.get_executions()
            if raw:
                new_count = exec_store.merge_executions(raw)
                print(f"  Fetched {len(raw)} executions from IBKR, {new_count} new (total: {exec_store.count})")
            else:
                print(f"  No executions returned from IBKR (session may have restarted)")
                print(f"  Cached: {exec_store.count} executions")
        else:
            print("  Provider does not support execution history")
            await _disconnect(provider)
            return 1
    finally:
        await _disconnect(provider)

    # Group and display
    groups = exec_store.get_grouped_by_order()

    # Filter by symbol if requested
    symbol_filter = getattr(args, "symbol", None)
    if symbol_filter:
        symbol_filter = symbol_filter.upper()
        groups = [g for g in groups if g["symbol"] == symbol_filter]

    if not groups:
        print("  No executions found")
        return 0

    _print_header(f"Execution History — {len(groups)} Orders ({exec_store.count} fills)")

    for g in groups:
        otype = g["order_type"]
        sym = g["symbol"]
        exp = g.get("expiration", "")
        time_str = g["time"][:19] if g["time"] else "?"
        net = g["net_amount"]
        comm = g["total_commission"]
        perm = g["perm_id"]
        legs = g["legs"]

        # Color code by type
        if "spread" in otype or "condor" in otype:
            type_color = "95"  # purple
        elif otype == "single_option":
            type_color = "96"  # cyan
        else:
            type_color = "97"  # white

        net_color = "92" if net >= 0 else "91"
        net_label = "credit" if net > 0 else "debit"

        print(f"\n  {_color(otype.upper().replace('_', ' '), type_color):>28} | "
              f"{sym:>6} | {time_str} | "
              f"{_color(f'${net:>+10,.2f}', net_color)} ({net_label}) | "
              f"comm=${comm:.2f} | perm={perm}")

        if exp:
            print(f"  {'':>28}   exp: {exp}")

        # Show legs
        for leg in legs:
            side = leg["side"]
            side_color = "91" if side == "SLD" else "92"
            sec = leg.get("sec_type", "")

            if sec in ("OPT", "FOP"):
                right = leg.get("right", "?")
                strike = leg.get("strike", 0)
                desc = f"{right}{strike:.0f}"
                print(f"  {'':>8} {_color(side, side_color):>12} "
                      f"{desc:>10} x{leg['shares']:.0f} @ ${leg['price']:.2f} "
                      f"(conId={leg.get('con_id', '?')})")
            else:
                print(f"  {'':>8} {_color(side, side_color):>12} "
                      f"{leg['symbol']:>10} x{leg['shares']:.0f} @ ${leg['price']:.2f}")

    print()
    return 0


# ── daemon ───────────────────────────────────────────────────────────────────

async def _cmd_daemon(args) -> int:
    """Start long-running daemon: IBKR connection + background loops + HTTP API."""
    import signal
    import uvicorn
    from app.core.provider import ProviderRegistry
    from app.services.ledger import init_ledger
    from app.services.position_store import init_position_store

    mode = _get_mode(args)
    if mode == "dry-run":
        mode = "paper"  # daemon should always connect

    data_dir = _resolve_data_dir(getattr(args, "data_dir", "data/utp"), mode)
    init_ledger(data_dir)
    init_position_store(data_dir)
    from app.services.execution_store import init_execution_store
    init_execution_store(data_dir)

    # Initialize IBKR provider
    live_provider = None
    if mode in ("live", "paper"):
        import app.config
        from app.config import Settings

        port = getattr(args, "port", None)
        if port is None:
            port = 7497 if mode == "paper" else 7496

        os.environ["IBKR_HOST"] = getattr(args, "host", "127.0.0.1")
        os.environ["IBKR_PORT"] = str(port)
        os.environ["IBKR_CLIENT_ID"] = str(getattr(args, "client_id", 10))
        os.environ["IBKR_READONLY"] = "false"
        exchange = getattr(args, "exchange", None)
        if exchange:
            os.environ["IBKR_EXCHANGE"] = exchange
        app.config.settings = Settings()

        from app.core.providers.ibkr import IBKRLiveProvider
        live_provider = IBKRLiveProvider(exchange=exchange or None)
        ProviderRegistry.register(live_provider)
        try:
            await live_provider.connect()
        except (ConnectionRefusedError, OSError, Exception) as e:
            print(f"  IBKR not available: {e}")
            print(f"  Starting in degraded mode — will retry connection in background.")
            live_provider._connected = False
    else:
        from app.core.providers.robinhood import RobinhoodProvider
        from app.core.providers.etrade import EtradeProvider
        from app.core.providers.ibkr import IBKRProvider
        for p in [RobinhoodProvider(), EtradeProvider(), IBKRProvider()]:
            ProviderRegistry.register(p)
            await p.connect()

    # Initialize LiveDataService (IBKR-primary with local fallback)
    from app.services.dashboard_service import DashboardService
    from app.services.live_data_service import init_live_data_service
    from app.services.position_store import get_position_store
    _daemon_store = get_position_store()
    if _daemon_store:
        _daemon_dashboard = DashboardService(_daemon_store)
        init_live_data_service(_daemon_store, _daemon_dashboard, live_provider)

    # Set daemon mode flag to prevent lifespan from re-initializing
    import app.main
    app.main._daemon_mode = True

    server_host = getattr(args, "server_host", "0.0.0.0")
    server_port = getattr(args, "server_port", 8000)

    print(f"\n  UTP Daemon starting [{_mode_label(mode)}]")
    print(f"  HTTP API: http://{server_host}:{server_port}")
    if live_provider:
        ibkr_status = "connected" if live_provider.is_healthy() else "degraded (retrying in background)"
        print(f"  IBKR: {ibkr_status}")
    advisor_profile = getattr(args, "advisor_profile", None)
    if advisor_profile:
        print(f"  Advisor: {advisor_profile}")
    print()

    # Start background tasks
    from app.services.ledger import get_ledger
    from app.websocket import ws_manager

    shutdown_event = asyncio.Event()
    bg_tasks: list[asyncio.Task] = []

    # Expiration loop
    async def _expiration_bg():
        from app.services.expiration_service import ExpirationService
        from datetime import UTC, datetime
        from app.config import settings
        while not shutdown_event.is_set():
            await asyncio.sleep(settings.expiration_check_interval_seconds)
            if shutdown_event.is_set():
                break
            store = get_position_store()
            ledger = get_ledger()
            if store and ledger:
                try:
                    svc = ExpirationService(store, ledger, ws_manager)
                    await svc.check_expirations(datetime.now(UTC).date())
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error("Expiration loop: %s", e)

    bg_tasks.append(asyncio.create_task(_expiration_bg()))

    # Position sync loop
    async def _sync_bg():
        from app.services.position_sync import PositionSyncService
        from datetime import UTC, datetime
        from app.config import settings
        while not shutdown_event.is_set():
            await asyncio.sleep(settings.position_sync_interval_seconds)
            if shutdown_event.is_set():
                break
            store = get_position_store()
            ledger = get_ledger()
            if store and ledger:
                svc = PositionSyncService(store, ledger, ws_manager)
                now = datetime.now(UTC)
                if svc.is_trading_hours(now):
                    try:
                        await svc.sync_all_brokers()
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).error("Sync loop: %s", e)

    bg_tasks.append(asyncio.create_task(_sync_bg()))

    # Market data streaming loop (if --streaming-config provided)
    streaming_config_path = getattr(args, "streaming_config", None)
    _streaming_svc = None
    if streaming_config_path:
        try:
            from app.services.streaming_config import load_streaming_config
            from app.services.market_data_streaming import init_streaming_service
            stream_cfg = load_streaming_config(streaming_config_path)
            _streaming_svc = init_streaming_service(stream_cfg, live_provider)

            async def _streaming_bg():
                import logging as _slog
                _slog_log = _slog.getLogger("utp.streaming")
                try:
                    await _streaming_svc.start()
                    # Keep running until shutdown
                    while not shutdown_event.is_set():
                        await asyncio.sleep(1)
                        # If IBKR reconnected and streaming lost subs, resubscribe
                        if (live_provider and live_provider.is_healthy()
                                and _streaming_svc.subscription_count == 0
                                and len(stream_cfg.symbols) > 0):
                            _slog_log.info("IBKR reconnected — resubscribing streaming")
                            await _streaming_svc.resubscribe_all()
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    _slog_log.error("Streaming error: %s", e)
                finally:
                    await _streaming_svc.stop()

            bg_tasks.append(asyncio.create_task(_streaming_bg()))
            print(f"  Streaming: {len(stream_cfg.symbols)} symbols from {streaming_config_path}")
        except Exception as e:
            print(f"  Streaming config error: {e}")
            logger.error("Failed to load streaming config: %s", e)

    # IBKR initial connection retry loop (if started in degraded mode)
    if live_provider and not live_provider.is_healthy():
        async def _ibkr_initial_connect_bg():
            import logging
            _log = logging.getLogger("utp.ibkr_connect")
            delay = 2.0
            backoff_cap = 10.0
            attempt = 0
            while not shutdown_event.is_set():
                attempt += 1
                _log.info("IBKR connect attempt %d (delay=%.1fs)", attempt, delay)
                await asyncio.sleep(delay)
                if shutdown_event.is_set():
                    return
                try:
                    await live_provider.connect()
                    _log.info("IBKR connected on attempt %d", attempt)
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    _log.warning("IBKR connect attempt %d failed: %s", attempt, e)
                    delay = min(delay * 2, backoff_cap)

        bg_tasks.append(asyncio.create_task(_ibkr_initial_connect_bg()))

    # Advisor loop (if profile specified)
    advisor_profile = getattr(args, "advisor_profile", None)
    auto_execute = getattr(args, "auto_execute", False)
    if advisor_profile:
        async def _advisor_bg():
            """Run advisor tier evaluation every 60s during market hours."""
            import logging
            from datetime import UTC, datetime, time as _time
            _log = logging.getLogger("utp.advisor")

            _daemon_state["advisor_profile"] = advisor_profile

            # Try to load the advisor profile
            try:
                project_root = str(Path(__file__).resolve().parent.parent)
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)

                from scripts.live_trading.advisor.profile_loader import load_profile
                from scripts.live_trading.advisor.tier_evaluator import TierEvaluator
                from scripts.live_trading.advisor.position_tracker import PositionTracker

                profile = load_profile(advisor_profile)
                evaluator = TierEvaluator(profile, PositionTracker(profile.name))
                evaluator.setup()
                _log.info("Advisor loaded: %s (%d tiers)", profile.name, len(profile.tiers))
            except ImportError as e:
                _log.warning("Advisor modules not available: %s — running in stub mode", e)
                profile = None
                evaluator = None
            except Exception as e:
                _log.error("Failed to load advisor profile '%s': %s", advisor_profile, e)
                profile = None
                evaluator = None

            while not shutdown_event.is_set():
                await asyncio.sleep(60)
                if shutdown_event.is_set():
                    break

                now = datetime.now(UTC)
                # Only during market hours (13:30-20:00 UTC)
                market_open = _time(13, 30)
                market_close = _time(20, 0)
                if not (market_open <= now.time() <= market_close):
                    continue

                if evaluator is None:
                    # Stub mode: just update timestamp
                    _daemon_state["advisor_last_eval"] = now.isoformat()
                    continue

                try:
                    price = evaluator.get_current_price()
                    if price is None:
                        continue

                    entries = evaluator.evaluate_entries(price, now)
                    exits = evaluator.evaluate_exits(price, now)

                    _daemon_state["advisor_entries"] = [
                        {
                            "tier_label": getattr(e, "tier_label", str(e)),
                            "direction": getattr(e, "direction", "unknown"),
                            "short_strike": getattr(e, "short_strike", None),
                            "long_strike": getattr(e, "long_strike", None),
                            "credit": getattr(e, "credit", None),
                            "dte": getattr(e, "dte", None),
                            "num_contracts": getattr(e, "num_contracts", 1),
                            "priority": i,
                        }
                        for i, e in enumerate(entries, 1)
                    ]
                    _daemon_state["advisor_exits"] = [
                        {
                            "tier_label": getattr(e, "tier_label", str(e)),
                            "action": getattr(e, "action", "unknown"),
                            "reason": getattr(e, "reason", ""),
                            "priority": i,
                        }
                        for i, e in enumerate(exits, 1)
                    ]
                    _daemon_state["advisor_last_eval"] = now.isoformat()

                    if entries or exits:
                        _log.info("Advisor: %d entries, %d exits at price=%.2f",
                                  len(entries), len(exits), price)

                    # Auto-execute if enabled
                    if auto_execute and entries:
                        for rec in _daemon_state["advisor_entries"]:
                            if rec.get("short_strike") and rec.get("long_strike"):
                                try:
                                    ticker = profile.ticker if profile else "SPX"
                                    dte = rec.get("dte", 0)
                                    direction = rec.get("direction", "put")
                                    option_type = "PUT" if direction == "put" else "CALL"

                                    instruction = {
                                        "id": f"advisor_{rec.get('tier_label', 'auto')}",
                                        "type": "credit_spread",
                                        "symbol": ticker,
                                        "expiration": (now + timedelta(days=dte)).strftime("%Y-%m-%d") if dte else _next_trading_day(),
                                        "short_strike": rec["short_strike"],
                                        "long_strike": rec["long_strike"],
                                        "option_type": option_type,
                                        "quantity": rec.get("num_contracts", 1),
                                        "net_price": rec.get("credit"),
                                    }
                                    await _execute_single_order(instruction, "ibkr", mode, 30, 1)
                                    _log.info("Auto-executed: %s", rec.get("tier_label"))
                                except Exception as e:
                                    _log.error("Auto-execute failed for %s: %s",
                                               rec.get("tier_label"), e)

                except Exception as e:
                    _log.error("Advisor eval error: %s", e)

        bg_tasks.append(asyncio.create_task(_advisor_bg()))

    # Signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: shutdown_event.set())

    # Start embedded uvicorn (non-blocking)
    config = uvicorn.Config(
        "app.main:app",
        host=server_host,
        port=server_port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    # Run server in background task
    server_task = asyncio.create_task(server.serve())

    # Wait for shutdown signal
    await shutdown_event.wait()

    print("\n  Shutting down daemon...")

    # Cancel background tasks
    for t in bg_tasks:
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

    # Stop HTTP server
    server.should_exit = True
    try:
        await asyncio.wait_for(server_task, timeout=5.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

    # Disconnect provider
    if live_provider:
        await live_provider.disconnect()
    ProviderRegistry.clear()

    print("  Daemon stopped.")
    return 0


# ── journal ──────────────────────────────────────────────────────────────────

async def _cmd_journal(args) -> int:
    """View trade history and ledger entries."""
    server = _detect_server(args)
    if server:
        rc = await _try_daemon(server, _cmd_journal_http, args)
        if rc is not None:
            return rc

    from app.services.ledger import get_ledger
    from app.models import LedgerQuery, LedgerEventType

    _init_read_only_services(args.data_dir, _get_mode(args))
    ledger = get_ledger()
    if not ledger:
        print("  Ledger not initialized")
        return 1

    limit = getattr(args, "limit", 50)
    event_type_str = getattr(args, "event_type", None)
    order_id = getattr(args, "order_id", None)

    # Build query
    query_kwargs = {"limit": limit}
    if event_type_str:
        try:
            query_kwargs["event_type"] = LedgerEventType(event_type_str)
        except ValueError:
            print(f"  Invalid event type: {event_type_str}")
            print(f"  Valid types: {', '.join(e.value for e in LedgerEventType)}")
            return 1
    if order_id:
        query_kwargs["order_id"] = order_id

    days = getattr(args, "days", None)
    if days:
        from datetime import timezone
        query_kwargs["start_date"] = datetime.now(timezone.utc) - timedelta(days=days)

    query = LedgerQuery(**query_kwargs)
    entries = await ledger.query(query)

    _print_header(f"Trade Journal ({len(entries)} entries)")

    if not entries:
        print("    (no entries found)")
        print()
        return 0

    # Display entries
    for entry in entries:
        ts = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S") if entry.timestamp else "?"
        evt = entry.event_type.value if hasattr(entry.event_type, 'value') else str(entry.event_type)
        broker_str = entry.broker.value if entry.broker and hasattr(entry.broker, 'value') else "—"
        dry_str = " [DRY]" if entry.dry_run else ""
        source_str = entry.source.value if hasattr(entry.source, 'value') else str(entry.source)

        # Color by event type
        if "OPENED" in evt or "SUBMITTED" in evt:
            evt_display = _color(evt, "92")
        elif "CLOSED" in evt:
            evt_display = _color(evt, "93")
        elif "FAIL" in evt or "ERROR" in evt:
            evt_display = _color(evt, "91")
        else:
            evt_display = evt

        print(f"  {ts} | {evt_display:>30} | {broker_str:>6} | {source_str:>12}{dry_str}")

        # Show key data fields
        if entry.data:
            data_parts = []
            for key in ["symbol", "legs", "playbook_name", "instruction_id", "status"]:
                if key in entry.data:
                    data_parts.append(f"{key}={entry.data[key]}")
            if data_parts:
                print(f"         {', '.join(data_parts)}")

        if entry.order_id:
            print(f"         order_id={entry.order_id[:12]}...")
        if entry.position_id:
            print(f"         position_id={entry.position_id[:12]}...")

    print()
    return 0


# ── performance ──────────────────────────────────────────────────────────────

async def _cmd_performance(args) -> int:
    """Show performance metrics."""
    server = _detect_server(args)
    if server:
        rc = await _try_daemon(server, _cmd_performance_http, args)
        if rc is not None:
            return rc

    from app.services.position_store import get_position_store
    from app.services.dashboard_service import DashboardService

    _init_read_only_services(args.data_dir, _get_mode(args))
    store = get_position_store()
    if not store:
        print("  Position store not initialized")
        return 1

    svc = DashboardService(store)

    start_date = None
    end_date = None
    if getattr(args, "start_date", None):
        start_date = date.fromisoformat(args.start_date)
    if getattr(args, "end_date", None):
        end_date = date.fromisoformat(args.end_date)
    if getattr(args, "days", None) and not start_date:
        start_date = date.today() - timedelta(days=args.days)

    metrics = svc.get_performance(start_date=start_date, end_date=end_date)

    _print_header("Performance Metrics")

    if metrics.total_trades == 0:
        print("    (no closed trades found)")
        print()
        return 0

    pnl_color = "92" if metrics.net_pnl >= 0 else "91"

    print(f"  Total Trades:    {metrics.total_trades}")
    print(f"  Wins:            {metrics.wins}")
    print(f"  Losses:          {metrics.losses}")
    print(f"  Win Rate:        {metrics.win_rate:.1%}")
    print(f"  Net P&L:         {_color(f'${metrics.net_pnl:+,.2f}', pnl_color)}")
    print(f"  Avg P&L:         ${metrics.avg_pnl:+,.2f}")
    print(f"  Profit Factor:   {metrics.profit_factor:.2f}")
    print(f"  Sharpe Ratio:    {metrics.sharpe:.2f}")
    print(f"  Max Drawdown:    ${metrics.max_drawdown:,.2f}")
    if metrics.roi:
        print(f"  ROI:             {metrics.roi:.1%}")

    # Daily P&L
    days_count = getattr(args, "days", 30)
    daily = svc.get_daily_pnl(days=days_count)
    if daily:
        _print_section(f"Daily P&L (last {days_count} days with activity)")
        print(f"  {'Date':>12} {'Realized':>12} {'Opened':>8} {'Closed':>8}")
        print(f"  {'─' * 12} {'─' * 12} {'─' * 8} {'─' * 8}")
        for d in daily:
            pc = "92" if d.realized_pnl >= 0 else "91"
            print(f"  {d.date.isoformat():>12} {_color(f'${d.realized_pnl:+,.2f}', pc):>20} "
                  f"{d.trades_opened:>8} {d.trades_closed:>8}")

    print()
    return 0


# ── orders ──────────────────────────────────────────────────────────────────

async def _cmd_orders(args) -> int:
    """Show working (open) orders at the broker."""
    server = _detect_server(args)
    if server:
        rc = await _try_daemon(server, _cmd_orders_http, args)
        if rc is not None:
            return rc

    from app.core.provider import ProviderRegistry
    from app.models import Broker

    mode = _get_mode(args)
    if mode == "dry-run":
        print("  orders command requires --paper or --live")
        return 1

    provider = await _init_ibkr_readonly(args)
    try:
        orders = await provider.get_open_orders()

        _print_header(f"Working Orders ({mode.upper()})")

        if not orders:
            print("    (no open orders)")
            print()
            return 0

        for o in orders:
            status_color = "93" if o.status.value == "SUBMITTED" else "96"
            print(f"  Order #{_color(o.order_id, '1')}")
            print(f"    Status:  {_color(o.status.value, status_color)}")
            print(f"    Details: {o.message}")
            if o.filled_quantity:
                print(f"    Filled:  {o.filled_quantity} @ ${o.filled_price:.2f}")
            print()

        print(f"  Total: {len(orders)} working order(s)")
        print(f"  To cancel: python utp.py cancel --order-id <ID> --{'live' if mode == 'live' else 'paper'}")
        print()
        return 0
    finally:
        await _disconnect(provider)


async def _cmd_cancel(args) -> int:
    """Cancel a working order at the broker."""
    server = _detect_server(args)
    if server:
        rc = await _try_daemon(server, _cmd_cancel_http, args)
        if rc is not None:
            return rc

    from app.core.provider import ProviderRegistry
    from app.models import Broker

    mode = _get_mode(args)
    if mode == "dry-run":
        print("  cancel command requires --paper or --live")
        return 1

    order_id = getattr(args, "order_id", None)
    cancel_all = getattr(args, "all", False)

    if not order_id and not cancel_all:
        print("  Specify --order-id <ID> or --all")
        return 1

    provider = await _init_ibkr_readonly(args)
    # Need to disable readonly for cancel
    import app.config
    app.config.settings.ibkr_readonly = False

    try:
        if cancel_all:
            orders = await provider.get_open_orders()
            if not orders:
                print("  No open orders to cancel")
                return 0

            print(f"  Cancelling {len(orders)} order(s)...")
            for o in orders:
                result = await provider.cancel_order(o.order_id)
                status_color = "92" if result.status.value == "CANCELLED" else "91"
                print(f"    Order #{o.order_id}: {_color(result.status.value, status_color)} — {result.message}")
        else:
            print(f"  Cancelling order #{order_id}...")
            result = await provider.cancel_order(order_id)
            status_color = "92" if result.status.value == "CANCELLED" else "91"
            print(f"    Status: {_color(result.status.value, status_color)}")
            print(f"    {result.message}")

        print()
        return 0
    finally:
        await _disconnect(provider)


# Re-export _match_broker_pnl from its canonical location in LiveDataService.
# This keeps backwards compatibility for any code that imports from utp.
from app.services.live_data_service import _match_broker_pnl  # noqa: F401


# ── trades (today's activity) ────────────────────────────────────────────────

async def _cmd_trades(args) -> int:
    """Show today's trades with order/position IDs and details."""
    server = _detect_server(args)
    if server:
        rc = await _try_daemon(server, _cmd_trades_http, args)
        if rc is not None:
            return rc

    from app.services.position_store import get_position_store

    mode = _get_mode(args)
    live_provider = None

    if mode in ("paper", "live"):
        live_provider = await _init_ibkr_readonly(args)
    else:
        _init_read_only_services(getattr(args, "data_dir", "data/utp"), mode)

    store = get_position_store()
    if not store:
        print("  Position store not initialized")
        if live_provider:
            await _disconnect(live_provider)
        return 1

    today = date.today()
    days = getattr(args, "days", 0)
    show_all = getattr(args, "show_all", False)

    detail_id = getattr(args, "detail", None)
    if detail_id:
        rc = _show_trade_detail(store, detail_id)
        if live_provider:
            await _disconnect(live_provider)
        return rc

    all_open = store.get_open_positions()

    if show_all:
        # Show everything: all open + all closed
        today_open = all_open
        all_closed = store.get_closed_positions()
        period = "All"
    else:
        cutoff = today - timedelta(days=days) if days else today
        all_closed = store.get_closed_positions(start_date=cutoff)
        # Filter open positions by entry_time
        today_open = []
        for pos in all_open:
            entry_str = pos.get("entry_time", "")
            if entry_str:
                try:
                    entry_date = datetime.fromisoformat(entry_str).date()
                    if entry_date >= cutoff:
                        today_open.append(pos)
                except (ValueError, TypeError):
                    pass
        period = "Today" if days == 0 else f"Last {days} day(s)"

    # Fetch broker-authoritative P&L for open positions
    broker_data = {}
    if live_provider:
        try:
            broker_data = _match_broker_pnl(
                await live_provider.get_portfolio_items(), today_open
            )
        except Exception as e:
            logger.debug("Failed to fetch broker P&L for trades: %s", e)

    all_trades = today_open + all_closed
    all_trades.sort(key=lambda p: p.get("entry_time", ""), reverse=True)

    _print_header(f"Trades — {period} ({mode.upper()})")

    if not all_trades:
        print("    (no trades found)")
        print()
        if live_provider:
            await _disconnect(live_provider)
        return 0

    has_broker = bool(broker_data)
    if has_broker:
        print(f"  {'Status':>8} {'Symbol':>8} {'Type':>14} {'Qty':>5} {'AvgCost':>10} "
              f"{'Mark':>10} {'P&L':>12} {'Order':>8} {'Position':>12}")
        print(f"  {'─' * 8} {'─' * 8} {'─' * 14} {'─' * 5} {'─' * 10} "
              f"{'─' * 10} {'─' * 12} {'─' * 8} {'─' * 12}")
    else:
        print(f"  {'Status':>8} {'Symbol':>8} {'Type':>14} {'Qty':>5} {'Entry':>10} "
              f"{'P&L':>12} {'Order':>8} {'Position':>12}")
        print(f"  {'─' * 8} {'─' * 8} {'─' * 14} {'─' * 5} {'─' * 10} "
              f"{'─' * 12} {'─' * 8} {'─' * 12}")

    for pos in all_trades:
        status = pos.get("status", "?")
        sym = pos.get("symbol", "?")
        otype = pos.get("order_type", "?")
        qty = pos.get("quantity", 0)
        entry = pos.get("entry_price", 0)
        pos_id = pos.get("position_id", "—")
        order_id = pos.get("order_id", "—")

        order_short = order_id[:8] if order_id and order_id != "—" else "—"
        pos_short = pos_id[:12] if pos_id and pos_id != "—" else "—"
        status_color = "92" if status == "open" else "93"

        if has_broker and pos_id in broker_data:
            bd = broker_data[pos_id]
            pnl = bd["unrealized_pnl"]
            avg_cost = bd["avg_cost"]
            mark = bd["market_price"]
            pnl_color = "92" if pnl >= 0 else "91"
            print(f"  {_color(status, status_color):>16} {sym:>8} {otype:>14} {qty:>5.0f} "
                  f"${avg_cost:>9.4f} ${mark:>9.4f} "
                  f"{_color(f'${pnl:>+10,.2f}', pnl_color):>20} "
                  f"{order_short:>8} {pos_short:>12}")
        elif has_broker:
            # No broker data for this position (e.g. closed)
            pnl = pos.get("pnl") or pos.get("unrealized_pnl") or 0
            pnl_color = "92" if pnl >= 0 else "91"
            print(f"  {_color(status, status_color):>16} {sym:>8} {otype:>14} {qty:>5.0f} "
                  f"${abs(entry):>9.4f} {'—':>10} "
                  f"{_color(f'${pnl:>+10,.2f}', pnl_color):>20} "
                  f"{order_short:>8} {pos_short:>12}")
        else:
            pnl = pos.get("pnl") or pos.get("unrealized_pnl") or 0
            pnl_color = "92" if pnl >= 0 else "91"
            print(f"  {_color(status, status_color):>16} {sym:>8} {otype:>14} {qty:>5.0f} "
                  f"${abs(entry):>9.2f} {_color(f'${pnl:>+10,.2f}', pnl_color):>20} "
                  f"{order_short:>8} {pos_short:>12}")

    print(f"\n  Total: {len(all_trades)} trade(s)")
    mode_flag = f" --{mode}" if mode in ("paper", "live") else ""
    print(f"  Detail: python utp.py trades --detail <position-id-prefix>{mode_flag}")
    print()
    if live_provider:
        await _disconnect(live_provider)
    return 0


def _show_trade_detail(store, detail_id: str) -> int:
    """Show detailed info for a single trade by position_id or order_id prefix."""
    match = None
    for pos in list(store.get_open_positions()) + list(store.get_closed_positions()):
        pid = pos.get("position_id", "")
        oid = pos.get("order_id", "")
        if pid.startswith(detail_id) or (oid and str(oid).startswith(detail_id)):
            match = pos
            break

    if not match:
        print(f"  No trade found matching '{detail_id}'")
        return 1

    _print_header("Trade Detail")
    print(f"  Position ID:  {match.get('position_id', '—')}")
    print(f"  Order ID:     {match.get('order_id', '—')}")
    print(f"  Status:       {match.get('status', '?')}")
    print(f"  Symbol:       {match.get('symbol', '?')}")
    print(f"  Type:         {match.get('order_type', '?')}")
    print(f"  Broker:       {match.get('broker', '?')}")
    print(f"  Source:       {match.get('source', '?')}")
    print(f"  Quantity:     {match.get('quantity', 0)}")
    print(f"  Entry Price:  ${match.get('entry_price', 0):.2f}")
    print(f"  Entry Time:   {match.get('entry_time', '—')}")
    if match.get("expiration"):
        print(f"  Expiration:   {match['expiration']}")

    legs = match.get("legs")
    if legs:
        _print_section("Legs")
        for i, leg in enumerate(legs, 1):
            action = leg.get("action", "?")
            strike = leg.get("strike", 0)
            opt_type = leg.get("option_type", "?")
            exp = leg.get("expiration", "?")
            print(f"    Leg {i}: {action} {strike} {opt_type} exp={exp}")

    _print_section("P&L")
    if match.get("status") == "closed":
        print(f"  Exit Price:   ${match.get('exit_price', 0):.2f}")
        print(f"  Exit Time:    {match.get('exit_time', '—')}")
        print(f"  Exit Reason:  {match.get('exit_reason', '—')}")
        pnl = match.get("pnl", 0)
        pc = "92" if pnl >= 0 else "91"
        print(f"  Realized P&L: {_color(f'${pnl:+,.2f}', pc)}")
    else:
        mark = match.get("current_mark")
        upnl = match.get("unrealized_pnl") or 0
        if mark is not None:
            print(f"  Current Mark: ${mark:.2f}")
        pc = "92" if upnl >= 0 else "91"
        print(f"  Unrealized:   {_color(f'${upnl:+,.2f}', pc)}")

    if legs and match.get("order_type") == "multi_leg":
        entry = match.get("entry_price", 0)
        qty = match.get("quantity", 1)
        strikes = [leg.get("strike", 0) for leg in legs]
        if len(strikes) == 2:
            width = abs(strikes[1] - strikes[0])
            credit = abs(entry)
            max_profit = credit * qty * 100
            max_loss = (width - credit) * qty * 100
            _print_section("Trade Economics")
            print(f"  Credit:       ${credit:.2f} x {qty} x 100 = ${max_profit:,.2f}")
            print(f"  Spread Width: ${width:.2f}")
            print(f"  Max Profit:   {_color(f'${max_profit:,.2f}', '92')}")
            print(f"  Max Loss:     {_color(f'${max_loss:,.2f}', '91')}")

    if match.get("status") == "open" and legs:
        sym = match.get("symbol", "?")
        exp = match.get("expiration", "?")
        opt_type = legs[0].get("option_type", "CALL")
        qty = match.get("quantity", 1)
        # Determine which leg is short and which is long
        short_strike = long_strike = None
        for leg in legs:
            if "SELL" in leg.get("action", ""):
                short_strike = leg.get("strike", 0)
            else:
                long_strike = leg.get("strike", 0)
        if short_strike is not None and long_strike is not None:
            print(f"\n  To close (dry-run first, then add --live):")
            print(f"    python utp.py trade credit-spread --symbol {sym} "
                  f"--short-strike {short_strike} --long-strike {long_strike} "
                  f"--option-type {opt_type} --expiration {exp} "
                  f"--quantity {int(qty)} --net-price 0.05 --close --live")

    print()
    return 0


# ── close (by position ID) ──────────────────────────────────────────────────

async def _cmd_close(args) -> int:
    """Close an open position by ID — auto-derives all trade parameters."""
    server = _detect_server(args)
    if server:
        rc = await _try_daemon(server, _cmd_close_http, args)
        if rc is not None:
            return rc

    from app.services.position_store import get_position_store

    mode = _get_mode(args)
    pos_id_prefix = args.position_id

    if mode in ("paper", "live"):
        provider = await _init_ibkr_readonly(args)
    else:
        _init_read_only_services(getattr(args, "data_dir", "data/utp"), mode)
        provider = None

    store = get_position_store()
    if not store:
        print("  Position store not initialized")
        if provider:
            await _disconnect(provider)
        return 1

    # Find matching position
    match = None
    for pos in store.get_open_positions():
        pid = pos.get("position_id", "")
        oid = pos.get("order_id", "")
        if pid.startswith(pos_id_prefix) or (oid and str(oid).startswith(pos_id_prefix)):
            match = pos
            break

    if not match:
        print(f"  No open position found matching '{pos_id_prefix}'")
        if provider:
            await _disconnect(provider)
        return 1

    legs = match.get("legs") or []
    otype = match.get("order_type", "")
    sym = match.get("symbol", "?")
    total_qty = int(match.get("quantity", 1))
    close_qty = getattr(args, "quantity", None) or total_qty
    if close_qty > total_qty:
        print(f"  Cannot close {close_qty} — position only has {total_qty} contract(s)")
        if provider:
            await _disconnect(provider)
        return 1
    partial = close_qty < total_qty
    exp = match.get("expiration", "")

    if otype != "multi_leg" or not legs:
        print(f"  Position {match.get('position_id', '?')} is {otype} — "
              f"only multi-leg (spread) close is supported via this command")
        if provider:
            await _disconnect(provider)
        return 1

    # Determine spread type and strikes from legs
    short_strike = long_strike = opt_type = None
    for leg in legs:
        action = leg.get("action", "")
        if "SELL" in action:
            short_strike = leg.get("strike")
        else:
            long_strike = leg.get("strike")
        opt_type = leg.get("option_type", "CALL")

    if short_strike is None or long_strike is None:
        print(f"  Cannot determine short/long strikes from position legs")
        if provider:
            await _disconnect(provider)
        return 1

    # Determine spread type (credit vs debit vs iron condor)
    if len(legs) == 4:
        spread_type = "iron-condor"
    elif len(legs) == 2:
        # Credit spread: first leg is SELL
        first_action = legs[0].get("action", "")
        spread_type = "credit-spread" if "SELL" in first_action else "debit-spread"
    else:
        print(f"  Unsupported leg count: {len(legs)}")
        if provider:
            await _disconnect(provider)
        return 1

    net_price = getattr(args, "net_price", None) or 0.05

    # Show what we're about to do
    _print_header("Close Position")
    print(f"  Position:   {match.get('position_id', '?')}")
    print(f"  Symbol:     {sym}")
    print(f"  Type:       {spread_type}")
    print(f"  Strikes:    {short_strike}/{long_strike} {opt_type}")
    if partial:
        print(f"  Closing:    {close_qty} of {total_qty} (partial)")
    else:
        print(f"  Quantity:   {close_qty}")
    print(f"  Expiration: {exp}")
    print(f"  Net Price:  ${net_price:.2f} (debit to close)")
    entry = match.get("entry_price", 0)
    credit = abs(entry)
    print(f"  Opened at:  ${credit:.2f} (credit received)")
    print(f"  Mode:       {_mode_label(mode)}")
    print()

    # Build args for _cmd_trade
    trade_args = argparse.Namespace(
        subcommand=spread_type,
        symbol=sym,
        short_strike=float(short_strike),
        long_strike=float(long_strike),
        option_type=opt_type,
        expiration=exp,
        quantity=close_qty,
        net_price=net_price,
        order_type="LIMIT",
        time_in_force="DAY",
        close=True,
        closing_position_id=match.get("position_id"),
        closing_quantity=close_qty if partial else None,
        verbose=True,
        validate_all=False,
        cleanup=False,
        data_dir=getattr(args, "data_dir", "data/utp"),
        host=getattr(args, "host", "127.0.0.1"),
        port=getattr(args, "port", None),
        client_id=getattr(args, "client_id", 10),
        broker=getattr(args, "broker", "ibkr"),
        exchange=getattr(args, "exchange", "SMART"),
        paper=getattr(args, "paper", False),
        live=getattr(args, "live", False),
        poll_timeout=getattr(args, "poll_timeout", 30),
        poll_interval=getattr(args, "poll_interval", 1.0),
    )

    if provider:
        await _disconnect(provider)

    # Execute via the existing trade command
    rc = await _cmd_trade(trade_args)

    if partial and rc == 0:
        remaining = total_qty - close_qty
        print(f"\n  Position {match.get('position_id', '?')[:8]}... "
              f"reduced to {remaining} contract(s)")

    return rc


# ── Argparse setup ───────────────────────────────────────────────────────────

def _add_connection_args(parser: argparse.ArgumentParser, *, default_paper: bool = False) -> None:
    """Add broker connection flags.

    Args:
        default_paper: If True, default to paper mode (IBKR port 7497) instead of dry-run.
    """
    default_mode = "paper" if default_paper else "dry-run"
    parser.set_defaults(_default_mode=default_mode, dry_run=False, paper=False, live=False)

    mode_group = parser.add_mutually_exclusive_group()
    dry_help = "Use stub providers, no broker connection"
    paper_help = "Connect to IBKR paper account (port 7497)"
    live_help = "Connect to IBKR live account (port 7496)"
    if default_paper:
        paper_help += " (default)"
    else:
        dry_help += " (default)"
    mode_group.add_argument("--dry-run", action="store_true",
                           help=dry_help)
    mode_group.add_argument("--paper", action="store_true",
                           help=paper_help)
    mode_group.add_argument("--live", action="store_true",
                           help=live_help)

    parser.add_argument("--host", default="127.0.0.1",
                        help="TWS/Gateway hostname (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None,
                        help="TWS/Gateway port (default: auto — 7497 paper, 7496 live)")
    parser.add_argument("--client-id", type=int, default=10,
                        help="IBKR client ID (default: 10)")
    parser.add_argument("--exchange", default=None,
                        help="Exchange routing (default: SMART)")
    parser.add_argument("--broker", default="ibkr", choices=["ibkr", "robinhood", "etrade"],
                        help="Broker (default: ibkr)")
    parser.add_argument("--data-dir", default="data/utp",
                        help="Persistence directory (default: data/utp)")
    parser.add_argument("--poll-timeout", type=float, default=30.0,
                        help="Max seconds to wait per order for fill (default: 30)")
    parser.add_argument("--poll-interval", type=float, default=1.0,
                        help="Seconds between fill status checks (default: 1.0)")
    # Add --server/--server-port unless already present (e.g. daemon subparser)
    existing = {a.option_strings[0] for a in parser._actions if a.option_strings}
    if "--server" not in existing:
        parser.add_argument("--server", default=None, metavar="URL",
                            help="Connect to UTP daemon at URL (e.g. http://localhost:8000). "
                                 "Auto-detected if daemon is running.")
    if "--server-port" not in existing:
        parser.add_argument("--server-port", type=int, default=8000,
                            help="Daemon port for auto-detect (default: 8000)")
    parser.add_argument("--allow-fallback", action="store_true", default=False,
                        help="If daemon is unreachable, fall back to direct IBKR connection "
                             "(default: off — reports error instead to avoid client-id conflicts)")


def _add_spread_args(parser: argparse.ArgumentParser, spread_type: str) -> None:
    """Add common spread arguments to a margin/trade subparser."""
    parser.add_argument("--symbol", required=True, help="Underlying symbol")
    parser.add_argument("--expiration", required=True, help="Expiration date YYYY-MM-DD")
    parser.add_argument("--quantity", type=int, default=1, help="Contracts (default: 1)")
    parser.add_argument("--net-price", type=float, default=None, help="Net price per contract")

    if spread_type in ("credit-spread", "debit-spread"):
        parser.add_argument("--short-strike", type=float, required=True, help="Short leg strike")
        parser.add_argument("--long-strike", type=float, required=True, help="Long leg strike")
        parser.add_argument("--option-type", required=True, choices=["CALL", "PUT"],
                           help="Option type")
    elif spread_type == "iron-condor":
        parser.add_argument("--put-short", type=float, required=True, help="Short put strike")
        parser.add_argument("--put-long", type=float, required=True, help="Long put strike")
        parser.add_argument("--call-short", type=float, required=True, help="Short call strike")
        parser.add_argument("--call-long", type=float, required=True, help="Long call strike")
    elif spread_type == "option":
        parser.add_argument("--strike", type=float, required=True, help="Strike price")
        parser.add_argument("--option-type", required=True, choices=["CALL", "PUT"],
                           help="Option type")
        parser.add_argument("--action", default="BUY_TO_OPEN",
                           choices=["BUY_TO_OPEN", "SELL_TO_OPEN", "BUY_TO_CLOSE", "SELL_TO_CLOSE"],
                           help="Action (default: BUY_TO_OPEN)")


def _run_daemon_with_restart(args) -> int:
    """Run daemon with auto-restart on crash. No restart on signal shutdown."""
    import time as _time

    backoff_cap = 10.0
    delay = 2.0
    max_consecutive_crashes = 20

    consecutive_crashes = 0
    while True:
        try:
            rc = asyncio.run(_cmd_daemon(args))
            # Clean exit (rc=0 means signal shutdown) — do not restart
            return rc
        except KeyboardInterrupt:
            # Ctrl-C — do not restart
            print("\n  Daemon stopped by keyboard interrupt.")
            return 0
        except SystemExit as e:
            return e.code if e.code is not None else 0
        except Exception as e:
            consecutive_crashes += 1
            if consecutive_crashes > max_consecutive_crashes:
                print(f"\n  Daemon crashed {max_consecutive_crashes} times consecutively. Giving up.")
                return 1
            print(f"\n  Daemon crashed: {e}")
            print(f"  Restarting in {delay:.0f}s (crash #{consecutive_crashes})...")
            _time.sleep(delay)
            delay = min(delay * 2, backoff_cap)

            # Reset module-level state for clean restart
            from app.core.provider import ProviderRegistry
            from app.services.ledger import reset_ledger
            from app.services.live_data_service import reset_live_data_service
            from app.services.position_store import reset_position_store
            ProviderRegistry.clear()
            reset_ledger()
            reset_position_store()
            reset_live_data_service()
            from app.services.market_data_streaming import reset_streaming_service
            from app.services.execution_store import reset_execution_store
            reset_streaming_service()
            reset_execution_store()
            import app.main
            app.main._daemon_mode = False


def main():
    parser = argparse.ArgumentParser(
        description="UTP — Universal Trade Platform CLI. One command for everything.",
        epilog="""
Subcommands:
  portfolio     Current positions, P&L (broker-authoritative), account summary
  quote         Real-time quotes for one or more symbols (fetched in parallel)
  options       Option chain strikes, expirations, live bid/ask/volume
  margin        Margin/cost check for a hypothetical trade (no execution)
  trade         Execute trades (equity, option, credit-spread, debit-spread, iron-condor)
  orders        Show working (open) orders at the broker
  cancel        Cancel a working order by ID or cancel all open orders
  trades        Today's trades with order/position IDs, P&L, and detail drill-down
  close         Close an open position by ID — submits real IBKR closing order
  executions    IBKR execution history grouped by order (identifies multi-leg trades)
  playbook      Execute or validate YAML trade playbooks (batch trading)
  status        System dashboard (positions, orders, connections, cache stats)
  reconcile     Compare system vs broker positions, flush/hard-reset
  readiness     Test IBKR connectivity and all 5 trade types
  daemon        Start long-running daemon (IBKR + HTTP API + background loops)
  repl          Interactive REPL connected to daemon
  server        Start standalone REST API server (no IBKR, no background tasks)
  journal       View trade history and ledger entries
  performance   Performance metrics (win rate, Sharpe, drawdown, profit factor)
  flush         Clear local positions/ledger data (blocked when daemon running)

Daemon-first routing:
  All commands auto-detect a running daemon and route through its HTTP API,
  sharing the IBKR connection. When no daemon is running, commands connect
  to IBKR directly (unless --allow-fallback is off). Use --server URL to
  point at a specific daemon.

Trade modes:
  (default)     Dry-run: stub providers, no broker connection
  --paper       Connect to IBKR paper account (port 7497)
  --live        Connect to IBKR live account (port 7496) — requires confirmation

Examples:
  %(prog)s portfolio --live                          # P&L from IBKR (authoritative)
  %(prog)s portfolio --paper                         # Paper account positions
  %(prog)s quote SPY AAPL QQQ                        # Parallel quotes
  %(prog)s quote SPX --live                          # Index quote with timestamp + source
  %(prog)s options RUT --type CALL --expiration 2026-03-16 --live
  %(prog)s options SPX --list-expirations --paper
  %(prog)s margin credit-spread --symbol RUT --short-strike 2550 \\
    --long-strike 2575 --option-type CALL --expiration 2026-03-16 --paper
  %(prog)s trade credit-spread --symbol SPX --short-strike 5500 \\
    --long-strike 5475 --option-type PUT --expiration 2026-03-20 \\
    --quantity 1 --net-price 3.50 --paper             # Open a credit spread
  %(prog)s trade credit-spread --symbol SPX --short-strike 5500 \\
    --long-strike 5475 --option-type PUT --expiration 2026-03-20 \\
    --quantity 1 --net-price 0.05 --close --live      # Close a credit spread
  %(prog)s trade --simulate --symbol RUT --live       # Margin check, no execution
  %(prog)s trade --validate-all --paper               # Test all 5 trade types
  %(prog)s orders --live                              # Show open/working orders
  %(prog)s cancel --order-id 123 --live               # Cancel order by ID
  %(prog)s cancel --all --live                        # Cancel all open orders
  %(prog)s trades --live                              # Today's transactions (IBKR P&L)
  %(prog)s trades --all --live                        # All trades (open + closed)
  %(prog)s trades --detail <pos-id> --live            # Drill down + close command
  %(prog)s close 2d9a --live                          # Close position (real IBKR order)
  %(prog)s close 2d9a --net-price 0.10 --live         # Close at specific debit
  %(prog)s close 2d9a --simulate --live               # Margin check only, no close
  %(prog)s executions --live                          # IBKR executions grouped by order
  %(prog)s playbook execute playbooks/example_mixed.yaml --paper
  %(prog)s status                                     # System dashboard
  %(prog)s reconcile --flush --show --live            # Flush + sync + display
  %(prog)s reconcile --hard-reset --live              # Full reset + rebuild
  %(prog)s readiness --symbol SPX --paper             # IBKR connectivity test
  %(prog)s daemon --paper                             # Start daemon (paper)
  %(prog)s daemon --live --advisor-profile tiered_v2  # Daemon with advisor
  %(prog)s repl                                       # Interactive REPL
  %(prog)s server --server-port 8000                  # Standalone HTTP server
  %(prog)s journal --days 7                           # Recent ledger entries
  %(prog)s performance --days 30                      # Win rate, Sharpe, drawdown
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # ── portfolio ──
    p_port = subparsers.add_parser("portfolio",
                                    help="Show current positions, P&L, account summary",
                                    aliases=["port", "positions", "pos"],
                                    description='''
Show current positions, unrealized P&L, account balances, and position summary.
When connected to IBKR (--paper or --live), displays broker-authoritative P&L
including average cost, mark price, and unrealized gains. Auto-detects a running
daemon and routes through HTTP if available.
                                    ''',
                                    epilog='''
Examples:
  %(prog)s --live                  Positions + P&L from IBKR live account
  %(prog)s --paper                 Paper account positions
  %(prog)s                         Local position store (default: paper mode)
  %(prog)s --live --host 10.0.0.5  Connect to remote TWS

Output includes:
  - Per-position: symbol, type, quantity, avg cost, mark, unrealized P&L
  - Account summary: net liquidation, buying power, cash balance
  - When daemon is running, no --live/--paper flag needed

Aliases: port, positions, pos
                                    ''',
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    p_port.add_argument("--recent", "-n", type=int, default=0,
                        help="Show N most recent closed trades at the bottom (default: 0 = hide)")
    _add_connection_args(p_port, default_paper=True)

    # ── quote ──
    p_quote = subparsers.add_parser("quote",
                                     help="Get real-time quotes for one or more symbols",
                                     aliases=["q"],
                                     description='''
Fetch real-time quotes for one or more symbols. Symbols are fetched in parallel
for speed. Output shows bid, ask, last price, and volume. When connected via
daemon or IBKR, also shows the quote timestamp and data source (streaming_cache,
ibkr, or delayed). For indices (SPX, NDX, RUT), uses IBKR streaming data.
                                     ''',
                                     epilog='''
Examples:
  %(prog)s SPY AAPL QQQ               Quotes for multiple equities (dry-run)
  %(prog)s SPX --live                  Index quote via IBKR
  %(prog)s SPX NDX RUT --live          Multiple index quotes
  %(prog)s SPY --paper                 Paper account quote

Output columns:
  Symbol   Ticker symbol
  Bid      Current best bid price
  Ask      Current best ask price
  Last     Last traded price
  Vol      Trading volume

Aliases: q
                                     ''',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    p_quote.add_argument("symbols", nargs="+", help="One or more ticker symbols")
    _add_connection_args(p_quote)

    # ── options ──
    p_opts = subparsers.add_parser("options",
                                    help="Show option chain strikes and quotes",
                                    aliases=["opts", "chain"],
                                    description='''
Show option chain data for an underlying symbol. Displays strikes with live
bid/ask/last/volume for calls, puts, or both. By default shows today's
expiration within +/-15% of current price. Use --strike-range to narrow the
view or --strike-min/--strike-max for an explicit range. Use --list-expirations
to see all available expiration dates instead of strike data.
                                    ''',
                                    epilog='''
Examples:
  %(prog)s SPX --live                              Today's exp, calls+puts, +/-15%%
  %(prog)s SPX --strike-range 2 --live             Tight range: +/-2%% of price
  %(prog)s SPX --type PUT --live                   Puts only, today's expiration
  %(prog)s SPX --type CALL --live                  Calls only
  %(prog)s SPX --strike-min 5400 --strike-max 5600 --live  Explicit range
  %(prog)s SPX --expiration 2026-03-21 --live      Specific expiration date
  %(prog)s SPX --list-expirations --live           List available expirations
  %(prog)s RUT --strike-range 5 --live             RUT +/-5%%
  %(prog)s NDX --type PUT --strike-range 3 --live  NDX puts +/-3%%

Output columns:
  Strike   Option strike price
  Bid      Best bid price
  Ask      Best ask price
  Last     Last traded price
  Vol      Trading volume

Aliases: opts, chain
                                    ''',
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    p_opts.add_argument("symbol", help="Underlying symbol (e.g. RUT, SPX, SPY)")
    p_opts.add_argument("--type", default="BOTH", choices=["CALL", "PUT", "BOTH"],
                        help="Option type (default: BOTH)")
    p_opts.add_argument("--expiration", "-e",
                        help="Expiration date YYYY-MM-DD (default: today)")
    p_opts.add_argument("--strike-range", type=float, default=15,
                        help="Strike range as %% of current price (default: 15)")
    p_opts.add_argument("--strike-min", type=float,
                        help="Minimum strike price (overrides --strike-range)")
    p_opts.add_argument("--strike-max", type=float,
                        help="Maximum strike price (overrides --strike-range)")
    p_opts.add_argument("--list-expirations", action="store_true",
                        help="List available expirations instead of showing strikes")
    _add_connection_args(p_opts, default_paper=True)

    # ── margin ──
    p_margin = subparsers.add_parser("margin",
                                      help="Check margin/cost for a hypothetical trade (no execution)",
                                      aliases=["m"],
                                      description='''
Check margin requirements and buying power impact for a hypothetical trade
without executing it. Supports credit spreads, debit spreads, iron condors,
and single options. Connects to IBKR to qualify contracts and compute exact
margin, but never submits an order.

Choose a trade type subcommand: credit-spread, debit-spread, iron-condor, or option.
                                      ''',
                                      epilog='''
Examples:
  %(prog)s credit-spread --symbol SPX --short-strike 5500 \\
    --long-strike 5475 --option-type PUT --expiration 2026-03-20
  %(prog)s iron-condor --symbol SPX --put-short 5500 --put-long 5475 \\
    --call-short 5700 --call-long 5725 --expiration 2026-03-20
  %(prog)s option --symbol SPY --strike 550 \\
    --option-type PUT --expiration 2026-03-20
  %(prog)s debit-spread --symbol QQQ --long-strike 480 \\
    --short-strike 490 --option-type CALL --expiration 2026-03-20

Trade type subcommands (aliases):
  credit-spread (cs)     Vertical credit spread margin check
  debit-spread (ds)      Vertical debit spread margin check
  iron-condor (ic)       Iron condor (4-leg) margin check
  option (opt)           Single option margin check

Aliases: m
                                      ''',
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    margin_sub = p_margin.add_subparsers(dest="margin_type", help="Trade type to check margin for")

    m_cs = margin_sub.add_parser("credit-spread", help="Credit spread margin check", aliases=["cs"])
    _add_spread_args(m_cs, "credit-spread")
    _add_connection_args(m_cs)

    m_ds = margin_sub.add_parser("debit-spread", help="Debit spread margin check", aliases=["ds"])
    _add_spread_args(m_ds, "debit-spread")
    _add_connection_args(m_ds)

    m_ic = margin_sub.add_parser("iron-condor", help="Iron condor margin check", aliases=["ic"])
    _add_spread_args(m_ic, "iron-condor")
    _add_connection_args(m_ic)

    m_opt = margin_sub.add_parser("option", help="Single option margin check", aliases=["opt"])
    _add_spread_args(m_opt, "option")
    _add_connection_args(m_opt)

    # ── trade ──
    p_trade = subparsers.add_parser("trade", help="Execute a trade",
                                     aliases=["t"],
                                     description='''
Execute equity, option, or multi-leg options trades. Choose a trade type
subcommand (equity, option, credit-spread, debit-spread, iron-condor).
Default mode is dry-run (no broker). Use --paper or --live for real execution.

Special flags:
  --simulate      Use live IBKR to qualify contracts and check margin, but do
                  NOT submit the order. Works with any trade type.
  --close         Reverse a spread position (BUY_TO_CLOSE / SELL_TO_CLOSE).
                  Available on credit-spread, debit-spread, and iron-condor.
  --validate-all  Run all 5 trade types as a validation suite (no real trades).
  --cleanup       Close positions created by --validate-all.

Auto-detects a running daemon and routes through HTTP if available.
                                     ''',
                                     epilog='''
Examples:
  # Equity
  %(prog)s equity --symbol SPY --side BUY --quantity 100
  %(prog)s equity --symbol GBTC --side SELL --quantity 4350 --live
  %(prog)s equity --symbol AAPL --side BUY --quantity 10 --order-type LIMIT --limit-price 200.00 --paper

  # Single option
  %(prog)s option --symbol SPY --strike 550 --option-type PUT --action BUY_TO_OPEN --quantity 1 --paper

  # Credit spread (sell premium)
  %(prog)s credit-spread --symbol SPX --short-strike 5500 --long-strike 5475 \\
    --option-type PUT --expiration 2026-03-20 --quantity 1 --net-price 3.50 --paper

  # Close a credit spread
  %(prog)s credit-spread --symbol SPX --short-strike 5500 --long-strike 5475 \\
    --option-type PUT --expiration 2026-03-20 --close --net-price 0.10 --live

  # Debit spread (buy premium)
  %(prog)s debit-spread --symbol QQQ --long-strike 480 --short-strike 490 \\
    --option-type CALL --expiration 2026-03-20 --quantity 3 --net-price 4.00 --paper

  # Iron condor
  %(prog)s iron-condor --symbol SPX --put-short 5500 --put-long 5475 \\
    --call-short 5700 --call-long 5725 --expiration 2026-03-20 --quantity 1 --paper

  # Simulate (margin check only, no execution)
  %(prog)s credit-spread --symbol RUT --short-strike 2460 --long-strike 2440 \\
    --option-type PUT --expiration 2026-03-18 --quantity 1 --live --simulate

  # Validate all 5 trade types (safe, no real execution)
  %(prog)s --validate-all --paper
  %(prog)s --validate-all --cleanup --paper    # Validate + clean up after

Trade types:
  equity          Buy or sell stocks (--side BUY or SELL)
  option          Buy or sell a single option contract
  credit-spread   Sell a vertical spread to collect premium
  debit-spread    Buy a vertical spread for directional exposure
  iron-condor     Sell a 4-leg neutral strategy (put spread + call spread)

Aliases: t
                                     ''',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    p_trade.add_argument("--verbose", "-v", action="store_true",
                         help="Show detailed P&L breakdown after execution")
    p_trade.add_argument("--validate-all", action="store_true",
                         help="Run all 5 trade types as validation")
    p_trade.add_argument("--cleanup", action="store_true",
                         help="Close positions after --validate-all")
    p_trade.add_argument("--simulate", action="store_true",
                         help="Use live IBKR connection but do NOT execute — "
                              "qualifies contracts, checks margin, shows what would happen")
    p_trade.add_argument("--confirm", action="store_true",
                         help="Confirm and execute the trade (without this, shows order summary only)")
    p_trade.add_argument("--symbol", default="SPY",
                         help="Symbol for --validate-all (default: SPY)")
    p_trade.add_argument("--expiration", default=None,
                         help="Expiration for --validate-all (default: next trading day)")
    _add_connection_args(p_trade)

    trade_sub = p_trade.add_subparsers(dest="subcommand", help="Trade type")

    # trade equity
    t_eq = trade_sub.add_parser("equity", help="Buy/sell stocks",
                                description="Buy or sell stock shares.",
                                epilog='''
Examples:
  %(prog)s --symbol SPY --side BUY --quantity 100              # Market buy (dry-run)
  %(prog)s --symbol GBTC --side SELL --quantity 4350 --live    # Sell all GBTC shares
  %(prog)s --symbol AAPL --side BUY --quantity 10 \\
    --order-type LIMIT --limit-price 200.00 --paper            # Limit buy on paper

Required flags:
  --symbol    Stock ticker (e.g., SPY, AAPL, GBTC)
  --side      BUY or SELL
  --quantity  Number of shares
                                ''',
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    t_eq.add_argument("--symbol", required=True, help="Stock symbol (e.g., SPY, AAPL, GBTC)")
    t_eq.add_argument("--side", required=True, choices=["BUY", "SELL"],
                      help="Order side: BUY to purchase, SELL to liquidate")
    t_eq.add_argument("--quantity", type=int, required=True, help="Number of shares")
    t_eq.add_argument("--order-type", default="MARKET", choices=["MARKET", "LIMIT"],
                      help="Order type (default: MARKET)")
    t_eq.add_argument("--limit-price", type=float, default=None,
                      help="Limit price (required for LIMIT orders)")
    t_eq.add_argument("--simulate", action="store_true",
                      help="Check margin only — do NOT execute")
    t_eq.add_argument("--confirm", action="store_true",
                      help="Confirm and execute (without this, shows order summary only)")
    _add_connection_args(t_eq)

    # trade option
    t_opt = trade_sub.add_parser("option", help="Buy/sell a single option",
                                 description="Buy or sell a single option contract.",
                                 epilog='''
Examples:
  %(prog)s --symbol SPY --strike 550 --option-type PUT \\
    --action BUY_TO_OPEN --quantity 1 --order-type LIMIT --limit-price 2.50 --paper
  %(prog)s --symbol SPX --strike 5500 --option-type CALL \\
    --action SELL_TO_OPEN --quantity 1 --live

Actions:
  BUY_TO_OPEN      Buy to open a new position
  SELL_TO_OPEN     Sell to open (write) a new position
  BUY_TO_CLOSE     Buy to close an existing short position
  SELL_TO_CLOSE    Sell to close an existing long position
                                 ''',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    t_opt.add_argument("--symbol", required=True, help="Underlying symbol (e.g., SPY, SPX)")
    t_opt.add_argument("--expiration", required=True, help="Expiration date YYYY-MM-DD")
    t_opt.add_argument("--strike", type=float, required=True, help="Strike price")
    t_opt.add_argument("--option-type", required=True, choices=["CALL", "PUT"],
                       help="Option type: CALL or PUT")
    t_opt.add_argument("--action", default="BUY_TO_OPEN",
                       choices=["BUY_TO_OPEN", "SELL_TO_OPEN", "BUY_TO_CLOSE", "SELL_TO_CLOSE"],
                       help="Order action (default: BUY_TO_OPEN)")
    t_opt.add_argument("--quantity", type=int, default=1, help="Contracts (default: 1)")
    t_opt.add_argument("--order-type", default="LIMIT", choices=["MARKET", "LIMIT"],
                       help="Order type (default: LIMIT)")
    t_opt.add_argument("--limit-price", type=float, default=None,
                       help="Limit price per contract (required for LIMIT orders)")
    t_opt.add_argument("--simulate", action="store_true",
                       help="Check margin only — do NOT execute")
    _add_connection_args(t_opt)

    # trade credit-spread
    t_cs = trade_sub.add_parser("credit-spread", help="Sell a credit spread",
                                description="Sell a vertical credit spread (collect premium).",
                                epilog='''
Examples:
  %(prog)s --symbol SPX --short-strike 5500 --long-strike 5475 \\
    --option-type PUT --expiration 2026-03-20 --quantity 1 --net-price 3.50 --paper
  %(prog)s --symbol NDX --short-strike 20000 --long-strike 19900 \\
    --option-type PUT --expiration 2026-03-20 --quantity 2 --live
  %(prog)s --symbol SPX --short-strike 5500 --long-strike 5475 \\
    --option-type PUT --expiration 2026-03-20 --close --net-price 0.10 --live

P&L: credit_received - cost_to_close (max profit = full credit, max loss = width - credit)
                                ''',
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    t_cs.add_argument("--symbol", required=True, help="Underlying symbol")
    t_cs.add_argument("--expiration", required=True, help="Expiration date YYYY-MM-DD")
    t_cs.add_argument("--short-strike", type=float, required=True)
    t_cs.add_argument("--long-strike", type=float, required=True)
    t_cs.add_argument("--option-type", required=True, choices=["CALL", "PUT"])
    t_cs.add_argument("--quantity", type=int, default=1)
    t_cs.add_argument("--net-price", type=float, default=None,
                      help="Net credit per contract. If omitted with --paper/--live, auto-prices at mid-point")
    t_cs.add_argument("--auto-price", action="store_true",
                      help="Fetch live quotes and set price (default when --net-price omitted)")
    t_cs.add_argument("--mid", action="store_true",
                      help="Use mid-point between market and best-case (default: market price)")
    t_cs.add_argument("--close", action="store_true",
                      help="Close an existing position (BUY_TO_CLOSE / SELL_TO_CLOSE)")
    t_cs.add_argument("--simulate", action="store_true",
                      help="Check margin only — do NOT execute")
    t_cs.add_argument("--confirm", action="store_true",
                      help="Confirm and execute (without this, shows order summary only)")
    _add_connection_args(t_cs)

    # trade debit-spread
    t_ds = trade_sub.add_parser("debit-spread", help="Buy a debit spread",
                                description="Buy a vertical debit spread (pay premium for directional exposure).",
                                epilog='''
Examples:
  %(prog)s --symbol QQQ --long-strike 480 --short-strike 490 \\
    --option-type CALL --expiration 2026-03-20 --quantity 3 --net-price 4.00 --paper

P&L: value_on_close - debit_paid (max profit = width - debit, max loss = debit paid)
                                ''',
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    t_ds.add_argument("--symbol", required=True, help="Underlying symbol")
    t_ds.add_argument("--expiration", required=True, help="Expiration date YYYY-MM-DD")
    t_ds.add_argument("--long-strike", type=float, required=True)
    t_ds.add_argument("--short-strike", type=float, required=True)
    t_ds.add_argument("--option-type", required=True, choices=["CALL", "PUT"])
    t_ds.add_argument("--quantity", type=int, default=1)
    t_ds.add_argument("--net-price", type=float, default=None,
                      help="Net debit per contract. If omitted with --paper/--live, auto-prices at mid-point")
    t_ds.add_argument("--auto-price", action="store_true",
                      help="Fetch live quotes and set price (default when --net-price omitted)")
    t_ds.add_argument("--mid", action="store_true",
                      help="Use mid-point between market and best-case (default: market price)")
    t_ds.add_argument("--close", action="store_true",
                      help="Close an existing position (SELL_TO_CLOSE / BUY_TO_CLOSE)")
    t_ds.add_argument("--simulate", action="store_true",
                      help="Check margin only — do NOT execute")
    t_ds.add_argument("--confirm", action="store_true",
                      help="Confirm and execute (without this, shows order summary only)")
    _add_connection_args(t_ds)

    # trade iron-condor
    t_ic = trade_sub.add_parser("iron-condor", help="Sell an iron condor",
                                description="Sell an iron condor (put credit spread + call credit spread).",
                                epilog='''
Examples:
  %(prog)s --symbol SPX --put-short 5500 --put-long 5475 \\
    --call-short 5700 --call-long 5725 --expiration 2026-03-20 \\
    --quantity 1 --net-price 3.50 --paper

P&L: combined credit - cost_to_close (max profit = total credit, max loss = wider wing width - credit)
                                ''',
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    t_ic.add_argument("--symbol", required=True, help="Underlying symbol")
    t_ic.add_argument("--expiration", required=True, help="Expiration date YYYY-MM-DD")
    t_ic.add_argument("--put-short", type=float, required=True)
    t_ic.add_argument("--put-long", type=float, required=True)
    t_ic.add_argument("--call-short", type=float, required=True)
    t_ic.add_argument("--call-long", type=float, required=True)
    t_ic.add_argument("--quantity", type=int, default=1)
    t_ic.add_argument("--net-price", type=float, default=None,
                      help="Net credit per contract. If omitted with --paper/--live, auto-prices at mid-point")
    t_ic.add_argument("--auto-price", action="store_true",
                      help="Fetch live quotes and set price (default when --net-price omitted)")
    t_ic.add_argument("--mid", action="store_true",
                      help="Use mid-point between market and best-case (default: market price)")
    t_ic.add_argument("--close", action="store_true",
                      help="Close an existing position (reverses all legs)")
    t_ic.add_argument("--simulate", action="store_true",
                      help="Check margin only — do NOT execute")
    t_ic.add_argument("--confirm", action="store_true",
                      help="Confirm and execute (without this, shows order summary only)")
    _add_connection_args(t_ic)

    # ── playbook ──
    p_pb = subparsers.add_parser("playbook",
                                  help="Execute or validate a YAML trade playbook",
                                  aliases=["pb"],
                                  description='''
Execute or validate YAML trade playbooks for batch trade execution. Playbooks
define a sequence of trades (equity, option, credit-spread, debit-spread,
iron-condor) in a single YAML file. Each instruction is executed in order.

Actions:
  execute (exec, run)   Execute all trades in the playbook
  validate (check)      Validate playbook structure without executing
  list (ls)             List available playbook files in a directory
                                  ''',
                                  epilog='''
Examples:
  %(prog)s execute playbooks/example_mixed.yaml          Execute (dry-run)
  %(prog)s execute playbooks/example_mixed.yaml --paper   Execute on paper
  %(prog)s execute playbooks/example_mixed.yaml --live    Execute on live
  %(prog)s validate playbooks/example_mixed.yaml          Validate only
  %(prog)s list                                           List available playbooks
  %(prog)s list --dir ./my_playbooks                      List from custom dir

Playbook YAML format:
  - Each trade is an instruction with: type, symbol, strikes, expiration, etc.
  - See playbooks/example_mixed.yaml for a full template

Aliases: pb
                                  ''',
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    pb_sub = p_pb.add_subparsers(dest="playbook_action", help="Playbook action")

    pb_exec = pb_sub.add_parser("execute", help="Execute a playbook", aliases=["exec", "run"])
    pb_exec.add_argument("playbook_file", help="Path to playbook YAML file")
    _add_connection_args(pb_exec)

    pb_val = pb_sub.add_parser("validate", help="Validate a playbook without executing",
                                aliases=["check"])
    pb_val.add_argument("playbook_file", help="Path to playbook YAML file")
    pb_val.add_argument("--data-dir", default="data/utp")

    pb_list = pb_sub.add_parser("list", help="List available playbooks", aliases=["ls"])
    pb_list.add_argument("--dir", default="playbooks", help="Playbooks directory (default: playbooks/)")

    # ── status ──
    p_status = subparsers.add_parser("status",
                                      help="System dashboard (positions, orders, connections)",
                                      aliases=["st", "dash"],
                                      description='''
Display a unified system dashboard showing active positions, pending orders,
recent trades, IBKR connection status, and cache statistics. Provides a
quick overview of the entire trading system state. Auto-detects a running
daemon and routes through HTTP if available.
                                      ''',
                                      epilog='''
Examples:
  %(prog)s                         Dashboard (default: paper mode)
  %(prog)s --live                  Dashboard with live IBKR data
  %(prog)s --paper                 Dashboard with paper account data

Dashboard sections:
  - Active positions with current P&L
  - Pending/working orders
  - Recent trade activity
  - IBKR connection health
  - Data cache statistics

Aliases: st, dash
                                      ''',
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_connection_args(p_status, default_paper=True)

    # ── flush ──
    p_flush = subparsers.add_parser("flush",
                                     help="Clear local position store and/or ledger",
                                     aliases=["reset"],
                                     description='''
Clear local persisted data (positions, ledger, or both). This is a local-only
operation that does NOT affect the broker. BLOCKED when a daemon is running —
use 'reconcile --flush' instead to flush through the daemon safely.

For a full system reset including closed P&L history, use 'reconcile --hard-reset'.
                                     ''',
                                     epilog='''
Examples:
  %(prog)s                         Flush all local data (positions + ledger)
  %(prog)s positions               Flush positions only (keep ledger)
  %(prog)s ledger                  Flush ledger only (keep positions)
  %(prog)s --data-dir data/custom  Flush from custom data directory

Note: This command is BLOCKED when a daemon is running to prevent
data conflicts. Use 'reconcile --flush' or 'reconcile --hard-reset'
to manage data through the daemon.

Aliases: reset
                                     ''',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    p_flush.add_argument("what", nargs="?", default="all",
                         choices=["positions", "ledger", "all"],
                         help="What to flush (default: all)")
    p_flush.add_argument("--data-dir", default="data/utp", help="Data directory")

    # ── reconcile ──
    p_recon = subparsers.add_parser("reconcile",
                                     help="Compare system vs broker positions",
                                     aliases=["recon"],
                                     description='''
Compare system-tracked positions against broker-reported positions to find
discrepancies. Optionally flush and rebuild from the broker as the source of
truth. Supports three levels of reset:

  (no flags)       Compare only — show mismatches but change nothing
  --flush          Clear open positions and re-sync from broker (preserves
                   closed positions for P&L history)
  --hard-reset     Full reset: clear ALL data (open + closed positions,
                   ledger, executions) and rebuild from scratch. When a
                   daemon is running, also clears the daemon's in-memory
                   position store.
                                     ''',
                                     epilog='''
Examples:
  %(prog)s --live                          Compare system vs broker (read-only)
  %(prog)s --flush --show --live           Flush open positions, sync, display
  %(prog)s --flush --portfolio --live      Flush, sync, full portfolio dump
  %(prog)s --hard-reset --live             Full reset + rebuild from scratch
  %(prog)s --paper                         Reconcile paper account
  %(prog)s --show --live                   Reconcile + show synced positions

Aliases: recon
                                     ''',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    p_recon.add_argument("--flush", action="store_true",
                         help="Flush open positions and re-sync from broker "
                              "(preserves closed positions for P&L history)")
    p_recon.add_argument("--hard-reset", action="store_true",
                         help="Full reset: clear ALL data (open + closed positions, "
                              "ledger, executions) and rebuild from scratch")
    p_recon.add_argument("--show", action="store_true",
                         help="After reconciling, show synced positions")
    p_recon.add_argument("--portfolio", action="store_true",
                         help="After reconciling, dump full portfolio summary")
    _add_connection_args(p_recon, default_paper=True)

    # ── readiness ──
    p_ready = subparsers.add_parser("readiness",
                                     help="Test IBKR connectivity and all 5 trade types",
                                     aliases=["ready", "test"],
                                     description='''
Test IBKR connectivity by validating all 5 trade types (equity, option,
credit-spread, debit-spread, iron-condor) without executing any orders.
Checks: IBKR connection, contract qualification, market data, and
optionally margin requirements.

WARNING: When a daemon is running, readiness needs its own IBKR client-id
to avoid conflicts. Use --client-id with a different value than the daemon.
                                     ''',
                                     epilog='''
Examples:
  %(prog)s --symbol SPX --paper                All 5 trade types (paper)
  %(prog)s --symbol NDX --skip-margin --live   Skip margin checks
  %(prog)s --symbol SPX --port 7496 --client-id 11  Custom connection
  %(prog)s --symbol RUT --market-data-type 1   Use live (paid) market data

Test steps:
  1. Connect to IBKR TWS/Gateway
  2. Qualify contracts for the symbol
  3. Fetch market data (bid/ask/last)
  4. Check margin for each trade type (unless --skip-margin)
  5. Report pass/fail for each step

Aliases: ready, test
                                     ''',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    p_ready.add_argument("--symbol", default="SPX", help="Symbol to test (default: SPX)")
    p_ready.add_argument("--expiration", default=None, help="Expiration date")
    p_ready.add_argument("--skip-margin", action="store_true", help="Skip margin checks")
    p_ready.add_argument("--market-data-type", type=int, default=4,
                         help="1=live(paid), 4=delayed(free) (default: 4)")
    _add_connection_args(p_ready)

    # ── server ──
    p_server = subparsers.add_parser("server",
                                      help="Start standalone REST API server",
                                      aliases=["serve", "api"],
                                      description='''
Start a standalone FastAPI HTTP server. This is a lightweight server without
IBKR connection or background tasks (no auto-expiration, no position sync).
For a full-featured server with IBKR, use 'daemon' instead.

The server binds to 0.0.0.0 by default so it is accessible from LAN.
LAN requests (private IPs) skip authentication automatically.
                                      ''',
                                      epilog='''
Examples:
  %(prog)s                             Start on 0.0.0.0:8000
  %(prog)s --server-port 9000          Custom port
  %(prog)s --server-host 127.0.0.1     Bind to localhost only
  %(prog)s --reload                    Auto-reload on code changes (dev mode)

For production use, prefer 'daemon' which includes:
  - Persistent IBKR connection
  - Auto-expiration of options
  - Position sync every 2 minutes
  - Process auto-restart on crash

Aliases: serve, api
                                      ''',
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    p_server.add_argument("--server-host", default="0.0.0.0",
                          help="API server bind address (default: 0.0.0.0)")
    p_server.add_argument("--server-port", type=int, default=8000,
                          help="API server port (default: 8000)")
    p_server.add_argument("--reload", action="store_true",
                          help="Enable auto-reload for development")

    # ── daemon ──
    p_daemon = subparsers.add_parser("daemon",
                                      help="Start long-running daemon (IBKR + HTTP API + background loops)",
                                      aliases=["d"],
                                      description='''
Start an always-on daemon that holds the IBKR connection, runs background
tasks, and serves the HTTP API. All CLI commands auto-detect the daemon and
route through HTTP, sharing the same IBKR connection.

Background tasks:
  - Auto-expiration of options at end of day
  - Position sync (polls IBKR every 2 minutes for out-of-band changes)
  - IBKR auto-reconnect with exponential backoff (2s to 10s cap)
  - Process auto-restart on crash (up to 20 consecutive failures)
  - Optional: advisor signal generation + auto-execution
  - Optional: real-time market data streaming to Redis/QuestDB/WS

Starts in degraded mode if IBKR is unavailable, retrying in background.
                                      ''',
                                      epilog='''
Examples:
  %(prog)s --paper                                    Paper trading daemon
  %(prog)s --live                                     Live trading daemon
  %(prog)s --live --advisor-profile tiered_v2         With advisor signals
  %(prog)s --live --advisor-profile tiered_v2 --auto-execute  Full auto
  %(prog)s --live --no-restart                        No auto-restart on crash
  %(prog)s --server-port 9000 --paper                 Custom API port
  %(prog)s --live --streaming-config configs/streaming_default.yaml

Stopping:
  Ctrl-C or SIGTERM — clean shutdown (no restart)
  Unhandled exception — auto-restarts with backoff (unless --no-restart)

Aliases: d
                                      ''',
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    p_daemon.add_argument("--server-host", default="0.0.0.0", help="API listen address (default: 0.0.0.0)")
    p_daemon.add_argument("--server-port", type=int, default=8000, help="API listen port (default: 8000)")
    p_daemon.add_argument("--advisor-profile", default=None, help="Advisor profile to run (e.g. tiered_v2)")
    p_daemon.add_argument("--auto-execute", action="store_true", help="Auto-execute advisor recommendations")
    p_daemon.add_argument("--no-restart", action="store_true",
                          help="Disable auto-restart on crash (exit immediately on failure)")
    p_daemon.add_argument("--streaming-config", default=None, metavar="YAML",
                          help="YAML config for real-time market data streaming (IBKR → Redis/QuestDB/WS)")
    _add_connection_args(p_daemon, default_paper=True)

    # ── repl ──
    p_repl = subparsers.add_parser("repl",
                                    help="Interactive REPL connected to daemon",
                                    aliases=["shell", "interactive"],
                                    description='''
Start an interactive Read-Eval-Print Loop connected to a running daemon.
All commands are sent to the daemon via HTTP. The prompt shows the current
trade mode ([LIVE], [PAPER], or [DRY]).

REPL commands:
  portfolio         Show positions and P&L
  quote <SYMBOL>    Get real-time quote
  options <SYM>     Show option chain (supports --strike-range, --type, etc.)
  trades --all      Show all trade history
  orders            Show open orders
  advisor           View advisor recommendations (if --advisor-profile set)
  y 1 3             Confirm advisor entries 1 and 3 for execution
  status            System dashboard
  help              Show available commands
  quit / exit       Close REPL
                                    ''',
                                    epilog='''
Examples:
  %(prog)s                                 Auto-detect daemon on localhost:8000
  %(prog)s --server http://192.168.1.50:8000  Connect to remote daemon
  %(prog)s --server-port 9000              Auto-detect on custom port

REPL session example:
  [LIVE] utp> portfolio
  [LIVE] utp> quote SPX NDX
  [LIVE] utp> options SPX --strike-range 2
  [LIVE] utp> advisor
  [LIVE] utp> y 1 3
  [LIVE] utp> quit

Aliases: shell, interactive
                                    ''',
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    p_repl.add_argument("--server", default=None, metavar="URL",
                         help="Daemon URL (default: auto-detect)")
    p_repl.add_argument("--server-port", type=int, default=8000,
                         help="Daemon port for auto-detect (default: 8000)")

    # ── journal ──
    p_journal = subparsers.add_parser("journal",
                                       help="View trade history and ledger entries",
                                       aliases=["log", "history"],
                                       description='''
View the append-only transaction ledger which records every trade event.
Each entry includes timestamp, event type, order details, and position
snapshots. Filter by date range, event type, or order ID. Auto-detects
a running daemon and routes through HTTP if available.
                                       ''',
                                       epilog='''
Examples:
  %(prog)s --days 7                    Last 7 days of entries
  %(prog)s --limit 100                 Show up to 100 entries
  %(prog)s --event-type TRADE_EXECUTED Only executed trades
  %(prog)s --event-type ORDER_SUBMITTED  Only submitted orders
  %(prog)s --event-type POSITION_OPENED  Position openings
  %(prog)s --event-type POSITION_CLOSED  Position closings
  %(prog)s --order-id abc123           Filter by order ID

Common event types:
  ORDER_SUBMITTED    Order sent to broker
  TRADE_EXECUTED     Order filled
  POSITION_OPENED    New position created
  POSITION_CLOSED    Position fully closed
  EXPIRATION         Option expired worthless

Aliases: log, history
                                       ''',
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p_journal.add_argument("--days", type=int, default=None,
                           help="Show entries from the last N days")
    p_journal.add_argument("--limit", type=int, default=50,
                           help="Max entries to show (default: 50)")
    p_journal.add_argument("--event-type", default=None,
                           help="Filter by event type (e.g., ORDER_SUBMITTED, POSITION_OPENED)")
    p_journal.add_argument("--order-id", default=None,
                           help="Filter by order ID")
    p_journal.add_argument("--data-dir", default="data/utp", help="Data directory")
    p_journal.add_argument("--server", default=None, metavar="URL",
                           help="Connect to UTP daemon at URL (auto-detected if running)")
    p_journal.add_argument("--server-port", type=int, default=8000,
                           help="Daemon port for auto-detect (default: 8000)")

    # ── performance ──
    p_perf = subparsers.add_parser("performance",
                                    help="Show performance metrics (win rate, Sharpe, drawdown)",
                                    aliases=["perf", "metrics"],
                                    description='''
Compute and display trading performance metrics from closed positions. Includes
win rate, Sharpe ratio, max drawdown, profit factor, average win/loss, and
total P&L. Can filter by date range. Auto-detects a running daemon and routes
through HTTP if available.
                                    ''',
                                    epilog='''
Examples:
  %(prog)s --days 30                   Last 30 days of performance
  %(prog)s --days 7                    Last week
  %(prog)s --start-date 2026-01-01     From January 1st to now
  %(prog)s --start-date 2026-01-01 --end-date 2026-03-01  Custom range

Metrics shown:
  Win Rate         Percentage of profitable trades
  Sharpe Ratio     Risk-adjusted return
  Max Drawdown     Largest peak-to-trough decline
  Profit Factor    Gross profit / gross loss
  Avg Win/Loss     Average P&L for winning and losing trades
  Total P&L        Net profit/loss for the period

Aliases: perf, metrics
                                    ''',
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    p_perf.add_argument("--days", type=int, default=30,
                        help="Show last N days (default: 30)")
    p_perf.add_argument("--start-date", default=None,
                        help="Start date YYYY-MM-DD")
    p_perf.add_argument("--end-date", default=None,
                        help="End date YYYY-MM-DD")
    p_perf.add_argument("--data-dir", default="data/utp", help="Data directory")
    p_perf.add_argument("--server", default=None, metavar="URL",
                        help="Connect to UTP daemon at URL (auto-detected if running)")
    p_perf.add_argument("--server-port", type=int, default=8000,
                        help="Daemon port for auto-detect (default: 8000)")

    # ── orders ──
    p_orders = subparsers.add_parser("orders",
                                      help="Show working (open) orders at the broker",
                                      aliases=["open-orders", "oo"],
                                      description='''
Show all working (open) orders at the broker. These are orders that have been
submitted but not yet filled, cancelled, or rejected. Auto-detects a running
daemon and routes through HTTP if available, otherwise connects to IBKR
directly with --paper or --live.
                                      ''',
                                      epilog='''
Examples:
  %(prog)s --live                  Show open orders on live account
  %(prog)s --paper                 Show open orders on paper account
  %(prog)s --live --host 10.0.0.5  Connect to remote TWS
  %(prog)s                         Via daemon (if running, no flags needed)

Output columns:
  Order ID     IBKR order identifier (use with 'cancel' command)
  Status       SUBMITTED, PENDING, or PARTIAL_FILL
  Description  Symbol, action, quantity, limit price

Aliases: open-orders, oo
                                      ''',
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_connection_args(p_orders)

    # ── cancel ──
    p_cancel = subparsers.add_parser("cancel",
                                      help="Cancel a working order",
                                      aliases=["cx"],
                                      description='''
Cancel a working order at the broker by order ID, or cancel all open orders.
Use 'orders' to find order IDs first. Auto-detects a running daemon and routes
through HTTP if available.
                                      ''',
                                      epilog='''
Examples:
  %(prog)s --order-id 123 --live   Cancel order #123 on live account
  %(prog)s --all --live            Cancel all open orders
  %(prog)s --order-id 456 --paper  Cancel on paper account
  %(prog)s --order-id 123         Via daemon (if running, no --live needed)

Workflow:
  1. python utp.py orders --live     # Find order IDs
  2. python utp.py cancel --order-id 123 --live  # Cancel specific order

Aliases: cx
                                      ''',
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    p_cancel.add_argument("--order-id", default=None,
                           help="Order ID to cancel")
    p_cancel.add_argument("--all", action="store_true",
                           help="Cancel all open orders")
    _add_connection_args(p_cancel)

    # ── close ──
    p_close = subparsers.add_parser("close",
                                     help="Close an open position by ID",
                                     aliases=["cl"],
                                     description='''
Close an open position by its position ID (or unique prefix). Automatically
derives the closing order parameters from the position's legs and submits a
REAL closing order to IBKR (not just local bookkeeping).

For credit spreads: submits a BUY_TO_CLOSE combo order at the specified debit.
For debit spreads: submits a SELL_TO_CLOSE combo order.
For iron condors: reverses all 4 legs.

Use --simulate to check margin without actually closing.
Auto-detects a running daemon and routes through HTTP if available.
                                     ''',
                                     epilog='''
Examples:
  %(prog)s 2d9a --paper                    Close position '2d9a' on paper account
  %(prog)s 2d9a --live                     Close on live at $0.05 debit (default)
  %(prog)s 2d9a --net-price 0.10 --live    Close at $0.10 debit
  %(prog)s 2d9a -q 1 --live               Partial close: 1 contract only
  %(prog)s 2d9a --simulate --live          Margin check only, no close
  %(prog)s 2d9a                            Dry-run close (no broker connection)

The position ID can be a prefix — if it uniquely matches one open position,
that position is closed. Use 'trades --all' to find position IDs.

Aliases: cl
                                     ''',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    p_close.add_argument("position_id", help="Position ID or unique prefix to close")
    p_close.add_argument("--quantity", "-q", type=int, default=None,
                          help="Number of contracts to close (default: all)")
    p_close.add_argument("--net-price", type=float, default=None,
                          help="Debit price to close (default: current mark from IBKR)")
    p_close.add_argument("--simulate", action="store_true",
                          help="Check margin only — do NOT execute the close")
    p_close.add_argument("--confirm", action="store_true",
                          help="Confirm and execute (without this, shows order summary only)")
    _add_connection_args(p_close)

    # ── trades ──
    p_trades = subparsers.add_parser("trades",
                                      help="Today's trades with order/position IDs",
                                      aliases=["activity"],
                                      description='''
Show trade activity: positions opened and closed. By default shows today's
closed positions. Use --days N for history or --all for everything.
With --live or --paper, enriches output with broker-authoritative P&L
(AvgCost, Mark, unrealized P&L from IBKR). Auto-detects a running daemon
and routes through HTTP if available.
                                      ''',
                                      epilog='''
Examples:
  %(prog)s --live                  Today's trades with IBKR P&L
  %(prog)s --days 7 --live         Last 7 days of trades
  %(prog)s --all --live            All trades (open + closed)
  %(prog)s --detail 2d9a --live    Full detail for a position ID prefix
  %(prog)s --paper                 Paper account trades

Output columns (with --live/--paper):
  Symbol       Underlying symbol
  Type         equity or multi_leg
  Qty          Number of contracts/shares
  AvgCost      Broker-reported average cost basis
  Mark         Current market price (from IBKR)
  P&L          Unrealized profit/loss (broker-authoritative)
  Status       open or closed
  Exit Reason  expired, stop_loss, profit_target, api_close, etc.

Aliases: activity
                                      ''',
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    p_trades.add_argument("--days", type=int, default=0,
                           help="Show trades from the last N days (default: 0 = today)")
    p_trades.add_argument("--all", action="store_true", dest="show_all",
                           help="Show all trades (open + closed) regardless of date")
    p_trades.add_argument("--detail", default=None,
                           help="Show full detail for a position ID or order ID prefix")
    _add_connection_args(p_trades)  # includes --data-dir

    # ── executions ──
    p_exec = subparsers.add_parser("executions",
                                    help="Show IBKR execution history grouped by order (identifies multi-leg trades)",
                                    aliases=["exec"],
                                    description='''
Show trade executions from IBKR (last ~7 days), grouped by permanent order ID
(perm_id). Multi-leg trades (credit spreads, iron condors) are automatically
identified by matching fills with the same perm_id, so you can see which legs
belong to the same order.

Executions are cached locally in data/utp/{mode}/executions.json and
deduplicated across runs. Use --flush to clear the cache and re-fetch.
Auto-detects a running daemon and routes through HTTP if available.
                                    ''',
                                    epilog='''
Examples:
  %(prog)s --live                  Fetch and show grouped executions
  %(prog)s --live --flush          Clear cache, re-fetch from IBKR
  %(prog)s --live --symbol RUT     Filter by symbol
  %(prog)s --paper                 Paper account executions

Output groups fills by perm_id:
  Order #12345 (perm_id: 67890)
    SELL SPX 5500P 2026-03-20  qty=1  price=3.50
    BUY  SPX 5475P 2026-03-20  qty=1  price=1.20
    Net credit: $2.30

Aliases: exec
                                    ''',
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    p_exec.add_argument("--flush", action="store_true",
                         help="Clear local execution cache before fetching")
    p_exec.add_argument("--symbol", default=None,
                         help="Filter to a specific symbol")
    _add_connection_args(p_exec)

    # Parse
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Dispatch
    cmd = args.command

    # Handle aliases
    alias_map = {
        "port": "portfolio", "positions": "portfolio", "pos": "portfolio",
        "q": "quote",
        "m": "margin",
        "t": "trade",
        "pb": "playbook",
        "st": "status", "dash": "status",
        "recon": "reconcile", "reset": "flush",
        "opts": "options", "chain": "options",
        "ready": "readiness", "test": "readiness",
        "serve": "server", "api": "server",
        "d": "daemon",
        "shell": "repl", "interactive": "repl",
        "log": "journal", "history": "journal",
        "perf": "performance", "metrics": "performance",
        "open-orders": "orders", "oo": "orders",
        "cx": "cancel",
        "cl": "close",
        "activity": "trades",
        "exec": "executions",
    }
    cmd = alias_map.get(cmd, cmd)

    if cmd == "server":
        rc = _cmd_server(args)
    elif cmd == "daemon":
        if getattr(args, "no_restart", False):
            rc = asyncio.run(_cmd_daemon(args))
        else:
            rc = _run_daemon_with_restart(args)
    elif cmd == "repl":
        rc = asyncio.run(_cmd_repl(args))
    elif cmd == "portfolio":
        rc = asyncio.run(_cmd_portfolio(args))
    elif cmd == "quote":
        rc = asyncio.run(_cmd_quote(args))
    elif cmd == "options":
        rc = asyncio.run(_cmd_options(args))
    elif cmd == "margin":
        rc = asyncio.run(_cmd_margin(args))
    elif cmd == "trade":
        rc = asyncio.run(_cmd_trade(args))
    elif cmd == "playbook":
        action = getattr(args, "playbook_action", None)
        if action == "list" or action == "ls":
            # List playbooks
            pb_dir = Path(getattr(args, "dir", "playbooks"))
            if pb_dir.exists():
                yamls = sorted(pb_dir.glob("*.yaml")) + sorted(pb_dir.glob("*.yml"))
                if yamls:
                    _print_header("Available Playbooks")
                    for f in yamls:
                        print(f"  {f}")
                else:
                    print(f"  No playbook files found in {pb_dir}")
            else:
                print(f"  Directory not found: {pb_dir}")
            rc = 0
        elif action in ("execute", "exec", "run", "validate", "check"):
            # Normalize action
            if action in ("exec", "run"):
                args.playbook_action = "execute"
            elif action == "check":
                args.playbook_action = "validate"
            rc = asyncio.run(_cmd_playbook(args))
        else:
            # No action specified — print playbook help
            subparsers.choices["playbook"].print_help()
            rc = 0
    elif cmd == "status":
        rc = asyncio.run(_cmd_status(args))
    elif cmd == "flush":
        rc = asyncio.run(_cmd_flush(args))
    elif cmd == "reconcile":
        rc = asyncio.run(_cmd_reconcile(args))
    elif cmd == "readiness":
        rc = asyncio.run(_cmd_readiness(args))
    elif cmd == "journal":
        rc = asyncio.run(_cmd_journal(args))
    elif cmd == "performance":
        rc = asyncio.run(_cmd_performance(args))
    elif cmd == "orders":
        rc = asyncio.run(_cmd_orders(args))
    elif cmd == "cancel":
        rc = asyncio.run(_cmd_cancel(args))
    elif cmd == "trades":
        rc = asyncio.run(_cmd_trades(args))
    elif cmd == "close":
        rc = asyncio.run(_cmd_close(args))
    elif cmd == "executions":
        rc = asyncio.run(_cmd_executions(args))
    else:
        parser.print_help()
        rc = 0

    sys.exit(rc)


if __name__ == "__main__":
    main()
