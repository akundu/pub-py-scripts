#!/usr/bin/env python3
"""UTP Voice — Natural Language Mobile Trading Interface.

A mobile-friendly web app that sits in front of the UTP daemon and lets you
execute trades via voice or text natural language input.

Usage:
    python utp_voice.py                              # Start server (default)
    python utp_voice.py serve                        # Start server
    python utp_voice.py serve --port 8800            # Custom port
    python utp_voice.py add-user akundu              # Add/update user credentials
    python utp_voice.py list-users                   # List configured users

Examples:
    # First time setup
    python utp_voice.py add-user akundu
    export ANTHROPIC_API_KEY="sk-ant-..."
    export UTP_VOICE_JWT_SECRET="my-secret-key"
    python utp_voice.py serve

    # Then open http://localhost:8800 on your phone
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import bcrypt as _bcrypt
import httpx
from fastapi import FastAPI, Request, Response, HTTPException, Depends, Cookie
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from jose import JWTError, jwt
import uvicorn

# ── Configuration ─────────────────────────────────────────────────────────────

UTP_DAEMON_URL = os.environ.get("UTP_DAEMON_URL", "http://localhost:8000")
UTP_VOICE_PORT = int(os.environ.get("UTP_VOICE_PORT", "8800"))
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
JWT_SECRET = os.environ.get("UTP_VOICE_JWT_SECRET", "")
JWT_EXPIRE_MINUTES = int(os.environ.get("UTP_VOICE_JWT_EXPIRE_MINUTES", "480"))
CREDENTIALS_FILE = os.environ.get(
    "UTP_VOICE_CREDENTIALS_FILE", "data/utp_voice/credentials.json"
)
CLAUDE_MODEL = os.environ.get("UTP_VOICE_MODEL", "claude-sonnet-4-20250514")
PUBLIC_MODE = False  # Set via --public flag; allows anonymous access to options/picks

logger = logging.getLogger("utp_voice")

# ── Credentials Management ────────────────────────────────────────────────────


def _load_credentials() -> list[dict]:
    path = Path(CREDENTIALS_FILE)
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _save_credentials(creds: list[dict]) -> None:
    path = Path(CREDENTIALS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(creds, f, indent=2)


def add_user(username: str, password: str) -> None:
    creds = _load_credentials()
    hashed = _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()
    for c in creds:
        if c["username"] == username:
            c["password_hash"] = hashed
            _save_credentials(creds)
            print(f"Updated password for user '{username}'.")
            return
    creds.append({"username": username, "password_hash": hashed})
    _save_credentials(creds)
    print(f"Added user '{username}'.")


def verify_user(username: str, password: str) -> bool:
    creds = _load_credentials()
    for c in creds:
        if c["username"] == username:
            return _bcrypt.checkpw(password.encode(), c["password_hash"].encode())
    return False


def list_users() -> list[str]:
    return [c["username"] for c in _load_credentials()]


# ── JWT Session ───────────────────────────────────────────────────────────────


def create_token(username: str) -> str:
    expire = datetime.now(UTC) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": username, "exp": expire},
        JWT_SECRET,
        algorithm="HS256",
    )


def decode_token(token: str) -> str | None:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload.get("sub")
    except JWTError:
        return None


COOKIE_NAME = "utp_voice_session"


async def require_session(
    request: Request,
    utp_voice_session: str | None = Cookie(default=None),
) -> str:
    token = utp_voice_session
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    username = decode_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Session expired")
    return username


async def optional_session(
    request: Request,
    utp_voice_session: str | None = Cookie(default=None),
) -> str | None:
    """Require auth unless PUBLIC_MODE is enabled."""
    if PUBLIC_MODE:
        return None  # Anonymous access allowed
    return await require_session(request, utp_voice_session)


# ── UTP Daemon Client ─────────────────────────────────────────────────────────


class UTPDaemonClient:
    """Async HTTP client for the UTP daemon."""

    def __init__(self, base_url: str = UTP_DAEMON_URL):
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=15.0
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _get(self, path: str, params: dict | None = None) -> dict:
        client = await self._get_client()
        resp = await client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    async def _post(self, path: str, json_data: dict | None = None) -> dict:
        client = await self._get_client()
        resp = await client.post(path, json=json_data)
        resp.raise_for_status()
        return resp.json()

    async def health(self) -> dict:
        return await self._get("/health")

    async def get_portfolio(self, include_quotes: bool = False) -> dict:
        params = {}
        if include_quotes:
            params["include_quotes"] = "true"
        return await self._get("/dashboard/portfolio", params=params or None)

    async def get_quote(self, symbol: str) -> dict:
        return await self._get(f"/market/quote/{symbol.upper()}")

    async def get_options(
        self,
        symbol: str,
        option_type: str | None = None,
        strike_range_pct: float | None = None,
        expiration: str | None = None,
        strike_min: float | None = None,
        strike_max: float | None = None,
        list_expirations: bool = False,
    ) -> dict:
        params: dict[str, Any] = {}
        if option_type:
            params["option_type"] = option_type.upper()
        if expiration:
            params["expiration"] = expiration
        if strike_min is not None:
            params["strike_min"] = strike_min
        if strike_max is not None:
            params["strike_max"] = strike_max
        if list_expirations:
            params["list_expirations"] = "true"
        # strike_range_pct is handled client-side by computing strike_min/max
        if strike_range_pct and not strike_min and not strike_max and not list_expirations:
            # Fetch quote first to compute range
            try:
                quote = await self.get_quote(symbol)
                price = quote.get("last") or quote.get("bid") or 0
                if price > 0:
                    params["strike_min"] = round(price * (1 - strike_range_pct / 100), 2)
                    params["strike_max"] = round(price * (1 + strike_range_pct / 100), 2)
            except Exception:
                pass
        return await self._get(f"/market/options/{symbol.upper()}", params=params)

    async def execute_trade(self, payload: dict) -> dict:
        return await self._post("/trade/execute", json_data=payload)

    async def close_position(
        self,
        position_id: str,
        quantity: int | None = None,
        net_price: float | None = None,
    ) -> dict:
        body: dict[str, Any] = {"position_id": position_id}
        if quantity is not None:
            body["quantity"] = quantity
        if net_price is not None:
            body["net_price"] = net_price
        return await self._post("/trade/close", json_data=body)

    async def get_trades(self, days: int = 0, include_all: bool = False) -> dict:
        params: dict[str, Any] = {}
        if days:
            params["days"] = days
        if include_all:
            params["all"] = "true"
        return await self._get("/account/trades", params=params)

    async def get_orders(self) -> dict:
        return await self._get("/account/orders")

    async def cancel_order(self, order_id: str) -> dict:
        return await self._post("/account/cancel", json_data={"order_id": order_id})

    async def get_performance(self, days: int = 30) -> dict:
        return await self._get("/dashboard/performance", params={"days": days})

    async def get_status(self) -> dict:
        return await self._get("/dashboard/status")

    async def flush_and_reconcile(self) -> dict:
        await self._post("/account/flush")
        return await self._get("/dashboard/summary")

    async def get_reconciliation(self) -> dict:
        return await self._get("/account/reconciliation")


# Module-level singleton
_daemon_client: UTPDaemonClient | None = None


def get_daemon_client() -> UTPDaemonClient:
    global _daemon_client
    if _daemon_client is None:
        _daemon_client = UTPDaemonClient(UTP_DAEMON_URL)
    return _daemon_client


# ── Claude Agent — Tool Definitions ───────────────────────────────────────────

TOOLS = [
    {
        "name": "get_portfolio",
        "description": (
            "Get the current portfolio summary including all open positions, "
            "unrealized P&L, and account balances. Each position shows symbol, "
            "option type, strikes, quantity, entry price, current mark, and P&L."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_quote",
        "description": (
            "Get a real-time quote for a symbol. Returns bid, ask, last price, "
            "and volume. Works for indices (SPX, NDX, RUT) and equities."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol (e.g., SPX, NDX, RUT, SPY)",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_options",
        "description": (
            "Get option chain data for a symbol. Returns strikes, bids, asks, "
            "volumes for calls and/or puts. Use this to find available spreads "
            "and analyze ROI. Can list available expirations or fetch quotes "
            "for a specific expiration."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Underlying symbol (e.g., SPX, RUT, NDX)",
                },
                "option_type": {
                    "type": "string",
                    "enum": ["PUT", "CALL"],
                    "description": "Filter by option type. Omit for both.",
                },
                "expiration": {
                    "type": "string",
                    "description": "Expiration date YYYY-MM-DD. Omit to list available expirations.",
                },
                "strike_range_pct": {
                    "type": "number",
                    "description": "Strike range as percentage of current price (e.g., 3 = +/-3%). Default 5.",
                },
                "strike_min": {
                    "type": "number",
                    "description": "Minimum strike price filter.",
                },
                "strike_max": {
                    "type": "number",
                    "description": "Maximum strike price filter.",
                },
                "list_expirations": {
                    "type": "boolean",
                    "description": "If true, only list available expiration dates.",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_trades",
        "description": (
            "Get trade history. Returns recent executed trades with P&L. "
            "Use days=0 for today only, or set days for a longer window."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days of history (0=today, default 0)",
                },
                "include_all": {
                    "type": "boolean",
                    "description": "Include all trades (open + closed) regardless of date",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_orders",
        "description": "Get all currently open/working orders.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_performance",
        "description": (
            "Get performance metrics: win rate, Sharpe ratio, max drawdown, "
            "profit factor, total P&L over the specified period."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days for performance calculation (default 30)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "execute_trade",
        "description": (
            "Execute a trade. IMPORTANT: This is a WRITE operation that will be "
            "shown to the user for confirmation before execution. Supports: "
            "credit-spread, debit-spread, equity, iron-condor, single-option. "
            "For credit spreads: short_strike is the strike you sell, long_strike "
            "is the strike you buy (further OTM for protection)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "trade_type": {
                    "type": "string",
                    "enum": [
                        "credit-spread",
                        "debit-spread",
                        "equity",
                        "iron-condor",
                        "single-option",
                    ],
                    "description": "Type of trade",
                },
                "symbol": {
                    "type": "string",
                    "description": "Underlying symbol (e.g., SPX, RUT, NDX)",
                },
                "option_type": {
                    "type": "string",
                    "enum": ["PUT", "CALL"],
                    "description": "Option type for spread trades",
                },
                "expiration": {
                    "type": "string",
                    "description": "Expiration date YYYY-MM-DD",
                },
                "short_strike": {
                    "type": "number",
                    "description": "Short strike price (the one you sell)",
                },
                "long_strike": {
                    "type": "number",
                    "description": "Long strike price (the one you buy for protection)",
                },
                "quantity": {
                    "type": "integer",
                    "description": "Number of contracts (default 1)",
                },
                "net_price": {
                    "type": "number",
                    "description": "Limit price per contract. Omit for MARKET order.",
                },
                "side": {
                    "type": "string",
                    "enum": ["BUY", "SELL"],
                    "description": "For equity trades: BUY or SELL",
                },
                "put_short": {"type": "number", "description": "Iron condor put short strike"},
                "put_long": {"type": "number", "description": "Iron condor put long strike"},
                "call_short": {"type": "number", "description": "Iron condor call short strike"},
                "call_long": {"type": "number", "description": "Iron condor call long strike"},
            },
            "required": ["trade_type", "symbol"],
        },
    },
    {
        "name": "close_position",
        "description": (
            "Close an open position by its ID (or ID prefix). "
            "This is a WRITE operation requiring user confirmation. "
            "Optionally specify a limit price (net_price) or partial quantity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "position_id": {
                    "type": "string",
                    "description": "Position ID or prefix (first 6+ chars)",
                },
                "quantity": {
                    "type": "integer",
                    "description": "Partial close quantity (omit for full close)",
                },
                "net_price": {
                    "type": "number",
                    "description": "Limit price for closing. Omit for MARKET order.",
                },
            },
            "required": ["position_id"],
        },
    },
    {
        "name": "cancel_order",
        "description": (
            "Cancel an open/working order by order ID. "
            "This is a WRITE operation requiring user confirmation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID to cancel",
                },
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "reconcile_flush",
        "description": (
            "Flush local position state and re-sync from the broker. "
            "This is a WRITE operation requiring user confirmation. "
            "Use when positions are out of sync."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

WRITE_TOOLS = {"execute_trade", "close_position", "cancel_order", "reconcile_flush"}
READ_TOOLS = {"get_portfolio", "get_quote", "get_options", "get_trades", "get_orders", "get_performance"}

# ── Claude Agent — Trade Payload Builder ──────────────────────────────────────


def build_trade_payload(tool_input: dict) -> dict:
    """Convert agent tool_input into a TradeRequest payload for POST /trade/execute."""
    trade_type = tool_input.get("trade_type", "")
    symbol = tool_input.get("symbol", "").upper()
    quantity = tool_input.get("quantity", 1)
    net_price = tool_input.get("net_price")
    expiration = tool_input.get("expiration", "")
    option_type = tool_input.get("option_type", "PUT")

    order_type = "LIMIT" if net_price is not None else "MARKET"

    if trade_type == "equity":
        return {
            "equity_order": {
                "broker": "ibkr",
                "symbol": symbol,
                "side": tool_input.get("side", "BUY"),
                "quantity": quantity,
                "order_type": order_type,
                "limit_price": net_price,
            }
        }

    if trade_type == "credit-spread":
        short_strike = tool_input["short_strike"]
        long_strike = tool_input["long_strike"]
        # Credit spread: sell short (closer to money), buy long (further OTM)
        legs = [
            {
                "symbol": symbol,
                "expiration": expiration,
                "strike": short_strike,
                "option_type": option_type,
                "action": "SELL_TO_OPEN",
                "quantity": quantity,
            },
            {
                "symbol": symbol,
                "expiration": expiration,
                "strike": long_strike,
                "option_type": option_type,
                "action": "BUY_TO_OPEN",
                "quantity": quantity,
            },
        ]
        payload: dict[str, Any] = {
            "multi_leg_order": {
                "broker": "ibkr",
                "legs": legs,
                "order_type": order_type,
                "quantity": quantity,
            }
        }
        if net_price is not None:
            payload["multi_leg_order"]["net_price"] = net_price
        return payload

    if trade_type == "debit-spread":
        long_strike = tool_input["long_strike"]
        short_strike = tool_input["short_strike"]
        legs = [
            {
                "symbol": symbol,
                "expiration": expiration,
                "strike": long_strike,
                "option_type": option_type,
                "action": "BUY_TO_OPEN",
                "quantity": quantity,
            },
            {
                "symbol": symbol,
                "expiration": expiration,
                "strike": short_strike,
                "option_type": option_type,
                "action": "SELL_TO_OPEN",
                "quantity": quantity,
            },
        ]
        payload = {
            "multi_leg_order": {
                "broker": "ibkr",
                "legs": legs,
                "order_type": order_type,
                "quantity": quantity,
            }
        }
        if net_price is not None:
            payload["multi_leg_order"]["net_price"] = net_price
        return payload

    if trade_type == "iron-condor":
        legs = [
            {
                "symbol": symbol,
                "expiration": expiration,
                "strike": tool_input["put_short"],
                "option_type": "PUT",
                "action": "SELL_TO_OPEN",
                "quantity": quantity,
            },
            {
                "symbol": symbol,
                "expiration": expiration,
                "strike": tool_input["put_long"],
                "option_type": "PUT",
                "action": "BUY_TO_OPEN",
                "quantity": quantity,
            },
            {
                "symbol": symbol,
                "expiration": expiration,
                "strike": tool_input["call_short"],
                "option_type": "CALL",
                "action": "SELL_TO_OPEN",
                "quantity": quantity,
            },
            {
                "symbol": symbol,
                "expiration": expiration,
                "strike": tool_input["call_long"],
                "option_type": "CALL",
                "action": "BUY_TO_OPEN",
                "quantity": quantity,
            },
        ]
        payload = {
            "multi_leg_order": {
                "broker": "ibkr",
                "legs": legs,
                "order_type": order_type,
                "quantity": quantity,
            }
        }
        if net_price is not None:
            payload["multi_leg_order"]["net_price"] = net_price
        return payload

    if trade_type == "single-option":
        action = tool_input.get("side", "BUY")
        option_action = "BUY_TO_OPEN" if action == "BUY" else "SELL_TO_OPEN"
        strike = tool_input.get("short_strike") or tool_input.get("long_strike", 0)
        legs = [
            {
                "symbol": symbol,
                "expiration": expiration,
                "strike": strike,
                "option_type": option_type,
                "action": option_action,
                "quantity": quantity,
            },
        ]
        payload = {
            "multi_leg_order": {
                "broker": "ibkr",
                "legs": legs,
                "order_type": order_type,
                "quantity": quantity,
            }
        }
        if net_price is not None:
            payload["multi_leg_order"]["net_price"] = net_price
        return payload

    raise ValueError(f"Unknown trade_type: {trade_type}")


def describe_trade(tool_input: dict) -> str:
    """Generate a human-readable description of a trade for confirmation."""
    trade_type = tool_input.get("trade_type", "unknown")
    symbol = tool_input.get("symbol", "?")
    quantity = tool_input.get("quantity", 1)
    expiration = tool_input.get("expiration", "?")
    option_type = tool_input.get("option_type", "?")
    net_price = tool_input.get("net_price")

    if trade_type == "credit-spread":
        short = tool_input.get("short_strike", "?")
        long = tool_input.get("long_strike", "?")
        width = abs(float(short) - float(long)) if short != "?" and long != "?" else "?"
        price_str = f"@ ${net_price:.2f}" if net_price else "@ MARKET"
        return (
            f"CREDIT SPREAD: {symbol} {option_type} {short}/{long} "
            f"x{quantity} exp {expiration} {price_str} (width: {width})"
        )
    elif trade_type == "debit-spread":
        short = tool_input.get("short_strike", "?")
        long = tool_input.get("long_strike", "?")
        price_str = f"@ ${net_price:.2f}" if net_price else "@ MARKET"
        return (
            f"DEBIT SPREAD: {symbol} {option_type} {long}/{short} "
            f"x{quantity} exp {expiration} {price_str}"
        )
    elif trade_type == "equity":
        side = tool_input.get("side", "BUY")
        price_str = f"@ ${net_price:.2f}" if net_price else "@ MARKET"
        return f"EQUITY: {side} {quantity} {symbol} {price_str}"
    elif trade_type == "iron-condor":
        ps = tool_input.get("put_short", "?")
        pl = tool_input.get("put_long", "?")
        cs = tool_input.get("call_short", "?")
        cl = tool_input.get("call_long", "?")
        price_str = f"@ ${net_price:.2f}" if net_price else "@ MARKET"
        return (
            f"IRON CONDOR: {symbol} P{ps}/{pl} C{cs}/{cl} "
            f"x{quantity} exp {expiration} {price_str}"
        )
    elif trade_type == "single-option":
        side = tool_input.get("side", "BUY")
        strike = tool_input.get("short_strike") or tool_input.get("long_strike", "?")
        price_str = f"@ ${net_price:.2f}" if net_price else "@ MARKET"
        return f"OPTION: {side} {symbol} {option_type} {strike} x{quantity} exp {expiration} {price_str}"
    else:
        return f"TRADE: {trade_type} {symbol} x{quantity}"


def describe_write_action(tool_name: str, tool_input: dict) -> str:
    """Describe any write action for the confirmation card."""
    if tool_name == "execute_trade":
        return describe_trade(tool_input)
    elif tool_name == "close_position":
        pos_id = tool_input.get("position_id", "?")
        qty = tool_input.get("quantity")
        net = tool_input.get("net_price")
        parts = [f"CLOSE POSITION: {pos_id}"]
        if qty:
            parts.append(f"qty={qty}")
        if net is not None:
            parts.append(f"@ ${net:.2f}")
        else:
            parts.append("@ MARKET")
        return " ".join(parts)
    elif tool_name == "cancel_order":
        return f"CANCEL ORDER: {tool_input.get('order_id', '?')}"
    elif tool_name == "reconcile_flush":
        return "FLUSH & RECONCILE: Reset local positions and re-sync from broker"
    return f"{tool_name}: {json.dumps(tool_input)}"


# ── Claude Agent — Pending Confirmations ──────────────────────────────────────


@dataclass
class PendingAction:
    tool_name: str
    tool_input: dict
    description: str
    confirmation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)


_pending_confirmations: dict[str, PendingAction] = {}
CONFIRMATION_EXPIRY_SECONDS = 300  # 5 minutes


def store_pending(action: PendingAction) -> str:
    _cleanup_expired()
    _pending_confirmations[action.confirmation_id] = action
    return action.confirmation_id


def get_pending(confirmation_id: str) -> PendingAction | None:
    _cleanup_expired()
    return _pending_confirmations.get(confirmation_id)


def remove_pending(confirmation_id: str) -> None:
    _pending_confirmations.pop(confirmation_id, None)


def _cleanup_expired() -> None:
    now = time.time()
    expired = [
        k for k, v in _pending_confirmations.items()
        if now - v.created_at > CONFIRMATION_EXPIRY_SECONDS
    ]
    for k in expired:
        del _pending_confirmations[k]


# ── Claude Agent — Execution Engine ───────────────────────────────────────────


@dataclass
class AgentResponse:
    type: str  # "text", "tool_result", "pending_confirmation", "error"
    content: str
    tool_name: str | None = None
    tool_input: dict | None = None
    raw_data: dict | None = None
    confirmation_id: str | None = None


SYSTEM_PROMPT = """You are a trading assistant for the Universal Trade Platform (UTP).
You help execute trades and monitor positions for an options trader who primarily
sells credit spreads on SPX, NDX, and RUT.

Today's date: {today}
Default tickers: SPX, RUT, NDX
Default spread widths: SPX=20pt, RUT=20pt, NDX=50pt (unless user specifies otherwise)

Key behaviors:
- For read-only queries (portfolio, quotes, options, history), call tools immediately
  and present results clearly in a concise, terminal-like format.
- For trade execution, ALWAYS use the execute_trade tool. The system will show a
  confirmation card to the user — you do NOT need to ask "are you sure?".
- When asked to "find best" spreads: get the current quote first, then fetch the
  option chain for today's expiration near the money, and analyze for highest ROI.
  ROI = credit / (spread_width - credit). Present the top 3-5 options.
- When asked about breaching positions: get the portfolio, then get current quotes
  for each underlying, and compare current price to short strikes.
- For puts: short strike < current price = safe. For calls: short strike > current price = safe.
- Credit received = net_price * 100 * quantity. Max loss = (spread_width - net_price) * 100 * quantity.
- Use concise formatting. Use tables for multi-row data. Bold key numbers.
- When the user says "execute" or "do it" or "complete the transaction" after you've
  suggested a trade, call execute_trade with the parameters you suggested.
- The user trades in quantities of 25-45 contracts typically.
- When listing option chain data, focus on the bid price for selling (credit spreads)
  and organize by strike with bid/ask/volume columns.
"""


async def execute_read_tool(tool_name: str, tool_input: dict) -> dict:
    """Execute a read-only tool and return the result."""
    client = get_daemon_client()
    try:
        if tool_name == "get_portfolio":
            return await client.get_portfolio()
        elif tool_name == "get_quote":
            return await client.get_quote(tool_input["symbol"])
        elif tool_name == "get_options":
            return await client.get_options(
                symbol=tool_input["symbol"],
                option_type=tool_input.get("option_type"),
                strike_range_pct=tool_input.get("strike_range_pct"),
                expiration=tool_input.get("expiration"),
                strike_min=tool_input.get("strike_min"),
                strike_max=tool_input.get("strike_max"),
                list_expirations=tool_input.get("list_expirations", False),
            )
        elif tool_name == "get_trades":
            return await client.get_trades(
                days=tool_input.get("days", 0),
                include_all=tool_input.get("include_all", False),
            )
        elif tool_name == "get_orders":
            return await client.get_orders()
        elif tool_name == "get_performance":
            return await client.get_performance(days=tool_input.get("days", 30))
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    except httpx.HTTPStatusError as e:
        return {"error": f"Daemon returned {e.response.status_code}: {e.response.text[:200]}"}
    except httpx.ConnectError:
        return {"error": "Cannot connect to UTP daemon. Is it running?"}
    except Exception as e:
        return {"error": str(e)}


async def execute_write_tool(tool_name: str, tool_input: dict) -> dict:
    """Execute a confirmed write tool."""
    client = get_daemon_client()
    try:
        if tool_name == "execute_trade":
            # Check position limits before execution
            limits = await _check_position_limits()
            if not limits["allowed"]:
                return {"error": limits["reason"], "limits": limits}

            payload = build_trade_payload(tool_input)
            result = await client.execute_trade(payload)
            # Log trade with all metadata to CSV — only real fills
            # Skip: errors, dry-run, stub responses (order_id="123" from test provider)
            status = result.get("status", "")
            order_id = str(result.get("order_id", ""))
            is_real = (
                not result.get("error")
                and not result.get("dry_run")
                and status in ("FILLED", "SUBMITTED")
                and order_id not in ("", "123")  # Stub provider returns "123"
                and len(order_id) > 3  # Real IBKR order IDs are longer
            )
            if is_real:
                log_trade_to_csv(tool_input, result, source="manual")
                # Register profit target via daemon if specified
                pt_pct = tool_input.get("profit_target_pct")
                if pt_pct and pt_pct > 0 and result.get("order_id"):
                    credit = tool_input.get("net_price") or tool_input.get("credit") or 0
                    asyncio.create_task(_set_profit_target_on_daemon(
                        position_id=str(result["order_id"]),
                        entry_credit=credit,
                        profit_target_pct=pt_pct,
                        symbol=tool_input.get("symbol", ""),
                        short_strike=tool_input.get("short_strike", 0),
                        long_strike=tool_input.get("long_strike", 0),
                        quantity=tool_input.get("quantity", 1),
                    ))
            return result
        elif tool_name == "close_position":
            return await client.close_position(
                position_id=tool_input["position_id"],
                quantity=tool_input.get("quantity"),
                net_price=tool_input.get("net_price"),
            )
        elif tool_name == "cancel_order":
            return await client.cancel_order(tool_input["order_id"])
        elif tool_name == "reconcile_flush":
            return await client.flush_and_reconcile()
        else:
            return {"error": f"Unknown write tool: {tool_name}"}
    except httpx.HTTPStatusError as e:
        return {"error": f"Daemon returned {e.response.status_code}: {e.response.text[:500]}"}
    except httpx.ConnectError:
        return {"error": "Cannot connect to UTP daemon. Is it running?"}
    except Exception as e:
        return {"error": str(e)}


async def run_agent(
    user_message: str,
    conversation_history: list[dict],
) -> list[AgentResponse]:
    """Run the Claude agent loop. Returns a list of AgentResponse objects."""
    try:
        import anthropic
    except ImportError:
        return [AgentResponse(
            type="error",
            content="anthropic package not installed. Run: pip install anthropic",
        )]

    if not ANTHROPIC_API_KEY:
        return [AgentResponse(
            type="error",
            content="ANTHROPIC_API_KEY not set. Export it as an environment variable.",
        )]

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    system_prompt = SYSTEM_PROMPT.format(today=datetime.now().strftime("%Y-%m-%d"))

    # Build messages: history + new user message
    messages = list(conversation_history)
    messages.append({"role": "user", "content": user_message})

    responses: list[AgentResponse] = []
    max_iterations = 10  # Safety limit for tool-calling loops

    for _ in range(max_iterations):
        try:
            response = await client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                system=system_prompt,
                tools=TOOLS,
                messages=messages,
            )
        except anthropic.APIError as e:
            responses.append(AgentResponse(type="error", content=f"Claude API error: {e}"))
            break

        # Process content blocks
        assistant_content = response.content
        tool_use_blocks = []

        for block in assistant_content:
            if block.type == "text":
                if block.text.strip():
                    responses.append(AgentResponse(type="text", content=block.text))
            elif block.type == "tool_use":
                tool_use_blocks.append(block)

        # If no tool calls, we're done
        if response.stop_reason != "tool_use" or not tool_use_blocks:
            break

        # Process tool calls
        # Add assistant message to conversation
        messages.append({"role": "assistant", "content": [b.model_dump() for b in assistant_content]})

        tool_results = []
        stop_after_this = False

        for tool_block in tool_use_blocks:
            tool_name = tool_block.name
            tool_input = tool_block.input

            if tool_name in WRITE_TOOLS:
                # Create pending confirmation
                description = describe_write_action(tool_name, tool_input)
                action = PendingAction(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    description=description,
                )
                store_pending(action)

                responses.append(AgentResponse(
                    type="pending_confirmation",
                    content=description,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    confirmation_id=action.confirmation_id,
                ))

                # Tell Claude the action is pending
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": json.dumps({
                        "status": "pending_user_confirmation",
                        "description": description,
                        "message": "The trade details have been shown to the user for confirmation. They will tap Execute to proceed or Cancel to abort.",
                    }),
                })
                stop_after_this = True
            else:
                # Execute read-only tool
                result = await execute_read_tool(tool_name, tool_input)
                responses.append(AgentResponse(
                    type="tool_result",
                    content="",  # Claude will format this
                    tool_name=tool_name,
                    tool_input=tool_input,
                    raw_data=result,
                ))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": json.dumps(result, default=str),
                })

        # Add tool results to conversation
        messages.append({"role": "user", "content": tool_results})

        if stop_after_this:
            break

    return responses


# ── FastAPI Application ───────────────────────────────────────────────────────

app = FastAPI(title="UTP Voice", docs_url=None, redoc_url=None)


@app.on_event("startup")
async def _resume_background_tasks_on_startup():
    """Resume auto-trade on startup. Profit monitoring is handled by the daemon."""
    global _auto_trade_task

    # Resume auto-trade if active
    try:
        state = _load_auto_trade_state()
        if state.get("active") and state.get("trading_day") == datetime.now().strftime("%Y-%m-%d"):
            logger.info("Resuming auto-trade from state file (trading_day=%s)", state["trading_day"])
            _auto_trade_task = asyncio.create_task(_auto_trade_loop(state))
    except Exception as e:
        logger.warning("Failed to resume auto-trade: %s", e)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions and return 500 instead of crashing."""
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
    )


def _get_base_path(request: Request) -> str:
    """Get the base path from X-Forwarded-Prefix header (set by envoy)."""
    return (request.headers.get("x-forwarded-prefix") or "").rstrip("/")


def _serve_template(request: Request) -> HTMLResponse:
    """Serve the HTML template with base path injected."""
    template_path = Path(__file__).parent / "templates" / "utp_voice.html"
    if not template_path.exists():
        return HTMLResponse("<h1>Template not found</h1>", status_code=500)
    html = template_path.read_text()
    base_path = _get_base_path(request)
    if base_path:
        inject = f'<script>window.__BASE_PATH="{base_path}";</script>'
        html = html.replace("<head>", f"<head>\n{inject}", 1)
    return HTMLResponse(html)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, utp_voice_session: str | None = Cookie(default=None)):
    """Serve the SPA. In default mode, requires login. With --public, serves anonymously."""
    if not PUBLIC_MODE:
        token = utp_voice_session
        if not token or not decode_token(token):
            return RedirectResponse(url="/login", status_code=302)
    return _serve_template(request)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return _serve_template(request)


class LoginRequest(json.__class__ if False else object):
    pass


@app.post("/api/login")
async def api_login(request: Request):
    body = await request.json()
    username = body.get("username", "")
    password = body.get("password", "")

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")

    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT secret not configured")

    if not verify_user(username, password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(username)
    base_path = _get_base_path(request)
    response = JSONResponse({"status": "ok", "username": username})
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=JWT_EXPIRE_MINUTES * 60,
        secure=bool(base_path),  # True when behind reverse proxy (HTTPS)
        path=f"{base_path}/" if base_path else "/",
    )
    return response


@app.post("/api/logout")
async def api_logout(request: Request):
    base_path = _get_base_path(request)
    response = JSONResponse({"status": "ok"})
    response.delete_cookie(COOKIE_NAME, path=f"{base_path}/" if base_path else "/")
    return response


@app.post("/api/chat")
async def api_chat(
    request: Request,
    username: str = Depends(require_session),
):
    body = await request.json()
    message = body.get("message", "").strip()
    history = body.get("history", [])

    if not message:
        raise HTTPException(status_code=400, detail="Message required")

    responses = await run_agent(message, history)

    # Serialize responses
    serialized = []
    for r in responses:
        d = {
            "type": r.type,
            "content": r.content,
        }
        if r.tool_name:
            d["tool_name"] = r.tool_name
        if r.tool_input:
            d["tool_input"] = r.tool_input
        if r.raw_data:
            d["raw_data"] = r.raw_data
        if r.confirmation_id:
            d["confirmation_id"] = r.confirmation_id
        serialized.append(d)

    # Build assistant content for conversation history (text blocks only)
    assistant_text_parts = [r.content for r in responses if r.type == "text" and r.content]
    assistant_content = "\n".join(assistant_text_parts) if assistant_text_parts else "(processed)"

    return JSONResponse({
        "responses": serialized,
        "assistant_content": assistant_content,
    })


@app.post("/api/confirm/{confirmation_id}")
async def api_confirm(
    confirmation_id: str,
    username: str = Depends(require_session),
):
    action = get_pending(confirmation_id)
    if not action:
        raise HTTPException(status_code=404, detail="Confirmation expired or not found")

    logger.info(
        "User %s confirmed %s: %s",
        username, action.tool_name, action.description,
    )

    result = await execute_write_tool(action.tool_name, action.tool_input)
    remove_pending(confirmation_id)

    return JSONResponse({
        "status": "executed",
        "tool_name": action.tool_name,
        "description": action.description,
        "result": result,
    })


@app.post("/api/execute-raw")
async def api_execute_raw(
    request: Request,
    username: str = Depends(require_session),
):
    """Raw mode: forward an arbitrary payload directly to the UTP daemon."""
    body = await request.json()
    endpoint = body.get("endpoint", "")
    method = body.get("method", "GET").upper()
    payload = body.get("payload")

    if not endpoint:
        raise HTTPException(status_code=400, detail="endpoint required")

    client = get_daemon_client()
    try:
        if method == "GET":
            result = await client._get(endpoint, params=payload)
        elif method == "POST":
            result = await client._post(endpoint, json_data=payload)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported method: {method}")
        return JSONResponse({"status": "ok", "result": result})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=502)


@app.get("/api/auth-check")
async def api_auth_check(username: str = Depends(require_session)):
    """Lightweight auth check — just validates the session cookie."""
    return {"authenticated": True, "username": username}


@app.get("/api/health")
async def api_health():
    client = get_daemon_client()
    try:
        daemon_health = await client.health()
        return {"status": "ok", "daemon": daemon_health}
    except Exception as e:
        return {"status": "degraded", "daemon": {"error": str(e)}}


# ── Pre-Built View API Endpoints ──────────────────────────────────────────────

DEFAULT_TICKERS = ["SPX", "NDX", "RUT"]

# ── Server-Side Options Cache & Background Pre-Fetch ─────────────────────────

# CSV exports directory for instant option data fallback
CSV_EXPORTS_DIR = os.environ.get(
    "CSV_EXPORTS_DIR",
    str(Path(__file__).parent.parent.parent / "csv_exports" / "options"),
)

# ── Trade CSV Logger ──────────────────────────────────────────────────────────

TRADES_CSV_PATH = Path("data/utp_voice/trades.csv")

TRADES_CSV_FIELDS = [
    "timestamp", "symbol", "trade_type", "option_type", "short_strike", "long_strike",
    "width", "quantity", "expiration", "dte", "credit_per_share", "credit_per_contract",
    "total_credit", "max_loss_per_contract", "total_max_loss", "roi_pct", "otm_pct",
    "current_price", "hist_percentile", "pred_percentile", "short_delta", "short_theta",
    "short_iv", "order_id", "fill_price", "slippage", "source", "status",
]


def log_trade_to_csv(tool_input: dict, result: dict, source: str = "manual") -> None:
    """Append a trade record to the trades CSV with full metadata."""
    import csv as csv_mod

    try:
        TRADES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_header = not TRADES_CSV_PATH.exists()

        ti = tool_input
        width = abs((ti.get("short_strike", 0) or 0) - (ti.get("long_strike", 0) or 0))
        qty = ti.get("quantity", 1)
        credit = ti.get("net_price") or ti.get("credit") or 0
        credit_pc = credit * 100 if credit else 0
        max_loss_pc = width * 100 - credit_pc if width else 0

        row = {
            "timestamp": _now_iso(),
            "symbol": ti.get("symbol", ""),
            "trade_type": ti.get("trade_type", ""),
            "option_type": ti.get("option_type", ""),
            "short_strike": ti.get("short_strike", ""),
            "long_strike": ti.get("long_strike", ""),
            "width": width,
            "quantity": qty,
            "expiration": ti.get("expiration", ""),
            "dte": ti.get("dte", ""),
            "credit_per_share": credit,
            "credit_per_contract": credit_pc,
            "total_credit": credit_pc * qty,
            "max_loss_per_contract": max_loss_pc,
            "total_max_loss": max_loss_pc * qty,
            "roi_pct": round(credit_pc / max_loss_pc * 100, 1) if max_loss_pc > 0 else 0,
            "otm_pct": ti.get("otm_pct", ""),
            "current_price": ti.get("current_price", ""),
            "hist_percentile": ti.get("hist_percentile", ""),
            "pred_percentile": ti.get("pred_percentile", ""),
            "short_delta": ti.get("short_delta", ""),
            "short_theta": ti.get("short_theta", ""),
            "short_iv": ti.get("short_iv", ""),
            "order_id": result.get("order_id", ""),
            "fill_price": result.get("filled_price", ""),
            "slippage": "",
            "source": source,
            "status": result.get("status", ""),
        }

        # Compute slippage if we have both estimated and filled
        if credit and result.get("filled_price"):
            row["slippage"] = round(credit - result["filled_price"], 4)

        with open(TRADES_CSV_PATH, "a", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=TRADES_CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        logger.info("Trade logged to CSV: %s %s %s/%s x%s",
                     ti.get("symbol"), ti.get("option_type"),
                     ti.get("short_strike"), ti.get("long_strike"), qty)
    except Exception as e:
        logger.warning("Failed to log trade to CSV: %s", e)


@dataclass
class CachedOptions:
    data: list[dict]
    cached_at: float           # time.time() when we stored it
    fetched_at_utc: str        # ISO timestamp of when the source data was captured
    symbol: str
    source: str = "unknown"    # "csv_exports", "ibkr", "streaming_cache"


_options_cache: dict[str, CachedOptions] = {}  # key: "{symbol}_{expiration}_{type}"
_expirations_cache: dict[str, tuple[list[str], float, str]] = {}  # symbol -> (exps, mono_ts, utc_iso)
_prefetch_in_progress: set[str] = set()

OPTIONS_CACHE_TTL_MARKET = 120  # 2 minutes during market hours


def _is_market_hours() -> bool:
    """Check if US equity markets are open (Mon-Fri 9:15a-4:15p ET).

    Uses 15-min buffer on both sides per user request:
    - Opens at 9:30 ET → we consider 'open' from 9:15 ET
    - Closes at 16:00 ET → we consider 'closed' from 16:15 ET
    """
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        now_et = datetime.now(UTC) - timedelta(hours=4)

    if now_et.weekday() >= 5:  # Sat/Sun
        return False

    hour, minute = now_et.hour, now_et.minute
    minutes_since_midnight = hour * 60 + minute
    return 555 <= minutes_since_midnight <= 975


def _cache_ttl() -> float:
    """During market hours: 0 (no voice-server cache, always go to daemon).
    Outside market hours: infinite (serve cached CSV/IBKR data forever)."""
    if _is_market_hours():
        return 0
    return float("inf")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _get_cached_options(symbol: str, expiration: str, option_type: str) -> CachedOptions | None:
    """Return the CachedOptions entry if valid, else None."""
    key = f"{symbol}_{expiration}_{option_type}"
    entry = _options_cache.get(key)
    if entry and (time.time() - entry.cached_at) < _cache_ttl():
        return entry
    return None


def _put_cached_options(
    symbol: str, expiration: str, option_type: str,
    data: list[dict], source: str = "unknown", fetched_at_utc: str | None = None,
) -> None:
    key = f"{symbol}_{expiration}_{option_type}"
    # Don't overwrite IBKR data with CSV data
    existing = _options_cache.get(key)
    if existing and existing.source == "ibkr" and source == "csv_exports":
        return
    _options_cache[key] = CachedOptions(
        data=data, cached_at=time.time(),
        fetched_at_utc=fetched_at_utc or _now_iso(),
        symbol=symbol, source=source,
    )


def _get_cached_expirations(symbol: str) -> list[str] | None:
    entry = _expirations_cache.get(symbol)
    if entry:
        exps, ts, _ = entry
        if (time.time() - ts) < _cache_ttl():
            return exps
    return None


def _put_cached_expirations(symbol: str, exps: list[str]) -> None:
    _expirations_cache[symbol] = (exps, time.time(), _now_iso())


# ── CSV Exports Reader ────────────────────────────────────────────────────────


def _get_csv_expirations(symbol: str) -> list[str]:
    """Get available expiration dates from CSV exports directory.

    Lists {CSV_EXPORTS_DIR}/{symbol}/*.csv, parses filenames as dates,
    returns sorted dates >= today.
    """
    csv_dir = Path(CSV_EXPORTS_DIR) / symbol.upper()
    if not csv_dir.is_dir():
        return []

    today = datetime.now().strftime("%Y-%m-%d")
    expirations = []
    for f in csv_dir.iterdir():
        if f.suffix == ".csv" and len(f.stem) == 10:  # YYYY-MM-DD
            try:
                # Validate date format
                datetime.strptime(f.stem, "%Y-%m-%d")
                if f.stem >= today:
                    expirations.append(f.stem)
            except ValueError:
                continue
    return sorted(expirations)


def _load_options_from_csv(
    symbol: str, expiration: str,
    strike_min: float | None = None, strike_max: float | None = None,
) -> tuple[list[dict], str]:
    """Load option quotes from CSV exports for a given symbol and expiration.

    Reads the latest snapshot (max timestamp) from the CSV file.
    Returns (quotes_list, snapshot_timestamp_iso).
    """
    import csv as csv_mod

    csv_path = Path(CSV_EXPORTS_DIR) / symbol.upper() / f"{expiration}.csv"
    if not csv_path.exists():
        return [], ""

    # Read efficiently: first pass to find max timestamp, second pass to extract rows
    # For large files (30-100MB), read only what we need
    max_ts = ""
    quotes_by_type: dict[str, list[dict]] = {"put": [], "call": []}

    with open(csv_path, "r") as f:
        reader = csv_mod.DictReader(f)
        rows = []
        for row in reader:
            ts = row.get("timestamp", "")
            strike = float(row.get("strike", 0) or 0)

            # Filter by strike range early to reduce memory
            if strike_min and strike < strike_min:
                continue
            if strike_max and strike > strike_max:
                continue

            if ts > max_ts:
                max_ts = ts
            rows.append(row)

    if not max_ts or not rows:
        return [], ""

    # Filter to latest snapshot only
    result = []
    for row in rows:
        if row.get("timestamp") != max_ts:
            continue

        strike = float(row.get("strike", 0) or 0)
        bid = float(row.get("bid", 0) or 0)
        ask = float(row.get("ask", 0) or 0)
        opt_type = row.get("type", "").lower()

        quote = {
            "symbol": row.get("ticker", ""),
            "strike": strike,
            "bid": bid,
            "ask": ask,
            "last": float(row.get("day_close", 0) or 0),
            "volume": int(float(row.get("volume", 0) or 0)),
            "open_interest": 0,
            "greeks": {},
        }

        # Add greeks if available
        greeks = {}
        for gk in ("delta", "gamma", "theta", "vega", "implied_volatility"):
            val = row.get(gk, "")
            if val:
                try:
                    key = "iv" if gk == "implied_volatility" else gk
                    greeks[key] = float(val)
                except (ValueError, TypeError):
                    pass
        if greeks:
            quote["greeks"] = greeks

        quote["_option_type"] = opt_type  # Temporary, for splitting
        result.append(quote)

    return result, max_ts


def _split_csv_quotes_by_type(quotes: list[dict]) -> dict[str, list[dict]]:
    """Split CSV quotes into separate put/call lists and remove temp field."""
    by_type: dict[str, list[dict]] = {"put": [], "call": []}
    for q in quotes:
        opt_type = q.pop("_option_type", "")
        if opt_type in by_type:
            by_type[opt_type].append(q)
    return by_type


def _normalize_date(d: str) -> str:
    """Normalize YYYYMMDD to YYYY-MM-DD. Pass through if already hyphenated."""
    if len(d) == 8 and d.isdigit():
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return d


def _merge_expirations(*expiration_lists: list[str]) -> list[str]:
    """Merge multiple expiration lists, normalize dates, deduplicate, sort.

    Filters out past dates — only returns expirations >= today.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    combined = set()
    for exp_list in expiration_lists:
        for d in exp_list:
            nd = _normalize_date(d)
            if nd >= today:
                combined.add(nd)
    return sorted(combined)


# ── Server-Side Spread Computation ────────────────────────────────────────────

DEFAULT_SPREAD_WIDTHS = {"SPX": 20, "NDX": 50, "RUT": 20}


def compute_spreads_server(
    chain: dict, symbol: str, current_price: float, width: int, filters: dict | None = None,
) -> list[dict]:
    """Server-side equivalent of the JS computeSpreads() + filters.

    chain: {"put": [{strike, bid, ask, greeks:{delta,theta,iv}, ...}], "call": [...]}
    filters: {min_roi, min_otm, min_pctl, min_pred, min_credit, max_delta, option_type, ...}
    Returns sorted by ROI descending.
    """
    filters = filters or {}
    min_roi = filters.get("min_roi_pct", 0)
    min_otm = filters.get("min_otm_pct", 0)
    max_delta = filters.get("max_delta")
    opt_type_filter = filters.get("option_type", "ALL")
    min_credit = filters.get("min_credit", 0)

    spreads = []
    for opt_type in ["PUT", "CALL"]:
        if opt_type_filter != "ALL" and opt_type != opt_type_filter:
            continue
        quotes = chain.get(opt_type.lower(), [])
        by_strike = {q["strike"]: q for q in quotes if q.get("strike")}
        for short_strike in sorted(by_strike.keys()):
            # OTM only
            if opt_type == "PUT" and short_strike >= current_price:
                continue
            if opt_type == "CALL" and short_strike <= current_price:
                continue

            long_strike = (short_strike - width) if opt_type == "PUT" else (short_strike + width)
            sq = by_strike.get(short_strike)
            lq = by_strike.get(long_strike)
            if not sq or not lq:
                continue

            short_bid = sq.get("bid", 0) or 0
            long_ask = lq.get("ask", 0) or 0
            if short_bid <= 0 or long_ask <= 0:
                continue

            credit = round(short_bid - long_ask, 2)
            if credit <= 0:
                continue
            if credit < min_credit:
                continue

            credit_pc = credit * 100
            max_loss_pc = width * 100 - credit_pc
            if max_loss_pc <= 0:
                continue

            roi = round(credit_pc / max_loss_pc * 100, 1)
            if roi < min_roi:
                continue

            otm = round(
                ((current_price - short_strike) / current_price * 100) if opt_type == "PUT"
                else ((short_strike - current_price) / current_price * 100),
                2,
            )
            if abs(otm) < min_otm:
                continue

            sg = sq.get("greeks") or {}
            delta = sg.get("delta")
            if max_delta is not None and delta is not None and abs(delta) > max_delta:
                continue

            spreads.append({
                "option_type": opt_type,
                "short_strike": short_strike,
                "long_strike": long_strike,
                "width": width,
                "credit": credit,
                "credit_per_contract": credit_pc,
                "max_loss": round(max_loss_pc),
                "roi_pct": roi,
                "otm_pct": otm,
                "short_bid": short_bid,
                "long_ask": long_ask,
                "short_delta": delta,
                "short_theta": sg.get("theta"),
                "short_iv": sg.get("iv"),
            })

    spreads.sort(key=lambda s: s["roi_pct"], reverse=True)
    return spreads


# ── Auto-Trade System ─────────────────────────────────────────────────────────

AUTO_TRADE_STATE_PATH = Path("data/utp_voice/auto_trade_state.json")


AUTO_TRADE_HISTORY_PATH = Path("data/utp_voice/auto_trade_history.jsonl")


def _load_auto_trade_state() -> dict:
    if AUTO_TRADE_STATE_PATH.exists():
        with open(AUTO_TRADE_STATE_PATH) as f:
            return json.load(f)
    return {"active": False}


def _archive_auto_trade_session(state: dict) -> None:
    """Append a completed session to the history JSONL file."""
    if not state.get("trading_day"):
        return
    try:
        AUTO_TRADE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Compute session summary
        executed = state.get("executed_today", [])
        total_credit = sum(e.get("credit", 0) * 100 * state.get("filters", {}).get("quantity", 25)
                          for e in executed)
        summary = {
            "trading_day": state.get("trading_day"),
            "created_at": state.get("created_at"),
            "stopped_at": _now_iso(),
            "config": state.get("config", {}),
            "filters": state.get("filters", {}),
            "trades_executed": len(executed),
            "executed": executed,
            "total_estimated_credit": round(total_credit, 2),
            "log_entries": len(state.get("log", [])),
        }
        with open(AUTO_TRADE_HISTORY_PATH, "a") as f:
            f.write(json.dumps(summary, default=str) + "\n")
        logger.info("Auto-trade session archived for %s (%d trades)", state.get("trading_day"), len(executed))
    except Exception as e:
        logger.warning("Failed to archive auto-trade session: %s", e)


def _load_auto_trade_history(limit: int = 30) -> list[dict]:
    """Load recent auto-trade sessions from history JSONL, newest first."""
    if not AUTO_TRADE_HISTORY_PATH.exists():
        return []
    sessions = []
    with open(AUTO_TRADE_HISTORY_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    sessions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    sessions.sort(key=lambda s: s.get("trading_day", ""), reverse=True)
    return sessions[:limit]


def _save_auto_trade_state(state: dict) -> None:
    AUTO_TRADE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUTO_TRADE_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


_auto_trade_task: asyncio.Task | None = None


async def _auto_trade_loop(state: dict) -> None:
    """Background loop: evaluate and execute top N spreads every interval."""
    config = state.get("config", {})
    filters = state.get("filters", {})
    interval = config.get("interval_minutes", 3) * 60
    top_n = config.get("top_n", 3)
    end_time_utc = config.get("end_time_utc", "16:00")
    qty = filters.get("quantity", 25)

    client = get_daemon_client()

    while True:
        # Check if still active
        current = _load_auto_trade_state()
        if not current.get("active"):
            break

        # Check trading day
        today = datetime.now().strftime("%Y-%m-%d")
        if current.get("trading_day") != today:
            _archive_auto_trade_session(current)
            current["active"] = False
            _save_auto_trade_state(current)
            logger.info("Auto-trade: trading day changed, stopping")
            break

        # Check end time
        try:
            from zoneinfo import ZoneInfo
            now_et = datetime.now(ZoneInfo("America/New_York"))
        except Exception:
            now_et = datetime.now(UTC) - timedelta(hours=4)

        end_parts = end_time_utc.split(":")
        end_minutes = int(end_parts[0]) * 60 + int(end_parts[1])
        now_minutes = now_et.hour * 60 + now_et.minute
        if now_minutes >= end_minutes:
            _archive_auto_trade_session(current)
            current["active"] = False
            _save_auto_trade_state(current)
            logger.info("Auto-trade: end time reached (%s ET), stopping", end_time_utc)
            break

        # Check market hours (9:30 AM - 3:50 PM ET)
        if now_minutes < 570 or now_minutes > 950:  # Before 9:30 or after 3:50
            await asyncio.sleep(30)
            continue

        # Evaluate spreads for each ticker
        tickers = filters.get("tickers", DEFAULT_TICKERS)
        executed = current.get("executed_today", [])
        log = current.get("log", [])

        all_candidates = []
        for sym in tickers:
            width = filters.get("width", {}).get(sym, DEFAULT_SPREAD_WIDTHS.get(sym, 20))
            if isinstance(filters.get("width"), (int, float)):
                width = int(filters["width"]) or DEFAULT_SPREAD_WIDTHS.get(sym, 20)

            # Get option chain from cache or fetch
            exps = _get_cached_expirations(sym)
            if not exps:
                continue
            dte = filters.get("dte", 0)
            exp = exps[min(dte, len(exps) - 1)]

            entry = _get_cached_options(sym, exp, "PUT")
            call_entry = _get_cached_options(sym, exp, "CALL")

            chain = {}
            if entry:
                chain["put"] = entry.data
            if call_entry:
                chain["call"] = call_entry.data

            if not chain:
                # Try fetching
                try:
                    data = await client.get_options(sym, expiration=exp, option_type="BOTH")
                    chain = {
                        "put": data.get("quotes", {}).get("put", []),
                        "call": data.get("quotes", {}).get("call", []),
                    }
                except Exception:
                    continue

            # Get current price
            try:
                quote = await client.get_quote(sym)
                cp = quote.get("last") or quote.get("bid") or 0
            except Exception:
                continue

            if not cp:
                continue

            spreads = compute_spreads_server(chain, sym, cp, width, filters)

            # Dedup config: normal cooldown and fallback (no-other-option) cooldown
            dedup_minutes = config.get("dedup_minutes", 10)
            dedup_fallback_minutes = config.get("dedup_fallback_minutes", 15)
            now_ts = time.time()

            for s in spreads:
                s["symbol"] = sym
                s["expiration"] = exp
                s["current_price"] = cp
                key = f"{sym}_{s['short_strike']}_{s['option_type']}_{exp}"
                # Check if this spread was executed recently
                last_exec = None
                for e in reversed(executed):
                    if e.get("key") == key:
                        last_exec = e
                        break
                if last_exec:
                    exec_ts = last_exec.get("epoch", 0)
                    age_min = (now_ts - exec_ts) / 60 if exec_ts else float("inf")
                    if age_min < dedup_minutes:
                        s["_dedup_blocked"] = True  # Blocked by normal cooldown
                    else:
                        s["_dedup_blocked"] = False  # Past normal cooldown
                else:
                    s["_dedup_blocked"] = False
                all_candidates.append(s)

        # Sort all candidates by ROI and pick top N
        all_candidates.sort(key=lambda s: s["roi_pct"], reverse=True)

        # First pass: pick from non-dedup-blocked candidates
        non_blocked = [s for s in all_candidates if not s.get("_dedup_blocked")]
        selected = non_blocked[:top_n]

        # Second pass: if not enough, allow dedup-blocked ones past fallback cooldown
        if len(selected) < top_n:
            now_ts = time.time()
            dedup_fallback_minutes = config.get("dedup_fallback_minutes", 15)
            for s in all_candidates:
                if len(selected) >= top_n:
                    break
                if s in selected:
                    continue
                if s.get("_dedup_blocked"):
                    key = f"{s['symbol']}_{s['short_strike']}_{s['option_type']}_{s['expiration']}"
                    last_exec = None
                    for e in reversed(executed):
                        if e.get("key") == key:
                            last_exec = e
                            break
                    if last_exec:
                        exec_ts = last_exec.get("epoch", 0)
                        age_min = (now_ts - exec_ts) / 60 if exec_ts else float("inf")
                        if age_min >= dedup_fallback_minutes:
                            selected.append(s)  # Allowed by fallback cooldown

        log_entry = {
            "time": _now_iso(),
            "action": "evaluate",
            "candidates": len(all_candidates),
            "selected": len(selected),
        }
        log.append(log_entry)

        # Execute selected trades
        for s in selected:
            profit_target_pct = config.get("profit_target_pct", 50)
            trade_input = {
                "trade_type": "credit-spread",
                "symbol": s["symbol"],
                "option_type": s["option_type"],
                "short_strike": s["short_strike"],
                "long_strike": s["long_strike"],
                "quantity": qty,
                "expiration": s["expiration"],
                "profit_target_pct": profit_target_pct,
                "otm_pct": s["otm_pct"],
                "current_price": s["current_price"],
                "short_delta": s.get("short_delta"),
                "short_theta": s.get("short_theta"),
                "short_iv": s.get("short_iv"),
            }

            try:
                # Check position limits
                limits = await _check_position_limits()
                if not limits["allowed"]:
                    log.append({"time": _now_iso(), "action": "skip", "sym": s["symbol"],
                                "short": s["short_strike"], "reason": limits["reason"]})
                    break  # Stop executing more trades this cycle

                payload = build_trade_payload(trade_input)
                result = await client.execute_trade(payload)
                status = result.get("status", "UNKNOWN")
                key = f"{s['symbol']}_{s['short_strike']}_{s['option_type']}_{s['expiration']}"

                executed.append({
                    "key": key,
                    "epoch": time.time(),
                    "sym": s["symbol"],
                    "short": s["short_strike"],
                    "long": s["long_strike"],
                    "type": s["option_type"],
                    "exp": s["expiration"],
                    "time": _now_iso(),
                    "credit": s["credit"],
                    "order_id": result.get("order_id", ""),
                    "status": status,
                })

                log.append({
                    "time": _now_iso(),
                    "action": "execute",
                    "sym": s["symbol"],
                    "short": s["short_strike"],
                    "type": s["option_type"],
                    "result": status,
                })

                # Log to CSV
                if not result.get("error"):
                    log_trade_to_csv(trade_input, result, source="auto")

                logger.info("Auto-trade executed: %s %s %s/%s → %s",
                            s["symbol"], s["option_type"], s["short_strike"], s["long_strike"], status)

            except Exception as e:
                log.append({
                    "time": _now_iso(),
                    "action": "error",
                    "sym": s["symbol"],
                    "short": s["short_strike"],
                    "error": str(e),
                })
                logger.warning("Auto-trade execution failed: %s", e)

        # Save state
        current["executed_today"] = executed
        current["log"] = log[-100:]  # Keep last 100 log entries
        _save_auto_trade_state(current)

        # Wait for next interval
        await asyncio.sleep(interval)


# ── Profit Targets (proxied to daemon) ────────────────────────────────────────
# Profit target monitoring runs in the UTP daemon, not here.
# Voice app just proxies CRUD operations to the daemon's endpoints.

MAX_OPEN_POSITIONS = int(os.environ.get("MAX_OPEN_POSITIONS", "10"))
MAX_DAILY_TRADES = int(os.environ.get("MAX_DAILY_TRADES", "20"))


async def _set_profit_target_on_daemon(
    position_id: str, entry_credit: float, profit_target_pct: float,
    symbol: str = "", short_strike: float = 0, long_strike: float = 0,
    quantity: int = 1,
) -> None:
    """Register a profit target via the daemon's endpoint."""
    if profit_target_pct <= 0:
        return
    client = get_daemon_client()
    try:
        await client._post("/account/profit-targets", json_data={
            "position_id": position_id,
            "entry_credit": entry_credit,
            "profit_target_pct": profit_target_pct,
            "symbol": symbol,
            "short_strike": short_strike,
            "long_strike": long_strike,
            "quantity": quantity,
        })
    except Exception as e:
        logger.warning("Failed to set profit target on daemon: %s", e)


async def _check_position_limits() -> dict:
    """Check if a new trade is allowed within position limits.
    Returns {allowed: bool, reason: str, open_count: int, daily_count: int}.
    """
    import csv as csv_mod

    # Count open positions from daemon
    open_count = 0
    try:
        client = get_daemon_client()
        portfolio = await client.get_portfolio(include_quotes=False)
        positions = portfolio.get("positions", [])
        open_count = sum(1 for p in positions if abs(p.get("quantity", 0)) > 0)
    except Exception:
        pass

    # Count today's trades from CSV
    daily_count = 0
    today = datetime.now().strftime("%Y-%m-%d")
    if TRADES_CSV_PATH.exists():
        try:
            with open(TRADES_CSV_PATH) as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    if (row.get("timestamp", "") or "")[:10] == today:
                        daily_count += 1
        except Exception:
            pass

    allowed = True
    reasons = []
    if MAX_OPEN_POSITIONS > 0 and open_count >= MAX_OPEN_POSITIONS:
        allowed = False
        reasons.append(f"Max open positions reached ({open_count}/{MAX_OPEN_POSITIONS})")
    if MAX_DAILY_TRADES > 0 and daily_count >= MAX_DAILY_TRADES:
        allowed = False
        reasons.append(f"Max daily trades reached ({daily_count}/{MAX_DAILY_TRADES})")

    return {
        "allowed": allowed,
        "reason": " | ".join(reasons) if reasons else "",
        "open_count": open_count,
        "max_open": MAX_OPEN_POSITIONS,
        "daily_count": daily_count,
        "max_daily": MAX_DAILY_TRADES,
    }


# ── Pre-Fetch Logic ──────────────────────────────────────────────────────────


def _prefetch_csv_for_symbol(symbol: str, price: float) -> list[str]:
    """Load CSV data instantly (synchronous). Returns list of CSV expirations loaded."""
    csv_exps = _get_csv_expirations(symbol)
    if not csv_exps or not price:
        return csv_exps

    strike_min = round(price * 0.93, 0)
    strike_max = round(price * 1.07, 0)

    for exp in csv_exps[:7]:
        for opt_type in ["PUT", "CALL"]:
            existing = _get_cached_options(symbol, exp, opt_type)
            if existing:
                continue  # Already have data (CSV or IBKR)

            all_quotes, snapshot_ts = _load_options_from_csv(
                symbol, exp, strike_min, strike_max,
            )
            if all_quotes:
                by_type = _split_csv_quotes_by_type(all_quotes)
                quotes_for_type = by_type.get(opt_type.lower(), [])
                if quotes_for_type:
                    _put_cached_options(
                        symbol, exp, opt_type, quotes_for_type,
                        source="csv_exports", fetched_at_utc=snapshot_ts,
                    )
                    logger.info(
                        "CSV loaded %s %s %s: %d quotes (snapshot %s)",
                        symbol, exp, opt_type, len(quotes_for_type), snapshot_ts[:19],
                    )
    return csv_exps


async def _prefetch_ibkr_for_symbol(symbol: str, price: float, ibkr_exps: list[str]) -> None:
    """Fetch IBKR data to upgrade CSV entries. Runs as a background task."""
    if symbol in _prefetch_in_progress:
        return
    _prefetch_in_progress.add(symbol)

    try:
        if not ibkr_exps or not price:
            return

        strike_min = round(price * 0.93, 0)
        strike_max = round(price * 1.07, 0)
        client = get_daemon_client()

        for exp in ibkr_exps[:7]:
            for opt_type in ["PUT", "CALL"]:
                existing = _get_cached_options(symbol, exp, opt_type)
                if existing and existing.source == "ibkr":
                    continue
                try:
                    data = await client.get_options(
                        symbol, option_type=opt_type, expiration=exp,
                        strike_min=strike_min, strike_max=strike_max,
                    )
                    quotes_list = data.get("quotes", {}).get(opt_type.lower(), [])
                    if quotes_list:
                        _put_cached_options(
                            symbol, exp, opt_type, quotes_list,
                            source="ibkr", fetched_at_utc=_now_iso(),
                        )
                        logger.info("IBKR loaded %s %s %s: %d quotes", symbol, exp, opt_type, len(quotes_list))
                except Exception as e:
                    logger.warning("IBKR prefetch failed %s %s %s: %s", symbol, exp, opt_type, e)
    finally:
        _prefetch_in_progress.discard(symbol)


async def _prefetch_options_for_symbol(symbol: str) -> None:
    """Two-phase prefetch: CSV instantly, IBKR in background."""
    client = get_daemon_client()

    # Get price
    try:
        quote = await client.get_quote(symbol)
        price = quote.get("last") or quote.get("bid") or 0
    except Exception:
        price = 0

    # Phase 1: CSV (instant, synchronous)
    csv_exps = _prefetch_csv_for_symbol(symbol, price)

    # Get IBKR expirations
    ibkr_exps: list[str] = []
    try:
        exp_data = await client.get_options(symbol, list_expirations=True)
        ibkr_exps = exp_data.get("expirations", [])
    except Exception as e:
        logger.warning("Prefetch IBKR expirations failed for %s: %s", symbol, e)

    # Merge and cache expirations
    merged_exps = _merge_expirations(csv_exps, ibkr_exps)
    if merged_exps:
        _put_cached_expirations(symbol, merged_exps)

    # Phase 2: IBKR (background task — does not block)
    if ibkr_exps and price:
        asyncio.create_task(_prefetch_ibkr_for_symbol(symbol, price, ibkr_exps))


async def prefetch_all_tickers() -> None:
    """Pre-fetch options data for all default tickers."""
    for symbol in DEFAULT_TICKERS:
        asyncio.create_task(_prefetch_options_for_symbol(symbol))






@app.get("/api/prefetch")
async def api_prefetch(_u: str | None = Depends(optional_session)):
    """Trigger background pre-fetch of options data for default tickers."""
    await prefetch_all_tickers()

    # Return cache status per ticker
    status = {}
    for sym in DEFAULT_TICKERS:
        exps = _get_cached_expirations(sym)
        cached_exps = 0
        if exps:
            for exp in exps[:5]:
                for ot in ["PUT", "CALL"]:
                    if _get_cached_options(sym, exp, ot):
                        cached_exps += 1
        status[sym] = {
            "expirations": len(exps) if exps else 0,
            "cached_chains": cached_exps,
            "prefetching": sym in _prefetch_in_progress,
        }
    return {"status": "prefetch_started", "tickers": status}


@app.get("/api/portfolio")
async def api_portfolio(username: str = Depends(require_session)):
    """Get portfolio from daemon. Daemon now computes spread metrics, breach
    status, and fetches quotes — we just pass through.
    """
    client = get_daemon_client()
    try:
        # Don't use include_quotes — it adds 4+ seconds for equity quote fetching.
        # The web UI already has live prices from the ticker bar and can compute
        # breach status client-side from those cached prices.
        data = await client.get_portfolio(include_quotes=False)

        # Trigger background pre-fetch (fire-and-forget, doesn't block response)
        asyncio.create_task(prefetch_all_tickers())

        return data
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Cannot connect to UTP daemon")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/quotes")
async def api_quotes(
    symbols: str = "SPX,NDX,RUT",
    _u: str | None = Depends(optional_session),
):
    """Get live quotes for multiple symbols."""
    client = get_daemon_client()
    tickers = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    results = {}
    for sym in tickers:
        try:
            results[sym] = await client.get_quote(sym)
        except Exception as e:
            results[sym] = {"error": str(e)}
    return results


@app.get("/api/options-grid")
async def api_options_grid(
    symbol: str = "SPX",
    expiration: str | None = None,
    option_type: str = "BOTH",
    strike_range_pct: float = 3.0,
    _u: str | None = Depends(optional_session),
):
    """Get option chain formatted for grid display. Uses server-side cache."""
    client = get_daemon_client()
    symbol = symbol.upper()

    try:
        # Get current price
        quote = await client.get_quote(symbol)
        current_price = quote.get("last") or quote.get("bid") or 0

        # Get expirations: merge CSV (has daily 0DTE) + IBKR (has further-out)
        expirations = _get_cached_expirations(symbol)
        if not expirations:
            csv_exps = _get_csv_expirations(symbol)
            ibkr_exps: list[str] = []
            try:
                exp_data = await client.get_options(symbol, list_expirations=True)
                ibkr_exps = exp_data.get("expirations", [])
            except Exception:
                pass
            expirations = _merge_expirations(csv_exps, ibkr_exps)
            if expirations:
                _put_cached_expirations(symbol, expirations)

        if not expirations:
            return {"symbol": symbol, "error": "No expirations available", "expirations": []}

        # Use first expiration if none specified
        if not expiration:
            expiration = expirations[0]

        # Compute strike range
        strike_min = round(current_price * (1 - strike_range_pct / 100), 0)
        strike_max = round(current_price * (1 + strike_range_pct / 100), 0)

        # Fetch option quotes — check cache first, then CSV, then daemon
        types = ["PUT", "CALL"] if option_type == "BOTH" else [option_type.upper()]
        chain_data = {}
        source = "unknown"
        fetched_at = None
        cached_at = None
        any_empty = False

        market_open = _is_market_hours()

        for ot in types:
            # 1. Check server-side cache (may have CSV or IBKR data)
            entry = _get_cached_options(symbol, expiration, ot)
            if entry is not None:
                # During market hours, don't serve stale CSV — let it expire (2 min TTL)
                # so IBKR data is fetched fresh below
                chain_data[ot.lower()] = _filter_strikes(entry.data, strike_min, strike_max)
                source = entry.source
                fetched_at = entry.fetched_at_utc
                cached_at = datetime.fromtimestamp(entry.cached_at, UTC).isoformat(timespec="seconds")
                continue

            # 2. During market hours: try daemon/IBKR FIRST (live prices + greeks)
            #    During market closed: try CSV first (instant, IBKR won't have fresh data anyway)
            if market_open:
                try:
                    data = await client.get_options(
                        symbol, option_type=ot, expiration=expiration,
                        strike_min=strike_min, strike_max=strike_max,
                    )
                    quotes_list = data.get("quotes", {}).get(ot.lower(), [])
                    if quotes_list:
                        _put_cached_options(
                            symbol, expiration, ot, quotes_list,
                            source="ibkr", fetched_at_utc=_now_iso(),
                        )
                        chain_data[ot.lower()] = quotes_list
                        source = "ibkr"
                        fetched_at = _now_iso()
                        cached_at = fetched_at
                        continue
                except Exception as e:
                    logger.warning("IBKR fetch failed %s %s %s: %s", symbol, expiration, ot, e)

            # 3. Fallback: CSV exports (instant, used when market closed or IBKR fails)
            csv_quotes, csv_ts = _load_options_from_csv(symbol, expiration, strike_min, strike_max)
            if csv_quotes:
                by_type = _split_csv_quotes_by_type(csv_quotes)
                quotes_for_type = by_type.get(ot.lower(), [])
                if quotes_for_type:
                    _put_cached_options(
                        symbol, expiration, ot, quotes_for_type,
                        source="csv_exports", fetched_at_utc=csv_ts,
                    )
                    chain_data[ot.lower()] = quotes_for_type
                    source = "csv_exports"
                    fetched_at = csv_ts
                    cached_at = _now_iso()
                    continue

            # 3. Try daemon/IBKR (slow)
            try:
                data = await client.get_options(
                    symbol, option_type=ot, expiration=expiration,
                    strike_min=strike_min, strike_max=strike_max,
                )
                quotes_list = data.get("quotes", {}).get(ot.lower(), [])
                if quotes_list:
                    _put_cached_options(
                        symbol, expiration, ot, quotes_list,
                        source="ibkr", fetched_at_utc=_now_iso(),
                    )
                chain_data[ot.lower()] = quotes_list
                source = "ibkr"
                fetched_at = _now_iso()
                cached_at = fetched_at
            except Exception as e:
                logger.warning("Options fetch failed %s %s %s: %s", symbol, expiration, ot, e)
                chain_data[ot.lower()] = []

            if not chain_data.get(ot.lower()):
                any_empty = True

        # If any type came back empty, kick off a background prefetch so
        # the next request will have data — the client auto-retries.
        if any_empty:
            asyncio.create_task(_prefetch_options_for_symbol(symbol))

        prefetching = symbol in _prefetch_in_progress

        return {
            "symbol": symbol,
            "current_price": current_price,
            "expiration": expiration,
            "expirations": expirations,
            "strike_range": {"min": strike_min, "max": strike_max},
            "chain": chain_data,
            "source": source,
            "fetched_at": fetched_at,
            "cached_at": cached_at,
            "prefetching": prefetching,
            "market_open": _is_market_hours(),
        }
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Cannot connect to UTP daemon")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


def _filter_strikes(quotes: list[dict], strike_min: float, strike_max: float) -> list[dict]:
    """Filter cached quotes by strike range."""
    if not strike_min and not strike_max:
        return quotes
    return [q for q in quotes if strike_min <= q.get("strike", 0) <= strike_max]


# /api/recommendations removed — spread computation moved to client-side JS
# The client uses chainCache + computeSpreads() to scan spreads locally


@app.get("/api/_legacy_recommendations")
async def api_recommendations(
    symbols: str = "SPX,NDX,RUT"
):
    """Serve pre-computed recommendations. Recomputed every 60s in background."""
    global _recommendations_cache

    # If we have cached results, serve them immediately
    if _recommendations_cache:
        age = time.time() - _recommendations_cache_at
        result = dict(_recommendations_cache)
        result["_cache_age_seconds"] = round(age)
        return result

    # First request — compute now (will be fast if option data is cached from CSV)
    await _recompute_recommendations()
    if _recommendations_cache:
        return _recommendations_cache

    # Fallback: compute on the fly from whatever we have
    client = get_daemon_client()
    tickers = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    results = {}
    for symbol in tickers:
        try:
            quote = await client.get_quote(symbol)
            current_price = quote.get("last") or quote.get("bid") or 0
            if not current_price:
                results[symbol] = {"error": "No price data"}
                continue

            # Get nearest expiration (cached)
            expirations = _get_cached_expirations(symbol)
            if not expirations:
                csv_exps = _get_csv_expirations(symbol)
                ibkr_exps: list[str] = []
                try:
                    exp_data = await client.get_options(symbol, list_expirations=True)
                    ibkr_exps = exp_data.get("expirations", [])
                except Exception:
                    pass
                expirations = _merge_expirations(csv_exps, ibkr_exps)
                if expirations:
                    _put_cached_expirations(symbol, expirations)
            if not expirations:
                results[symbol] = {"error": "No expirations"}
                continue
            expiration = expirations[0]

            spread_widths = {"SPX": 20, "NDX": 50, "RUT": 20}
            width = spread_widths.get(symbol, 20)

            spreads = []
            for opt_type in ["PUT", "CALL"]:
                try:
                    strike_min = round(current_price * 0.95, 0)
                    strike_max = round(current_price * 1.05, 0)

                    quotes_list = None
                    cached = _get_cached_options(symbol, expiration, opt_type)
                    if cached is not None:
                        quotes_list = _filter_strikes(cached.data, strike_min, strike_max)
                    else:
                        # Try CSV
                        csv_quotes, csv_ts = _load_options_from_csv(symbol, expiration, strike_min, strike_max)
                        if csv_quotes:
                            by_type = _split_csv_quotes_by_type(csv_quotes)
                            quotes_list = by_type.get(opt_type.lower(), [])
                        else:
                            data = await client.get_options(
                                symbol, option_type=opt_type, expiration=expiration,
                                strike_min=strike_min, strike_max=strike_max,
                            )
                            quotes_list = data.get("quotes", {}).get(opt_type.lower(), [])

                    # Build spreads from the option chain
                    by_strike = {q["strike"]: q for q in quotes_list if q.get("strike")}
                    strikes = sorted(by_strike.keys())

                    for short_strike in strikes:
                        if opt_type == "PUT":
                            long_strike = short_strike - width
                        else:
                            long_strike = short_strike + width

                        short_q = by_strike.get(short_strike)
                        long_q = by_strike.get(long_strike)

                        if not short_q or not long_q:
                            continue

                        short_bid = short_q.get("bid", 0) or 0
                        long_ask = long_q.get("ask", 0) or 0

                        if short_bid <= 0 or long_ask <= 0:
                            continue

                        credit = round(short_bid - long_ask, 2)
                        if credit <= 0:
                            continue

                        max_loss = width - credit
                        if max_loss <= 0:
                            continue

                        roi = round(credit / max_loss * 100, 1)

                        # Distance from current price
                        if opt_type == "PUT":
                            otm_pct = round((current_price - short_strike) / current_price * 100, 2)
                        else:
                            otm_pct = round((short_strike - current_price) / current_price * 100, 2)

                        spreads.append({
                            "option_type": opt_type,
                            "short_strike": short_strike,
                            "long_strike": long_strike,
                            "width": width,
                            "credit": credit,
                            "max_loss": round(max_loss, 2),
                            "roi_pct": roi,
                            "otm_pct": round(otm_pct, 2),
                            "expiration": expiration,
                            "short_bid": short_bid,
                            "long_ask": long_ask,
                            "short_delta": (short_q.get("greeks") or {}).get("delta"),
                        })
                except Exception:
                    continue

            # Sort by ROI descending, take top 10
            spreads.sort(key=lambda s: s["roi_pct"], reverse=True)

            results[symbol] = {
                "current_price": current_price,
                "expiration": expiration,
                "spreads": spreads[:10],
                "quote": quote,
            }
        except Exception as e:
            results[symbol] = {"error": str(e)}

    return results


@app.get("/api/performance-summary")
async def api_performance_summary(
    days: int = 30,
    _u: str | None = Depends(optional_session),
):
    """Get performance metrics for the dashboard."""
    client = get_daemon_client()
    try:
        return await client.get_performance(days=days)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


PERCENTILE_SERVER_URL = os.environ.get("PERCENTILE_SERVER_URL", "http://localhost:9100")
_percentile_cache: dict | None = None
_percentile_cache_at: float = 0


@app.get("/api/percentiles")
async def api_percentiles(_u: str | None = Depends(optional_session)):
    """Proxy to range_percentiles server. Cached 60s during market hours, infinite when closed."""
    global _percentile_cache, _percentile_cache_at
    ttl = 120 if _is_market_hours() else float("inf")
    if _percentile_cache and (time.time() - _percentile_cache_at) < ttl:
        return _percentile_cache

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{PERCENTILE_SERVER_URL}/range_percentiles",
                params={"ticker": "SPX,NDX,RUT", "windows": "0,1,2,3,5", "format": "json"},
            )
            if resp.status_code == 200:
                _percentile_cache = resp.json()
                _percentile_cache_at = time.time()
                return _percentile_cache
    except Exception as e:
        logger.warning("Percentile server unavailable: %s", e)

    if _percentile_cache:
        return _percentile_cache
    return {"error": "Percentile server unavailable"}


_predictions_cache: dict[str, dict] = {}  # {ticker: response}
_predictions_cache_at: float = 0


@app.get("/api/predictions")
async def api_predictions(_u: str | None = Depends(optional_session)):
    """Proxy to predictions server for all default tickers. Cached same as percentiles."""
    global _predictions_cache, _predictions_cache_at
    ttl = 120 if _is_market_hours() else float("inf")
    # Re-fetch if cache is missing any default tickers (incomplete previous fetch)
    cache_complete = _predictions_cache and all(sym in _predictions_cache for sym in DEFAULT_TICKERS)
    if cache_complete and (time.time() - _predictions_cache_at) < ttl:
        return _predictions_cache

    result = {}
    try:
        # Predictions responses are large (300-400KB each) — use longer timeout
        # and fetch concurrently. Only keep today + future fields (skip band_history).
        async with httpx.AsyncClient(timeout=30.0) as client:
            async def _fetch_pred(sym: str) -> tuple[str, dict | None]:
                try:
                    resp = await client.get(
                        f"{PERCENTILE_SERVER_URL}/predictions/{sym}",
                        params={"format": "json"},
                    )
                    if resp.status_code == 200:
                        full = resp.json()
                        # Extract only what we need (today + future bands)
                        trimmed = {
                            "ticker": full.get("ticker"),
                            "today": full.get("today"),
                            "future": full.get("future"),
                        }
                        return sym, trimmed
                except Exception as e:
                    logger.debug("Prediction fetch failed for %s: %s", sym, e)
                return sym, None

            results = await asyncio.gather(*[_fetch_pred(s) for s in DEFAULT_TICKERS])
            for sym, data in results:
                if data:
                    result[sym] = data
    except Exception as e:
        logger.warning("Predictions server unavailable: %s", e)

    if result:
        _predictions_cache = result
        _predictions_cache_at = time.time()
        return result

    if _predictions_cache:
        return _predictions_cache
    return {"error": "Predictions server unavailable"}


@app.get("/api/trades/export")
async def api_trades_export(username: str = Depends(require_session)):
    """Download the trades CSV."""
    from fastapi.responses import FileResponse
    if not TRADES_CSV_PATH.exists():
        raise HTTPException(status_code=404, detail="No trades recorded yet")
    return FileResponse(
        path=str(TRADES_CSV_PATH),
        media_type="text/csv",
        filename="utp_voice_trades.csv",
    )


@app.get("/api/profit-targets")
async def api_profit_targets(username: str = Depends(require_session)):
    """List active profit targets — proxied to daemon."""
    client = get_daemon_client()
    try:
        return await client._get("/account/profit-targets")
    except Exception as e:
        return {"positions": {}, "error": str(e)}


@app.put("/api/profit-targets/{position_id}")
async def api_update_profit_target(
    position_id: str,
    request: Request,
    username: str = Depends(require_session),
):
    """Update a position's profit target — proxied to daemon."""
    body = await request.json()
    client = get_daemon_client()
    try:
        return await client._get(
            f"/account/profit-targets/{position_id}",
            params={"profit_target_pct": body.get("profit_target_pct", 50)},
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/position-limits")
async def api_position_limits(username: str = Depends(require_session)):
    """Get current position limit status."""
    return await _check_position_limits()


@app.get("/api/trades/list")
async def api_trades_list(
    days: int = 0,
    page: int = 1,
    per_page: int = 20,
    username: str = Depends(require_session),
):
    """List trades from CSV with optional date filter, paginated, newest first."""
    import csv as csv_mod
    if not TRADES_CSV_PATH.exists():
        return {"trades": [], "total": 0, "page": 1, "pages": 0, "summary": {}}

    trades = []
    with open(TRADES_CSV_PATH) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            trades.append(row)

    # Date filter
    if days > 0:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        trades = [t for t in trades if (t.get("timestamp", "") or "")[:10] >= cutoff]

    # Sort newest first
    trades.sort(key=lambda t: t.get("timestamp", ""), reverse=True)

    # Summary KPIs
    total_credit = sum(float(t.get("total_credit", 0) or 0) for t in trades)
    total_max_loss = sum(float(t.get("total_max_loss", 0) or 0) for t in trades)
    auto_count = sum(1 for t in trades if t.get("source") == "auto")
    manual_count = sum(1 for t in trades if t.get("source") != "auto")

    # Pagination
    total = len(trades)
    start = (page - 1) * per_page
    page_trades = trades[start:start + per_page]

    return {
        "trades": page_trades,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page if total else 0,
        "summary": {
            "total_trades": total,
            "auto_trades": auto_count,
            "manual_trades": manual_count,
            "total_credit": round(total_credit, 2),
            "total_max_loss": round(total_max_loss, 2),
        },
    }


@app.post("/api/auto-trade/start")
async def api_auto_trade_start(
    request: Request,
    username: str = Depends(require_session),
):
    """Start auto-trading with the given filters and config."""
    global _auto_trade_task
    body = await request.json()
    filters = body.get("filters", {})
    config = body.get("config", {})

    # Validate
    if not filters.get("tickers"):
        filters["tickers"] = list(DEFAULT_TICKERS)

    today = datetime.now().strftime("%Y-%m-%d")
    state = {
        "active": True,
        "created_at": _now_iso(),
        "trading_day": today,
        "filters": filters,
        "config": config,
        "executed_today": [],
        "log": [{"time": _now_iso(), "action": "started", "by": username}],
    }
    _save_auto_trade_state(state)

    # Start background task
    if _auto_trade_task and not _auto_trade_task.done():
        _auto_trade_task.cancel()
    _auto_trade_task = asyncio.create_task(_auto_trade_loop(state))

    logger.info("Auto-trade started by %s: top_%s every %sm until %s",
                username, config.get("top_n", 3), config.get("interval_minutes", 3),
                config.get("end_time_utc", "16:00"))

    return {"status": "started", "state": state}


@app.post("/api/auto-trade/stop")
async def api_auto_trade_stop(username: str = Depends(require_session)):
    """Stop auto-trading."""
    global _auto_trade_task
    state = _load_auto_trade_state()
    state.get("log", []).append({"time": _now_iso(), "action": "stopped", "by": username})
    _archive_auto_trade_session(state)
    state["active"] = False
    _save_auto_trade_state(state)
    if _auto_trade_task and not _auto_trade_task.done():
        _auto_trade_task.cancel()
    logger.info("Auto-trade stopped by %s", username)
    return {"status": "stopped"}


@app.get("/api/auto-trade/status")
async def api_auto_trade_status(username: str = Depends(require_session)):
    """Get current auto-trade state."""
    state = _load_auto_trade_state()
    state["task_running"] = _auto_trade_task is not None and not _auto_trade_task.done()
    return state


@app.get("/api/auto-trade/history")
async def api_auto_trade_history(
    page: int = 1,
    per_page: int = 10,
    username: str = Depends(require_session),
):
    """Get auto-trade session history with trade details, paged, newest first."""
    import csv as csv_mod

    sessions = _load_auto_trade_history(limit=100)

    # Enrich sessions with P&L from trades CSV if available
    trades_by_day: dict[str, list[dict]] = {}
    if TRADES_CSV_PATH.exists():
        with open(TRADES_CSV_PATH) as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                if row.get("source") == "auto":
                    day = row.get("timestamp", "")[:10]
                    trades_by_day.setdefault(day, []).append(row)

    for session in sessions:
        day = session.get("trading_day", "")
        day_trades = trades_by_day.get(day, [])
        session["csv_trades"] = day_trades
        session["total_trades_csv"] = len(day_trades)
        # Compute P&L from CSV data
        total_credit = sum(float(t.get("total_credit", 0) or 0) for t in day_trades)
        total_max_loss = sum(float(t.get("total_max_loss", 0) or 0) for t in day_trades)
        session["csv_total_credit"] = round(total_credit, 2)
        session["csv_total_max_loss"] = round(total_max_loss, 2)

    # Pagination
    total = len(sessions)
    start = (page - 1) * per_page
    page_sessions = sessions[start:start + per_page]

    return {
        "sessions": page_sessions,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page,
    }


@app.post("/api/quick-trade")
async def api_quick_trade(
    request: Request,
    username: str = Depends(require_session),
):
    """Quick trade from the options grid — creates a pending confirmation.

    Body: {trade_type, symbol, option_type, short_strike, long_strike,
           quantity, expiration, net_price?}
    """
    body = await request.json()
    required = ["trade_type", "symbol", "short_strike", "long_strike", "expiration"]
    for field in required:
        if field not in body:
            raise HTTPException(status_code=400, detail=f"Missing: {field}")

    # Default quantity to 1
    if "quantity" not in body:
        body["quantity"] = 1

    description = describe_write_action("execute_trade", body)
    action = PendingAction(
        tool_name="execute_trade",
        tool_input=body,
        description=description,
    )
    store_pending(action)

    return {
        "confirmation_id": action.confirmation_id,
        "description": description,
        "tool_input": body,
        "payload_preview": build_trade_payload(body),
    }


# ── CLI Entry Point ───────────────────────────────────────────────────────────


def _update_daemon_url(url: str) -> None:
    global UTP_DAEMON_URL, _daemon_client
    UTP_DAEMON_URL = url
    _daemon_client = None


def main():
    parser = argparse.ArgumentParser(
        description="UTP Voice — Natural Language Mobile Trading Interface",
        epilog="""
Examples:
  %(prog)s add-user akundu         Add or update a user
  %(prog)s list-users              List all configured users
  %(prog)s serve                   Start the web server
  %(prog)s serve --port 9000       Start on custom port
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the UTP Voice web server")
    serve_parser.add_argument("--port", type=int, default=None, help=f"Port (default: {UTP_VOICE_PORT})")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    serve_parser.add_argument("--daemon-url", default=None, help=f"UTP daemon URL (default: {UTP_DAEMON_URL})")
    serve_parser.add_argument("--log-level", default="info", help="Log level (default: info)")
    serve_parser.add_argument("--workers", type=int, default=1,
                              help="Number of worker processes (default: 1). Use 2-4 for multi-core.")
    serve_parser.add_argument("--public", action="store_true", default=False,
                              help="Allow anonymous access to options/picks without login (default: require login for all)")
    serve_parser.add_argument("--max-open-positions", type=int, default=10,
                              help="Max open positions at any time (default: 10, 0=unlimited)")
    serve_parser.add_argument("--max-daily-trades", type=int, default=20,
                              help="Max new trades per day (default: 20, 0=unlimited)")

    # add-user
    add_user_parser = subparsers.add_parser("add-user", help="Add or update a user")
    add_user_parser.add_argument("username", help="Username to add")
    add_user_parser.add_argument("--password", help="Password (will prompt if not provided)")

    # list-users
    subparsers.add_parser("list-users", help="List configured users")

    args = parser.parse_args()
    cmd = args.command

    if cmd == "add-user":
        password = args.password
        if not password:
            password = getpass.getpass("Enter password: ")
            confirm = getpass.getpass("Confirm password: ")
            if password != confirm:
                print("Passwords do not match.")
                sys.exit(1)
        add_user(args.username, password)

    elif cmd == "list-users":
        users = list_users()
        if users:
            print("Configured users:")
            for u in users:
                print(f"  - {u}")
        else:
            print(f"No users configured. Run: python {sys.argv[0]} add-user <username>")

    elif cmd == "serve" or cmd is None:
        # Update globals from args if provided
        port = getattr(args, "port", None) or UTP_VOICE_PORT
        daemon_url = getattr(args, "daemon_url", None)
        log_level = getattr(args, "log_level", "info")

        if daemon_url:
            _update_daemon_url(daemon_url)

        # Set public mode from --public flag
        global PUBLIC_MODE
        PUBLIC_MODE = getattr(args, "public", False)

        global MAX_OPEN_POSITIONS, MAX_DAILY_TRADES
        MAX_OPEN_POSITIONS = getattr(args, "max_open_positions", 10)
        MAX_DAILY_TRADES = getattr(args, "max_daily_trades", 20)

        if not JWT_SECRET:
            print("ERROR: UTP_VOICE_JWT_SECRET environment variable is required.")
            print("  export UTP_VOICE_JWT_SECRET='your-secret-key'")
            sys.exit(1)

        if not ANTHROPIC_API_KEY:
            print("WARNING: ANTHROPIC_API_KEY not set. NL commands will not work.")

        # Check credentials
        if not list_users():
            print(f"WARNING: No users configured. Run: python {sys.argv[0]} add-user <username>")

        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )

        host = getattr(args, "host", "0.0.0.0")
        workers = getattr(args, "workers", 1)
        print(f"Starting UTP Voice on {host}:{port}")
        print(f"  Daemon URL: {UTP_DAEMON_URL}")
        print(f"  Credentials: {CREDENTIALS_FILE}")
        print(f"  Workers: {workers}")
        if PUBLIC_MODE:
            print(f"  Public mode: ON (options/picks accessible without login)")
        print(f"  Open http://localhost:{port} in your browser")

        _run_server_with_restart(host, port, log_level, workers)


def _run_server_with_restart(host: str, port: int, log_level: str, workers: int) -> None:
    """Run uvicorn with auto-restart on crash. Ctrl-C/SIGTERM exits cleanly."""
    import signal
    import threading

    def _force_exit():
        """Force kill after 5 seconds if graceful shutdown hangs."""
        print("\n  Force killing (shutdown took too long)...")
        os._exit(1)

    max_restarts = 50
    restart_count = 0
    backoff = 2.0
    max_backoff = 30.0

    while restart_count < max_restarts:
        try:
            if workers > 1:
                uvicorn.run(
                    "utp_voice:app",
                    host=host, port=port, log_level=log_level,
                    workers=workers,
                )
            else:
                uvicorn.run(app, host=host, port=port, log_level=log_level)

            # Clean exit (uvicorn returned normally) — don't restart
            break

        except KeyboardInterrupt:
            print("\n  Shutting down (Ctrl-C)... force kill in 5s if stuck")
            # Start a daemon thread that force-kills after 5 seconds
            t = threading.Timer(5.0, _force_exit)
            t.daemon = True
            t.start()
            break
        except SystemExit as e:
            if e.code == 0:
                break
            # Non-zero exit — restart
            restart_count += 1
            wait = min(backoff * restart_count, max_backoff)
            print(f"\n  Server exited with code {e.code}. Restarting in {wait:.0f}s... ({restart_count}/{max_restarts})")
            try:
                time.sleep(wait)
            except KeyboardInterrupt:
                print("\n  Shutting down (Ctrl-C during restart)...")
                break
        except Exception as e:
            restart_count += 1
            wait = min(backoff * restart_count, max_backoff)
            logger.error("Server crashed: %s", e, exc_info=True)
            print(f"\n  Server crashed: {e}")
            print(f"  Restarting in {wait:.0f}s... ({restart_count}/{max_restarts})")
            try:
                time.sleep(wait)
            except KeyboardInterrupt:
                print("\n  Shutting down (Ctrl-C during restart)...")
                break

    if restart_count >= max_restarts:
        print(f"\n  Max restarts ({max_restarts}) reached. Giving up.")
        sys.exit(1)


if __name__ == "__main__":
    main()
