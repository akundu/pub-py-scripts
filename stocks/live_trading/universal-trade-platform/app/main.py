"""FastAPI application — wires up routes, providers, and lifecycle."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime

# Ensure common/ package is importable (for market_hours, etc.)
from pathlib import Path as _Path
_stocks_root = str(_Path(__file__).resolve().parents[3])
if _stocks_root not in sys.path:
    sys.path.insert(0, _stocks_root)
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request

from app.config import settings
from app.core.provider import ProviderRegistry
from app.core.providers.etrade import EtradeProvider
from app.core.providers.ibkr import IBKRProvider, IBKRLiveProvider
from app.core.providers.robinhood import RobinhoodProvider
from app.routes import account, auth_routes, market, trade, ws
from app.routes import ledger as ledger_routes
from app.routes import dashboard as dashboard_routes
from app.routes import import_routes
from app.routes import playbook as playbook_routes
from app.services.dashboard_service import DashboardService
from app.services.ledger import init_ledger, get_ledger
from app.services.live_data_service import init_live_data_service, reset_live_data_service
from app.services.position_store import init_position_store, get_position_store
from app.websocket import ws_manager

_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, _log_level, logging.INFO),
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Module-level daemon mode flag. When True, lifespan() skips provider init
# because the daemon process already did it.  Also check env var for
# subprocess workers that can't inherit the Python module-level flag.
_daemon_mode = os.environ.get("_UTP_DAEMON_MODE") == "1"


async def _expiration_loop(interval: int) -> None:
    """Background loop: check expirations and EOD exits."""
    from app.services.expiration_service import ExpirationService

    while True:
        await asyncio.sleep(interval)
        store = get_position_store()
        ledger = get_ledger()
        if not store or not ledger:
            continue

        exp_service = ExpirationService(store, ledger, ws_manager)
        today = datetime.now(UTC).date()
        try:
            await exp_service.check_expirations(today)
            if settings.eod_auto_close:
                await exp_service.check_eod_exits(datetime.now(UTC))
        except Exception as e:
            logger.error("Expiration loop error: %s", e)


async def _position_sync_loop(interval: int) -> None:
    """Background loop: sync positions from all brokers."""
    from app.services.position_sync import PositionSyncService

    while True:
        await asyncio.sleep(interval)
        store = get_position_store()
        ledger = get_ledger()
        if not store or not ledger:
            continue

        sync_service = PositionSyncService(store, ledger, ws_manager)
        now = datetime.now(UTC)
        if sync_service.is_trading_hours(now):
            try:
                result = await sync_service.sync_all_brokers()
                if result.new_positions > 0:
                    logger.info("Sync found %d new positions", result.new_positions)
            except Exception as e:
                logger.error("Position sync loop error: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Register and connect broker providers on startup; disconnect on shutdown."""
    if _daemon_mode:
        # Daemon already initialized providers, ledger, position store, and background tasks.
        # If this is a forked worker (not the main daemon process), the IBKR connection
        # doesn't survive the fork. Register a proxy provider that routes IBKR calls
        # back to the main daemon process via HTTP.
        if os.environ.get("_UTP_DAEMON_WORKER") == "1":
            # Workers are pure reverse proxies — all requests forwarded to IBKR
            # process by worker_proxy_middleware. No local services needed.
            logger.info("Worker process: all requests proxy to IBKR process (port=%s)",
                        os.environ.get("_UTP_DAEMON_PORT", "8001"))
        yield
        return

    # Initialize persistence services
    data_dir = Path(settings.data_dir)
    init_ledger(data_dir)
    init_position_store(data_dir)

    # Select IBKR provider: REST (CPG), live (ib_insync), or stub
    if settings.ibkr_api_mode == "rest":
        from app.core.providers.ibkr_rest import IBKRRestProvider
        ibkr_provider = IBKRRestProvider(
            gateway_url=settings.ibkr_gateway_url,
            account_id=settings.ibkr_account_id,
        )
    elif settings.ibkr_account_id:
        ibkr_provider = IBKRLiveProvider()
    else:
        ibkr_provider = IBKRProvider()

    providers = [RobinhoodProvider(), EtradeProvider(), ibkr_provider]
    for p in providers:
        ProviderRegistry.register(p)
        await p.connect()

    # Initialize LiveDataService (IBKR-primary with local fallback)
    store = get_position_store()
    if store:
        dashboard_svc = DashboardService(store)
        # Only pass IBKR provider if it's the live variant
        ibkr_for_live = ibkr_provider if settings.ibkr_account_id else None
        init_live_data_service(store, dashboard_svc, ibkr_for_live)

    # Start background tasks
    tasks: list[asyncio.Task] = []
    tasks.append(
        asyncio.create_task(_expiration_loop(settings.expiration_check_interval_seconds))
    )
    if settings.position_sync_enabled:
        tasks.append(
            asyncio.create_task(_position_sync_loop(settings.position_sync_interval_seconds))
        )

    yield

    # Shutdown
    for t in tasks:
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

    for p in providers:
        await p.disconnect()
    ProviderRegistry.clear()
    reset_live_data_service()


PRIVATE_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
]

app = FastAPI(
    title="Universal Trade Platform",
    version="2.0.0",
    description="Unified multi-broker trading API with real-time WebSocket updates, "
                "transaction ledger, dashboard, and paper trading.",
    lifespan=lifespan,
)


# Shared httpx client for worker → IBKR process proxying (connection pooling)
_worker_proxy_client: "httpx.AsyncClient | None" = None


def _get_worker_proxy_client():
    global _worker_proxy_client
    if _worker_proxy_client is None or _worker_proxy_client.is_closed:
        import httpx
        port = int(os.environ.get("_UTP_DAEMON_PORT", "8001"))
        _worker_proxy_client = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}",
            timeout=60.0,
        )
    return _worker_proxy_client


@app.middleware("http")
async def worker_proxy_middleware(request: Request, call_next):
    """In multi-worker mode, route requests through Redis cache or IBKR process.

    Workers have no local state. For read (GET) requests:
    1. Check Redis for a cached response → serve instantly if fresh
    2. On miss → proxy to IBKR process → cache the response in Redis

    For write (POST/PUT/DELETE) requests: always proxy to IBKR process.
    /health is handled locally for uvicorn worker health checks.
    """
    if os.environ.get("_UTP_DAEMON_WORKER") != "1" or request.url.path == "/health":
        return await call_next(request)

    from starlette.responses import Response
    from fastapi.responses import JSONResponse

    path = request.url.path
    query = str(request.url.query) if request.url.query else ""
    target = f"{path}?{query}" if query else path

    # GET requests: try Redis cache first
    if request.method == "GET":
        try:
            from app.services.redis_cache import get_redis
            r = await get_redis()
            if r:
                cache_key = f"utp:http_cache:{path}?{query}" if query else f"utp:http_cache:{path}"
                cached = await r.get(cache_key)
                if cached:
                    return Response(
                        content=cached.encode() if isinstance(cached, str) else cached,
                        status_code=200,
                        media_type="application/json",
                    )
        except Exception:
            pass  # Redis down — fall through to proxy

    # Proxy to IBKR process (shared connection pool)
    try:
        client = _get_worker_proxy_client()
        body = await request.body()
        resp = await client.request(
            method=request.method,
            url=target,
            headers={k: v for k, v in request.headers.items()
                     if k.lower() not in ("host", "content-length")},
            content=body if body else None,
        )

        # Cache successful GET responses in Redis (short TTL)
        if request.method == "GET" and resp.status_code == 200:
            try:
                from app.services.redis_cache import get_redis
                r = await get_redis()
                if r:
                    cache_key = f"utp:http_cache:{path}?{query}" if query else f"utp:http_cache:{path}"
                    await r.setex(cache_key, 5, resp.content)
            except Exception:
                pass

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"detail": f"IBKR process proxy error: {e}"},
        )


@app.middleware("http")
async def trust_lan_middleware(request: Request, call_next):
    """Mark requests from private/LAN IPs as trusted, skipping auth."""
    if settings.trust_local_network:
        try:
            client_ip = ipaddress.ip_address(request.client.host)
            if any(client_ip in net for net in PRIVATE_NETWORKS):
                request.state.lan_trusted = True
        except (ValueError, AttributeError):
            pass
    response = await call_next(request)
    return response


app.include_router(auth_routes.router)
app.include_router(trade.router)
app.include_router(market.router)
app.include_router(account.router)
app.include_router(ws.router)
app.include_router(ledger_routes.router)
app.include_router(dashboard_routes.router)
app.include_router(import_routes.router)
app.include_router(playbook_routes.router)


@app.get("/health")
async def health() -> dict:
    result: dict = {"status": "ok", "daemon_mode": _daemon_mode}
    if _daemon_mode:
        from app.models import Broker
        try:
            provider = ProviderRegistry.get(Broker.IBKR)
            result["ibkr_connected"] = getattr(provider, 'is_healthy', lambda: True)()
        except Exception:
            result["ibkr_connected"] = False
    return result
