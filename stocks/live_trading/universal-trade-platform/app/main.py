"""FastAPI application — wires up routes, providers, and lifecycle."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
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
# because the daemon process already did it.
_daemon_mode = False


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
        # Daemon already initialized providers, ledger, position store, and background tasks
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
