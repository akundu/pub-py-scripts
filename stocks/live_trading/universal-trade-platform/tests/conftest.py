"""Shared fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core.provider import ProviderRegistry
from app.core.providers.etrade import EtradeProvider
from app.core.providers.ibkr import IBKRProvider
from app.core.providers.robinhood import RobinhoodProvider
from app.main import app
from app.services.ledger import TransactionLedger, init_ledger, reset_ledger
from app.services.dashboard_service import DashboardService
from app.services.live_data_service import init_live_data_service, reset_live_data_service
from app.services.position_store import (
    PlatformPositionStore,
    get_position_store,
    init_position_store,
    reset_position_store,
)


@pytest.fixture(autouse=True)
def _disable_server_detection(monkeypatch):
    """Prevent CLI commands from detecting a running daemon during tests."""
    import utp
    monkeypatch.setattr(utp, "_detect_server", lambda args: None)


@pytest.fixture(autouse=True)
async def _setup_providers(tmp_path):
    """Register and connect all providers before each test; clean up after."""
    ProviderRegistry.clear()
    providers = [RobinhoodProvider(), EtradeProvider(), IBKRProvider()]
    for p in providers:
        ProviderRegistry.register(p)
        await p.connect()

    # Initialize persistence services with temp dirs
    init_ledger(tmp_path)
    init_position_store(tmp_path)

    # Initialize LiveDataService (no IBKR in tests → always fallback path)
    store = get_position_store()
    if store:
        init_live_data_service(store, DashboardService(store))

    yield

    for p in providers:
        await p.disconnect()
    ProviderRegistry.clear()
    reset_ledger()
    reset_position_store()
    reset_live_data_service()
    from app.services.market_data_streaming import reset_streaming_service
    reset_streaming_service()

    # Reset daemon shared state between tests
    from utp import _daemon_state
    _daemon_state["advisor_entries"] = []
    _daemon_state["advisor_exits"] = []
    _daemon_state["advisor_profile"] = None
    _daemon_state["advisor_last_eval"] = None


@pytest.fixture
def api_key_headers() -> dict[str, str]:
    return {"X-API-Key": settings.api_key_secret}


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def tmp_data_dir(tmp_path) -> Path:
    """Isolated data directory for file I/O tests."""
    d = tmp_path / "utp"
    d.mkdir()
    return d


@pytest.fixture
def ledger(tmp_data_dir) -> TransactionLedger:
    """Transaction ledger backed by temp directory."""
    return TransactionLedger(tmp_data_dir / "ledger")


@pytest.fixture
def position_store(tmp_data_dir) -> PlatformPositionStore:
    """Position store backed by temp directory."""
    return PlatformPositionStore(tmp_data_dir / "positions.json")
