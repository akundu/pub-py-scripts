"""Market data endpoints — quotes, batch quotes, margin checks."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Security
from pydantic import BaseModel, Field

from fastapi import HTTPException

from app.auth import TokenData, require_auth
from app.core.provider import ProviderRegistry
from app.models import Broker, MultiLegOrder, Quote

router = APIRouter(prefix="/market", tags=["market data"])


@router.get("/quote/{symbol}", response_model=Quote)
async def get_quote(
    symbol: str,
    _user: Annotated[TokenData, Security(require_auth, scopes=["market:read"])],
    broker: Broker = Broker.IBKR,
) -> Quote:
    """Fetch a real-time quote for a symbol using the symbology engine."""
    provider = ProviderRegistry.get(broker)
    return await provider.get_quote(symbol.upper())


class BatchQuoteRequest(BaseModel):
    symbols: list[str] = Field(..., min_length=1, description="List of ticker symbols")
    broker: Broker = Broker.IBKR


class BatchQuoteResponse(BaseModel):
    quotes: list[Quote]
    errors: dict[str, str] = Field(default_factory=dict)


@router.post("/quotes", response_model=BatchQuoteResponse)
async def get_quotes_batch(
    request: BatchQuoteRequest,
    _user: Annotated[TokenData, Security(require_auth, scopes=["market:read"])],
) -> BatchQuoteResponse:
    """Fetch quotes for multiple symbols in a single request."""
    provider = ProviderRegistry.get(request.broker)
    quotes: list[Quote] = []
    errors: dict[str, str] = {}

    for symbol in request.symbols:
        try:
            q = await provider.get_quote(symbol.upper())
            quotes.append(q)
        except Exception as e:
            errors[symbol] = str(e)

    return BatchQuoteResponse(quotes=quotes, errors=errors)


class MarginCheckRequest(BaseModel):
    order: MultiLegOrder
    timeout: float = Field(default=10.0, ge=1.0, le=60.0,
                           description="Timeout in seconds for margin check")


class MarginCheckResponse(BaseModel):
    init_margin: float = 0.0
    maint_margin: float = 0.0
    commission: float = 0.0
    equity_with_loan: float | None = None
    init_margin_before: float | None = None
    init_margin_after: float | None = None
    error: str | None = None


@router.post("/margin", response_model=MarginCheckResponse)
async def check_margin(
    request: MarginCheckRequest,
    _user: Annotated[TokenData, Security(require_auth, scopes=["market:read"])],
) -> MarginCheckResponse:
    """Check margin requirements for a hypothetical multi-leg order.

    Uses the broker's whatIfOrder API to determine initial/maintenance margin
    and commission without placing the order.
    """
    provider = ProviderRegistry.get(request.order.broker)

    try:
        margin = await asyncio.wait_for(
            provider.check_margin(request.order),
            timeout=request.timeout,
        )
    except asyncio.TimeoutError:
        return MarginCheckResponse(
            error=f"Margin check timed out after {request.timeout}s — may work during market hours"
        )
    except Exception as e:
        return MarginCheckResponse(error=str(e))

    return MarginCheckResponse(
        init_margin=margin.get("init_margin", 0.0),
        maint_margin=margin.get("maint_margin", 0.0),
        commission=margin.get("commission", 0.0),
        equity_with_loan=margin.get("equity_with_loan"),
        init_margin_before=margin.get("init_margin_before"),
        init_margin_after=margin.get("init_margin_after"),
        error=margin.get("error"),
    )


@router.get("/options/{symbol}")
async def get_options(
    symbol: str,
    _user: Annotated[TokenData, Security(require_auth, scopes=["market:read"])],
    broker: Broker = Broker.IBKR,
    expiration: str | None = None,
    option_type: str | None = None,
    strike_min: float | None = None,
    strike_max: float | None = None,
    list_expirations: bool = False,
) -> dict:
    """Get option chain data for a symbol."""
    provider = ProviderRegistry.get(broker)
    chain = await provider.get_option_chain(symbol.upper())

    if list_expirations:
        return {"symbol": symbol.upper(), "expirations": chain.get("expirations", [])}

    result = {"symbol": symbol.upper(), "chain": chain}

    if expiration and hasattr(provider, 'get_option_quotes'):
        quotes = {}
        types_to_fetch = []
        if option_type:
            types_to_fetch = [option_type.upper()]
        else:
            types_to_fetch = ["CALL", "PUT"]

        for ot in types_to_fetch:
            try:
                q = await provider.get_option_quotes(
                    symbol.upper(), expiration, ot,
                    strike_min=strike_min, strike_max=strike_max,
                )
                quotes[ot.lower()] = q
            except Exception as e:
                quotes[ot.lower()] = {"error": str(e)}

        result["quotes"] = quotes

    return result


# ── Streaming management ──────────────────────────────────────────────────────


@router.get("/streaming/status")
async def streaming_status(
    _user: Annotated[TokenData, Security(require_auth, scopes=["market:read"])],
) -> dict:
    """Get real-time streaming service status: active subscriptions, stats."""
    from app.services.market_data_streaming import get_streaming_service

    svc = get_streaming_service()
    if not svc:
        return {"running": False, "message": "Streaming service not initialized. Start daemon with --streaming-config."}
    return svc.stats


class StreamingSubscribeRequest(BaseModel):
    symbols: list[str] = Field(..., min_length=1, description="Symbols to subscribe to")


@router.post("/streaming/subscribe")
async def streaming_subscribe(
    request: StreamingSubscribeRequest,
    _user: Annotated[TokenData, Security(require_auth, scopes=["market:read"])],
) -> dict:
    """Subscribe to additional symbols for real-time streaming."""
    from app.services.market_data_streaming import get_streaming_service
    from app.services.streaming_config import StreamingSymbolConfig, _INDEX_EXCHANGES

    svc = get_streaming_service()
    if not svc:
        raise HTTPException(status_code=503, detail="Streaming service not initialized")
    if not svc.is_running:
        raise HTTPException(status_code=503, detail="Streaming service not running")

    configs = []
    for sym in request.symbols:
        sym = sym.upper()
        if sym in _INDEX_EXCHANGES:
            configs.append(StreamingSymbolConfig(symbol=sym, sec_type="IND", exchange=_INDEX_EXCHANGES[sym]))
        else:
            configs.append(StreamingSymbolConfig(symbol=sym, sec_type="STK", exchange="SMART"))

    subscribed = await svc.subscribe(configs)
    return {"subscribed": subscribed, "total_subscriptions": svc.subscription_count}


@router.post("/streaming/unsubscribe")
async def streaming_unsubscribe(
    request: StreamingSubscribeRequest,
    _user: Annotated[TokenData, Security(require_auth, scopes=["market:read"])],
) -> dict:
    """Unsubscribe from symbols."""
    from app.services.market_data_streaming import get_streaming_service

    svc = get_streaming_service()
    if not svc:
        raise HTTPException(status_code=503, detail="Streaming service not initialized")

    removed = await svc.unsubscribe([s.upper() for s in request.symbols])
    return {"unsubscribed": removed, "total_subscriptions": svc.subscription_count}
