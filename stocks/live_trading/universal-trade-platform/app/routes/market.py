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
    max_age: float | None = None,
    force_refresh: bool = False,
) -> Quote:
    """Fetch a real-time quote — streaming cache first, provider fallback.

    Parameters:
        max_age: Reject cached quotes older than this (seconds); force a
            provider refresh before falling through to stale cache.
        force_refresh: Skip all caches and always hit the provider.
    """
    from app.services.market_data import get_quote as _get_quote
    return await _get_quote(symbol, broker, max_age=max_age, force_refresh=force_refresh)


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
    from app.services.market_data import get_quote as _get_quote
    quotes: list[Quote] = []
    errors: dict[str, str] = {}

    for symbol in request.symbols:
        try:
            q = await _get_quote(symbol.upper(), request.broker)
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
    max_age: float | None = None,
    force_refresh: bool = False,
) -> dict:
    """Get option chain data — centralized cache → provider fallback.

    Parameters:
        max_age: Reject cached quotes older than this (seconds); force a
            provider refresh before falling through to stale cache.
        force_refresh: Skip all caches and always hit the provider.
    """
    from app.services.market_data import get_option_quotes_with_age as _get_opts_age
    from app.services.market_data import get_option_chain as _get_chain

    if list_expirations:
        chain = await _get_chain(symbol.upper(), broker)
        ibkr_exps = chain.get("expirations", [])

        # Merge with CSV-derived expirations from the streaming service
        # (IBKR only lists weekly/monthly; CSV has daily 0DTE expirations)
        from app.services.option_quote_streaming import get_option_quote_streaming
        oqs = get_option_quote_streaming()
        csv_exps = []
        if oqs:
            csv_exps = oqs._get_expirations_from_csv(symbol.upper())

        # Merge, normalize, deduplicate
        all_exps = set()
        # In simulation mode, use the sim date (not actual today)
        from app.services.simulation_clock import get_sim_clock as _get_sim_clock
        _sim_clock = _get_sim_clock()
        today_str = _sim_clock.sim_date.isoformat() if _sim_clock else __import__("datetime").date.today().isoformat()
        for e in list(ibkr_exps) + csv_exps:
            norm = e.replace("-", "") if len(e) == 10 else e
            iso = f"{norm[:4]}-{norm[4:6]}-{norm[6:8]}" if len(norm) == 8 else norm
            if iso >= today_str:
                all_exps.add(iso)
        return {"symbol": symbol.upper(), "expirations": sorted(all_exps)}

    if not expiration:
        chain = await _get_chain(symbol.upper(), broker)
        return {"symbol": symbol.upper(), "chain": chain}

    # Fetch quotes through centralized data layer (cache → stale → provider)
    types_to_fetch = [option_type.upper()] if option_type else ["CALL", "PUT"]
    quotes = {}
    # Track the most-stale internal source label (fresh_cache, stale_cache,
    # provider, empty).  ``source`` (legacy field) maps this to the older
    # "streaming_cache" / "streaming_cache_stale" / "provider" vocabulary.
    internal_source = "provider"
    worst_age: float | None = None
    per_type_meta: dict[str, dict] = {}
    for ot in types_to_fetch:
        try:
            q, age, type_source = await _get_opts_age(
                symbol, expiration, ot,
                strike_min=strike_min, strike_max=strike_max,
                broker=broker,
                max_age=max_age, force_refresh=force_refresh,
            )
            quotes[ot.lower()] = q
            per_type_meta[ot.lower()] = {"age_seconds": age, "source": type_source}
            # Track worst (oldest) age across types for a conservative summary
            if age is not None:
                worst_age = age if worst_age is None else max(worst_age, age)
            # Prefer the most-stale source label so callers can classify
            if type_source == "stale_cache":
                internal_source = "stale_cache"
            elif type_source == "empty" and internal_source == "provider":
                internal_source = "empty"
            elif type_source == "fresh_cache" and internal_source == "provider":
                internal_source = "fresh_cache"
        except Exception as e:
            quotes[ot.lower()] = {"error": str(e)}
            per_type_meta[ot.lower()] = {"age_seconds": None, "source": "error"}

    # Legacy source mapping — preserve previous API contract.
    if internal_source == "fresh_cache":
        legacy_source = "streaming_cache"
    elif internal_source == "stale_cache":
        legacy_source = "streaming_cache_stale"
    else:
        legacy_source = internal_source  # "provider" or "empty"

    return {
        "symbol": symbol.upper(),
        "chain": {"expirations": [], "strikes": []},
        "quotes": quotes,
        "source": legacy_source,
        "quote_age_seconds": worst_age,
        "quote_source": internal_source,
        "meta": {
            "age_seconds": worst_age,
            "source": internal_source,
            "per_type": per_type_meta,
        },
    }


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


@router.get("/streaming/option-quotes/status")
async def option_quote_streaming_status(
    _user: Annotated[TokenData, Security(require_auth, scopes=["market:read"])],
) -> dict:
    """Get option quote streaming cache status: size, cycle count, symbols tracked."""
    from app.services.option_quote_streaming import get_option_quote_streaming

    svc = get_option_quote_streaming()
    if not svc:
        return {"running": False, "message": "Option quote streaming not initialized. Enable option_quotes_enabled in streaming config."}
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
