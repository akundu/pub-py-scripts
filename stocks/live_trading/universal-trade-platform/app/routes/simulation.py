"""Simulation clock control + picks + sweep endpoints."""

from __future__ import annotations

import itertools
import logging
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.provider import ProviderRegistry
from app.models import (
    Broker,
    MultiLegOrder,
    OptionAction,
    OptionLeg,
    OptionType,
    OrderType,
)
from app.services.ledger import get_ledger
from app.services.position_store import get_position_store
from app.services.simulation_clock import get_sim_clock

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sim", tags=["simulation"])

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SetTimeRequest(BaseModel):
    time: Optional[str] = None
    advance_minutes: Optional[int] = None
    timestamp: Optional[str] = None


class LoadDateRequest(BaseModel):
    date: str


class AutoAdvanceRequest(BaseModel):
    enabled: bool = True
    interval: float = 3.0


class PicksRequest(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    option_types: list[str] = Field(default=["put", "call"])
    min_otm_pct: float = 0.015
    min_credit: float = 0.25
    max_loss_per_spread: float = 10000.0
    spread_width: Optional[float] = None
    num_contracts: int = 10
    dte: list[int] = Field(default=[0])
    roi_min: float = 0.0
    sort_by: str = "roi"
    limit: int = 20
    auto_execute: bool = False
    top_n: int = 5


class SweepRequest(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    sweep_params: dict = Field(default_factory=dict)
    fixed_params: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_status() -> dict:
    """Build the simulation status dict from the current clock state."""
    clock = get_sim_clock()
    if clock is None:
        return {"active": False}

    sim_time = clock.now()
    et_time = sim_time.astimezone(ET)

    ts_range: dict = {}
    if clock.timestamps:
        ts_range = {
            "first": clock.timestamps[0].isoformat(),
            "last": clock.timestamps[-1].isoformat(),
        }

    store = get_position_store()
    pos_count = (
        len(store.get_open_positions()) if store else 0
    )

    return {
        "active": True,
        "date": clock.sim_date.isoformat(),
        "current_time_utc": sim_time.isoformat(),
        "current_time_et": et_time.strftime("%Y-%m-%d %H:%M:%S"),
        "tickers": getattr(clock, "_tickers", []),
        "timestamp_count": len(clock.timestamps),
        "timestamp_range": ts_range,
        "cursor_position": clock._cursor,
        "auto_advancing": clock.auto_advancing,
        "position_count": pos_count,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/status")
async def sim_status() -> dict:
    """Return current simulation state."""
    return _build_status()


@router.post("/set-time")
async def sim_set_time(body: SetTimeRequest) -> dict:
    """Advance, jump, or set the simulation clock."""
    clock = get_sim_clock()
    if clock is None:
        raise HTTPException(status_code=409, detail="No simulation active")

    fields_set = sum(
        1 for v in (body.time, body.advance_minutes, body.timestamp) if v is not None
    )
    if fields_set == 0:
        raise HTTPException(
            status_code=422,
            detail="Provide one of: time, advance_minutes, timestamp",
        )
    if fields_set > 1:
        raise HTTPException(
            status_code=422,
            detail="Provide only one of: time, advance_minutes, timestamp",
        )

    if body.time is not None:
        clock.jump_to_et(body.time)
    elif body.advance_minutes is not None:
        clock.advance(minutes=body.advance_minutes)
    elif body.timestamp is not None:
        ts = datetime.fromisoformat(body.timestamp)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        clock.set_time(ts)

    return _build_status()


@router.post("/reset")
async def sim_reset() -> dict:
    """Clear positions and ledger, reset clock to market open."""
    clock = get_sim_clock()
    if clock is None:
        raise HTTPException(status_code=409, detail="No simulation active")

    # Close all open positions
    positions_cleared = 0
    store = get_position_store()
    if store:
        for pos in store.get_open_positions():
            pid = pos.get("position_id")
            if pid:
                try:
                    store.close_position(pid, exit_price=0.0, reason="sim_reset")
                    positions_cleared += 1
                except Exception:
                    logger.debug("Failed to close position %s during reset", pid)

    clock.reset()

    return {
        "reset": True,
        "time": clock.now().isoformat(),
        "positions_cleared": positions_cleared,
    }


@router.get("/timestamps")
async def sim_timestamps() -> dict:
    """Return all available timestamps as ISO strings."""
    clock = get_sim_clock()
    if clock is None:
        raise HTTPException(status_code=409, detail="No simulation active")

    return {
        "timestamps": [ts.isoformat() for ts in clock.timestamps],
        "count": len(clock.timestamps),
    }


@router.post("/load-date")
async def sim_load_date(body: LoadDateRequest) -> dict:
    """Hot-swap simulation to a different date.

    Creates a new CSVSimulationProvider for the requested date, replaces the
    old provider in the registry, resets the sim clock, and clears positions.
    """
    from datetime import date as _date_cls

    from app.core.providers.csv_simulation import CSVSimulationProvider
    from app.services.simulation_clock import init_sim_clock, reset_sim_clock

    try:
        new_date = _date_cls.fromisoformat(body.date)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid date format: {body.date}")

    # Get existing provider config
    old_provider = ProviderRegistry.get(Broker.IBKR)
    if not isinstance(old_provider, CSVSimulationProvider):
        raise HTTPException(
            status_code=409,
            detail="load-date only works in simulation mode (--sim-date)",
        )

    # Create new provider for the new date
    new_provider = CSVSimulationProvider(
        sim_date=new_date,
        tickers=old_provider.tickers,
        equities_dir=old_provider.equities_dir,
        options_dir=old_provider.options_dir,
    )
    await new_provider.connect()

    # Check we got data
    all_ts = new_provider.get_all_equity_timestamps()
    if not all_ts:
        raise HTTPException(
            status_code=404,
            detail=f"No equity data found for {body.date}",
        )

    # Swap provider
    ProviderRegistry.register(new_provider)

    # Reset clock
    reset_sim_clock()
    clock = init_sim_clock(new_date, all_ts)
    clock._tickers = old_provider.tickers

    # Clear positions
    positions_cleared = 0
    store = get_position_store()
    if store:
        for pos in store.get_open_positions():
            pid = pos.get("position_id")
            if pid:
                try:
                    store.close_position(pid, exit_price=0.0, reason="load_date_reset")
                    positions_cleared += 1
                except Exception:
                    pass

    logger.info("Loaded simulation date %s (%d timestamps, %d positions cleared)",
                body.date, len(all_ts), positions_cleared)

    return {
        **_build_status(),
        "positions_cleared": positions_cleared,
    }


@router.post("/auto-advance")
async def sim_auto_advance(body: AutoAdvanceRequest) -> dict:
    """Start or stop auto-advancing the simulation clock."""
    clock = get_sim_clock()
    if clock is None:
        raise HTTPException(status_code=409, detail="No simulation active")

    if body.enabled:
        clock.start_auto_advance(interval_sec=body.interval)
    else:
        clock.stop_auto_advance()

    return _build_status()


# ---------------------------------------------------------------------------
# Picks generation
# ---------------------------------------------------------------------------

async def _generate_picks(
    tickers: list[str],
    option_types: list[str],
    min_otm_pct: float,
    min_credit: float,
    max_loss_per_spread: float,
    spread_width: float | None,
    num_contracts: int,
    dte: list[int],
    roi_min: float,
    sort_by: str,
    limit: int,
) -> list[dict]:
    """Build candidate credit spreads from CSV option data at current sim time.

    Uses the CSVSimulationProvider via ProviderRegistry to pull option quotes,
    then constructs spread candidates directly (no pandas dependency).
    """
    clock = get_sim_clock()
    if not clock:
        return []

    provider = ProviderRegistry.get(Broker.IBKR)

    # Get current price for each ticker
    picks: list[dict] = []
    for ticker in tickers:
        quote = await provider.get_quote(ticker)
        current_price = quote.last
        if current_price <= 0:
            continue

        chain = await provider.get_option_chain(ticker)
        expirations = chain.get("expirations", [])

        # Filter expirations by DTE
        sim_date = clock.sim_date
        target_exps = []
        for exp_str in expirations:
            try:
                from datetime import date as _date_cls
                exp_date = _date_cls.fromisoformat(exp_str)
                exp_dte = (exp_date - sim_date).days
                if exp_dte in dte:
                    target_exps.append(exp_str)
            except ValueError:
                continue

        for exp in target_exps:
            for otype in option_types:
                opt_type_upper = otype.upper()
                opts = await provider.get_option_quotes(
                    ticker, exp, opt_type_upper,
                )
                if not opts:
                    continue

                # Build spreads from option quotes
                strikes = sorted(set(q["strike"] for q in opts if q["strike"] > 0))
                strike_map = {q["strike"]: q for q in opts}

                for short_strike in strikes:
                    short_q = strike_map.get(short_strike)
                    if not short_q or short_q["bid"] <= 0:
                        continue

                    # OTM check
                    if opt_type_upper == "PUT":
                        otm_pct = (current_price - short_strike) / current_price
                        if otm_pct < min_otm_pct:
                            continue
                        # Long strike is below short
                        if spread_width:
                            long_target = short_strike - spread_width
                        else:
                            # Find next strike below
                            below = [s for s in strikes if s < short_strike]
                            if not below:
                                continue
                            long_target = below[-1]
                    else:  # CALL
                        otm_pct = (short_strike - current_price) / current_price
                        if otm_pct < min_otm_pct:
                            continue
                        if spread_width:
                            long_target = short_strike + spread_width
                        else:
                            above = [s for s in strikes if s > short_strike]
                            if not above:
                                continue
                            long_target = above[0]

                    # Find closest long strike
                    long_q = strike_map.get(long_target)
                    if not long_q:
                        # Find nearest available
                        candidates = sorted(strikes, key=lambda s: abs(s - long_target))
                        for c in candidates:
                            if c != short_strike:
                                long_q = strike_map.get(c)
                                if long_q:
                                    long_target = c
                                    break
                    if not long_q:
                        continue

                    width = abs(short_strike - long_target)
                    if width <= 0:
                        continue

                    credit = short_q["bid"] - long_q["ask"]
                    if credit < min_credit:
                        continue

                    max_loss = (width - credit) * 100 * num_contracts
                    if max_loss > max_loss_per_spread:
                        continue

                    total_credit = credit * 100 * num_contracts
                    roi_pct = (credit / (width - credit)) * 100 if (width - credit) > 0 else 0

                    if roi_pct < roi_min:
                        continue

                    pick = {
                        "ticker": ticker,
                        "option_type": otype.lower(),
                        "short_strike": short_strike,
                        "long_strike": long_target,
                        "spread_width": width,
                        "expiration": exp,
                        "bid": short_q["bid"],
                        "ask": short_q["ask"],
                        "credit": round(credit, 4),
                        "max_loss": round(max_loss, 2),
                        "total_credit": round(total_credit, 2),
                        "roi_pct": round(roi_pct, 2),
                        "otm_pct": round(otm_pct * 100, 2),
                        "num_contracts": num_contracts,
                        "current_price": current_price,
                        "greeks": short_q.get("greeks", {}),
                    }
                    picks.append(pick)

    # Sort
    if sort_by == "roi":
        picks.sort(key=lambda p: p["roi_pct"], reverse=True)
    elif sort_by == "credit":
        picks.sort(key=lambda p: p["credit"], reverse=True)
    elif sort_by == "otm_pct":
        picks.sort(key=lambda p: p["otm_pct"], reverse=True)

    return picks[:limit]


async def _execute_pick(pick: dict) -> dict:
    """Execute a single pick as a credit spread trade through the standard path."""
    from app.services.trade_service import execute_trade
    from app.models import TradeRequest

    otype = pick["option_type"].upper()
    is_put = otype == "PUT"

    trade_req = TradeRequest(
        trade_type="credit_spread",
        broker="ibkr",
        symbol=pick["ticker"],
        expiration=pick["expiration"],
        option_type=otype,
        quantity=pick["num_contracts"],
        short_strike=pick["short_strike"],
        long_strike=pick["long_strike"],
    )
    result = await execute_trade(trade_req, dry_run=False)
    return {
        "pick": pick,
        "order_id": result.order_id,
        "status": result.status.value,
        "filled_price": result.filled_price,
    }


@router.post("/picks")
async def sim_picks(body: PicksRequest) -> dict:
    """Generate candidate credit spreads at the current simulation time."""
    clock = get_sim_clock()
    if clock is None:
        raise HTTPException(status_code=409, detail="No simulation active")

    tickers = body.tickers or getattr(clock, "_tickers", [])
    picks = await _generate_picks(
        tickers=tickers,
        option_types=body.option_types,
        min_otm_pct=body.min_otm_pct,
        min_credit=body.min_credit,
        max_loss_per_spread=body.max_loss_per_spread,
        spread_width=body.spread_width,
        num_contracts=body.num_contracts,
        dte=body.dte,
        roi_min=body.roi_min,
        sort_by=body.sort_by,
        limit=body.limit,
    )

    return {
        "time": clock.now().isoformat(),
        "picks": picks,
        "count": len(picks),
    }


@router.post("/execute-picks")
async def sim_execute_picks(body: PicksRequest) -> dict:
    """Generate picks and auto-execute the top N."""
    clock = get_sim_clock()
    if clock is None:
        raise HTTPException(status_code=409, detail="No simulation active")

    tickers = body.tickers or getattr(clock, "_tickers", [])
    picks = await _generate_picks(
        tickers=tickers,
        option_types=body.option_types,
        min_otm_pct=body.min_otm_pct,
        min_credit=body.min_credit,
        max_loss_per_spread=body.max_loss_per_spread,
        spread_width=body.spread_width,
        num_contracts=body.num_contracts,
        dte=body.dte,
        roi_min=body.roi_min,
        sort_by=body.sort_by,
        limit=body.limit,
    )

    top = picks[: body.top_n]
    results = []
    for pick in top:
        try:
            result = await _execute_pick(pick)
            results.append(result)
        except Exception as e:
            results.append({"pick": pick, "error": str(e)})

    return {
        "time": clock.now().isoformat(),
        "executed": len(results),
        "results": results,
        "total_picks_available": len(picks),
    }


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------


@router.post("/sweep")
async def sim_sweep(body: SweepRequest) -> dict:
    """Run the full trading day with different parameter combinations.

    Steps through every timestamp, executing picks at each step.
    Resets and repeats for each parameter combo.
    """
    clock = get_sim_clock()
    if clock is None:
        raise HTTPException(status_code=409, detail="No simulation active")

    tickers = body.tickers or getattr(clock, "_tickers", [])

    # Build cartesian product of sweep params
    param_names = list(body.sweep_params.keys())
    param_values = [body.sweep_params[k] for k in param_names]
    if not param_names:
        raise HTTPException(status_code=422, detail="sweep_params must not be empty")

    combos = list(itertools.product(*param_values))

    # Fixed params as defaults
    fixed = body.fixed_params or {}

    all_results: list[dict] = []

    for combo in combos:
        params = dict(zip(param_names, combo))

        # Reset state
        store = get_position_store()
        if store:
            for pos in store.get_open_positions():
                pid = pos.get("position_id")
                if pid:
                    try:
                        store.close_position(pid, exit_price=0.0, reason="sweep_reset")
                    except Exception:
                        pass
        clock.reset()

        # Merge params
        merged = {**fixed, **params}
        option_types = merged.get("option_types", ["put", "call"])
        if isinstance(option_types, str):
            option_types = [option_types]
        num_contracts = merged.get("num_contracts", 10)
        min_credit = merged.get("min_credit", 0.25)
        min_otm_pct = merged.get("min_otm_pct", 0.015)
        spread_width = merged.get("spread_width", None)
        max_loss = merged.get("max_loss_per_spread", 10000)
        dte_list = merged.get("dte", [0])
        entry_start_et = merged.get("entry_start_et", "09:45")
        entry_end_et = merged.get("entry_end_et", "15:00")
        top_n = merged.get("top_n", 3)

        # Parse entry window as minutes from midnight ET
        def _et_minutes(tstr: str) -> int:
            h, m = tstr.split(":")
            return int(h) * 60 + int(m)

        start_min = _et_minutes(entry_start_et)
        end_min = _et_minutes(entry_end_et)

        trades_executed = []

        # Step through all timestamps
        while True:
            sim_et = clock.now().astimezone(ET)
            et_min = sim_et.hour * 60 + sim_et.minute

            if start_min <= et_min <= end_min:
                picks = await _generate_picks(
                    tickers=tickers,
                    option_types=option_types,
                    min_otm_pct=min_otm_pct,
                    min_credit=min_credit,
                    max_loss_per_spread=max_loss,
                    spread_width=spread_width,
                    num_contracts=num_contracts,
                    dte=dte_list,
                    roi_min=0,
                    sort_by="roi",
                    limit=top_n,
                )
                for pick in picks[:top_n]:
                    try:
                        result = await _execute_pick(pick)
                        trades_executed.append(result)
                    except Exception:
                        pass

            if clock.advance() is None:
                break

        # Compute P&L at end of day from positions
        net_pnl = 0.0
        wins = 0
        total_risk = 0.0
        if store:
            for pos in store.get_open_positions():
                # Mark to market at last bar — for 0DTE credit spreads,
                # if expired OTM, full credit is profit
                entry_price = pos.get("entry_price", 0)
                if entry_price:
                    credit = abs(entry_price)
                    qty = pos.get("quantity", 1)
                    net_pnl += credit * 100 * qty
                    wins += 1
                    w = pos.get("spread_width") or merged.get("spread_width", 25)
                    total_risk += w * 100 * qty

        combo_result = {
            "params": params,
            "trades": len(trades_executed),
            "wins": wins,
            "win_rate": wins / max(len(trades_executed), 1),
            "net_pnl": round(net_pnl, 2),
            "total_risk": round(total_risk, 2),
            "roi_pct": round(net_pnl / max(total_risk, 1) * 100, 2),
        }
        all_results.append(combo_result)

    # Sort by net P&L
    all_results.sort(key=lambda r: r["net_pnl"], reverse=True)

    return {
        "combinations": len(all_results),
        "results": all_results,
    }
