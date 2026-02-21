import argparse
import asyncio
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Project Path Setup
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.stock_db import get_stock_db, StockDBBase
from common.logging_utils import get_logger


@dataclass
class MarketContext:
    ticker: str
    current_price: Optional[float]
    prev_close: Optional[float]
    hourly_volatility: Optional[float]
    latest_hour_open: Optional[float]
    latest_hour_close: Optional[float]
    current_price_move_pct: Optional[float]  # (current_price - prev_close) / prev_close
    price_move_percentile: Optional[float]  # Percentile of current move in historical distribution
    dow_percentile: Optional[float]  # Percentile of current move in same day-of-week distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print top option combinations for iron condors or credit/debit spreads "
            "using latest options CSV and QuestDB market data."
        )
    )
    parser.add_argument("--csv-path", required=True, help="Path to options CSV snapshot")
    parser.add_argument("--underlying-ticker", help="Override underlying ticker for all rows")
    parser.add_argument("--db-path", dest="db_path", help="QuestDB connection string")
    parser.add_argument("--db-type", default="questdb", help="Database type (default: questdb)")
    parser.add_argument("--no-cache", action="store_true", help="Disable Redis cache")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    parser.add_argument("--mode", choices=["iron_condor", "credit_spread"], default="iron_condor")
    parser.add_argument("--direction", choices=["sell", "buy"], default="sell")
    parser.add_argument("--spread-type", choices=["call", "put", "both"], default="both")
    parser.add_argument("--top-n", type=int, default=5)

    parser.add_argument(
        "--min-dte",
        type=int,
        default=0,
        help="Minimum Days To Expiration (DTE). Filters out options expiring sooner than this.",
    )
    parser.add_argument(
        "--max-dte",
        type=int,
        default=30,
        help="Maximum Days To Expiration (DTE). Filters out options expiring later than this.",
    )
    parser.add_argument(
        "--min-width",
        type=float,
        default=5.0,
        help="Minimum spread width (strike price difference between long and short legs).",
    )
    parser.add_argument(
        "--max-width",
        type=float,
        default=200.0,
        help="Maximum spread width (strike price difference between long and short legs).",
    )
    parser.add_argument(
        "--max-leg-candidates",
        type=int,
        default=60,
        help="Maximum number of option candidates per side (put/call) to consider. Limits computation.",
    )

    parser.add_argument(
        "--min-distance-pct",
        type=float,
        default=0.03,
        help=(
            "Minimum percent distance from CURRENT price. "
            "Calls must be >= current_price * (1 + min_distance_pct). "
            "Puts must be <= current_price * (1 - min_distance_pct)."
        ),
    )
    parser.add_argument(
        "--max-distance-pct",
        type=float,
        default=None,
        help=(
            "Maximum percent distance from CURRENT price. "
            "Calls must be <= current_price * (1 + max_distance_pct). "
            "Puts must be >= current_price * (1 - max_distance_pct)."
        ),
    )

    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help=(
            "Number of days to look back for historical price move analysis. "
            "Used with --price-move-percentile-min/max to filter based on percentile of historical moves."
        ),
    )
    parser.add_argument(
        "--price-move-percentile-min",
        type=float,
        default=None,
        help=(
            "Minimum percentile (0.0-1.0) for current price move vs historical moves. "
            "Requires --lookback-days. Filters options if today's move is below this percentile of historical moves."
        ),
    )
    parser.add_argument(
        "--price-move-percentile-max",
        type=float,
        default=None,
        help=(
            "Maximum percentile (0.0-1.0) for current price move vs historical moves. "
            "Requires --lookback-days. Filters options if today's move is above this percentile of historical moves."
        ),
    )
    parser.add_argument(
        "--dow-percentile-min",
        type=float,
        default=None,
        help=(
            "Minimum percentile (0.0-1.0) for current price move vs same day-of-week historical moves. "
            "Filters options if today's move (vs prev close) is below this percentile of same weekday moves."
        ),
    )
    parser.add_argument(
        "--dow-percentile-max",
        type=float,
        default=None,
        help=(
            "Maximum percentile (0.0-1.0) for current price move vs same day-of-week historical moves. "
            "Filters options if today's move (vs prev close) is above this percentile of same weekday moves."
        ),
    )

    parser.add_argument(
        "--hourly-lookback",
        type=int,
        default=6,
        help=(
            "Number of hours to look back for hourly volatility calculation. "
            "Volatility is computed as average of (high-low)/close for each hour in the lookback period."
        ),
    )
    parser.add_argument(
        "--value-weight",
        type=float,
        default=1.0,
        help="Scoring weight for maximum profit value (higher = prioritize higher profit).",
    )
    parser.add_argument(
        "--rr-weight",
        type=float,
        default=1.0,
        help="Scoring weight for reward/risk ratio (higher = prioritize better risk/reward).",
    )
    parser.add_argument(
        "--distance-weight",
        type=float,
        default=1.0,
        help="Scoring weight for distance from current price (higher = prioritize further OTM).",
    )
    parser.add_argument(
        "--vol-weight",
        type=float,
        default=1.0,
        help="Penalty weight for hourly volatility (higher = penalize volatile stocks more).",
    )
    parser.add_argument(
        "--use-mid",
        action="store_true",
        help=(
            "Use mid price (average of bid and ask) for option pricing. "
            "By default, uses bid for sells and ask for buys (more realistic execution)."
        ),
    )

    return parser.parse_args()


def _infer_underlying(option_ticker: str) -> Optional[str]:
    if not isinstance(option_ticker, str) or not option_ticker:
        return None
    symbol = option_ticker
    if symbol.startswith("O:"):
        symbol = symbol[2:]
    match = re.match(r"([A-Z]+)", symbol)
    return match.group(1) if match else None


def _normalize_option_type(option_type: str) -> Optional[str]:
    if not isinstance(option_type, str):
        return None
    value = option_type.strip().lower()
    if value in ("call", "c"):
        return "CALL"
    if value in ("put", "p"):
        return "PUT"
    return value.upper()


def load_options_csv(path: str, override_underlying: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Preserve original file row order (0-based, with header as row 0, so first data row is 1)
    df["_file_row"] = range(1, len(df) + 1)

    if "option_ticker" not in df.columns:
        if "ticker" in df.columns:
            df = df.rename(columns={"ticker": "option_ticker"})
        else:
            raise ValueError("CSV must include 'option_ticker' or 'ticker' column.")

    if "timestamp" not in df.columns:
        raise ValueError("CSV must include 'timestamp' column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    if "option_type" in df.columns:
        df["option_type"] = df["option_type"].apply(_normalize_option_type)
    elif "type" in df.columns:
        df["option_type"] = df["type"].apply(_normalize_option_type)
    else:
        raise ValueError("CSV must include 'type' or 'option_type' column.")

    if "strike_price" not in df.columns:
        if "strike" in df.columns:
            df = df.rename(columns={"strike": "strike_price"})
        else:
            raise ValueError("CSV must include 'strike' or 'strike_price' column.")

    if "expiration_date" not in df.columns:
        if "expiration" in df.columns:
            df = df.rename(columns={"expiration": "expiration_date"})
        else:
            raise ValueError("CSV must include 'expiration' or 'expiration_date' column.")

    df["expiration_date"] = pd.to_datetime(df["expiration_date"], errors="coerce")
    df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")
    df["bid"] = pd.to_numeric(df.get("bid"), errors="coerce")
    df["ask"] = pd.to_numeric(df.get("ask"), errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce")

    if override_underlying:
        df["underlying"] = override_underlying
    elif "underlying" not in df.columns:
        df["underlying"] = df["option_ticker"].apply(_infer_underlying)

    df = df.dropna(subset=["option_ticker", "option_type", "strike_price", "expiration_date", "underlying"])

    # Sort by timestamp (ascending), then by file row (descending) to prefer rows closer to bottom of file
    # This ensures if timestamps are equal, we use the row that appears later in the CSV
    df = df.sort_values(["timestamp", "_file_row"], ascending=[True, False])
    # Take the last row per option_ticker (latest timestamp, or if equal, row closest to bottom of file)
    latest_rows = (
        df.groupby("option_ticker", as_index=False)
        .tail(1)
        .drop(columns=["_file_row"])
        .reset_index(drop=True)
    )
    return latest_rows


async def calculate_price_move_percentile(
    db: StockDBBase, ticker: str, current_price: float, prev_close: float, lookback_days: int
) -> Optional[float]:
    """Calculate percentile of current price move in historical distribution.
    
    Returns percentile (0.0-1.0) of current move vs historical moves over lookback_days.
    """
    if prev_close is None or prev_close <= 0:
        return None
    
    current_move = (current_price - prev_close) / prev_close
    
    # Get daily data for lookback period
    now_utc = datetime.now(timezone.utc)
    start_date = (now_utc - timedelta(days=lookback_days + 10)).strftime("%Y-%m-%d")  # Extra buffer
    end_date = now_utc.strftime("%Y-%m-%d")
    
    daily_df = await db.get_stock_data(ticker, start_date=start_date, end_date=end_date, interval="daily")
    
    if daily_df.empty or "close" not in daily_df.columns:
        return None
    
    # Calculate daily moves: (close_today - close_yesterday) / close_yesterday
    daily_df = daily_df.sort_index()
    daily_df["prev_close"] = daily_df["close"].shift(1)
    daily_df = daily_df.dropna(subset=["close", "prev_close"])
    daily_df["move_pct"] = (daily_df["close"] - daily_df["prev_close"]) / daily_df["prev_close"]
    
    moves = daily_df["move_pct"].dropna().tolist()
    if len(moves) < 5:  # Need at least 5 data points
        return None
    
    # Calculate percentile
    moves_sorted = sorted(moves)
    percentile = sum(1 for m in moves_sorted if m <= current_move) / len(moves_sorted)
    return percentile


async def calculate_dow_percentile(
    db: StockDBBase, ticker: str, current_price: float, prev_close: float, min_days: int = 20
) -> Optional[float]:
    """Calculate percentile of current price move vs same day-of-week historical moves.
    
    Returns percentile (0.0-1.0) of current move vs historical moves on the same weekday.
    """
    if prev_close is None or prev_close <= 0:
        return None
    
    current_move = (current_price - prev_close) / prev_close
    
    # Get current day of week (0=Monday, 6=Sunday)
    now_utc = datetime.now(timezone.utc)
    current_dow = now_utc.weekday()
    
    # Get enough daily data to have multiple instances of this weekday
    # Need at least min_days worth of data, so look back further
    lookback_days = max(min_days * 7, 60)  # At least 60 days to get multiple instances
    start_date = (now_utc - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end_date = now_utc.strftime("%Y-%m-%d")
    
    daily_df = await db.get_stock_data(ticker, start_date=start_date, end_date=end_date, interval="daily")
    
    if daily_df.empty or "close" not in daily_df.columns:
        return None
    
    # Calculate daily moves
    daily_df = daily_df.sort_index()
    daily_df["prev_close"] = daily_df["close"].shift(1)
    daily_df = daily_df.dropna(subset=["close", "prev_close"])
    daily_df["move_pct"] = (daily_df["close"] - daily_df["prev_close"]) / daily_df["prev_close"]
    
    # Get day of week for each date
    if isinstance(daily_df.index, pd.DatetimeIndex):
        daily_df["dow"] = daily_df.index.weekday
    else:
        # If index is not datetime, try to parse a date column
        return None
    
    # Filter to same day of week
    same_dow_moves = daily_df[daily_df["dow"] == current_dow]["move_pct"].dropna().tolist()
    
    if len(same_dow_moves) < 3:  # Need at least 3 instances of same weekday
        return None
    
    # Calculate percentile
    same_dow_sorted = sorted(same_dow_moves)
    percentile = sum(1 for m in same_dow_sorted if m <= current_move) / len(same_dow_sorted)
    return percentile


async def get_market_context(
    db: StockDBBase,
    ticker: str,
    hourly_lookback: int,
    lookback_days: Optional[int] = None,
) -> MarketContext:
    current_price = await db.get_latest_price(ticker, use_market_time=True)
    prev_close_map = await db.get_previous_close_prices([ticker])
    prev_close = prev_close_map.get(ticker)

    now_utc = datetime.now(timezone.utc)
    start_dt = now_utc - timedelta(hours=hourly_lookback)
    hourly_df = await db.get_stock_data(
        ticker,
        start_date=start_dt.isoformat(),
        end_date=now_utc.isoformat(),
        interval="hourly",
    )

    hourly_vol = None
    latest_hour_open = None
    latest_hour_close = None
    if not hourly_df.empty and {"high", "low", "close"}.issubset(hourly_df.columns):
        hourly_df = hourly_df.copy()
        hourly_df["close"] = pd.to_numeric(hourly_df["close"], errors="coerce")
        hourly_df["high"] = pd.to_numeric(hourly_df["high"], errors="coerce")
        hourly_df["low"] = pd.to_numeric(hourly_df["low"], errors="coerce")
        hourly_df = hourly_df.dropna(subset=["high", "low", "close"])
        if not hourly_df.empty:
            hourly_df["range_pct"] = (hourly_df["high"] - hourly_df["low"]) / hourly_df["close"]
            hourly_vol = float(hourly_df["range_pct"].tail(hourly_lookback).mean())

            latest_row = hourly_df.iloc[-1]
            latest_hour_open = float(latest_row["open"]) if "open" in hourly_df.columns else None
            latest_hour_close = float(latest_row["close"])

    # Calculate current price move
    current_price_move_pct = None
    if current_price is not None and prev_close is not None and prev_close > 0:
        current_price_move_pct = (current_price - prev_close) / prev_close

    # Calculate percentiles if requested
    price_move_percentile = None
    if lookback_days and current_price is not None and prev_close is not None:
        price_move_percentile = await calculate_price_move_percentile(
            db, ticker, current_price, prev_close, lookback_days
        )

    dow_percentile = None
    if current_price is not None and prev_close is not None:
        dow_percentile = await calculate_dow_percentile(db, ticker, current_price, prev_close)

    return MarketContext(
        ticker=ticker,
        current_price=current_price,
        prev_close=prev_close,
        hourly_volatility=hourly_vol,
        latest_hour_open=latest_hour_open,
        latest_hour_close=latest_hour_close,
        current_price_move_pct=current_price_move_pct,
        price_move_percentile=price_move_percentile,
        dow_percentile=dow_percentile,
    )


def _option_price(row: pd.Series, action: str, use_mid: bool) -> Optional[float]:
    bid = row.get("bid")
    ask = row.get("ask")
    if use_mid and pd.notna(bid) and pd.notna(ask):
        return float((bid + ask) / 2.0)
    if action == "sell":
        return float(bid) if pd.notna(bid) else None
    return float(ask) if pd.notna(ask) else None


def _filter_options_by_distance(
    df: pd.DataFrame,
    current_price: Optional[float],
    prev_close: Optional[float],  # Not used, kept for API compatibility
    min_distance_pct: Optional[float],
    max_distance_pct: Optional[float] = None,
) -> pd.DataFrame:
    """Filter options by distance from current price.
    
    Calls must be at least min_distance_pct above current price.
    Puts must be at least min_distance_pct below current price.
    If max_distance_pct is provided, also enforces maximum distance.
    """
    if current_price is None:
        return df

    def _distance_ok(row: pd.Series) -> bool:
        strike = row["strike_price"]
        if row["option_type"] == "CALL":
            min_ok = True
            max_ok = True
            if min_distance_pct is not None:
                min_ok = strike >= current_price * (1 + min_distance_pct)
            if max_distance_pct is not None:
                max_ok = strike <= current_price * (1 + max_distance_pct)
            return min_ok and max_ok
        else:  # PUT
            min_ok = True
            max_ok = True
            if min_distance_pct is not None:
                min_ok = strike <= current_price * (1 - min_distance_pct)
            if max_distance_pct is not None:
                max_ok = strike >= current_price * (1 - max_distance_pct)
            return min_ok and max_ok

    return df[df.apply(_distance_ok, axis=1)]


def _filter_by_price_move_percentile(
    df: pd.DataFrame,
    price_move_percentile: Optional[float],
    percentile_min: Optional[float],
    percentile_max: Optional[float],
) -> pd.DataFrame:
    """Filter options based on percentile of current price move vs historical moves.
    
    Only keeps options if current move percentile is within [percentile_min, percentile_max].
    """
    if price_move_percentile is None:
        return df
    
    if percentile_min is None and percentile_max is None:
        return df
    
    def _percentile_ok(_row: pd.Series) -> bool:
        if percentile_min is not None and price_move_percentile < percentile_min:
            return False
        if percentile_max is not None and price_move_percentile > percentile_max:
            return False
        return True
    
    return df[df.apply(_percentile_ok, axis=1)]


def _filter_by_dow_percentile(
    df: pd.DataFrame,
    dow_percentile: Optional[float],
    percentile_min: Optional[float],
    percentile_max: Optional[float],
) -> pd.DataFrame:
    """Filter options based on percentile of current price move vs same day-of-week historical moves.
    
    Only keeps options if current move percentile (vs same weekday) is within [percentile_min, percentile_max].
    """
    if dow_percentile is None:
        return df
    
    if percentile_min is None and percentile_max is None:
        return df
    
    def _percentile_ok(_row: pd.Series) -> bool:
        if percentile_min is not None and dow_percentile < percentile_min:
            return False
        if percentile_max is not None and dow_percentile > percentile_max:
            return False
        return True
    
    return df[df.apply(_percentile_ok, axis=1)]


def _filter_by_dte(df: pd.DataFrame, min_dte: int, max_dte: int, ref_ts: datetime) -> pd.DataFrame:
    df = df.copy()
    df["dte"] = (df["expiration_date"] - ref_ts).dt.days
    return df[(df["dte"] >= min_dte) & (df["dte"] <= max_dte)]


def _limit_candidates(df: pd.DataFrame, max_candidates: int, current_price: Optional[float]) -> pd.DataFrame:
    if current_price is None or df.empty:
        return df.head(max_candidates)
    df = df.copy()
    df["distance"] = (df["strike_price"] - current_price).abs()
    return df.sort_values("distance").head(max_candidates)


def _score_trade(
    max_profit: float,
    max_loss: float,
    distance_pct: float,
    hourly_vol: Optional[float],
    value_weight: float,
    rr_weight: float,
    distance_weight: float,
    vol_weight: float,
) -> float:
    reward_risk = (max_profit / max_loss) if max_loss > 0 else 0.0
    vol_penalty = hourly_vol if hourly_vol is not None else 0.0
    return (
        value_weight * max_profit
        + rr_weight * reward_risk
        + distance_weight * distance_pct
        - vol_weight * vol_penalty
    )


def _format_money(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def _format_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def build_credit_spreads(
    options_df: pd.DataFrame,
    current_price: Optional[float],
    direction: str,
    spread_type: str,
    min_width: float,
    max_width: float,
    use_mid: bool,
    max_candidates: int,
) -> List[Dict[str, Any]]:
    results = []
    filtered = options_df

    if spread_type in ("call", "both"):
        calls = filtered[filtered["option_type"] == "CALL"].sort_values("strike_price")
        calls = _limit_candidates(calls, max_candidates, current_price)
        for i in range(len(calls)):
            short_candidate = calls.iloc[i]
            for j in range(i + 1, len(calls)):
                long_candidate = calls.iloc[j]
                short_leg = short_candidate if direction == "sell" else long_candidate
                long_leg = long_candidate if direction == "sell" else short_candidate

                width = abs(long_leg["strike_price"] - short_leg["strike_price"])
                if width < min_width or width > max_width:
                    continue
                short_price = _option_price(short_leg, "sell", use_mid)
                long_price = _option_price(long_leg, "buy", use_mid)
                if short_price is None or long_price is None:
                    continue
                net_credit = short_price - long_price
                if direction == "sell" and net_credit <= 0:
                    continue
                net_debit = long_price - short_price
                trade = {
                    "structure": "CALL_SPREAD",
                    "short_leg": short_leg,
                    "long_leg": long_leg,
                    "width": width,
                    "net_credit": net_credit if direction == "sell" else None,
                    "net_debit": net_debit if direction == "buy" else None,
                }
                results.append(trade)

    if spread_type in ("put", "both"):
        puts = filtered[filtered["option_type"] == "PUT"].sort_values("strike_price")
        puts = _limit_candidates(puts, max_candidates, current_price)
        for i in range(len(puts)):
            long_candidate = puts.iloc[i]
            for j in range(i + 1, len(puts)):
                short_candidate = puts.iloc[j]
                short_leg = short_candidate if direction == "sell" else long_candidate
                long_leg = long_candidate if direction == "sell" else short_candidate

                width = abs(long_leg["strike_price"] - short_leg["strike_price"])
                if width < min_width or width > max_width:
                    continue
                short_price = _option_price(short_leg, "sell", use_mid)
                long_price = _option_price(long_leg, "buy", use_mid)
                if short_price is None or long_price is None:
                    continue
                net_credit = short_price - long_price
                if direction == "sell" and net_credit <= 0:
                    continue
                net_debit = long_price - short_price
                trade = {
                    "structure": "PUT_SPREAD",
                    "short_leg": short_leg,
                    "long_leg": long_leg,
                    "width": width,
                    "net_credit": net_credit if direction == "sell" else None,
                    "net_debit": net_debit if direction == "buy" else None,
                }
                results.append(trade)

    return results


def build_iron_condors(
    options_df: pd.DataFrame,
    current_price: Optional[float],
    direction: str,
    min_width: float,
    max_width: float,
    use_mid: bool,
    max_candidates: int,
) -> List[Dict[str, Any]]:
    results = []
    filtered = options_df
    puts = filtered[filtered["option_type"] == "PUT"].sort_values("strike_price")
    calls = filtered[filtered["option_type"] == "CALL"].sort_values("strike_price")

    puts = _limit_candidates(puts, max_candidates, current_price)
    calls = _limit_candidates(calls, max_candidates, current_price)

    for i in range(len(puts)):
        long_put_candidate = puts.iloc[i]
        for j in range(i + 1, len(puts)):
            short_put_candidate = puts.iloc[j]
            short_put = short_put_candidate if direction == "sell" else long_put_candidate
            long_put = long_put_candidate if direction == "sell" else short_put_candidate

            put_width = abs(long_put["strike_price"] - short_put["strike_price"])
            if put_width < min_width or put_width > max_width:
                continue

            for k in range(len(calls)):
                short_call_candidate = calls.iloc[k]
                for l in range(k + 1, len(calls)):
                    long_call_candidate = calls.iloc[l]
                    short_call = short_call_candidate if direction == "sell" else long_call_candidate
                    long_call = long_call_candidate if direction == "sell" else short_call_candidate

                    call_width = abs(long_call["strike_price"] - short_call["strike_price"])
                    if call_width < min_width or call_width > max_width:
                        continue

                    if not (
                        long_put["strike_price"]
                        < short_put["strike_price"]
                        < short_call["strike_price"]
                        < long_call["strike_price"]
                    ):
                        continue

                    short_put_price = _option_price(short_put, "sell", use_mid)
                    long_put_price = _option_price(long_put, "buy", use_mid)
                    short_call_price = _option_price(short_call, "sell", use_mid)
                    long_call_price = _option_price(long_call, "buy", use_mid)

                    if None in (short_put_price, long_put_price, short_call_price, long_call_price):
                        continue

                    net_credit = (short_put_price + short_call_price) - (long_put_price + long_call_price)
                    if direction == "sell" and net_credit <= 0:
                        continue

                    net_debit = (long_put_price + long_call_price) - (short_put_price + short_call_price)
                    trade = {
                        "structure": "IRON_CONDOR",
                        "short_put": short_put,
                        "long_put": long_put,
                        "short_call": short_call,
                        "long_call": long_call,
                        "put_width": put_width,
                        "call_width": call_width,
                        "net_credit": net_credit if direction == "sell" else None,
                        "net_debit": net_debit if direction == "buy" else None,
                    }
                    results.append(trade)

    return results


def rank_trades(
    trades: List[Dict[str, Any]],
    mode: str,
    direction: str,
    current_price: Optional[float],
    hourly_vol: Optional[float],
    weights: Dict[str, float],
) -> List[Dict[str, Any]]:
    ranked = []
    for trade in trades:
        if mode == "credit_spread":
            short_leg = trade["short_leg"]
            long_leg = trade["long_leg"]
            width = trade["width"]

            max_profit = trade["net_credit"] if direction == "sell" else width - trade["net_debit"]
            max_loss = width - trade["net_credit"] if direction == "sell" else trade["net_debit"]

            risk_leg = short_leg if direction == "sell" else long_leg
            if current_price:
                if trade["structure"].startswith("CALL"):
                    distance_pct = max((risk_leg["strike_price"] - current_price) / current_price, 0.0)
                else:
                    distance_pct = max((current_price - risk_leg["strike_price"]) / current_price, 0.0)
            else:
                distance_pct = 0.0

        else:
            short_put = trade["short_put"]
            long_put = trade["long_put"]
            short_call = trade["short_call"]
            long_call = trade["long_call"]
            max_width = max(trade["put_width"], trade["call_width"])

            max_profit = trade["net_credit"] if direction == "sell" else max_width - trade["net_debit"]
            max_loss = max_width - trade["net_credit"] if direction == "sell" else trade["net_debit"]

            if current_price:
                if direction == "sell":
                    dist_put = (current_price - short_put["strike_price"]) / current_price
                    dist_call = (short_call["strike_price"] - current_price) / current_price
                else:
                    dist_put = (current_price - long_put["strike_price"]) / current_price
                    dist_call = (long_call["strike_price"] - current_price) / current_price
                distance_pct = max(min(dist_put, dist_call), 0.0)
            else:
                distance_pct = 0.0

        if max_profit is None or max_loss is None or max_loss <= 0:
            continue
        score = _score_trade(
            max_profit=max_profit,
            max_loss=max_loss,
            distance_pct=distance_pct,
            hourly_vol=hourly_vol,
            value_weight=weights["value"],
            rr_weight=weights["rr"],
            distance_weight=weights["distance"],
            vol_weight=weights["vol"],
        )
        trade["max_profit"] = max_profit
        trade["max_loss"] = max_loss
        trade["distance_pct"] = distance_pct
        trade["score"] = score
        ranked.append(trade)

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def print_trades(
    ticker: str,
    ctx: MarketContext,
    mode: str,
    direction: str,
    trades: List[Dict[str, Any]],
    top_n: int,
) -> None:
    print("=" * 80)
    print(f"{ticker} | mode={mode} | direction={direction}")
    move_info = ""
    if ctx.current_price_move_pct is not None:
        move_info = f"Move={_format_pct(ctx.current_price_move_pct)} "
    if ctx.price_move_percentile is not None:
        move_info += f"MovePctl={ctx.price_move_percentile:.3f} "
    if ctx.dow_percentile is not None:
        move_info += f"DoWPctl={ctx.dow_percentile:.3f} "
    print(
        "Current="
        f"{_format_money(ctx.current_price)} "
        f"PrevClose={_format_money(ctx.prev_close)} "
        f"HourOpen={_format_money(ctx.latest_hour_open)} "
        f"HourClose={_format_money(ctx.latest_hour_close)} "
        f"HourlyVol={_format_pct(ctx.hourly_volatility)}"
    )
    if move_info:
        print(move_info.strip())
    print("-" * 80)

    if not trades:
        print("No trades matched the filters.")
        return

    for idx, trade in enumerate(trades[:top_n], start=1):
        if mode == "credit_spread":
            short_leg = trade["short_leg"]
            long_leg = trade["long_leg"]
            price = trade["net_credit"] if direction == "sell" else trade["net_debit"]
            print(
                f"{idx}. {trade['structure']} "
                f"{short_leg['strike_price']} / {long_leg['strike_price']} "
                f"exp={short_leg['expiration_date'].date()} "
                f"price={_format_money(price)} "
                f"maxP={_format_money(trade['max_profit'])} "
                f"maxL={_format_money(trade['max_loss'])} "
                f"dist={_format_pct(trade['distance_pct'])} "
                f"score={trade['score']:.4f}"
            )
        else:
            price = trade["net_credit"] if direction == "sell" else trade["net_debit"]
            print(
                f"{idx}. IRON_CONDOR "
                f"{trade['long_put']['strike_price']}<"
                f"{trade['short_put']['strike_price']}<"
                f"{trade['short_call']['strike_price']}<"
                f"{trade['long_call']['strike_price']} "
                f"exp={trade['short_call']['expiration_date'].date()} "
                f"price={_format_money(price)} "
                f"maxP={_format_money(trade['max_profit'])} "
                f"maxL={_format_money(trade['max_loss'])} "
                f"dist={_format_pct(trade['distance_pct'])} "
                f"score={trade['score']:.4f}"
            )


async def run(args: argparse.Namespace) -> None:
    logger = get_logger("options_combo_picker", level=args.log_level)

    # Validate percentile arguments
    if (args.price_move_percentile_min is not None or args.price_move_percentile_max is not None) and args.lookback_days is None:
        logger.error("--price-move-percentile-min/max requires --lookback-days")
        return
    
    if args.price_move_percentile_min is not None and (args.price_move_percentile_min < 0 or args.price_move_percentile_min > 1):
        logger.error("--price-move-percentile-min must be between 0.0 and 1.0")
        return
    
    if args.price_move_percentile_max is not None and (args.price_move_percentile_max < 0 or args.price_move_percentile_max > 1):
        logger.error("--price-move-percentile-max must be between 0.0 and 1.0")
        return
    
    if args.dow_percentile_min is not None and (args.dow_percentile_min < 0 or args.dow_percentile_min > 1):
        logger.error("--dow-percentile-min must be between 0.0 and 1.0")
        return
    
    if args.dow_percentile_max is not None and (args.dow_percentile_max < 0 or args.dow_percentile_max > 1):
        logger.error("--dow-percentile-max must be between 0.0 and 1.0")
        return

    db_config = args.db_path
    if not db_config:
        db_config = os.getenv("QUESTDB_URL") or os.getenv("DB_PATH")
        if not db_config:
            db_config = "questdb://user:password@localhost:9009/stock_data"

    enable_cache = not args.no_cache
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0") if enable_cache else None

    options_df = load_options_csv(args.csv_path, args.underlying_ticker)
    if options_df.empty:
        logger.warning("No options data found in CSV after normalization.")
        return

    tickers = sorted(set(options_df["underlying"].dropna().unique().tolist()))
    if not tickers:
        logger.warning("No underlying tickers inferred. Provide --underlying-ticker.")
        return

    weights = {
        "value": args.value_weight,
        "rr": args.rr_weight,
        "distance": args.distance_weight,
        "vol": args.vol_weight,
    }

    db = get_stock_db(
        args.db_type,
        db_config=db_config,
        log_level=args.log_level,
        enable_cache=enable_cache,
        redis_url=redis_url,
    )

    async with db:
        for ticker in tickers:
            ctx = await get_market_context(
                db, ticker, args.hourly_lookback, lookback_days=args.lookback_days
            )

            ticker_df = options_df[options_df["underlying"] == ticker].copy()
            if ticker_df.empty:
                continue

            ref_ts = ticker_df["timestamp"].max()
            ticker_df = _filter_by_dte(ticker_df, args.min_dte, args.max_dte, ref_ts)
            
            # Apply distance filtering
            ticker_df = _filter_options_by_distance(
                ticker_df,
                ctx.current_price,
                ctx.prev_close,
                args.min_distance_pct,
                args.max_distance_pct,
            )
            
            # Apply percentile-based filtering
            if args.lookback_days:
                ticker_df = _filter_by_price_move_percentile(
                    ticker_df,
                    ctx.price_move_percentile,
                    args.price_move_percentile_min,
                    args.price_move_percentile_max,
                )
            
            if args.dow_percentile_min is not None or args.dow_percentile_max is not None:
                ticker_df = _filter_by_dow_percentile(
                    ticker_df,
                    ctx.dow_percentile,
                    args.dow_percentile_min,
                    args.dow_percentile_max,
                )
            
            ticker_df = ticker_df.dropna(subset=["bid", "ask"])

            if ticker_df.empty:
                print_trades(ticker, ctx, args.mode, args.direction, [], args.top_n)
                continue

            if args.mode == "credit_spread":
                trades = build_credit_spreads(
                    ticker_df,
                    ctx.current_price,
                    args.direction,
                    args.spread_type,
                    args.min_width,
                    args.max_width,
                    args.use_mid,
                    args.max_leg_candidates,
                )
            else:
                trades = build_iron_condors(
                    ticker_df,
                    ctx.current_price,
                    args.direction,
                    args.min_width,
                    args.max_width,
                    args.use_mid,
                    args.max_leg_candidates,
                )

            ranked = rank_trades(
                trades,
                args.mode,
                args.direction,
                ctx.current_price,
                ctx.hourly_volatility,
                weights,
            )

            print_trades(ticker, ctx, args.mode, args.direction, ranked, args.top_n)


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
