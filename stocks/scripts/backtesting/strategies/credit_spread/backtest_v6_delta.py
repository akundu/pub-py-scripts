"""BacktestV6DeltaStrategy -- delta-based credit spread selection across DTEs.

Instead of percentile-based strike selection (V3-V5), this strategy uses
option delta to place short strikes. Target delta (e.g., 0.10 = 10-delta)
directly controls how far OTM the short strike is.

Key differences from V5:
  1. Strike selection by delta -- no percentile signal needed.
  2. Multi-DTE native -- evaluates multiple DTEs per interval and picks best.
  3. Delta-aware scoring -- ranks candidates by delta-adjusted credit/risk.
  4. Per-transaction and daily budget limits.

Inherits V4 infrastructure: multi-ticker evaluation, liquidity scoring,
constraint checking, exit rules, rolling.
"""

import math
from datetime import date, time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .base_credit_spread import BaseCreditSpreadStrategy
from ..base import DayContext
from ..registry import BacktestStrategyRegistry
from ...constraints.base import ConstraintContext
from ...constraints.exit_rules.smart_roll_exit import RollingConfig, SmartRollExit
from ...instruments.base import InstrumentPosition

from scripts.credit_spread_utils.delta_utils import (
    calculate_bs_delta,
    calculate_delta_for_option,
    DeltaFilterConfig,
    get_vix1d_at_timestamp,
)


def _snap_options_to_time(options_df: pd.DataFrame, target_ts) -> pd.DataFrame:
    """Snap multi-timestamp options data to the closest timestamp per strike.

    Options CSVs contain multiple snapshots per strike (e.g., every 5 min).
    This picks the snapshot closest to target_ts for each (strike, type) pair.
    """
    if options_df.empty or "timestamp" not in options_df.columns:
        return options_df

    df = options_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    target = pd.to_datetime(target_ts, utc=True) if not hasattr(target_ts, 'tzinfo') or target_ts.tzinfo is None else pd.to_datetime(target_ts)

    # Find closest timestamp overall
    unique_ts = df["timestamp"].unique()
    if len(unique_ts) <= 1:
        return df

    diffs = abs(pd.to_datetime(unique_ts) - target)
    closest_ts = unique_ts[diffs.argmin()]

    # Filter to that timestamp
    snapped = df[df["timestamp"] == closest_ts]
    if snapped.empty:
        # Fallback: just deduplicate
        return df.drop_duplicates(subset=["strike", "type"], keep="first")

    return snapped


def find_strike_by_delta(
    options_df: pd.DataFrame,
    option_type: str,
    target_delta: float,
    underlying_price: float,
    dte_days: int,
    default_iv: float = 0.20,
    vix1d_value: Optional[float] = None,
) -> Optional[float]:
    """Find the strike closest to a target absolute delta.

    Args:
        options_df: Options chain (may contain multiple timestamps per strike).
        option_type: 'put' or 'call'.
        target_delta: Target absolute delta (e.g., 0.10 for 10-delta).
        underlying_price: Current underlying price.
        dte_days: Days to expiration (for BS calculation).
        default_iv: Fallback IV if option data lacks it.
        vix1d_value: VIX1D value (decimal) for IV fallback.

    Returns:
        Strike price closest to the target delta, or None.
    """
    if options_df.empty:
        return None

    typed = options_df[options_df["type"].str.upper() == option_type.upper()]
    if typed.empty:
        return None

    # Deduplicate to unique strikes — use the row with best (highest) bid
    if "bid" in typed.columns:
        typed = typed.sort_values("bid", ascending=False)
    typed = typed.drop_duplicates(subset=["strike"], keep="first")

    T = max(dte_days, 1) / 365.0
    best_strike = None
    best_diff = float("inf")

    for _, row in typed.iterrows():
        strike = float(row["strike"])
        if strike <= 0:
            continue

        # Get IV: option IV > VIX1D > default
        iv = default_iv
        if "implied_volatility" in row.index:
            opt_iv = row.get("implied_volatility")
            if pd.notna(opt_iv) and float(opt_iv) > 0:
                iv = float(opt_iv)
                if iv > 1:
                    iv = iv / 100.0
            elif vix1d_value is not None:
                iv = vix1d_value
        elif vix1d_value is not None:
            iv = vix1d_value

        delta = calculate_bs_delta(underlying_price, strike, T, iv, option_type)
        abs_delta = abs(delta)
        diff = abs(abs_delta - target_delta)

        if diff < best_diff:
            best_diff = diff
            best_strike = strike

    return best_strike


def find_strikes_by_delta_range(
    options_df: pd.DataFrame,
    option_type: str,
    min_delta: float,
    max_delta: float,
    underlying_price: float,
    dte_days: int,
    default_iv: float = 0.20,
    vix1d_value: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Find all strikes within a delta range.

    Returns:
        List of (strike, abs_delta) tuples sorted by delta descending.
    """
    if options_df.empty:
        return []

    typed = options_df[options_df["type"].str.upper() == option_type.upper()]
    if typed.empty:
        return []

    T = max(dte_days, 1) / 365.0
    results = []

    for _, row in typed.iterrows():
        strike = float(row["strike"])
        if strike <= 0:
            continue

        iv = default_iv
        if "implied_volatility" in row.index:
            opt_iv = row.get("implied_volatility")
            if pd.notna(opt_iv) and float(opt_iv) > 0:
                iv = float(opt_iv)
                if iv > 1:
                    iv = iv / 100.0
            elif vix1d_value is not None:
                iv = vix1d_value
        elif vix1d_value is not None:
            iv = vix1d_value

        delta = calculate_bs_delta(underlying_price, strike, T, iv, option_type)
        abs_delta = abs(delta)

        if min_delta <= abs_delta <= max_delta:
            results.append((strike, abs_delta))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


class BacktestV6DeltaStrategy(BaseCreditSpreadStrategy):
    """Delta-based credit spread strategy with multi-DTE evaluation.

    Selects short strikes by target delta instead of percentile boundaries.
    Evaluates multiple DTEs per interval and picks the best risk/reward.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tickers: List[str] = []
        self._ticker_equity_cache: Dict[Tuple[str, date], Any] = {}
        self._ticker_options_cache: Dict[Tuple[str, date], Any] = {}
        self._ticker_prev_close: Dict[Tuple[str, date], float] = {}
        self._rolling_config: Optional[RollingConfig] = None
        self._smart_roll_exit: Optional[SmartRollExit] = None

    @property
    def name(self) -> str:
        return "backtest_v6_delta"

    def setup(self) -> None:
        super().setup()
        params = self.config.params
        self._tickers = params.get("tickers", ["NDX"])

        # VIX regime signal (for IC fallback)
        try:
            from ...signals.vix_regime import VIXRegimeSignal
            vix_sg = VIXRegimeSignal()
            vix_sg.setup(None, {
                "vix_csv_dir": params.get("vix_csv_dir", "equities_output/I:VIX"),
                "lookback": params.get("vix_lookback", 60),
            })
            self.attach_signal_generator("vix_regime", vix_sg)
        except Exception:
            pass

        # Rolling config
        rolling_dict = params.get("rolling", {})
        self._rolling_config = RollingConfig.from_dict(rolling_dict) if rolling_dict else RollingConfig()
        if not params.get("roll_enabled", False):
            self._rolling_config.enabled = False

        if self._rolling_config.enabled and self.exit_manager is not None:
            self._smart_roll_exit = SmartRollExit(self._rolling_config)
            existing_rules = list(self.exit_manager.rules)
            new_rules = [self._smart_roll_exit]
            for rule in existing_rules:
                if rule.name in ("roll_trigger", "expiry_day_roll"):
                    continue
                new_rules.append(rule)
            self.exit_manager._rules = new_rules

    def on_day_start(self, day_context: DayContext) -> None:
        super().on_day_start(day_context)
        self._ticker_equity_cache = {}
        self._ticker_options_cache = {}
        self._ticker_prev_close = {}

        for ticker in self._tickers:
            self._preload_ticker_data(ticker, day_context)

    def _preload_ticker_data(self, ticker: str, day_context: DayContext) -> None:
        """Preload equity and options data for a ticker."""
        trading_date = day_context.trading_date
        primary_ticker = day_context.ticker

        if ticker == primary_ticker or ticker == primary_ticker.replace("I:", ""):
            self._ticker_equity_cache[(ticker, trading_date)] = day_context.equity_bars
            self._ticker_options_cache[(ticker, trading_date)] = day_context.options_data
            self._ticker_prev_close[(ticker, trading_date)] = day_context.prev_close or 0
        else:
            try:
                equity_provider = self.provider.equity if hasattr(self.provider, 'equity') else self.provider
                bars = equity_provider.get_bars(ticker, trading_date)
                self._ticker_equity_cache[(ticker, trading_date)] = bars
                prev = equity_provider.get_previous_close(ticker, trading_date)
                self._ticker_prev_close[(ticker, trading_date)] = prev or 0
            except Exception:
                self._ticker_equity_cache[(ticker, trading_date)] = pd.DataFrame()
                self._ticker_prev_close[(ticker, trading_date)] = 0

            try:
                options_provider = self.provider.options if hasattr(self.provider, 'options') else None
                if options_provider is not None:
                    opts = options_provider.get_options_chain(ticker, trading_date)
                    self._ticker_options_cache[(ticker, trading_date)] = opts
                else:
                    self._ticker_options_cache[(ticker, trading_date)] = None
            except Exception:
                self._ticker_options_cache[(ticker, trading_date)] = None

    def generate_signals(self, day_context: DayContext) -> List[Dict]:
        """Delta-based signal generation across tickers and DTEs."""
        params = self.config.params

        # Day-of-week filter (0=Mon, 1=Tue, ..., 4=Fri; None=all days)
        trading_days = params.get("trading_days", None)
        if trading_days is not None:
            dow = day_context.trading_date.weekday()
            if dow not in trading_days:
                return []

        # Delta parameters
        target_delta = params.get("target_delta", 0.10)
        delta_tolerance = params.get("delta_tolerance", 0.05)
        default_iv = params.get("default_iv", 0.20)
        use_vix1d = params.get("use_vix1d", False)
        vix1d_dir = params.get("vix1d_dir", "equities_output/I:VIX1D")

        # DTE parameters
        dte_list = params.get("dte_list", [0])
        if isinstance(dte_list, int):
            dte_list = [dte_list]

        # Spread parameters
        default_spread_width = params.get("spread_width", 50)
        spread_width_by_ticker = params.get("spread_width_by_ticker", {})
        num_contracts = params.get("num_contracts", 1)
        min_credit = params.get("min_credit", 0.30)
        min_credit_by_ticker = params.get("min_credit_per_option_by_ticker", {})

        # Dynamic sizing (V5-style)
        min_total_credit = params.get("min_total_credit", 0)
        max_risk_per_txn = params.get("max_risk_per_transaction", 100000)
        width_search_factors = params.get("width_search_factors", [1.0])

        # Timing
        interval_minutes = params.get("interval_minutes", 5)
        entry_start = params.get("entry_start_utc", "14:00")
        entry_end = params.get("entry_end_utc", "17:00")
        option_types = params.get("option_types", ["put", "call"])
        directional_mode = params.get("directional_entry", "both")

        # Selection limits
        max_positions_per_interval = params.get("max_positions_per_interval", 2)
        deployment_target = params.get("deployment_target_per_interval", 50000)

        signals = []

        bars = day_context.equity_bars
        if bars is None or bars.empty:
            return signals
        if "timestamp" not in bars.columns:
            return signals

        start_parts = entry_start.split(":")
        end_parts = entry_end.split(":")
        start_time = time(int(start_parts[0]), int(start_parts[1]))
        end_time = time(int(end_parts[0]), int(end_parts[1]))

        last_signal_time = None
        for _, bar in bars.iterrows():
            ts = bar["timestamp"]
            if not hasattr(ts, "time"):
                continue
            bar_time = ts.time()
            if bar_time < start_time or bar_time > end_time:
                continue
            if last_signal_time is not None:
                elapsed = (
                    ts.hour * 60 + ts.minute
                ) - (
                    last_signal_time.hour * 60 + last_signal_time.minute
                )
                if elapsed < interval_minutes:
                    continue

            # Get VIX1D value if configured
            vix1d_value = None
            if use_vix1d:
                try:
                    vix1d_value = get_vix1d_at_timestamp(ts, vix1d_dir)
                except Exception:
                    pass

            candidates = []

            for ticker in self._tickers:
                trading_date = day_context.trading_date
                prev_close = self._ticker_prev_close.get((ticker, trading_date), 0)
                if prev_close <= 0:
                    continue

                options_data = self._ticker_options_cache.get((ticker, trading_date))
                if options_data is None or (hasattr(options_data, "empty") and options_data.empty):
                    continue

                # Current price for directional filtering
                ticker_bars = self._ticker_equity_cache.get((ticker, trading_date))
                current_price = prev_close
                if ticker_bars is not None and not ticker_bars.empty:
                    if "timestamp" in ticker_bars.columns:
                        mask = ticker_bars["timestamp"] <= ts
                        if mask.any():
                            current_price = float(ticker_bars.loc[mask, "close"].iloc[-1])

                # Directional filtering
                if directional_mode in ("momentum", "contrarian", "pursuit"):
                    price_above_close = current_price > prev_close
                    if directional_mode == "momentum":
                        active_types = ["put"] if price_above_close else ["call"]
                    else:
                        active_types = ["call"] if price_above_close else ["put"]
                    active_types = [t for t in active_types if t in option_types]
                else:
                    active_types = list(option_types)

                if not active_types:
                    continue

                ticker_min_credit = min_credit_by_ticker.get(ticker, min_credit)
                base_width = spread_width_by_ticker.get(ticker, default_spread_width)

                # Evaluate across DTEs
                for dte in dte_list:
                    # Filter options to this DTE
                    filtered_opts = options_data
                    if "dte" in filtered_opts.columns:
                        filtered_opts = filtered_opts[filtered_opts["dte"] == dte]
                    elif "expiration" in filtered_opts.columns:
                        from datetime import timedelta
                        target_exp = trading_date + timedelta(days=dte)
                        target_exp_str = target_exp.strftime("%Y-%m-%d")
                        filtered_opts = filtered_opts[
                            filtered_opts["expiration"].astype(str).str[:10] == target_exp_str
                        ]

                    if filtered_opts.empty:
                        continue

                    # Snap to closest timestamp for accurate pricing
                    filtered_opts = _snap_options_to_time(filtered_opts, ts)

                    for opt_type in active_types:
                        # Find short strike by delta
                        target_strike = find_strike_by_delta(
                            filtered_opts, opt_type, target_delta,
                            current_price, dte, default_iv, vix1d_value,
                        )
                        if target_strike is None:
                            continue

                        # Width search
                        best_cs = None
                        best_width = base_width
                        best_contracts = num_contracts

                        for factor in width_search_factors:
                            try_width = max(5, int(round(base_width * factor / 5) * 5))

                            signal = self._probe_delta_spread(
                                ticker, dte, target_strike, opt_type,
                                try_width, 1, ts, day_context, params,
                                filtered_opts, current_price,
                            )

                            if signal is None or signal.get("credit", 0) < ticker_min_credit:
                                continue

                            # Dynamic sizing
                            if min_total_credit > 0:
                                n = _compute_dynamic_contracts(
                                    signal["credit"], try_width,
                                    min_total_credit, max_risk_per_txn,
                                )
                                if n is None:
                                    continue
                            else:
                                n = num_contracts

                            best_cs = signal
                            best_width = try_width
                            best_contracts = n

                        if best_cs is None:
                            continue

                        # Score and finalize
                        total_risk = best_width * best_contracts * 100
                        total_credit = best_cs["credit"] * best_contracts * 100
                        best_cs["num_contracts"] = best_contracts
                        best_cs["max_loss"] = total_risk
                        best_cs["max_width"] = (best_width, best_width)
                        best_cs["min_width"] = max(5, best_width // 2)
                        best_cs["ticker"] = ticker
                        best_cs["total_credit"] = total_credit
                        best_cs["actual_width"] = best_width
                        best_cs["target_delta"] = target_delta
                        best_cs["actual_dte"] = dte

                        score = self._score_candidate(
                            best_cs, base_width, spread_width_by_ticker,
                        )
                        best_cs["score"] = score
                        candidates.append(best_cs)

            # Rank and select
            candidates.sort(key=lambda x: x.get("score", 0), reverse=True)

            deployed = 0.0
            picked = 0
            for candidate in candidates:
                if picked >= max_positions_per_interval:
                    break
                if deployed >= deployment_target:
                    break
                cand_loss = candidate.get("max_loss", 0)
                if cand_loss > max_risk_per_txn:
                    continue
                if min_total_credit > 0:
                    cand_total = candidate.get("total_credit", 0)
                    if cand_total > 0 and cand_total < min_total_credit:
                        continue
                signals.append(candidate)
                deployed += cand_loss
                picked += 1

            last_signal_time = ts

        return signals

    def execute_signals(
        self, signals: List[Dict], day_context: DayContext
    ) -> List[Dict]:
        """Override to use pre-filtered options data from signal generation.

        The base class passes the full day's options_data to build_position,
        which causes O(n²) spread building on thousands of rows.  We attach
        the pre-filtered (snapped + strike-range) DataFrame to each signal
        so build_position only sees ~5 rows.
        """
        from datetime import datetime as dt

        positions = []
        for signal in signals:
            timestamp = signal.get("timestamp", dt.now())
            max_loss = signal.get("max_loss", 0)

            ctx = ConstraintContext(
                timestamp=timestamp,
                trading_date=day_context.trading_date,
                position_capital=max_loss,
                daily_capital_used=self._daily_capital_used,
                positions_open=len(self._open_positions),
            )
            result = self.constraints.check_all(ctx)
            if not result.allowed:
                continue

            instrument_name = signal.get("instrument", "credit_spread")
            instrument = self.get_instrument(instrument_name)

            # Use pre-filtered options if available, else fall back to full data
            options = signal.pop("_filtered_options", None)
            if options is None or (hasattr(options, "empty") and options.empty):
                options = day_context.options_data

            ticker = signal.get("ticker", day_context.ticker)
            prev_close = self._ticker_prev_close.get(
                (ticker, day_context.trading_date), day_context.prev_close
            )

            position = instrument.build_position(options, signal, prev_close)

            if position:
                position.metadata["ticker"] = ticker
                self.constraints.notify_opened(position.max_loss, timestamp)
                self._daily_capital_used += position.max_loss
                self._open_positions.append(position)
                positions.append({
                    "position": position,
                    "signal": signal,
                })

        return positions

    def _probe_delta_spread(
        self, ticker: str, dte: int, target_strike: float,
        opt_type: str, spread_width: int, num_contracts: int,
        timestamp: Any, day_context: DayContext, params: Dict,
        options_data: pd.DataFrame, current_price: float,
    ) -> Optional[Dict]:
        """Build a credit spread with the short strike at the delta-selected strike."""
        trading_date = day_context.trading_date
        prev_close = self._ticker_prev_close.get((ticker, trading_date), 0)
        if prev_close <= 0:
            return None

        # Filter near target strike
        margin = spread_width + 5
        if opt_type == "put":
            filtered = options_data[
                (options_data["strike"] >= target_strike - margin)
                & (options_data["strike"] <= target_strike + 5)
            ]
        else:
            filtered = options_data[
                (options_data["strike"] >= target_strike - 5)
                & (options_data["strike"] <= target_strike + margin)
            ]

        if filtered.empty:
            return None

        # Filter to correct type
        if "type" in filtered.columns:
            filtered = filtered[filtered["type"].str.upper() == opt_type.upper()]
        if filtered.empty:
            return None

        # Compute liquidity metrics
        liquidity = self._compute_liquidity_metrics(filtered, opt_type, target_strike)
        if liquidity["valid_quotes"] == 0:
            return None

        min_volume = params.get("min_volume", None)
        if min_volume is not None and liquidity["avg_volume"] < min_volume:
            return None

        # Always deduplicate to unique strikes (options CSVs have multiple timestamps)
        if "bid" in filtered.columns:
            filtered = filtered.sort_values("bid", ascending=False)
        filtered = filtered.drop_duplicates(subset=["strike", "type"], keep="first")

        # Build spread via instrument
        instrument = self.get_instrument("credit_spread")
        probe_signal = {
            "option_type": opt_type,
            "percent_beyond": (0.0, 0.0),
            "instrument": "credit_spread",
            "num_contracts": num_contracts,
            "timestamp": timestamp,
            "max_loss": params.get("max_loss_estimate", 10000),
            "max_width": (spread_width, spread_width),
            "min_width": max(5, spread_width // 2),
            "dte": dte,
            "entry_date": trading_date,
            "use_mid": params.get("use_mid", True),
            "min_volume": min_volume,
            "percentile_target_strike": target_strike,
        }
        position = instrument.build_position(filtered, probe_signal, prev_close)

        if position is None:
            return None

        real_credit = position.initial_credit
        real_max_loss = position.max_loss / max(1, position.num_contracts) / 100

        liq_score = liquidity["liquidity_score"]
        credit_risk = real_credit / max(real_max_loss, 0.01)
        composite_score = credit_risk * (0.6 + 0.4 * liq_score)

        return {
            "option_type": opt_type,
            "percent_beyond": (0.0, 0.0),
            "instrument": "credit_spread",
            "num_contracts": num_contracts,
            "timestamp": timestamp,
            "max_loss": params.get("max_loss_estimate", 10000),
            "max_width": (spread_width, spread_width),
            "min_width": max(5, spread_width // 2),
            "dte": dte,
            "entry_date": trading_date,
            "use_mid": params.get("use_mid", True),
            "min_volume": min_volume,
            "percentile_target_strike": target_strike,
            "credit": real_credit,
            "liquidity": liquidity,
            "composite_score": composite_score,
            "_filtered_options": filtered,  # Pre-filtered for fast execute_signals
        }

    def _compute_liquidity_metrics(
        self, options_data: pd.DataFrame, option_type: str, target_strike: float,
    ) -> Dict[str, Any]:
        """Compute liquidity metrics from options data near the target strike."""
        import numpy as np

        if "type" in options_data.columns:
            typed = options_data[
                options_data["type"].str.upper() == option_type.upper()
            ]
        else:
            typed = options_data

        if typed.empty:
            return {
                "valid_quotes": 0, "avg_bid_ask_pct": 1.0,
                "avg_volume": 0, "avg_iv": 0, "avg_delta": 0,
                "liquidity_score": 0.0,
            }

        has_bid = has_ask = False
        if "bid" in typed.columns and "ask" in typed.columns:
            bids = pd.to_numeric(typed["bid"], errors="coerce").fillna(0)
            asks = pd.to_numeric(typed["ask"], errors="coerce").fillna(0)
            valid = (bids > 0) & (asks > 0) & (asks > bids)
            has_bid = has_ask = True
        else:
            valid = pd.Series([False] * len(typed), index=typed.index)

        valid_count = int(valid.sum())

        if valid_count > 0 and has_bid and has_ask:
            v_bids = bids[valid]
            v_asks = asks[valid]
            mids = (v_bids + v_asks) / 2.0
            ba_pcts = ((v_asks - v_bids) / mids).replace([np.inf, -np.inf], np.nan).dropna()
            avg_ba_pct = float(ba_pcts.mean()) if len(ba_pcts) > 0 else 1.0
        else:
            avg_ba_pct = 1.0

        if "volume" in typed.columns:
            vols = pd.to_numeric(typed["volume"], errors="coerce").fillna(0)
            avg_volume = float(vols.mean())
        else:
            avg_volume = 0.0

        if "implied_volatility" in typed.columns:
            ivs = pd.to_numeric(typed["implied_volatility"], errors="coerce").dropna()
            ivs = ivs[ivs > 0]
            avg_iv = float(ivs.mean()) if len(ivs) > 0 else 0.0
        else:
            avg_iv = 0.0

        avg_delta = 0.0
        if "delta" in typed.columns:
            deltas = pd.to_numeric(typed["delta"], errors="coerce").dropna()
            if len(deltas) > 0:
                avg_delta = float(deltas.abs().mean())

        # Composite liquidity score
        ba_score = max(0, min(1, 1 - (avg_ba_pct - 0.02) / 0.18)) if avg_ba_pct < 0.20 else 0.0
        vol_score = min(1, avg_volume / 100.0) if avg_volume > 0 else 0.0
        quote_score = min(1, valid_count / 5.0)
        iv_score = 1.0 if avg_iv > 0 else 0.0
        liquidity_score = 0.40 * ba_score + 0.30 * vol_score + 0.20 * quote_score + 0.10 * iv_score

        return {
            "valid_quotes": valid_count,
            "avg_bid_ask_pct": avg_ba_pct,
            "avg_volume": avg_volume,
            "avg_iv": avg_iv,
            "avg_delta": avg_delta,
            "liquidity_score": liquidity_score,
        }

    @staticmethod
    def _score_candidate(
        candidate: Dict,
        base_width: int,
        spread_width_by_ticker: Dict,
    ) -> float:
        """Score a candidate by credit efficiency, width, liquidity, and DTE preference.

        Components:
        - Capital efficiency (35%): total_credit / total_risk
        - Width efficiency (20%): narrower = better
        - Liquidity (25%): bid-ask, volume, IV
        - DTE preference (20%): shorter DTE = higher theta decay = better
        """
        total_credit = candidate.get("total_credit", 0)
        total_risk = candidate.get("max_loss", 1)
        actual_width = candidate.get("actual_width", base_width)
        dte = candidate.get("actual_dte", 0)
        liq = candidate.get("liquidity", {})
        liq_score = liq.get("liquidity_score", 0.5) if isinstance(liq, dict) else 0.5

        cap_eff = min(total_credit / max(total_risk, 1), 1.0)

        max_width = max(spread_width_by_ticker.values()) if spread_width_by_ticker else base_width
        width_eff = 1.0 - (actual_width / max(max_width, 1)) if max_width > 0 else 0.5

        # DTE preference: 0DTE = 1.0, 10DTE = 0.0
        dte_score = max(0, 1.0 - dte / 10.0)

        return 0.35 * cap_eff + 0.20 * width_eff + 0.25 * liq_score + 0.20 * dte_score

def _compute_dynamic_contracts(
    initial_credit: float,
    spread_width: int,
    min_total_credit: float,
    max_risk: float,
) -> Optional[int]:
    """Compute num_contracts to meet min total credit within max risk."""
    if initial_credit <= 0 or spread_width <= 0:
        return None

    credit_per_contract = initial_credit * 100
    loss_per_contract = spread_width * 100

    if credit_per_contract <= 0 or loss_per_contract <= 0:
        return None

    min_contracts = math.ceil(min_total_credit / credit_per_contract)
    max_contracts = int(max_risk / loss_per_contract)

    if min_contracts > max_contracts:
        return None

    return min_contracts


BacktestStrategyRegistry.register("backtest_v6_delta", BacktestV6DeltaStrategy)
