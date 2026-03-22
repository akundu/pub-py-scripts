"""BacktestV4Strategy -- multi-ticker evaluation, IC fallback, smart rolling.

Inherits from PercentileEntryCreditSpreadStrategy (v3) and adds:
  1. Multi-ticker evaluation — at each interval, evaluates NDX, SPX, and RUT
     simultaneously and picks the best opportunities by credit/risk ratio.
  2. Liquidity-aware scoring — uses real bid/ask spreads, volume, and IV from
     the options chain to assess true availability before ranking candidates.
  3. Iron condor fallback when credit spreads are insufficient in low-IV regimes.
  4. Smart rolling — 0DTE proximity/ITM triggers, P85 percentile strike selection,
     width expansion only when BTC cost exceeds new credit, DTE search 1→10.
  5. Chain loss cap to stop rolling when losses exceed a threshold.

Always prefers credit spreads over iron condors.
"""

import copy
import math
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base_credit_spread import BaseCreditSpreadStrategy
from .percentile_entry import PercentileEntryCreditSpreadStrategy
from ..base import DayContext
from ..registry import BacktestStrategyRegistry
from ...constraints.base import ConstraintContext
from ...constraints.exit_rules.expiration_day_roll_exit import ExpirationDayRollExit
from ...constraints.exit_rules.roll_trigger import RollTriggerExit
from ...constraints.exit_rules.smart_roll_exit import RollingConfig, SmartRollExit
from ...instruments.base import InstrumentPosition


class BacktestV4Strategy(PercentileEntryCreditSpreadStrategy):
    """Multi-ticker credit spread strategy with IC fallback and advanced rolling.

    Features beyond v3:
    - Evaluates all configured tickers at each interval
    - Picks best opportunities by credit/risk ratio across tickers
    - Iron condor fallback in low-VIX regimes when CS credit is insufficient
    - Chain-aware profit exit (net chain P&L must be positive to close)
    - ExpirationDayRollExit for breach + expiry-day triggers
    - OTM-maximizing roll strikes with next-day-first DTE progression
    - Roll analytics tracking by category (breach, expiry, p95)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tickers: List[str] = []
        self._ticker_signals: Dict[str, Any] = {}
        self._ticker_equity_cache: Dict[Tuple[str, date], Any] = {}
        self._ticker_options_cache: Dict[Tuple[str, date], Any] = {}
        self._ticker_prev_close: Dict[Tuple[str, date], float] = {}
        self._ticker_pct_data: Dict[str, Dict] = {}
        self._rolling_config: Optional[RollingConfig] = None
        self._smart_roll_exit: Optional[SmartRollExit] = None
        self._roll_analytics: Dict[str, Dict] = {
            "proximity_roll": {"count": 0, "successes": 0, "chain_pnls": []},
            "itm_roll": {"count": 0, "successes": 0, "chain_pnls": []},
        }

    @property
    def name(self) -> str:
        return "backtest_v4"

    def setup(self) -> None:
        super().setup()

        params = self.config.params
        self._tickers = params.get("tickers", ["NDX"])

        # Set up VIX regime signal
        from ...signals.vix_regime import VIXRegimeSignal
        vix_sg = VIXRegimeSignal()
        vix_sg.setup(None, {
            "vix_csv_dir": params.get("vix_csv_dir", "equities_output/I:VIX"),
            "lookback": params.get("vix_lookback", 60),
        })
        self.attach_signal_generator("vix_regime", vix_sg)

        # Set up PercentileRangeSignal for each additional ticker
        from ...signals.percentile_range import PercentileRangeSignal

        percentile = params.get("percentile", 95)
        ic_percentile = params.get("iron_condor_percentile", 85)
        lookback = params.get("lookback", 120)
        dte = params.get("dte", 0)
        roll_min_dte = params.get("roll_min_dte", 1)
        roll_max_dte = params.get("roll_max_dte", 10)
        # Load RollingConfig from params
        rolling_dict = params.get("rolling", {})
        self._rolling_config = RollingConfig.from_dict(rolling_dict) if rolling_dict else RollingConfig()
        if not params.get("roll_enabled", False):
            self._rolling_config.enabled = False
        roll_percentile = self._rolling_config.roll_percentile

        percentiles_needed = sorted(set([percentile, ic_percentile, roll_percentile]))
        dte_windows = sorted(set(
            ([dte] if isinstance(dte, int) else dte)
            + list(range(roll_min_dte, roll_max_dte + 1))
        ))
        fallback_dtes = params.get("fallback_dte_list", [1, 2, 3, 5, 7, 10])
        dte_windows = sorted(set(dte_windows + fallback_dtes))

        for ticker in self._tickers:
            sg = PercentileRangeSignal()
            provider = self.provider.equity if hasattr(self.provider, 'equity') else self.provider
            sg.setup(provider, {
                "lookback": lookback,
                "percentiles": percentiles_needed,
                "dte_windows": dte_windows,
                "ticker": ticker,
            })
            self._ticker_signals[ticker] = sg

        # Inject SmartRollExit — replaces both ExpirationDayRollExit and RollTriggerExit
        # Must be FIRST in the rule list so it gets priority at the check time
        # (otherwise profit_target/stop_loss close positions before the roll check)
        if self._rolling_config.enabled and self.exit_manager is not None:
            self._smart_roll_exit = SmartRollExit(self._rolling_config)
            # Remove old roll exit rules, then prepend SmartRollExit
            existing_rules = list(self.exit_manager.rules)
            new_rules = [self._smart_roll_exit]
            for rule in existing_rules:
                if rule.name in ("roll_trigger", "expiry_day_roll"):
                    continue  # skip old roll rules
                new_rules.append(rule)
            self.exit_manager._rules = new_rules

    def on_day_start(self, day_context: DayContext) -> None:
        super().on_day_start(day_context)
        # Clear per-day caches
        self._ticker_equity_cache = {}
        self._ticker_options_cache = {}
        self._ticker_prev_close = {}
        self._ticker_pct_data = {}

        # Preload data for all tickers
        for ticker in self._tickers:
            self._preload_ticker_data(ticker, day_context)

    def _preload_ticker_data(self, ticker: str, day_context: DayContext) -> None:
        """Preload equity, options, and signal data for a ticker."""
        trading_date = day_context.trading_date

        # Use main day_context data for the primary ticker
        primary_ticker = day_context.ticker
        if ticker == primary_ticker or ticker == primary_ticker.replace("I:", ""):
            self._ticker_equity_cache[(ticker, trading_date)] = day_context.equity_bars
            self._ticker_options_cache[(ticker, trading_date)] = day_context.options_data
            self._ticker_prev_close[(ticker, trading_date)] = day_context.prev_close or 0
        else:
            # Try loading from provider for other tickers
            try:
                equity_provider = self.provider.equity if hasattr(self.provider, 'equity') else self.provider
                bars = equity_provider.get_bars(ticker, trading_date)
                self._ticker_equity_cache[(ticker, trading_date)] = bars
                prev = equity_provider.get_previous_close(ticker, trading_date)
                self._ticker_prev_close[(ticker, trading_date)] = prev or 0
            except Exception:
                import pandas as pd
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

        # Generate percentile signal data for this ticker
        if ticker in self._ticker_signals:
            try:
                mini_ctx = DayContext(
                    trading_date=trading_date,
                    ticker=ticker,
                    equity_bars=self._ticker_equity_cache.get((ticker, trading_date)),
                    options_data=self._ticker_options_cache.get((ticker, trading_date)),
                    prev_close=self._ticker_prev_close.get((ticker, trading_date), 0),
                    signals={},
                    metadata={},
                )
                pct_data = self._ticker_signals[ticker].generate(mini_ctx)
                self._ticker_pct_data[ticker] = pct_data
            except Exception:
                self._ticker_pct_data[ticker] = {}

    def generate_signals(self, day_context: DayContext) -> List[Dict]:
        """Multi-ticker signal generation with IC fallback."""
        params = self.config.params
        dte = params.get("dte", 0)
        percentile = params.get("percentile", 95)
        ic_percentile = params.get("iron_condor_percentile", 85)
        default_spread_width = params.get("spread_width", 50)
        spread_width_by_ticker = params.get("spread_width_by_ticker", {})
        interval_minutes = params.get("interval_minutes", 10)
        entry_start = params.get("entry_start_utc", "14:00")
        entry_end = params.get("entry_end_utc", "17:00")
        num_contracts = params.get("num_contracts", 1)
        min_credit = params.get("min_credit", 0.30)
        ic_credit_multiplier = params.get("ic_credit_multiplier", 2.0)
        max_positions_per_interval = params.get("max_positions_per_interval", 2)
        deployment_target = params.get("deployment_target_per_interval", 50000)
        fallback_dte_enabled = params.get("fallback_dte_enabled", True)
        fallback_dte_list = params.get("fallback_dte_list", [1, 2, 3, 5, 7, 10])
        option_types = params.get("option_types", ["put", "call"])

        signals = []

        # Get primary ticker bars for time iteration
        bars = day_context.equity_bars
        if bars is None or bars.empty:
            return signals
        if "timestamp" not in bars.columns:
            return signals

        start_parts = entry_start.split(":")
        end_parts = entry_end.split(":")
        start_time = time(int(start_parts[0]), int(start_parts[1]))
        end_time = time(int(end_parts[0]), int(end_parts[1]))

        # VIX regime
        vix_data = day_context.signals.get("vix_regime", {})
        vix_regime = vix_data.get("regime", "normal")

        # Directional filtering (matches V3 logic)
        directional_mode = params.get("directional_entry", "both")

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

            # Evaluate all tickers at this interval
            candidates = []

            for ticker in self._tickers:
                trading_date = day_context.trading_date
                pct_data = self._ticker_pct_data.get(ticker, {})
                strikes_by_dte = pct_data.get("strikes", {})
                prev_close = self._ticker_prev_close.get((ticker, trading_date), 0)

                if prev_close <= 0:
                    continue

                # Apply directional filtering per-ticker using ticker-specific price
                ticker_bars = self._ticker_equity_cache.get((ticker, trading_date))
                if ticker_bars is not None and not ticker_bars.empty:
                    # Find the bar closest to current time for THIS ticker
                    if "timestamp" in ticker_bars.columns:
                        ts_col = ticker_bars["timestamp"]
                        mask = ts_col <= ts
                        if mask.any():
                            current_price = float(ticker_bars.loc[mask, "close"].iloc[-1])
                        else:
                            current_price = float(ticker_bars["close"].iloc[0])
                    else:
                        current_price = float(ticker_bars["close"].iloc[-1])
                else:
                    current_price = prev_close

                if directional_mode in ("momentum", "contrarian", "pursuit"):
                    price_above_close = current_price > prev_close
                    if directional_mode == "momentum":
                        active_types = ["put"] if price_above_close else ["call"]
                    else:  # contrarian or pursuit
                        active_types = ["call"] if price_above_close else ["put"]
                    active_types = [t for t in active_types if t in option_types]
                else:
                    active_types = list(option_types)

                if not active_types:
                    continue

                # Per-ticker spread width
                spread_width = spread_width_by_ticker.get(ticker, default_spread_width)

                # Try credit spread at configured DTE
                best_cs = self._probe_credit_spread(
                    ticker, dte, percentile, strikes_by_dte, spread_width,
                    num_contracts, ts, day_context, active_types, params,
                )

                # DTE fallback if credit too low
                if fallback_dte_enabled and (best_cs is None or best_cs.get("credit", 0) < min_credit):
                    for fb_dte in fallback_dte_list:
                        if fb_dte == dte:
                            continue
                        fb_cs = self._probe_credit_spread(
                            ticker, fb_dte, percentile, strikes_by_dte, spread_width,
                            num_contracts, ts, day_context, active_types, params,
                        )
                        if fb_cs is not None and fb_cs.get("credit", 0) >= min_credit:
                            best_cs = fb_cs
                            break

                cs_credit = best_cs.get("credit", 0) if best_cs else 0

                # IC fallback: only if VIX low AND CS credit insufficient
                if vix_regime == "low" and cs_credit < min_credit:
                    ic_signal = self._probe_iron_condor(
                        ticker, dte, ic_percentile, strikes_by_dte, spread_width,
                        num_contracts, ts, day_context, params,
                    )
                    if ic_signal is not None:
                        ic_credit = ic_signal.get("credit", 0)
                        ic_max_loss = ic_signal.get("max_loss", 1)
                        ic_liq = ic_signal.get("liquidity", {})
                        ic_liq_score = ic_liq.get("combined_score", 0.5) if isinstance(ic_liq, dict) else 0.5
                        if ic_credit > ic_credit_multiplier * cs_credit and ic_max_loss > 0:
                            ic_signal["ticker"] = ticker
                            # Score IC with liquidity weighting
                            raw_score = ic_credit / ic_max_loss
                            ic_signal["score"] = raw_score * (0.6 + 0.4 * ic_liq_score)
                            candidates.append(ic_signal)
                            continue

                # Add best CS for this ticker (score already computed by _probe with liquidity)
                if best_cs is not None and cs_credit > 0:
                    cs_max_loss = best_cs.get("max_loss", 1)
                    cs_liq = best_cs.get("liquidity", {})
                    cs_liq_score = cs_liq.get("liquidity_score", 0.5) if isinstance(cs_liq, dict) else 0.5
                    best_cs["ticker"] = ticker
                    raw_score = cs_credit / cs_max_loss if cs_max_loss > 0 else 0
                    best_cs["score"] = raw_score * (0.6 + 0.4 * cs_liq_score)
                    candidates.append(best_cs)

            # Rank candidates by credit/risk ratio
            candidates.sort(key=lambda x: x.get("score", 0), reverse=True)

            # Select best fitting within constraints
            deployed = 0
            picked = 0
            max_spend = params.get("max_loss_estimate", 10000)
            for candidate in candidates:
                if picked >= max_positions_per_interval:
                    break
                if deployed >= deployment_target:
                    break
                cand_loss = candidate.get("max_loss", 0)
                if cand_loss > max_spend:
                    continue
                signals.append(candidate)
                deployed += cand_loss
                picked += 1

            last_signal_time = ts

        return signals

    def _probe_credit_spread(
        self, ticker: str, dte: int, percentile: int,
        strikes_by_dte: Dict, spread_width: int, num_contracts: int,
        timestamp: Any, day_context: DayContext, option_types: List[str],
        params: Dict,
    ) -> Optional[Dict]:
        """Probe credit spread for a ticker at a given DTE using real options data.

        Builds actual spreads from the options chain for this ticker/DTE,
        computes a liquidity score from bid-ask width, volume, and IV,
        and returns the best signal with real credit and liquidity metadata.
        """
        dte_strikes = strikes_by_dte.get(dte, {})
        pct_strikes = dte_strikes.get(percentile, {})

        trading_date = day_context.trading_date
        options_data = self._ticker_options_cache.get((ticker, trading_date))
        if options_data is None or (hasattr(options_data, 'empty') and options_data.empty):
            return None

        prev_close = self._ticker_prev_close.get((ticker, trading_date), 0)
        if prev_close <= 0:
            return None

        best_signal = None
        best_score = 0

        for opt_type in option_types:
            target_strike = pct_strikes.get(opt_type)
            if target_strike is None:
                continue

            # Filter options to this DTE and near the target strike
            filtered = options_data
            if "dte" in filtered.columns:
                filtered = filtered[filtered["dte"] == dte]
            if filtered.empty:
                continue

            if "strike" in filtered.columns:
                margin = spread_width + 5
                if opt_type == "put":
                    filtered = filtered[
                        (filtered["strike"] >= target_strike - margin)
                        & (filtered["strike"] <= target_strike + 5)
                    ]
                else:
                    filtered = filtered[
                        (filtered["strike"] >= target_strike - 5)
                        & (filtered["strike"] <= target_strike + margin)
                    ]

            if filtered.empty:
                continue

            # Compute liquidity metrics from the options near the target strike
            liquidity = self._compute_liquidity_metrics(filtered, opt_type, target_strike)

            # Skip if liquidity is too poor (no valid bid/ask quotes)
            if liquidity["valid_quotes"] == 0:
                continue

            # Apply minimum volume filter
            min_volume = params.get("min_volume", None)
            if min_volume is not None and liquidity["avg_volume"] < min_volume:
                continue

            # Deduplicate before building spreads
            if "type" in filtered.columns and len(filtered) > 20:
                if "bid" in filtered.columns:
                    filtered = filtered.sort_values("bid", ascending=False)
                filtered = filtered.drop_duplicates(
                    subset=["strike", "type"], keep="first"
                )

            # Try to build an actual spread to get real credit
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
                continue

            real_credit = position.initial_credit
            real_max_loss = position.max_loss / max(1, position.num_contracts) / 100

            # Composite score: credit/risk ratio weighted by liquidity quality
            # Liquidity score: 0.0 (terrible) to 1.0 (excellent)
            liq_score = liquidity["liquidity_score"]
            credit_risk = real_credit / max(real_max_loss, 0.01)
            composite_score = credit_risk * (0.6 + 0.4 * liq_score)

            signal = {
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
            }

            if composite_score > best_score:
                best_score = composite_score
                best_signal = signal

        return best_signal

    def _compute_liquidity_metrics(
        self, options_data, option_type: str, target_strike: float,
    ) -> Dict[str, Any]:
        """Compute liquidity metrics from options data near the target strike.

        Examines bid/ask spread width, volume, implied volatility, and delta
        to produce a composite liquidity score (0.0 to 1.0).

        Returns dict with:
            valid_quotes: count of options with positive bid AND ask
            avg_bid_ask_pct: average bid-ask spread as % of mid price
            avg_volume: average volume across strikes near target
            avg_iv: average implied volatility
            avg_delta: average absolute delta (for OTM confirmation)
            liquidity_score: composite 0.0-1.0 score
        """
        # Filter to matching option type
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

        # Identify valid quotes: positive bid AND ask
        has_bid = False
        has_ask = False
        if "bid" in typed.columns and "ask" in typed.columns:
            bids = pd.to_numeric(typed["bid"], errors="coerce").fillna(0)
            asks = pd.to_numeric(typed["ask"], errors="coerce").fillna(0)
            valid = (bids > 0) & (asks > 0) & (asks > bids)
            has_bid = True
            has_ask = True
        else:
            valid = pd.Series([False] * len(typed), index=typed.index)

        valid_count = int(valid.sum())

        # Bid-ask spread as % of mid price
        if valid_count > 0 and has_bid and has_ask:
            v_bids = bids[valid]
            v_asks = asks[valid]
            mids = (v_bids + v_asks) / 2.0
            ba_pcts = ((v_asks - v_bids) / mids).replace([np.inf, -np.inf], np.nan).dropna()
            avg_ba_pct = float(ba_pcts.mean()) if len(ba_pcts) > 0 else 1.0
        else:
            avg_ba_pct = 1.0  # 100% spread = illiquid

        # Volume
        if "volume" in typed.columns:
            vols = pd.to_numeric(typed["volume"], errors="coerce").fillna(0)
            avg_volume = float(vols.mean())
        else:
            avg_volume = 0.0

        # Implied volatility
        if "implied_volatility" in typed.columns:
            ivs = pd.to_numeric(typed["implied_volatility"], errors="coerce").dropna()
            ivs = ivs[ivs > 0]
            avg_iv = float(ivs.mean()) if len(ivs) > 0 else 0.0
        else:
            avg_iv = 0.0

        # Delta (absolute)
        if "delta" in typed.columns:
            deltas = pd.to_numeric(typed["delta"], errors="coerce").dropna()
            avg_delta = float(deltas.abs().mean()) if len(deltas) > 0 else 0.0
        else:
            avg_delta = 0.0

        # Composite liquidity score (0.0 to 1.0)
        # Components:
        #   1. Bid-ask tightness (weight 0.40): tighter = better
        #      <2% spread = 1.0, >20% spread = 0.0
        #   2. Volume score (weight 0.30): higher = better
        #      >100 = 1.0, 0 = 0.0
        #   3. Quote availability (weight 0.20): more valid quotes = better
        #      >5 valid = 1.0, 0 = 0.0
        #   4. IV availability (weight 0.10): having IV data = better pricing
        ba_score = max(0.0, min(1.0, 1.0 - (avg_ba_pct - 0.02) / 0.18))
        vol_score = max(0.0, min(1.0, avg_volume / 100.0))
        quote_score = max(0.0, min(1.0, valid_count / 5.0))
        iv_score = 1.0 if avg_iv > 0 else 0.0

        liquidity_score = (
            0.40 * ba_score
            + 0.30 * vol_score
            + 0.20 * quote_score
            + 0.10 * iv_score
        )

        return {
            "valid_quotes": valid_count,
            "avg_bid_ask_pct": round(avg_ba_pct, 4),
            "avg_volume": round(avg_volume, 1),
            "avg_iv": round(avg_iv, 4),
            "avg_delta": round(avg_delta, 4),
            "liquidity_score": round(liquidity_score, 3),
        }

    def _probe_iron_condor(
        self, ticker: str, dte: int, ic_percentile: int,
        strikes_by_dte: Dict, spread_width: int, num_contracts: int,
        timestamp: Any, day_context: DayContext, params: Dict,
    ) -> Optional[Dict]:
        """Probe iron condor for a ticker using real options data.

        Checks both put and call sides for liquidity before returning.
        """
        dte_strikes = strikes_by_dte.get(dte, {})
        pct_strikes = dte_strikes.get(ic_percentile, {})

        put_target = pct_strikes.get("put")
        call_target = pct_strikes.get("call")
        if put_target is None or call_target is None:
            return None

        trading_date = day_context.trading_date
        options_data = self._ticker_options_cache.get((ticker, trading_date))
        if options_data is None or (hasattr(options_data, 'empty') and options_data.empty):
            return None

        # Filter to this DTE
        filtered = options_data
        if "dte" in filtered.columns:
            filtered = filtered[filtered["dte"] == dte]
        if filtered.empty:
            return None

        # Check liquidity on both sides
        put_liq = self._compute_liquidity_metrics(filtered, "put", put_target)
        call_liq = self._compute_liquidity_metrics(filtered, "call", call_target)

        # Skip if either side has no valid quotes
        if put_liq["valid_quotes"] == 0 or call_liq["valid_quotes"] == 0:
            return None

        # Use the average liquidity score from both sides
        combined_liq_score = (put_liq["liquidity_score"] + call_liq["liquidity_score"]) / 2.0

        # Estimate credit from actual bid/ask if available
        credit_estimate = 0.0
        for side_type, target in [("put", put_target), ("call", call_target)]:
            if "type" in filtered.columns:
                side_opts = filtered[filtered["type"].str.upper() == side_type.upper()]
            else:
                side_opts = filtered
            if side_opts.empty:
                continue
            if "bid" in side_opts.columns and "strike" in side_opts.columns:
                # Find the option closest to target strike
                side_opts = side_opts.copy()
                side_opts["_dist"] = (side_opts["strike"] - target).abs()
                nearest = side_opts.nsmallest(1, "_dist")
                if not nearest.empty:
                    bid_val = pd.to_numeric(nearest["bid"].iloc[0], errors="coerce")
                    if pd.notna(bid_val) and bid_val > 0:
                        credit_estimate += float(bid_val)

        signal = {
            "option_type": "iron_condor",
            "instrument": "iron_condor",
            "put_target_strike": put_target,
            "call_target_strike": call_target,
            "percentile_target_strike": put_target,
            "spread_width": spread_width,
            "num_contracts": num_contracts,
            "timestamp": timestamp,
            "max_loss": params.get("max_loss_estimate", 10000),
            "dte": dte,
            "entry_date": day_context.trading_date,
            "use_mid": params.get("use_mid", True),
            "credit": credit_estimate,
            "liquidity": {
                "put": put_liq,
                "call": call_liq,
                "combined_score": round(combined_liq_score, 3),
            },
        }

        return signal

    def execute_signals(
        self, signals: List[Dict], day_context: DayContext
    ) -> List[Dict]:
        """Execute multi-ticker CS and IC signals."""
        positions = []

        for signal in signals:
            timestamp = signal.get("timestamp", datetime.now())
            max_loss = signal.get("max_loss", 0)
            ticker = signal.get("ticker", self._tickers[0])

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
            trading_date = day_context.trading_date

            # Get ticker-specific data
            options_data = self._ticker_options_cache.get((ticker, trading_date))
            if options_data is None:
                options_data = day_context.options_data
            prev_close = self._ticker_prev_close.get((ticker, trading_date))
            if prev_close is None:
                prev_close = day_context.prev_close

            if instrument_name == "iron_condor":
                position = self._build_ic_position(signal, options_data, prev_close)
            else:
                position = self._build_cs_position(
                    signal, options_data, prev_close, day_context
                )

            if position is None:
                continue

            min_credit = self.config.params.get("min_credit", 0)
            if min_credit > 0 and position.initial_credit < min_credit:
                continue

            # Tag position metadata with ticker and liquidity
            position.metadata["ticker"] = ticker
            if "liquidity" in signal:
                position.metadata["liquidity"] = signal["liquidity"]

            self.constraints.notify_opened(position.max_loss, timestamp)
            self._daily_capital_used += position.max_loss
            self._open_positions.append(position)
            positions.append({
                "position": position,
                "signal": signal,
            })

        return positions

    def _build_cs_position(
        self, signal: Dict, options_data, prev_close: float,
        day_context: DayContext,
    ) -> Optional[InstrumentPosition]:
        """Build a credit spread position from ticker-specific data."""
        option_type = signal.get("option_type", "put")
        target_strike = signal.get("percentile_target_strike")
        target_dte = signal.get("dte", self.config.params.get("dte", 0))

        cache_key = f"{signal.get('ticker', '')}_{option_type}_{target_strike}_{target_dte}"
        if cache_key in self._position_cache:
            template = self._position_cache[cache_key]
            if template is None:
                return None
            position = copy.copy(template)
            position.entry_time = signal.get("timestamp")
            return position

        instrument = self.get_instrument("credit_spread")

        if options_data is not None and not options_data.empty:
            filtered = options_data
            if "dte" in filtered.columns:
                filtered = filtered[filtered["dte"] == target_dte]
            if target_strike is not None and "strike" in filtered.columns:
                spread_width = self.config.params.get("spread_width", 50)
                margin = spread_width + 5
                if option_type == "put":
                    filtered = filtered[
                        (filtered["strike"] >= target_strike - margin)
                        & (filtered["strike"] <= target_strike + 5)
                    ]
                else:
                    filtered = filtered[
                        (filtered["strike"] >= target_strike - 5)
                        & (filtered["strike"] <= target_strike + margin)
                    ]
                if "type" in filtered.columns and len(filtered) > 20:
                    if "bid" in filtered.columns:
                        filtered = filtered.sort_values("bid", ascending=False)
                    filtered = filtered.drop_duplicates(
                        subset=["strike", "type"], keep="first"
                    )
            options_data = filtered

        position = instrument.build_position(options_data, signal, prev_close)
        self._position_cache[cache_key] = position

        if position is None:
            return None

        result = copy.copy(position)
        result.entry_time = signal.get("timestamp")
        return result

    def _build_ic_position(
        self, signal: Dict, options_data, prev_close: float,
    ) -> Optional[InstrumentPosition]:
        """Build an iron condor position from ticker-specific data."""
        try:
            from scripts.credit_spread_utils.iron_condor_builder import IronCondorBuilder
        except ImportError:
            return None

        if options_data is None or options_data.empty:
            return None

        target_dte = signal.get("dte", 0)
        if "dte" in options_data.columns:
            options_data = options_data[options_data["dte"] == target_dte]
        if options_data.empty:
            return None

        spread_width = signal.get("spread_width", self.config.params.get("spread_width", 50))
        put_target = signal.get("put_target_strike")
        call_target = signal.get("call_target_strike")
        if put_target is None or call_target is None:
            return None

        builder = IronCondorBuilder(
            min_credit=self.config.params.get("min_credit", 0.30),
            use_mid_price=signal.get("use_mid", True),
            min_wing_width=max(5, spread_width // 2),
            max_wing_width=spread_width * 2,
        )

        condors = builder.build_iron_condor(
            options_df=options_data,
            call_target_strike=call_target,
            put_target_strike=put_target,
            call_spread_width=spread_width,
            put_spread_width=spread_width,
            prev_close=prev_close,
        )
        if not condors:
            return None

        best = builder.get_best_iron_condor(condors, sort_by="total_credit")
        if best is None:
            return None

        num_contracts = signal.get("num_contracts", 1)
        return InstrumentPosition(
            instrument_type="iron_condor",
            entry_time=signal.get("timestamp", datetime.now()),
            option_type="iron_condor",
            short_strike=best["short_put_strike"],
            long_strike=best["long_put_strike"],
            initial_credit=best["total_credit"],
            max_loss=best["max_loss"] * num_contracts,
            num_contracts=num_contracts,
            metadata={
                "call_short_strike": best["short_call_strike"],
                "call_long_strike": best["long_call_strike"],
                "put_credit": best["put_spread"]["credit"],
                "call_credit": best["call_spread"]["credit"],
                "regime": signal.get("regime", "low"),
                "ticker": signal.get("ticker", ""),
            },
        )

    def evaluate(
        self, positions: List[Dict], day_context: DayContext
    ) -> List[Dict]:
        """Evaluate with chain-aware profit exit and roll category tracking."""
        results = []
        params = self.config.params
        dte = params.get("dte", 0)
        roll_enabled = params.get("roll_enabled", False)
        chain_aware = params.get("chain_aware_profit_exit", True)

        if day_context.equity_bars.empty:
            return results

        # Add new positions to multi-day tracking
        for pos_dict in positions:
            pos_dict["entry_date"] = day_context.trading_date
            pos_dict["dte"] = pos_dict.get("signal", {}).get("dte", dte)
            pos_dict.setdefault("roll_count", 0)
            pos_dict.setdefault("total_pnl_chain", 0.0)
            self._multi_day_positions.append(pos_dict)

        still_open = []
        bars = day_context.equity_bars

        for pos_dict in self._multi_day_positions:
            position = pos_dict["position"]
            entry_date = pos_dict.get("entry_date", day_context.trading_date)
            pos_dte = pos_dict.get("dte", dte)
            ticker = position.metadata.get("ticker", self._tickers[0])

            # Use ticker-specific price for exit evaluation
            ticker_bars = self._ticker_equity_cache.get(
                (ticker, day_context.trading_date)
            )
            if ticker_bars is None or (hasattr(ticker_bars, 'empty') and ticker_bars.empty):
                ticker_bars = bars

            days_held = (day_context.trading_date - entry_date).days
            if pos_dte > 0 and days_held < pos_dte - 1:
                exit_signal = self._check_intraday_exits(pos_dict, ticker_bars, day_context)
                if exit_signal and exit_signal.triggered:
                    result = self._handle_exit(
                        pos_dict, exit_signal, day_context, roll_enabled,
                        chain_aware, still_open, ticker_bars,
                    )
                    if result:
                        results.extend(result)
                else:
                    still_open.append(pos_dict)
                continue

            # Last day or 0DTE
            exit_signal = self._check_intraday_exits(pos_dict, ticker_bars, day_context)

            if exit_signal and exit_signal.triggered:
                result = self._handle_exit(
                    pos_dict, exit_signal, day_context, roll_enabled,
                    chain_aware, still_open, ticker_bars,
                )
                if result:
                    results.extend(result)
            else:
                close_price = float(ticker_bars["close"].iloc[-1])
                instrument = self.get_instrument(position.instrument_type)
                pnl_result = instrument.calculate_pnl(position, close_price)
                pnl_result.exit_reason = "eod_close" if pos_dte == 0 else "expiration"
                pnl_result.exit_time = (
                    ticker_bars["timestamp"].iloc[-1]
                    if "timestamp" in ticker_bars.columns
                    else datetime.now()
                )
                total_chain_pnl = pos_dict.get("total_pnl_chain", 0.0)
                pnl_result.metadata["total_chain_pnl"] = total_chain_pnl + pnl_result.pnl
                pnl_result.metadata["roll_count"] = pos_dict.get("roll_count", 0)
                pnl_result.metadata["ticker"] = ticker
                pnl_result.metadata["dte"] = pos_dte
                self.constraints.notify_closed(position.max_loss, pnl_result.exit_time)
                self._daily_capital_used -= position.max_loss

                # Track final chain P&L in roll analytics
                if pos_dict.get("roll_count", 0) > 0:
                    self._track_roll_completion(pos_dict, pnl_result.pnl)

                results.append(pnl_result.to_dict())

        self._multi_day_positions = still_open
        return results

    def _handle_exit(
        self, pos_dict: Dict, exit_signal, day_context: DayContext,
        roll_enabled: bool, chain_aware: bool,
        still_open: List[Dict], ticker_bars,
    ) -> List[Dict]:
        """Handle an exit signal with chain-aware rolling.

        When chain_aware_profit_exit is True and the position has been rolled:
        - Chain P&L < 0: suppress profit exits (keep recovering), and on stop_loss
          attempt a roll instead of closing flat (keep trying to recover credit).
        - Chain P&L >= 0: normal profit-taking and exit rules apply.
        - max_rolls still constrains total rolls.
        - Time/EOD exits always close (can't avoid expiration).
        """
        results = []
        position = pos_dict["position"]

        is_roll_trigger = exit_signal.reason.startswith("roll_trigger")
        is_ic = position.instrument_type == "iron_condor"

        rc = self._rolling_config
        if rc is not None:
            chain_aware = rc.chain_aware_profit_exit

        # Compute chain P&L for rolled positions
        is_rolled = pos_dict.get("roll_count", 0) > 0
        chain_pnl_negative = False
        if chain_aware and is_rolled:
            instrument = self.get_instrument(position.instrument_type)
            current_leg_pnl = instrument.calculate_pnl(
                position, exit_signal.exit_price
            ).pnl
            chain_pnl = pos_dict.get("total_pnl_chain", 0.0) + current_leg_pnl
            chain_pnl_negative = chain_pnl < 0

        if roll_enabled and is_roll_trigger and not is_ic:
            # Roll trigger (proximity/ITM) → always roll
            self._track_roll_trigger(exit_signal.reason)

            roll_result = self._execute_smart_roll(pos_dict, exit_signal, day_context)
            if roll_result is not None:
                closed_result, new_pos = roll_result
                results.append(closed_result)
                if new_pos is not None:
                    still_open.append(new_pos)
            else:
                result = self._close_position(pos_dict, exit_signal, day_context)
                if result:
                    results.append(result)
        elif exit_signal.reason.startswith("profit_target"):
            # Chain-aware: suppress profit exit while chain P&L is negative
            if chain_pnl_negative:
                still_open.append(pos_dict)
                return results

            # Chain recovered (>= 0) or not rolled → normal profit-taking
            if not self._passes_roi_gate(pos_dict, position, exit_signal.exit_price):
                still_open.append(pos_dict)
                return results

            result = self._close_position(pos_dict, exit_signal, day_context)
            if result:
                if is_rolled:
                    self._track_roll_completion(pos_dict, result.get("pnl", 0))
                results.append(result)
        elif exit_signal.reason.startswith("stop_loss") and chain_pnl_negative and roll_enabled and not is_ic:
            # Stop loss on a rolled position with negative chain → try to roll
            # instead of closing at a loss (keep trying to recover credit)
            self._track_roll_trigger(exit_signal.reason)

            roll_result = self._execute_smart_roll(pos_dict, exit_signal, day_context)
            if roll_result is not None:
                closed_result, new_pos = roll_result
                results.append(closed_result)
                if new_pos is not None:
                    still_open.append(new_pos)
            else:
                # Can't roll (max_rolls, no options, chain_loss_cap) → close flat
                result = self._close_position(pos_dict, exit_signal, day_context)
                if result:
                    results.append(result)
        else:
            # Time exit, EOD, or stop_loss on non-rolled/recovered positions → close
            result = self._close_position(pos_dict, exit_signal, day_context)
            if result:
                results.append(result)

        return results

    def _execute_smart_roll(
        self, pos_dict: Dict, exit_signal, day_context: DayContext
    ) -> Optional[tuple]:
        """Execute smart roll with P85 percentile strike selection and conditional width expansion.

        Strike selection uses the P85 (configurable) boundary at the rolled DTE from
        the existing PercentileRangeSignal data. Width expansion only happens when
        BTC cost exceeds new credit at original width.
        """
        rc = self._rolling_config or RollingConfig()
        position = pos_dict["position"]
        roll_count = pos_dict.get("roll_count", 0)
        total_pnl = pos_dict.get("total_pnl_chain", 0.0)
        ticker = position.metadata.get("ticker", self._tickers[0])

        # Chain loss cap: stop rolling if cumulative losses exceed cap
        if rc.chain_loss_cap > 0 and abs(total_pnl) >= rc.chain_loss_cap:
            return None  # caller will close flat

        # Close current position
        instrument = self.get_instrument(position.instrument_type)
        pnl_result = instrument.calculate_pnl(position, exit_signal.exit_price)
        pnl_result.exit_reason = exit_signal.reason
        pnl_result.exit_time = exit_signal.exit_time
        pnl_result.metadata["roll_count"] = roll_count
        pnl_result.metadata["ticker"] = ticker
        pnl_result.metadata["dte"] = pos_dict.get("dte", 0)
        total_pnl += pnl_result.pnl
        pnl_result.metadata["total_chain_pnl"] = total_pnl

        self.constraints.notify_closed(position.max_loss, exit_signal.exit_time)
        self._daily_capital_used -= position.max_loss
        closed_result = pnl_result.to_dict()

        # Compute BTC cost (cost to close current position)
        btc_cost = max(0, -pnl_result.pnl / max(1, position.num_contracts) / 100)

        # Get ticker-specific options — look ahead to future dates for DTE 1+
        trading_date = day_context.trading_date
        all_options = self._ticker_options_cache.get((ticker, trading_date))
        if all_options is None:
            all_options = day_context.options_data

        # Build combined options across future dates for DTE search
        # Today's file may only have DTE 0; next day's file has DTE 1 (from its perspective)
        combined_options = self._get_roll_target_options(
            ticker, trading_date, all_options, rc.max_dte
        )
        if combined_options is None or combined_options.empty:
            return (closed_result, None)

        prev_close = self._ticker_prev_close.get((ticker, trading_date))
        if prev_close is None:
            prev_close = day_context.prev_close

        original_width = abs(position.short_strike - position.long_strike)
        min_credit = self.config.params.get("min_credit", 0.30)

        new_position = None
        new_dte = None

        # DTE search: 1 → max_dte, prefer lower DTE
        for try_dte in range(rc.min_dte, rc.max_dte + 1):
            if "dte" not in combined_options.columns:
                continue

            dte_options = combined_options[combined_options["dte"] == try_dte]
            if dte_options.empty:
                continue

            # P85 strike selection from percentile signal data
            target_strike = self._get_percentile_target_strike(
                ticker, try_dte, rc.roll_percentile, position.option_type,
                exit_signal.exit_price,
            )

            # Try original width first, then expand only if BTC > credit
            pos_result = self._build_roll_spread(
                dte_options, position.option_type, target_strike,
                original_width, rc.max_width_multiplier, min_credit,
                position.num_contracts, exit_signal.exit_time,
                try_dte, day_context, ticker, btc_cost,
            )
            if pos_result is not None:
                new_position = pos_result
                new_dte = try_dte
                break

        if new_position is None:
            return (closed_result, None)

        # Track capital for new position
        self.constraints.notify_opened(new_position.max_loss, exit_signal.exit_time)
        self._daily_capital_used += new_position.max_loss

        new_pos_dict = {
            "position": new_position,
            "signal": {
                "instrument": "credit_spread",
                "option_type": position.option_type,
                "ticker": ticker,
                "original_roll_reason": exit_signal.reason,
            },
            "entry_date": day_context.trading_date,
            "dte": new_dte,
            "roll_count": roll_count + 1,
            "total_pnl_chain": total_pnl,
        }

        return (closed_result, new_pos_dict)

    def _get_roll_target_options(
        self, ticker: str, trading_date, today_options, max_dte: int,
    ):
        """Get options data with DTE 1+ for rolling targets.

        Today's options file typically only has DTE 0 (same-day expiration).
        To roll into future expirations, we load the next trading day's file
        which contains options with multiple expiration dates, then recompute
        DTE relative to today.
        """
        import pandas as pd
        from datetime import timedelta

        frames = []

        # Include today's options if they have DTE > 0
        if today_options is not None and not today_options.empty:
            if "dte" in today_options.columns:
                future = today_options[today_options["dte"] > 0]
                if not future.empty:
                    frames.append(future)

        # Look ahead up to 5 calendar days to find next trading day's file
        options_provider = getattr(self.provider, "options", None)

        if options_provider is not None:
            for offset in range(1, 6):
                next_date = trading_date + timedelta(days=offset)
                next_options = options_provider.get_options_chain(
                    ticker, next_date, dte_buckets=None
                )
                if next_options is not None and not next_options.empty:
                    # Recompute DTE relative to today (not next_date)
                    if "expiration" in next_options.columns:
                        next_options = next_options.copy()
                        exp_dates = pd.to_datetime(next_options["expiration"]).dt.date
                        next_options["dte"] = exp_dates.apply(
                            lambda exp: (exp - trading_date).days if exp else None
                        )
                        # Keep only DTE 1 to max_dte
                        valid = next_options[
                            (next_options["dte"] >= 1) & (next_options["dte"] <= max_dte)
                        ]
                        if not valid.empty:
                            frames.append(valid)
                    break  # only need the first available future date

        if not frames:
            return today_options  # fallback to today's data

        return pd.concat(frames, ignore_index=True)

    def _get_percentile_target_strike(
        self, ticker: str, dte: int, percentile: int,
        option_type: str, current_price: float,
    ) -> float:
        """Look up the percentile boundary strike from preloaded signal data.

        For puts: returns the P85 lower boundary (short strike for put spread).
        For calls: returns the P85 upper boundary (short strike for call spread).
        Falls back to current_price if data unavailable.
        """
        pct_data = self._ticker_pct_data.get(ticker, {})
        strikes_by_dte = pct_data.get("strikes", {})
        dte_strikes = strikes_by_dte.get(dte, {})
        pct_strikes = dte_strikes.get(percentile, {})

        if option_type == "put":
            target = pct_strikes.get("put", 0)
        else:
            target = pct_strikes.get("call", 0)

        return target if target > 0 else current_price

    def _build_roll_spread(
        self, options_data, option_type: str, target_strike: float,
        original_width: float, max_width_multiplier: float, min_credit: float,
        num_contracts: int, timestamp: Any, dte: int,
        day_context: DayContext, ticker: str, btc_cost: float,
    ) -> Optional[InstrumentPosition]:
        """Build a spread at the target strike, expanding width only if needed.

        Delegates to shared roll_builder.build_roll_spread().
        """
        from scripts.credit_spread_utils.roll_builder import build_roll_spread

        params = self.config.params
        prev_close = self._ticker_prev_close.get(
            (ticker, day_context.trading_date), 0
        ) or day_context.prev_close

        result = build_roll_spread(
            options_df=options_data,
            option_type=option_type,
            target_strike=target_strike,
            base_width=original_width,
            max_width_multiplier=max_width_multiplier,
            btc_cost=btc_cost,
            use_mid=params.get("use_mid", True),
            min_credit=min_credit,
            prev_close=prev_close,
        )

        if result is None:
            return None

        position = InstrumentPosition(
            instrument_type="credit_spread",
            entry_time=timestamp,
            option_type=option_type,
            short_strike=result.short_strike,
            long_strike=result.long_strike,
            initial_credit=result.credit,
            max_loss=(result.width - result.credit) * 100 * num_contracts,
            num_contracts=num_contracts,
            metadata={
                "width": result.width,
                "net_credit_per_contract": result.credit * 100,
                "short_price": result.short_price,
                "long_price": result.long_price,
                "ticker": ticker,
            },
        )
        return position

    def _track_roll_trigger(self, reason: str) -> None:
        """Categorize and count a roll trigger."""
        if "proximity" in reason:
            self._roll_analytics["proximity_roll"]["count"] += 1
        elif "itm" in reason:
            self._roll_analytics["itm_roll"]["count"] += 1

    def _track_roll_completion(self, pos_dict: Dict, final_leg_pnl: float) -> None:
        """Track completion of a rolled position chain."""
        chain_pnl = pos_dict.get("total_pnl_chain", 0.0) + final_leg_pnl
        # Use the FIRST roll reason to categorize the chain
        # We store this in the position metadata from the first roll
        exit_reason = pos_dict.get("signal", {}).get("original_roll_reason", "")
        if not exit_reason:
            exit_reason = "p95"  # default

        if "proximity" in exit_reason:
            cat = "proximity_roll"
        elif "itm" in exit_reason:
            cat = "itm_roll"
        else:
            cat = "itm_roll"  # default

        self._roll_analytics[cat]["chain_pnls"].append(chain_pnl)
        if chain_pnl > 0:
            self._roll_analytics[cat]["successes"] += 1

    def get_roll_analytics(self) -> Dict[str, Dict]:
        """Return roll analytics summary."""
        summary = {}
        for cat, data in self._roll_analytics.items():
            count = data["count"]
            successes = data["successes"]
            chain_pnls = data["chain_pnls"]
            summary[cat] = {
                "count": count,
                "success_rate": successes / max(1, len(chain_pnls)) * 100,
                "avg_chain_pnl": sum(chain_pnls) / max(1, len(chain_pnls)),
                "total_chain_pnl": sum(chain_pnls),
            }
        return summary

    def _close_position(
        self, pos_dict: Dict, exit_signal, day_context: DayContext
    ) -> Optional[Dict]:
        """Close a position with ticker metadata."""
        result = super()._close_position(pos_dict, exit_signal, day_context)
        if result is not None:
            position = pos_dict["position"]
            result["ticker"] = position.metadata.get("ticker", self._tickers[0])
        return result

    def teardown(self) -> None:
        super().teardown()
        self._ticker_signals.clear()
        self._ticker_equity_cache.clear()
        self._ticker_options_cache.clear()
        self._ticker_prev_close.clear()
        self._ticker_pct_data.clear()


BacktestStrategyRegistry.register("backtest_v4", BacktestV4Strategy)
