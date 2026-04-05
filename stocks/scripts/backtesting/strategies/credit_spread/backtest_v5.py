"""BacktestV5Strategy -- multi-ticker combined V4 with smart selection.

V5 extends V4 with:
  1. Dynamic num_contracts — computed per candidate to meet min total credit
     ($2K default) while staying under max risk ($45K default).
  2. Per-ticker min credit/option — NDX >= $0.75, SPX/RUT >= $0.35.
  3. Width optimization — tries narrower widths per ticker, preferring
     the narrowest that still meets credit thresholds.
  4. Capital-efficiency scoring — ranks candidates by total_credit/total_risk
     weighted by width efficiency and liquidity.

All V4 features carry over: ExpirationDayRollExit, chain-aware profit exit,
OTM-maximizing roll, DTE fallback, liquidity scoring, IC fallback.
Rolling is always same-ticker (ticker comes from position.metadata["ticker"]).
"""

import math
from datetime import time
from typing import Any, Dict, List, Optional

from .backtest_v4 import BacktestV4Strategy
from ..base import DayContext
from ..registry import BacktestStrategyRegistry


class BacktestV5Strategy(BacktestV4Strategy):
    """Multi-ticker combined strategy with smart selection and dynamic sizing.

    Overrides generate_signals() to add:
    - Width search: tries narrower widths per ticker
    - Dynamic num_contracts: sized to meet min_total_credit within max_risk
    - Per-ticker min credit/option thresholds
    - Capital-efficiency scoring for cross-ticker ranking
    """

    @property
    def name(self) -> str:
        return "backtest_v5"

    def generate_signals(self, day_context: DayContext) -> List[Dict]:
        """Multi-ticker signal generation with smart selection and dynamic sizing."""
        params = self.config.params
        dte = params.get("dte", 0)
        percentile = params.get("percentile", 95)
        ic_percentile = params.get("iron_condor_percentile", 85)
        default_spread_width = params.get("spread_width", 50)
        spread_width_by_ticker = params.get("spread_width_by_ticker", {})
        interval_minutes = params.get("interval_minutes", 10)
        entry_start = params.get("entry_start_utc", "14:00")
        entry_end = params.get("entry_end_utc", "17:00")
        base_num_contracts = params.get("num_contracts", 1)
        min_credit = params.get("min_credit", 0.30)
        ic_credit_multiplier = params.get("ic_credit_multiplier", 2.0)
        max_positions_per_interval = params.get("max_positions_per_interval", 2)
        deployment_target = params.get("deployment_target_per_interval", 50000)
        fallback_dte_enabled = params.get("fallback_dte_enabled", True)
        fallback_dte_list = params.get("fallback_dte_list", [1, 2, 3, 5, 7, 10])
        option_types = params.get("option_types", ["put", "call"])

        # V5-specific params
        min_total_credit = params.get("min_total_credit", 2000)
        max_risk_per_txn = params.get("max_risk_per_transaction", 45000)
        min_credit_by_ticker = params.get("min_credit_per_option_by_ticker", {})
        width_search_factors = params.get("width_search_factors", [1.0, 0.75, 0.50])

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

        vix_data = day_context.signals.get("vix_regime", {})
        vix_regime = vix_data.get("regime", "normal")
        vix_close = vix_data.get("vix_close")  # prev day VIX close
        vix_pct_rank = vix_data.get("percentile_rank", 50.0)

        # V5 VIX filtering: skip entry if VIX below minimum
        vix_entry_min = params.get("vix_entry_min")  # e.g., 18
        vix_entry_max = params.get("vix_entry_max")  # e.g., 35
        vix_regime_skip = params.get("vix_regime_skip", [])  # e.g., ["extreme"]

        if vix_close is not None:
            if vix_entry_min and vix_close < vix_entry_min:
                return signals  # skip day — VIX too low
            if vix_entry_max and vix_close > vix_entry_max:
                return signals  # skip day — VIX too high
        if vix_regime in vix_regime_skip:
            return signals  # skip day — regime excluded

        # VIX-scaled credit threshold: require higher credit in low-VIX (thin premium)
        vix_credit_scaling = params.get("vix_credit_scaling", False)
        if vix_credit_scaling and vix_close is not None:
            if vix_close < 15:
                min_credit = max(min_credit, min_credit * 1.5)  # 50% higher bar
            elif vix_close > 25:
                min_credit = max(0.30, min_credit * 0.75)  # relax in high VIX

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

            candidates = []

            for ticker in self._tickers:
                trading_date = day_context.trading_date
                pct_data = self._ticker_pct_data.get(ticker, {})
                strikes_by_dte = pct_data.get("strikes", {})
                prev_close = self._ticker_prev_close.get((ticker, trading_date), 0)

                if prev_close <= 0:
                    continue

                # Directional filtering using ticker-specific price
                ticker_bars = self._ticker_equity_cache.get((ticker, trading_date))
                if ticker_bars is not None and not ticker_bars.empty:
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
                    else:
                        active_types = ["call"] if price_above_close else ["put"]
                    active_types = [t for t in active_types if t in option_types]
                else:
                    active_types = list(option_types)

                if not active_types:
                    continue

                # Per-ticker min credit threshold
                ticker_min_credit = min_credit_by_ticker.get(ticker, min_credit)
                base_width = spread_width_by_ticker.get(ticker, default_spread_width)

                # Width search: try narrower widths to minimize risk
                best_cs = None
                best_width = base_width
                best_contracts = base_num_contracts

                for factor in width_search_factors:
                    try_width = max(5, int(round(base_width * factor / 5) * 5))

                    cs = self._probe_credit_spread(
                        ticker, dte, percentile, strikes_by_dte, try_width,
                        1, ts, day_context, active_types, params,
                    )

                    # DTE fallback at this width
                    if fallback_dte_enabled and (cs is None or cs.get("credit", 0) < ticker_min_credit):
                        for fb_dte in fallback_dte_list:
                            if fb_dte == dte:
                                continue
                            fb_cs = self._probe_credit_spread(
                                ticker, fb_dte, percentile, strikes_by_dte,
                                try_width, 1, ts, day_context, active_types, params,
                            )
                            if fb_cs is not None and fb_cs.get("credit", 0) >= ticker_min_credit:
                                cs = fb_cs
                                break

                    if cs is None or cs.get("credit", 0) < ticker_min_credit:
                        continue

                    # Dynamic contract sizing
                    n_contracts = self._compute_dynamic_contracts(
                        cs["credit"], try_width, min_total_credit, max_risk_per_txn,
                    )
                    if n_contracts is None:
                        continue

                    # Found a valid width — use it (narrowest wins since factors decrease)
                    best_cs = cs
                    best_width = try_width
                    best_contracts = n_contracts
                    # Don't break — keep trying narrower widths

                if best_cs is None:
                    cs_credit = 0
                else:
                    cs_credit = best_cs.get("credit", 0)

                # IC fallback: only if VIX low AND CS credit insufficient
                if vix_regime == "low" and cs_credit < ticker_min_credit:
                    ic_signal = self._probe_iron_condor(
                        ticker, dte, ic_percentile, strikes_by_dte, base_width,
                        base_num_contracts, ts, day_context, params,
                    )
                    if ic_signal is not None:
                        ic_credit = ic_signal.get("credit", 0)
                        if ic_credit >= ticker_min_credit:
                            # Dynamic sizing for IC too
                            ic_contracts = self._compute_dynamic_contracts(
                                ic_credit, base_width, min_total_credit, max_risk_per_txn,
                            )
                            if ic_contracts is not None:
                                ic_total_risk = base_width * ic_contracts * 100
                                ic_total_credit = ic_credit * ic_contracts * 100
                                ic_signal["num_contracts"] = ic_contracts
                                ic_signal["max_loss"] = ic_total_risk
                                ic_signal["ticker"] = ticker
                                ic_signal["total_credit"] = ic_total_credit
                                ic_liq = ic_signal.get("liquidity", {})
                                ic_liq_score = ic_liq.get("combined_score", 0.5) if isinstance(ic_liq, dict) else 0.5
                                raw_score = ic_total_credit / max(ic_total_risk, 1)
                                ic_signal["score"] = min(raw_score, 1.0) * 0.4 + 0.3 * ic_liq_score + 0.3 * 0.5
                                candidates.append(ic_signal)
                                continue

                # Score and add CS candidate
                if best_cs is not None and cs_credit > 0:
                    # Update signal with dynamic sizing
                    total_risk = best_width * best_contracts * 100
                    total_credit = cs_credit * best_contracts * 100
                    best_cs["num_contracts"] = best_contracts
                    best_cs["max_loss"] = total_risk
                    best_cs["max_width"] = (best_width, best_width)
                    best_cs["min_width"] = max(5, best_width // 2)
                    best_cs["ticker"] = ticker
                    best_cs["actual_width"] = best_width
                    best_cs["total_credit"] = total_credit

                    # Capital-efficiency scoring
                    score = self._score_candidate_v5(
                        best_cs, base_width, spread_width_by_ticker, params,
                    )
                    best_cs["score"] = score
                    best_cs["vix_close"] = vix_close
                    best_cs["vix_regime"] = vix_regime
                    best_cs["vix_pct_rank"] = vix_pct_rank
                    candidates.append(best_cs)

            # Rank by score
            candidates.sort(key=lambda x: x.get("score", 0), reverse=True)

            # Select best within constraints
            deployed = 0
            picked = 0
            for candidate in candidates:
                if picked >= max_positions_per_interval:
                    break
                if deployed >= deployment_target:
                    break
                cand_loss = candidate.get("max_loss", 0)
                if cand_loss > max_risk_per_txn:
                    continue
                # Enforce min total credit at selection time
                cand_total_credit = candidate.get("total_credit", 0)
                if cand_total_credit > 0 and cand_total_credit < min_total_credit:
                    continue
                signals.append(candidate)
                deployed += cand_loss
                picked += 1

            last_signal_time = ts

        return signals

    @staticmethod
    def _compute_dynamic_contracts(
        initial_credit: float,
        spread_width: int,
        min_total_credit: float,
        max_risk: float,
    ) -> Optional[int]:
        """Compute num_contracts to meet min total credit within max risk.

        Args:
            initial_credit: Per-share credit (e.g., $0.75).
            spread_width: Spread width in points.
            min_total_credit: Minimum total credit in dollars (e.g., $2000).
            max_risk: Maximum total risk in dollars (e.g., $45000).

        Returns:
            Number of contracts, or None if constraints can't be satisfied.
        """
        if initial_credit <= 0 or spread_width <= 0:
            return None

        credit_per_contract = initial_credit * 100  # $0.75 → $75
        loss_per_contract = spread_width * 100  # width=50 → $5,000

        if credit_per_contract <= 0 or loss_per_contract <= 0:
            return None

        min_contracts = math.ceil(min_total_credit / credit_per_contract)
        max_contracts = int(max_risk / loss_per_contract)

        if min_contracts > max_contracts:
            return None

        return min_contracts

    @staticmethod
    def _score_candidate_v5(
        candidate: Dict,
        base_width: int,
        spread_width_by_ticker: Dict,
        params: Dict,
    ) -> float:
        """Score a candidate by capital efficiency, width efficiency, and liquidity.

        Components:
        - Capital efficiency (40%): total_credit / total_risk — more credit per $ at risk
        - Width efficiency (30%): narrower width = less risk for same credit
        - Liquidity (30%): combined score from bid-ask, volume, IV availability
        """
        total_credit = candidate.get("total_credit", 0)
        total_risk = candidate.get("max_loss", 1)
        actual_width = candidate.get("actual_width", base_width)
        liq = candidate.get("liquidity", {})
        liq_score = liq.get("liquidity_score", 0.5) if isinstance(liq, dict) else 0.5

        # Capital efficiency: credit per dollar of risk, capped at 1.0
        cap_eff = min(total_credit / max(total_risk, 1), 1.0)

        # Width efficiency: how much narrower than max configured width
        max_width = max(spread_width_by_ticker.values()) if spread_width_by_ticker else base_width
        width_eff = 1.0 - (actual_width / max(max_width, 1)) if max_width > 0 else 0.5

        return 0.40 * cap_eff + 0.30 * width_eff + 0.30 * liq_score


BacktestStrategyRegistry.register("backtest_v5", BacktestV5Strategy)
