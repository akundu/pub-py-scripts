"""CombinedCollector -- merges per-instance results with attribution."""

import math
from collections import defaultdict
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from .evaluator import Proposal


class CombinedCollector:
    """Merges orchestrated results from Phase 2 selection.

    Tracks:
    - Combined trades with algo attribution
    - Per-algo contribution stats
    - Budget utilization
    - Overlap analysis (days with multiple algo candidates)
    - Exit events (interval mode)
    - Interval-level selection log
    """

    def __init__(self):
        self._accepted: List[Dict[str, Any]] = []
        self._rejected: List[Dict[str, Any]] = []
        self._daily_budget_used: Dict[str, float] = {}
        self._selection_log: List[Dict[str, Any]] = []
        self._exit_events: List[Dict[str, Any]] = []
        self._adaptive_budget_log: List[Dict[str, Any]] = []

    def record_selection(
        self,
        trading_date: date,
        accepted: List[Proposal],
        all_proposals: List[Proposal],
        budget_remaining: float,
        interval_key: Optional[str] = None,
    ) -> None:
        """Record a per-slot selection decision.

        Args:
            trading_date: The trading date.
            accepted: Accepted proposals.
            all_proposals: All proposals (accepted + rejected).
            budget_remaining: Budget remaining at time of selection.
            interval_key: If interval mode, the interval key (e.g., "2026-03-10_1430").
        """
        for proposal in accepted:
            trade = proposal.metadata.get("original_trade", {}).copy()
            trade["orchestrator_instance_id"] = proposal.instance_id
            trade["orchestrator_algo"] = proposal.algo_name
            trade["orchestrator_score"] = proposal.score
            trade["orchestrator_accepted"] = True
            if interval_key is not None:
                trade["interval_key"] = interval_key
            self._accepted.append(trade)

        # Log rejected candidates
        accepted_ids = {p.instance_id for p in accepted}
        for proposal in all_proposals:
            if proposal.instance_id not in accepted_ids:
                trade = proposal.metadata.get("original_trade", {}).copy()
                trade["orchestrator_instance_id"] = proposal.instance_id
                trade["orchestrator_algo"] = proposal.algo_name
                trade["orchestrator_score"] = proposal.score
                trade["orchestrator_accepted"] = False
                trade["reject_reason"] = "not_selected"
                if interval_key is not None:
                    trade["interval_key"] = interval_key
                self._rejected.append(trade)

        # Log selection
        competing = list({p.instance_id for p in all_proposals})
        winners = [p.instance_id for p in accepted]
        log_entry = {
            "date": str(trading_date),
            "num_candidates": len(all_proposals),
            "num_accepted": len(accepted),
            "winners": winners,
            "competing": competing,
            "budget_remaining": budget_remaining,
        }
        if interval_key is not None:
            log_entry["interval_key"] = interval_key
        self._selection_log.append(log_entry)

        # Track budget
        date_key = str(trading_date)
        if date_key not in self._daily_budget_used:
            self._daily_budget_used[date_key] = 0.0
        for p in accepted:
            self._daily_budget_used[date_key] += p.total_max_loss

    def record_exit(
        self,
        position: Dict[str, Any],
        exit_signal: Any,
        interval_key: Optional[str] = None,
    ) -> None:
        """Record an exit event (interval mode position tracking)."""
        event = {
            "instance_id": position.get("instance_id", ""),
            "algo_name": position.get("algo_name", ""),
            "ticker": position.get("ticker", ""),
            "entry_interval": position.get("entry_interval", ""),
            "exit_reason": getattr(exit_signal, "reason", str(exit_signal)),
            "exit_time": str(getattr(exit_signal, "exit_time", "")),
            "exit_price": getattr(exit_signal, "exit_price", 0),
        }
        if interval_key is not None:
            event["exit_interval"] = interval_key
        self._exit_events.append(event)

    def record_adaptive_budget_log(self, entries: List[Dict[str, Any]]) -> None:
        """Record per-interval adaptive budget analytics."""
        self._adaptive_budget_log.extend(entries)

    @property
    def adaptive_budget_log(self) -> List[Dict[str, Any]]:
        return list(self._adaptive_budget_log)

    @property
    def accepted_trades(self) -> List[Dict[str, Any]]:
        return list(self._accepted)

    @property
    def rejected_trades(self) -> List[Dict[str, Any]]:
        return list(self._rejected)

    @property
    def selection_log(self) -> List[Dict[str, Any]]:
        return list(self._selection_log)

    @property
    def exit_events(self) -> List[Dict[str, Any]]:
        return list(self._exit_events)

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute combined metrics from accepted trades."""
        from scripts.backtesting.results.metrics import StandardMetrics
        return StandardMetrics.compute(self._accepted)

    def per_algo_attribution(self) -> Dict[str, Dict[str, Any]]:
        """Break down contributions by algo instance."""
        from scripts.backtesting.results.metrics import StandardMetrics

        by_algo: Dict[str, List[Dict]] = defaultdict(list)
        for trade in self._accepted:
            key = trade.get("orchestrator_instance_id", "unknown")
            by_algo[key].append(trade)

        result = {}
        for instance_id, trades in sorted(by_algo.items()):
            metrics = StandardMetrics.compute(trades)
            result[instance_id] = {
                "trades": len(trades),
                "metrics": metrics,
                "algo_name": trades[0].get("orchestrator_algo", ""),
                "ticker": trades[0].get("ticker", ""),
            }

        return result

    def overlap_analysis(self) -> Dict[str, Any]:
        """Analyze how often multiple algos competed for the same slot."""
        total_slots = len(self._selection_log)
        contested = sum(1 for s in self._selection_log if s["num_candidates"] > 1)
        uncontested = total_slots - contested

        algo_win_counts: Dict[str, int] = defaultdict(int)
        for s in self._selection_log:
            for winner in s["winners"]:
                algo_win_counts[winner] += 1

        return {
            "total_slots": total_slots,
            "contested_slots": contested,
            "uncontested_slots": uncontested,
            "contest_rate": contested / total_slots if total_slots > 0 else 0,
            "wins_by_instance": dict(algo_win_counts),
        }

    def interval_analysis(self) -> Dict[str, Any]:
        """Analyze selection distribution by time-of-day (interval mode only)."""
        by_hour: Dict[int, int] = defaultdict(int)
        by_interval: Dict[str, int] = defaultdict(int)

        for trade in self._accepted:
            ik = trade.get("interval_key", "")
            if ik and "_" in ik:
                time_part = ik.split("_")[1]
                if len(time_part) >= 4:
                    hour = int(time_part[:2])
                    by_hour[hour] += 1
                    by_interval[time_part] += 1

        return {
            "trades_by_hour": dict(sorted(by_hour.items())),
            "trades_by_interval": dict(sorted(by_interval.items())),
            "total_exit_events": len(self._exit_events),
            "exit_reasons": self._exit_reason_counts(),
        }

    def _exit_reason_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for evt in self._exit_events:
            reason = evt.get("exit_reason", "unknown")
            counts[reason] += 1
        return dict(counts)

    def summarize(self) -> Dict[str, Any]:
        """Full summary of orchestrated results."""
        summary = {
            "combined_metrics": self.compute_metrics(),
            "per_algo_attribution": self.per_algo_attribution(),
            "overlap_analysis": self.overlap_analysis(),
            "total_accepted": len(self._accepted),
            "total_rejected": len(self._rejected),
            "selection_log": self._selection_log,
        }

        # Add interval analysis if we have interval data
        if any(t.get("interval_key") for t in self._accepted):
            summary["interval_analysis"] = self.interval_analysis()

        if self._exit_events:
            summary["exit_events"] = self._exit_events

        if self._adaptive_budget_log:
            summary["adaptive_budget_log"] = self._adaptive_budget_log

        return summary
