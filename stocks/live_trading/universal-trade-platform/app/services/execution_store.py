"""Persistent execution cache — stores IBKR trade executions for grouping and dedup.

Executions are keyed by exec_id (globally unique per fill) and grouped by
perm_id (permanent order ID — consistent across TWS sessions, identifies
all fills for one order including multi-leg combos).

This enables:
- Identifying multi-leg trades (same perm_id = same order = same spread)
- Deduplicating across multiple reconcile runs
- Preserving history beyond IBKR's ~7-day execution window
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ExecutionStore:
    """JSON-backed persistent store for IBKR executions."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._executions: dict[str, dict] = {}  # exec_id -> execution dict
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self._executions = data.get("executions", {})
                logger.info(
                    "Execution store loaded: %d executions from %s",
                    len(self._executions), self._path,
                )
            except Exception as e:
                logger.error("Failed to load execution store: %s", e)
                self._executions = {}
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def _save(self) -> None:
        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self._path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump({"executions": self._executions}, f, indent=2)
            os.replace(tmp_path, str(self._path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def merge_executions(self, executions: list[dict]) -> int:
        """Merge new executions into the store. Returns count of new entries."""
        new_count = 0
        with self._lock:
            for ex in executions:
                exec_id = ex.get("exec_id", "")
                if not exec_id:
                    continue
                if exec_id not in self._executions:
                    self._executions[exec_id] = ex
                    new_count += 1
            if new_count > 0:
                self._save()
                logger.info("Merged %d new executions (total: %d)", new_count, len(self._executions))
        return new_count

    def get_all(self) -> list[dict]:
        """Return all stored executions sorted by time."""
        with self._lock:
            execs = list(self._executions.values())
        execs.sort(key=lambda e: e.get("time", ""))
        return execs

    def get_grouped_by_order(self) -> list[dict]:
        """Group executions by perm_id (permanent order ID).

        Each group represents one order submission. Multi-leg orders
        (credit spreads, iron condors) will have multiple executions
        with the same perm_id.

        Returns list of order groups sorted by time, each with:
        - perm_id, order_id, time, symbol, order_type (inferred)
        - legs: list of individual fills
        - total_commission, net_amount
        """
        with self._lock:
            execs = list(self._executions.values())

        # Group by perm_id (permanent order ID — same across sessions)
        by_perm: dict[int, list[dict]] = defaultdict(list)
        for ex in execs:
            perm_id = ex.get("perm_id", 0)
            if perm_id:
                by_perm[perm_id].append(ex)
            else:
                # Fallback: use order_id if no perm_id
                by_perm[ex.get("order_id", id(ex))].append(ex)

        groups = []
        for perm_id, fills in by_perm.items():
            fills.sort(key=lambda f: f.get("time", ""))
            first = fills[0]

            # Determine order type from the legs
            sec_types = {f.get("sec_type", "") for f in fills}
            unique_con_ids = {f.get("con_id") for f in fills}
            sides = {f.get("side", "") for f in fills}

            if "OPT" in sec_types or "FOP" in sec_types:
                if len(unique_con_ids) >= 4:
                    order_type = "iron_condor"
                elif len(unique_con_ids) >= 2:
                    # Check if it's credit or debit
                    if "SLD" in sides and "BOT" in sides:
                        order_type = "credit_spread"  # or debit — refine below
                    else:
                        order_type = "multi_leg"
                else:
                    order_type = "single_option"
            else:
                order_type = "equity"

            # Build leg details
            legs = []
            total_commission = 0.0
            total_realized_pnl = 0.0
            net_amount = 0.0
            for f in fills:
                side = f.get("side", "")
                price = f.get("price", 0)
                shares = f.get("shares", 0)
                # BOT = bought (debit), SLD = sold (credit)
                sign = -1 if side == "BOT" else 1
                amount = sign * price * shares
                if f.get("sec_type") in ("OPT", "FOP"):
                    amount *= 100  # options multiplier

                strike = f.get("strike", 0)
                right = f.get("right", "")
                exp = f.get("expiration", "")
                if exp and len(exp) == 8:
                    exp = f"{exp[:4]}-{exp[4:6]}-{exp[6:8]}"

                legs.append({
                    "exec_id": f.get("exec_id"),
                    "con_id": f.get("con_id"),
                    "symbol": f.get("symbol"),
                    "sec_type": f.get("sec_type"),
                    "side": side,
                    "shares": shares,
                    "price": price,
                    "strike": strike,
                    "right": right,
                    "expiration": exp,
                    "local_symbol": f.get("local_symbol", ""),
                    "amount": round(amount, 2),
                    "commission": f.get("commission", 0),
                    "time": f.get("time"),
                })

                total_commission += f.get("commission", 0)
                total_realized_pnl += f.get("realized_pnl", 0)
                net_amount += amount

            # Refine credit vs debit spread
            if order_type == "credit_spread":
                if net_amount > 0:
                    order_type = "credit_spread"
                else:
                    order_type = "debit_spread"

            # Format expiration from first option leg
            expiration = ""
            for leg in legs:
                if leg.get("expiration"):
                    expiration = leg["expiration"]
                    break

            groups.append({
                "perm_id": perm_id,
                "order_id": first.get("order_id"),
                "time": first.get("time"),
                "symbol": first.get("symbol"),
                "order_type": order_type,
                "expiration": expiration,
                "legs": legs,
                "leg_count": len(unique_con_ids),
                "fill_count": len(fills),
                "net_amount": round(net_amount, 2),
                "total_commission": round(total_commission, 2),
                "total_realized_pnl": round(total_realized_pnl, 2),
            })

        groups.sort(key=lambda g: g.get("time", ""), reverse=True)
        return groups

    def flush(self) -> int:
        """Clear all stored executions. Returns count removed."""
        with self._lock:
            count = len(self._executions)
            self._executions.clear()
            self._save()
        return count

    @property
    def count(self) -> int:
        return len(self._executions)


# ── Module-level singleton ────────────────────────────────────────────────────

_execution_store: Optional[ExecutionStore] = None


def get_execution_store() -> Optional[ExecutionStore]:
    return _execution_store


def init_execution_store(data_dir: Path) -> ExecutionStore:
    global _execution_store
    path = data_dir / "executions.json"
    _execution_store = ExecutionStore(path)
    return _execution_store


def reset_execution_store() -> None:
    global _execution_store
    _execution_store = None
