from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


class SimulationClock:
    """Time control singleton for UTP simulation mode.

    Controls what time "now" is when replaying historical data from CSV files.
    Timestamps come from the equity CSV (loaded externally) and advance() steps
    to the next real data point in the sorted list.
    """

    def __init__(self, sim_date: date, timestamps: list[datetime]) -> None:
        self.sim_date = sim_date
        self.timestamps = sorted(timestamps)
        self.sim_time: datetime = self.timestamps[0] if self.timestamps else datetime.combine(
            sim_date, datetime.min.time(), tzinfo=timezone.utc
        )
        self._cursor: int = 0
        self.auto_advancing: bool = False
        self._auto_task: asyncio.Task | None = None

    def set_time(self, ts: datetime) -> None:
        """Jump to an exact UTC timestamp."""
        self.sim_time = ts
        # Snap cursor to nearest timestamp at or after ts
        for i, t in enumerate(self.timestamps):
            if t >= ts:
                self._cursor = i
                return
        self._cursor = len(self.timestamps) - 1

    def advance(self, minutes: int = 5) -> datetime | None:
        """Step forward to the next available timestamp.

        The minutes parameter is ignored; we always step to the next real data
        point in the sorted timestamps list. Returns the new timestamp or None
        if at end.
        """
        if self._cursor + 1 >= len(self.timestamps):
            return None
        self._cursor += 1
        self.sim_time = self.timestamps[self._cursor]
        return self.sim_time

    def jump_to_et(self, time_str: str) -> None:
        """Jump to a wall-clock ET time, e.g. '10:30', resolved to UTC."""
        hour, minute = (int(p) for p in time_str.split(":"))
        et_dt = datetime(
            self.sim_date.year, self.sim_date.month, self.sim_date.day,
            hour, minute, tzinfo=ET,
        )
        utc_dt = et_dt.astimezone(timezone.utc)
        self.set_time(utc_dt)

    def reset(self) -> None:
        """Back to the first available timestamp (market open)."""
        self._cursor = 0
        if self.timestamps:
            self.sim_time = self.timestamps[0]

    def is_active(self) -> bool:
        """Always True when simulation is loaded."""
        return True

    def now(self) -> datetime:
        """Return the current simulation time."""
        return self.sim_time

    def start_auto_advance(self, interval_sec: float = 3.0) -> None:
        """Tick forward every interval_sec seconds via asyncio task."""
        if self.auto_advancing:
            return
        self.auto_advancing = True
        self._auto_task = asyncio.ensure_future(self._auto_advance_loop(interval_sec))

    def stop_auto_advance(self) -> None:
        """Stop auto-advance."""
        self.auto_advancing = False
        if self._auto_task is not None:
            self._auto_task.cancel()
            self._auto_task = None

    async def _auto_advance_loop(self, interval_sec: float) -> None:
        try:
            while self.auto_advancing:
                await asyncio.sleep(interval_sec)
                ts = self.advance()
                if ts is None:
                    logger.info("SimulationClock: reached end of timestamps, stopping auto-advance")
                    self.auto_advancing = False
                    break
                logger.debug("SimulationClock: advanced to %s", ts.isoformat())
        except asyncio.CancelledError:
            pass
        finally:
            self.auto_advancing = False


# ---------------------------------------------------------------------------
# Module-level accessors
# ---------------------------------------------------------------------------

_sim_clock: SimulationClock | None = None


def init_sim_clock(sim_date: date, timestamps: list[datetime]) -> SimulationClock:
    global _sim_clock
    _sim_clock = SimulationClock(sim_date, timestamps)
    logger.info(
        "SimulationClock initialised: date=%s, %d timestamps, first=%s, last=%s",
        sim_date,
        len(timestamps),
        timestamps[0].isoformat() if timestamps else "N/A",
        timestamps[-1].isoformat() if timestamps else "N/A",
    )
    return _sim_clock


def get_sim_clock() -> SimulationClock | None:
    return _sim_clock


def reset_sim_clock() -> None:
    global _sim_clock
    if _sim_clock is not None:
        _sim_clock.stop_auto_advance()
    _sim_clock = None
