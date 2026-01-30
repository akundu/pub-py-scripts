"""
Time-block based rate limiter for transaction throttling.

Allows different rate limits for different time blocks within the trading day,
with evenly spaced transaction slots within each block.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, date
from typing import List, Optional, Dict, Tuple
import logging


@dataclass
class TimeBlock:
    """
    Defines a time block with its transaction limit.

    Attributes:
        start: Start time of the block (e.g., time(6, 30))
        end: End time of the block (e.g., time(7, 0))
        max_transactions: Maximum transactions allowed in this block
    """
    start: time
    end: time
    max_transactions: int

    def duration_seconds(self) -> float:
        """Get block duration in seconds."""
        start_dt = datetime.combine(date.today(), self.start)
        end_dt = datetime.combine(date.today(), self.end)
        return (end_dt - start_dt).total_seconds()

    def contains_time(self, t: time) -> bool:
        """Check if a time falls within this block."""
        return self.start <= t < self.end

    def __hash__(self):
        return hash((self.start, self.end, self.max_transactions))


@dataclass
class TimeBlockRateLimiter:
    """
    Rate limiter with different limits for different time blocks.

    Transactions are evenly spaced within each block. For example,
    3 transactions in a 30-minute block means slots at 0, 10, and 20 minutes.

    Attributes:
        blocks: List of TimeBlock configurations
        logger: Optional logger for status messages
    """
    blocks: List[TimeBlock]
    logger: Optional[logging.Logger] = None
    _slots_used: Dict[Tuple[date, time, time], List[datetime]] = field(default_factory=dict)

    def __post_init__(self):
        # Sort blocks by start time
        self.blocks = sorted(self.blocks, key=lambda b: b.start)

    @classmethod
    def from_string(cls, config_str: str, logger: Optional[logging.Logger] = None) -> 'TimeBlockRateLimiter':
        """
        Parse rate limit blocks from CLI string format.

        Format: "HH:MM-HH:MM=N,HH:MM-HH:MM=N,..."
        Example: "06:30-07:00=3,07:00-08:00=4,08:00-09:00=2"

        Args:
            config_str: Configuration string
            logger: Optional logger

        Returns:
            TimeBlockRateLimiter instance
        """
        blocks = []
        for block_str in config_str.split(','):
            block_str = block_str.strip()
            if not block_str:
                continue

            # Parse "HH:MM-HH:MM=N"
            time_part, count_part = block_str.split('=')
            start_str, end_str = time_part.split('-')

            start_h, start_m = map(int, start_str.split(':'))
            end_h, end_m = map(int, end_str.split(':'))
            max_trans = int(count_part)

            blocks.append(TimeBlock(
                start=time(start_h, start_m),
                end=time(end_h, end_m),
                max_transactions=max_trans
            ))

        return cls(blocks=blocks, logger=logger)

    def get_current_block(self, now: datetime) -> Optional[TimeBlock]:
        """Get the time block containing the current time."""
        current_time = now.time()
        for block in self.blocks:
            if block.contains_time(current_time):
                return block
        return None

    def get_evenly_spaced_slots(self, block: TimeBlock, target_date: date) -> List[datetime]:
        """
        Calculate evenly spaced transaction slots within a block.

        For N transactions in a block, slots are at 0%, 1/(N), 2/(N), ... of the block.
        Example: 3 transactions in 30 min → slots at 0, 10, 20 minutes from start.

        Args:
            block: The time block
            target_date: The date for the slots

        Returns:
            List of datetime slots
        """
        if block.max_transactions <= 0:
            return []

        block_start = datetime.combine(target_date, block.start)
        duration = block.duration_seconds()

        # Space evenly: first slot at start, then every (duration / max_trans)
        interval = duration / block.max_transactions

        slots = []
        for i in range(block.max_transactions):
            slot_offset = timedelta(seconds=i * interval)
            slots.append(block_start + slot_offset)

        return slots

    def _get_block_key(self, block: TimeBlock, target_date: date) -> Tuple[date, time, time]:
        """Get a hashable key for tracking slots used in a block on a date."""
        return (target_date, block.start, block.end)

    def get_slots_used(self, block: TimeBlock, target_date: date) -> List[datetime]:
        """Get list of slots already used for a block on a date."""
        key = self._get_block_key(block, target_date)
        return self._slots_used.get(key, [])

    def mark_slot_used(self, block: TimeBlock, slot: datetime):
        """Mark a slot as used."""
        key = self._get_block_key(block, slot.date())
        if key not in self._slots_used:
            self._slots_used[key] = []
        self._slots_used[key].append(slot)

    def get_available_slots(self, block: TimeBlock, target_date: date, now: datetime) -> List[datetime]:
        """
        Get available (unused and not in future) slots for a block.

        Args:
            block: The time block
            target_date: The date
            now: Current datetime

        Returns:
            List of available slots
        """
        all_slots = self.get_evenly_spaced_slots(block, target_date)
        used = self.get_slots_used(block, target_date)

        # Available: not used and slot time has passed (or is now)
        available = [s for s in all_slots if s not in used and s <= now]
        return available

    def get_next_future_slot(self, block: TimeBlock, target_date: date, now: datetime) -> Optional[datetime]:
        """Get the next future slot that hasn't been used."""
        all_slots = self.get_evenly_spaced_slots(block, target_date)
        used = self.get_slots_used(block, target_date)

        future_unused = [s for s in all_slots if s not in used and s > now]
        return future_unused[0] if future_unused else None

    async def acquire(self) -> bool:
        """
        Wait until a slot is available, then mark it as used.

        Returns:
            True if a slot was acquired, False if outside all blocks
        """
        now = datetime.now()
        block = self.get_current_block(now)

        if not block:
            # Outside all defined blocks - no rate limiting
            return True

        # Check for available slots (past or current time, not used)
        available = self.get_available_slots(block, now.date(), now)

        if available:
            # Use the earliest available slot
            slot = available[0]
            self.mark_slot_used(block, slot)
            if self.logger:
                self.logger.debug(f"Acquired slot at {slot.strftime('%H:%M:%S')}")
            return True

        # No available slots - check for future slots
        next_slot = self.get_next_future_slot(block, now.date(), now)

        if next_slot:
            wait_seconds = (next_slot - now).total_seconds()
            if self.logger:
                self.logger.info(
                    f"Rate limit reached for {block.start}-{block.end}. "
                    f"Waiting {wait_seconds:.1f}s until {next_slot.strftime('%H:%M:%S')}"
                )
            await asyncio.sleep(wait_seconds + 0.1)  # Small buffer
            self.mark_slot_used(block, next_slot)
            return True

        # All slots used in this block
        if self.logger:
            self.logger.warning(
                f"All {block.max_transactions} slots used for block {block.start}-{block.end}"
            )
        return False

    def check_can_acquire(self, now: Optional[datetime] = None) -> Tuple[bool, Optional[float]]:
        """
        Check if a slot can be acquired without waiting.

        Args:
            now: Current time (defaults to datetime.now())

        Returns:
            Tuple of (can_acquire_now, seconds_until_next_slot_or_None)
        """
        if now is None:
            now = datetime.now()

        block = self.get_current_block(now)

        if not block:
            return (True, None)  # Outside blocks, no limit

        available = self.get_available_slots(block, now.date(), now)
        if available:
            return (True, None)

        next_slot = self.get_next_future_slot(block, now.date(), now)
        if next_slot:
            wait_seconds = (next_slot - now).total_seconds()
            return (False, wait_seconds)

        return (False, None)  # All slots used

    def get_stats(self) -> Dict:
        """Get current rate limiter statistics."""
        now = datetime.now()
        block = self.get_current_block(now)

        stats = {
            'current_block': None,
            'blocks_configured': len(self.blocks),
            'total_slots_used_today': 0,
        }

        if block:
            used = self.get_slots_used(block, now.date())
            stats['current_block'] = {
                'start': block.start.strftime('%H:%M'),
                'end': block.end.strftime('%H:%M'),
                'max_transactions': block.max_transactions,
                'used': len(used),
                'remaining': block.max_transactions - len(used),
            }

        # Count total slots used today
        today = now.date()
        for key, slots in self._slots_used.items():
            if key[0] == today:
                stats['total_slots_used_today'] += len(slots)

        return stats

    def reset_day(self, target_date: Optional[date] = None):
        """Reset slots for a specific date (or all dates if None)."""
        if target_date is None:
            self._slots_used.clear()
        else:
            keys_to_remove = [k for k in self._slots_used if k[0] == target_date]
            for k in keys_to_remove:
                del self._slots_used[k]
