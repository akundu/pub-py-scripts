"""Sliding window rate limiter for transaction throttling."""

import asyncio
import time
from collections import deque
from typing import Optional
import logging


class SlidingWindowRateLimiter:
    """
    Limits to N transactions in M seconds.
    When N=0 or M=0, rate limiting is disabled.
    """

    def __init__(self, max_transactions: int = 0, window_seconds: float = 0, logger: Optional[logging.Logger] = None):
        self.max_transactions = max_transactions
        self.window_seconds = window_seconds
        self.logger = logger
        self._timestamps: deque = deque()
        self._enabled = max_transactions > 0 and window_seconds > 0

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def _cleanup_old(self) -> None:
        if not self._enabled:
            return
        cutoff = time.monotonic() - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    async def acquire(self) -> None:
        """Wait until a slot is available, then record the transaction."""
        if not self._enabled:
            return

        while True:
            self._cleanup_old()
            if len(self._timestamps) < self.max_transactions:
                self._timestamps.append(time.monotonic())
                return

            # Wait until oldest expires
            oldest = self._timestamps[0]
            wait_time = (oldest + self.window_seconds) - time.monotonic()
            if wait_time > 0:
                if self.logger:
                    self.logger.info(f"Rate limit reached ({self.max_transactions} in {self.window_seconds}s). Waiting {wait_time:.2f}s...")
                await asyncio.sleep(wait_time + 0.01)

    def get_stats(self) -> dict:
        self._cleanup_old()
        return {
            'enabled': self._enabled,
            'current': len(self._timestamps),
            'max': self.max_transactions,
            'window': self.window_seconds,
        }
