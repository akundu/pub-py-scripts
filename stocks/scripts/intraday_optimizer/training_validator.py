"""
Training and validation period management.

Handles splitting data into training and validation periods,
with support for various time-based splits.
"""

from datetime import datetime, timedelta
from typing import Tuple
from pathlib import Path
import sys

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TrainingValidator:
    """Split data into training and validation periods."""

    def __init__(
        self,
        training_months: int = 3,
        validation_days: int = 30
    ):
        """
        Initialize with training/validation split.

        Args:
            training_months: Number of months for training
            validation_days: Number of days for validation
        """
        self.training_months = training_months
        self.validation_days = validation_days

    def get_periods(self, end_date: str) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        """
        Get training and validation periods.

        Args:
            end_date: Final date (YYYY-MM-DD)

        Returns:
            Tuple of ((train_start, train_end), (val_start, val_end))
        """
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        # Validation period: last N days
        val_end = end_dt
        val_start = end_dt - timedelta(days=self.validation_days - 1)

        # Training period: N months before validation
        train_end = val_start - timedelta(days=1)
        train_start = train_end - timedelta(days=30 * self.training_months)

        return (
            (train_start.strftime('%Y-%m-%d'), train_end.strftime('%Y-%m-%d')),
            (val_start.strftime('%Y-%m-%d'), val_end.strftime('%Y-%m-%d'))
        )

    def get_training_period(self, end_date: str) -> Tuple[str, str]:
        """Get training period only."""
        (train_start, train_end), _ = self.get_periods(end_date)
        return (train_start, train_end)

    def get_validation_period(self, end_date: str) -> Tuple[str, str]:
        """Get validation period only."""
        _, (val_start, val_end) = self.get_periods(end_date)
        return (val_start, val_end)
