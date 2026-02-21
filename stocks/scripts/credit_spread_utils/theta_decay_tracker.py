"""
Theta decay tracking using actual option prices from CSV data.

This module loads intraday option prices to calculate real theta decay
(not theoretical Black-Scholes), which is critical for accurate exit price estimation.
"""

from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import logging
import csv

logger = logging.getLogger(__name__)


class ThetaDecayTracker:
    """Track theta decay using actual option prices from CSV data."""

    def __init__(self, csv_base_dir: str = "options_csv_output"):
        """
        Initialize theta decay tracker.

        Args:
            csv_base_dir: Base directory for CSV files
                - 0 DTE: options_csv_output/{ticker}/
                - Non-0 DTE: options_csv_output_full/{ticker}/
        """
        self.csv_base_dir = Path(csv_base_dir)

    def get_csv_path(
        self,
        ticker: str,
        date: datetime,
        dte: int
    ) -> Path:
        """
        Get path to CSV file for a specific date and DTE.

        Args:
            ticker: Underlying ticker (e.g., 'NDX')
            date: Trading date
            dte: Days to expiration

        Returns:
            Path to CSV file
        """
        date_str = date.strftime('%Y-%m-%d')

        if dte == 0:
            # 0 DTE files in options_csv_output/{ticker}/
            base_dir = self.csv_base_dir / ticker
        else:
            # Non-0 DTE files in options_csv_output_full/{ticker}/
            base_dir = self.csv_base_dir.parent / "options_csv_output_full" / ticker

        csv_path = base_dir / f"{date_str}.csv"
        return csv_path

    def load_option_prices_intraday(
        self,
        ticker: str,
        date: datetime,
        dte: int,
        strike: float,
        option_type: str,
        expiration_date: Optional[datetime] = None
    ) -> List[Tuple[datetime, float]]:
        """
        Load intraday option prices from CSV.

        Args:
            ticker: Underlying ticker
            date: Trading date
            dte: Days to expiration
            strike: Strike price
            option_type: 'call' or 'put'
            expiration_date: Optional expiration date (for non-0 DTE)

        Returns:
            List of (timestamp, mid_price) tuples sorted by time
        """
        csv_path = self.get_csv_path(ticker, date, dte)

        if not csv_path.exists():
            logger.warning(f"CSV file not found: {csv_path}")
            return []

        prices = []

        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    # Filter by strike and option type
                    row_strike = float(row.get('strike', 0))
                    row_type = row.get('option_type', '').lower()

                    if row_strike != strike or row_type != option_type.lower():
                        continue

                    # For non-0 DTE, also filter by expiration
                    if dte > 0 and expiration_date:
                        row_exp = row.get('expiration')
                        if row_exp:
                            exp_str = expiration_date.strftime('%Y-%m-%d')
                            if row_exp != exp_str:
                                continue

                    # Parse timestamp and price
                    timestamp_str = row.get('timestamp')
                    bid = float(row.get('bid', 0))
                    ask = float(row.get('ask', 0))

                    if bid > 0 and ask > 0:
                        mid_price = (bid + ask) / 2
                        timestamp = datetime.fromisoformat(timestamp_str)
                        prices.append((timestamp, mid_price))

            prices.sort(key=lambda x: x[0])  # Sort by timestamp
            logger.debug(
                f"Loaded {len(prices)} price points for {ticker} "
                f"{option_type} {strike} on {date.date()}"
            )

        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")
            return []

        return prices

    def calculate_theta_decay_rate(
        self,
        entry_time: datetime,
        entry_price: float,
        intraday_prices: List[Tuple[datetime, float]]
    ) -> Dict[str, float]:
        """
        Calculate actual theta decay rate from price data.

        Args:
            entry_time: Entry timestamp
            entry_price: Entry option price
            intraday_prices: List of (timestamp, price) tuples

        Returns:
            Dictionary with:
                - avg_hourly_decay: Average decay per hour (dollars)
                - avg_hourly_decay_pct: Average decay per hour (percentage)
                - total_decay_pct: Total decay percentage
                - predicted_eod_price: Predicted end-of-day price
                - hours_tracked: Number of hours tracked
        """
        if not intraday_prices or entry_price <= 0:
            logger.warning("Insufficient data for theta decay calculation")
            return {
                'avg_hourly_decay': 0.0,
                'avg_hourly_decay_pct': 0.0,
                'total_decay_pct': 0.0,
                'predicted_eod_price': entry_price,
                'hours_tracked': 0
            }

        # Filter prices after entry time
        future_prices = [
            (ts, price) for ts, price in intraday_prices
            if ts > entry_time
        ]

        if not future_prices:
            logger.debug("No price data after entry time")
            return {
                'avg_hourly_decay': 0.0,
                'avg_hourly_decay_pct': 0.0,
                'total_decay_pct': 0.0,
                'predicted_eod_price': entry_price,
                'hours_tracked': 0
            }

        # Calculate decay from entry to last observed price
        last_time, last_price = future_prices[-1]
        hours_elapsed = (last_time - entry_time).total_seconds() / 3600

        if hours_elapsed <= 0:
            return {
                'avg_hourly_decay': 0.0,
                'avg_hourly_decay_pct': 0.0,
                'total_decay_pct': 0.0,
                'predicted_eod_price': entry_price,
                'hours_tracked': 0
            }

        # Calculate decay (negative if price decreased)
        total_decay = last_price - entry_price
        total_decay_pct = (total_decay / entry_price) * 100
        avg_hourly_decay = total_decay / hours_elapsed
        avg_hourly_decay_pct = (avg_hourly_decay / entry_price) * 100

        # Predict EOD price (assume market closes at 4 PM = 16:00)
        eod_time = entry_time.replace(hour=16, minute=0, second=0, microsecond=0)
        hours_to_eod = (eod_time - entry_time).total_seconds() / 3600

        if hours_to_eod > 0:
            predicted_eod_price = entry_price + (avg_hourly_decay * hours_to_eod)
        else:
            predicted_eod_price = last_price

        logger.debug(
            f"Theta decay: {avg_hourly_decay:.2f}/hr ({avg_hourly_decay_pct:.1f}%/hr), "
            f"total={total_decay_pct:.1f}% over {hours_elapsed:.1f}hrs"
        )

        return {
            'avg_hourly_decay': avg_hourly_decay,
            'avg_hourly_decay_pct': avg_hourly_decay_pct,
            'total_decay_pct': total_decay_pct,
            'predicted_eod_price': predicted_eod_price,
            'hours_tracked': hours_elapsed
        }

    def estimate_exit_price(
        self,
        entry_price: float,
        hours_held: float,
        theta_decay_rate: float
    ) -> float:
        """
        Estimate exit price based on theta decay rate.

        Args:
            entry_price: Entry option price
            hours_held: Hours position held
            theta_decay_rate: Hourly decay rate (from calculate_theta_decay_rate)

        Returns:
            Estimated exit price
        """
        exit_price = entry_price + (theta_decay_rate * hours_held)

        # Floor at $0 (options can't go negative)
        exit_price = max(0.0, exit_price)

        logger.debug(
            f"Estimated exit price: {entry_price:.2f} + "
            f"({theta_decay_rate:.2f} * {hours_held:.1f}hrs) = {exit_price:.2f}"
        )

        return exit_price

    def get_price_at_time(
        self,
        intraday_prices: List[Tuple[datetime, float]],
        target_time: datetime,
        tolerance_minutes: int = 5
    ) -> Optional[float]:
        """
        Get option price at a specific time (or closest match).

        Args:
            intraday_prices: List of (timestamp, price) tuples
            target_time: Target timestamp
            tolerance_minutes: Max time difference allowed (minutes)

        Returns:
            Price at target time, or None if no match within tolerance
        """
        if not intraday_prices:
            return None

        # Find closest timestamp
        closest = min(
            intraday_prices,
            key=lambda x: abs((x[0] - target_time).total_seconds())
        )

        time_diff_seconds = abs((closest[0] - target_time).total_seconds())
        time_diff_minutes = time_diff_seconds / 60

        if time_diff_minutes <= tolerance_minutes:
            logger.debug(
                f"Found price {closest[1]:.2f} at {closest[0]} "
                f"(target: {target_time}, diff: {time_diff_minutes:.1f}min)"
            )
            return closest[1]
        else:
            logger.debug(
                f"No price within {tolerance_minutes}min of {target_time} "
                f"(closest: {time_diff_minutes:.1f}min)"
            )
            return None


# Convenience function
def track_theta_decay(
    ticker: str,
    date: datetime,
    dte: int,
    strike: float,
    option_type: str,
    entry_time: datetime,
    entry_price: float,
    csv_base_dir: str = "options_csv_output"
) -> Dict[str, float]:
    """
    Simple function to track theta decay for an option.

    Returns:
        Theta decay metrics dictionary
    """
    tracker = ThetaDecayTracker(csv_base_dir)

    prices = tracker.load_option_prices_intraday(
        ticker, date, dte, strike, option_type
    )

    return tracker.calculate_theta_decay_rate(entry_time, entry_price, prices)
