"""
Common trading strategies and technical indicators for stock analysis.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime


def calculate_moving_average(
    symbol: str,
    records: List[Dict[str, Any]],
    ma_period: int,
    price_field: str = "price",
) -> List[Dict[str, Any]]:
    """
    Calculate simple moving average for stock data and add it to the records.

    Args:
        symbol: Stock symbol (for reference/logging)
        records: List of dictionaries containing date and price data
        ma_period: Number of periods for moving average calculation
        price_field: Field name containing the price data (default: 'price')

    Returns:
        List of records with moving average added where sufficient data exists
    """
    if not records or ma_period <= 0:
        return records

    # Sort records by date to ensure proper chronological order
    sorted_records = sorted(records, key=lambda x: x.get("date", ""))

    # Calculate moving averages
    for i in range(len(sorted_records)):
        if i >= ma_period - 1:  # We have enough data points
            # Get the last ma_period prices
            prices = []
            for j in range(i - ma_period + 1, i + 1):
                price = sorted_records[j].get(price_field)
                if price is not None:
                    prices.append(float(price))

            if len(prices) == ma_period:
                ma_value = sum(prices) / ma_period
                sorted_records[i][f"ma_{ma_period}"] = round(ma_value, 4)

    return sorted_records


def calculate_exponential_moving_average(
    symbol: str,
    records: List[Dict[str, Any]],
    ema_period: int,
    price_field: str = "price",
) -> List[Dict[str, Any]]:
    """
    Calculate exponential moving average for stock data and add it to the records.

    Args:
        symbol: Stock symbol (for reference/logging)
        records: List of dictionaries containing date and price data
        ema_period: Number of periods for EMA calculation
        price_field: Field name containing the price data (default: 'price')

    Returns:
        List of records with exponential moving average added where sufficient data exists
    """
    if not records or ema_period <= 0:
        return records

    # Sort records by date to ensure proper chronological order
    sorted_records = sorted(records, key=lambda x: x.get("date", ""))

    # Calculate smoothing factor (alpha)
    alpha = 2 / (ema_period + 1)

    # Initialize EMA with the first available price or simple moving average
    ema_value = None

    for i in range(len(sorted_records)):
        current_price = sorted_records[i].get(price_field)

        if current_price is not None:
            current_price = float(current_price)

            if ema_value is None:
                # Initialize EMA with first price or SMA of first ema_period prices
                if i >= ema_period - 1:
                    # Use SMA for initialization
                    prices = []
                    for j in range(i - ema_period + 1, i + 1):
                        price = sorted_records[j].get(price_field)
                        if price is not None:
                            prices.append(float(price))

                    if len(prices) == ema_period:
                        ema_value = sum(prices) / ema_period
                        sorted_records[i][f"ema_{ema_period}"] = round(ema_value, 4)
                else:
                    # Not enough data yet, continue
                    continue
            else:
                # Calculate EMA: EMA = (Current Price * Alpha) + (Previous EMA * (1 - Alpha))
                ema_value = (current_price * alpha) + (ema_value * (1 - alpha))
                sorted_records[i][f"ema_{ema_period}"] = round(ema_value, 4)

    return sorted_records


def add_multiple_moving_averages(
    symbol: str,
    records: List[Dict[str, Any]],
    ma_periods: List[int],
    price_field: str = "price",
) -> List[Dict[str, Any]]:
    """
    Calculate multiple simple moving averages for stock data.

    Args:
        symbol: Stock symbol (for reference/logging)
        records: List of dictionaries containing date and price data
        ma_periods: List of periods for moving average calculations
        price_field: Field name containing the price data (default: 'price')

    Returns:
        List of records with all moving averages added
    """
    result_records = records.copy()

    for period in ma_periods:
        result_records = calculate_moving_average(
            symbol, result_records, period, price_field
        )

    return result_records


def add_multiple_exponential_moving_averages(
    symbol: str,
    records: List[Dict[str, Any]],
    ema_periods: List[int],
    price_field: str = "price",
) -> List[Dict[str, Any]]:
    """
    Calculate multiple exponential moving averages for stock data.

    Args:
        symbol: Stock symbol (for reference/logging)
        records: List of dictionaries containing date and price data
        ema_periods: List of periods for EMA calculations
        price_field: Field name containing the price data (default: 'price')

    Returns:
        List of records with all exponential moving averages added
    """
    result_records = records.copy()

    for period in ema_periods:
        result_records = calculate_exponential_moving_average(
            symbol, result_records, period, price_field
        )

    return result_records


def add_rsi(
    symbol: str,
    records: List[Dict[str, Any]],
    rsi_period: int = 14,
    price_field: str = "price",
    output_field_prefix: str = "rsi_",
) -> List[Dict[str, Any]]:
    """
    Compute Relative Strength Index (RSI) over a list of records and add it per record.

    Args:
        symbol: Stock symbol (for reference/logging)
        records: List of dicts with at least a date and price field
        rsi_period: RSI lookback period (default: 14)
        price_field: Field name used as price (default: 'price')
        output_field_prefix: Output field name prefix (default: 'rsi_')

    Returns:
        List of records with computed RSI values placed in key f"{output_field_prefix}{rsi_period}"
    """
    if not records or rsi_period <= 0:
        return records

    # Sort by date to ensure order
    sorted_records = sorted(records, key=lambda x: x.get("date", ""))

    prices: List[Optional[float]] = []
    for r in sorted_records:
        val = r.get(price_field)
        prices.append(float(val) if val is not None else None)

    # Compute deltas
    deltas: List[Optional[float]] = [None]
    for i in range(1, len(prices)):
        if prices[i] is None or prices[i - 1] is None:
            deltas.append(None)
        else:
            deltas.append(prices[i] - prices[i - 1])

    # Rolling average gains and losses (simple mean over window)
    gains: List[Optional[float]] = [None] * len(deltas)
    losses: List[Optional[float]] = [None] * len(deltas)

    for i in range(len(deltas)):
        if i < rsi_period:
            continue
        window = deltas[i - rsi_period + 1 : i + 1]
        pos = [d for d in window if d is not None and d > 0]
        neg = [(-d) for d in window if d is not None and d < 0]
        avg_gain = (sum(pos) / rsi_period) if len(window) == rsi_period else None
        avg_loss = (sum(neg) / rsi_period) if len(window) == rsi_period else None
        gains[i] = avg_gain
        losses[i] = avg_loss

    # Compute RSI
    out_field = f"{output_field_prefix}{rsi_period}"
    for i in range(len(sorted_records)):
        rsi_value: Optional[float] = None
        if i >= rsi_period and gains[i] is not None and losses[i] is not None:
            if losses[i] == 0:
                rsi_value = 100.0
            else:
                rs = gains[i] / losses[i]
                rsi_value = 100.0 - (100.0 / (1.0 + rs))
        if rsi_value is not None:
            sorted_records[i][out_field] = round(rsi_value, 2)

    return sorted_records


def compute_rsi_series(prices: "pd.Series", window: int = 14) -> "pd.Series":
    """
    Compute RSI for a pandas Series of prices. This is provided for convenience
    when working with DataFrames elsewhere in the codebase.
    """
    # Local import to avoid hard dependency when pandas is not used
    import pandas as pd  # type: ignore

    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=window, min_periods=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window, min_periods=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Example usage and testing functions
def example_usage():
    """
    Example of how to use the moving average functions.
    """
    # Sample data
    sample_records = [
        {"date": "2024-01-01", "price": 100.0},
        {"date": "2024-01-02", "price": 102.0},
        {"date": "2024-01-03", "price": 101.0},
        {"date": "2024-01-04", "price": 103.0},
        {"date": "2024-01-05", "price": 105.0},
        {"date": "2024-01-06", "price": 104.0},
        {"date": "2024-01-07", "price": 106.0},
        {"date": "2024-01-08", "price": 108.0},
        {"date": "2024-01-09", "price": 107.0},
        {"date": "2024-01-10", "price": 109.0},
    ]

    # Calculate 5-day moving average
    ma_records = calculate_moving_average("AAPL", sample_records, 5)

    # Calculate 5-day exponential moving average
    ema_records = calculate_exponential_moving_average("AAPL", sample_records, 5)

    # Calculate multiple moving averages
    multi_ma_records = add_multiple_moving_averages("AAPL", sample_records, [3, 5, 10])

    # Calculate multiple EMAs
    multi_ema_records = add_multiple_exponential_moving_averages(
        "AAPL", sample_records, [3, 5, 10]
    )

    return {
        "ma_records": ma_records,
        "ema_records": ema_records,
        "multi_ma_records": multi_ma_records,
        "multi_ema_records": multi_ema_records,
    }


if __name__ == "__main__":
    # Run example
    results = example_usage()

    print("Sample records with 5-day MA:")
    for record in results["ma_records"]:
        print(
            f"Date: {record['date']}, Price: {record['price']}, MA_5: {record.get('ma_5', 'N/A')}"
        )

    print("\nSample records with 5-day EMA:")
    for record in results["ema_records"]:
        print(
            f"Date: {record['date']}, Price: {record['price']}, EMA_5: {record.get('ema_5', 'N/A')}"
        )
