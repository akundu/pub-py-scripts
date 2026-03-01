"""Shared P&L calculations for spread instruments.

Wraps credit_spread_utils/backtest_engine.py calculate_spread_pnl().
"""


def calculate_spread_pnl(
    initial_credit: float,
    short_strike: float,
    long_strike: float,
    underlying_price: float,
    option_type: str,
) -> float:
    """Calculate P&L per share for a credit spread at a given underlying price.

    Args:
        initial_credit: Credit received per share when opening spread.
        short_strike: Strike of the short leg.
        long_strike: Strike of the long leg.
        underlying_price: Current/settlement price.
        option_type: 'put' or 'call'.

    Returns:
        P&L per share (positive = profit, negative = loss).
    """
    if option_type.lower() == "put":
        if underlying_price >= short_strike:
            spread_value = 0.0
        elif underlying_price <= long_strike:
            spread_value = short_strike - long_strike
        else:
            spread_value = short_strike - underlying_price
    else:  # call
        if underlying_price <= short_strike:
            spread_value = 0.0
        elif underlying_price >= long_strike:
            spread_value = long_strike - short_strike
        else:
            spread_value = underlying_price - short_strike

    return initial_credit - spread_value


def calculate_iron_condor_pnl(
    put_credit: float,
    call_credit: float,
    put_short_strike: float,
    put_long_strike: float,
    call_short_strike: float,
    call_long_strike: float,
    underlying_price: float,
) -> float:
    """Calculate P&L per share for an iron condor."""
    put_pnl = calculate_spread_pnl(
        put_credit, put_short_strike, put_long_strike, underlying_price, "put"
    )
    call_pnl = calculate_spread_pnl(
        call_credit, call_short_strike, call_long_strike, underlying_price, "call"
    )
    return put_pnl + call_pnl


def calculate_strangle_pnl(
    put_credit: float,
    call_credit: float,
    put_strike: float,
    call_strike: float,
    underlying_price: float,
) -> float:
    """Calculate P&L per share for a short strangle."""
    # Put intrinsic
    put_value = max(0.0, put_strike - underlying_price)
    # Call intrinsic
    call_value = max(0.0, underlying_price - call_strike)

    total_credit = put_credit + call_credit
    return total_credit - put_value - call_value


def calculate_straddle_pnl(
    total_credit: float,
    strike: float,
    underlying_price: float,
) -> float:
    """Calculate P&L per share for a short straddle."""
    intrinsic = abs(underlying_price - strike)
    return total_credit - intrinsic
