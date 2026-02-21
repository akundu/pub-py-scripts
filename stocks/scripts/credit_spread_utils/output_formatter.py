"""
Output formatting and display utilities for credit spread analysis.

Functions for formatting and printing analysis results, including
summary views, detailed views, and best-option displays.
"""

from typing import Dict, List, Optional, Any

from .timezone_utils import format_timestamp


def format_best_current_option(result: Dict, output_tz, use_mid_price: bool = False) -> str:
    """Format a single result as a best-current-option one-liner.

    Args:
        result: Analysis result dictionary
        output_tz: Output timezone
        use_mid_price: Whether mid-price was used

    Returns:
        Formatted string for display
    """
    timestamp_str = format_timestamp(result['timestamp'], output_tz)
    max_credit = result['best_spread'].get('total_credit')
    if max_credit is None:
        max_credit = result['best_spread'].get('net_credit_per_contract', 0)
    num_contracts = result['best_spread'].get('num_contracts', 0)
    if num_contracts is None:
        num_contracts = 0
    opt_type_upper = result.get('option_type', 'UNKNOWN').upper()
    short_strike = result['best_spread']['short_strike']
    long_strike = result['best_spread']['long_strike']
    short_premium = result['best_spread']['short_price']
    long_premium = result['best_spread']['long_price']
    _short_lbl = "Short (bid)" if not use_mid_price else "Short"
    _long_lbl = "Long (ask)" if not use_mid_price else "Long"
    return (
        f"BEST CURRENT OPTION: {timestamp_str} | Type: {opt_type_upper} | "
        f"Max Credit: ${max_credit:.2f} | Contracts: {num_contracts} | "
        f"Spread: ${short_strike:.2f}/${long_strike:.2f} | "
        f"{_short_lbl}: ${short_premium:.2f} {_long_lbl}: ${long_premium:.2f}"
    )


def format_summary_line(result: Dict, output_tz, use_mid_price: bool = False) -> str:
    """Format a single result as a summary line.

    Args:
        result: Analysis result dictionary
        output_tz: Output timezone
        use_mid_price: Whether mid-price was used

    Returns:
        Formatted summary line
    """
    timestamp_str = format_timestamp(result['timestamp'], output_tz)
    max_credit = result['best_spread'].get('total_credit')
    if max_credit is None:
        max_credit = result['best_spread'].get('net_credit_per_contract', 0)
    num_contracts = result['best_spread'].get('num_contracts', 0)
    if num_contracts is None:
        num_contracts = 0
    opt_type_upper = result.get('option_type', 'UNKNOWN').upper()
    short_strike = result['best_spread']['short_strike']
    long_strike = result['best_spread']['long_strike']

    # Backtest indicators
    backtest_result = result.get('backtest_successful')
    backtest_indicator = ""
    profit_target_indicator = ""
    pnl_str = ""

    if backtest_result is True:
        backtest_indicator = " ✓"
        if result.get('profit_target_hit') is True:
            profit_target_indicator = " [PT]"
    elif backtest_result is False:
        backtest_indicator = " ✗"

    actual_pnl_per_share = result.get('actual_pnl_per_share')
    if actual_pnl_per_share is not None and num_contracts:
        total_pnl = actual_pnl_per_share * num_contracts * 100
        pnl_str = f" | P&L: ${total_pnl:+.2f}"

    return (
        f"{timestamp_str} | Type: {opt_type_upper} | Max Credit: ${max_credit:.2f} | "
        f"Contracts: {num_contracts} | Spread: ${short_strike:.2f}/${long_strike:.2f}"
        f"{backtest_indicator}{profit_target_indicator}{pnl_str}"
    )


def build_summary_parts(
    results: List[Dict],
    output_tz,
    option_type: str,
    top_n: Optional[int] = None,
    use_mid_price: bool = False,
) -> List[str]:
    """Build summary parts for final one-line summary.

    Args:
        results: Analysis results
        output_tz: Output timezone
        option_type: 'call', 'put', or 'both'
        top_n: Optional top-N filter applied
        use_mid_price: Whether mid-price was used

    Returns:
        List of summary part strings
    """
    if not results:
        return []

    overall_best_call = None
    overall_best_put = None
    max_credit_call = 0
    max_credit_put = 0
    total_options = len(results)
    backtest_success_count = 0
    backtest_failure_count = 0

    for result in results:
        max_credit = result['best_spread'].get('total_credit')
        if max_credit is None:
            max_credit = result['best_spread'].get('net_credit_per_contract', 0)

        backtest_result = result.get('backtest_successful')
        if backtest_result is True:
            backtest_success_count += 1
        elif backtest_result is False:
            backtest_failure_count += 1

        opt_type = result.get('option_type', 'UNKNOWN').lower()
        if opt_type == 'call':
            if max_credit > max_credit_call:
                max_credit_call = max_credit
                overall_best_call = result
        elif opt_type == 'put':
            if max_credit > max_credit_put:
                max_credit_put = max_credit
                overall_best_put = result

    summary_parts = []
    if top_n:
        summary_parts.append(f"Total Options: {total_options} (Top-{top_n} per day)")
    else:
        summary_parts.append(f"Total Options: {total_options}")

    if backtest_success_count > 0 or backtest_failure_count > 0:
        backtest_total = backtest_success_count + backtest_failure_count
        success_pct = (backtest_success_count / backtest_total * 100) if backtest_total > 0 else 0
        summary_parts.append(f"Backtest: {backtest_success_count}✓ / {backtest_failure_count}✗ ({success_pct:.1f}% success)")

    if overall_best_call:
        call_price_diff = overall_best_call.get('price_diff_pct')
        call_price_diff_str = f"{call_price_diff:+.2f}%" if call_price_diff is not None else "N/A"
        summary_parts.append(f"CALL Max Credit: ${max_credit_call:.2f} (Price Diff: {call_price_diff_str})")

    if overall_best_put:
        put_price_diff = overall_best_put.get('price_diff_pct')
        put_price_diff_str = f"{put_price_diff:+.2f}%" if put_price_diff is not None else "N/A"
        summary_parts.append(f"PUT Max Credit: ${max_credit_put:.2f} (Price Diff: {put_price_diff_str})")

    # Best across both types
    if option_type == "both" and overall_best_call and overall_best_put:
        if max_credit_call > max_credit_put:
            overall_best = overall_best_call
            best_type = "CALL"
            best_credit = max_credit_call
        else:
            overall_best = overall_best_put
            best_type = "PUT"
            best_credit = max_credit_put

        best_price_diff = overall_best.get('price_diff_pct')
        best_price_diff_str = f"{best_price_diff:+.2f}%" if best_price_diff is not None else "N/A"
        best_timestamp = format_timestamp(overall_best['timestamp'], output_tz)
        best_contracts = overall_best['best_spread'].get('num_contracts', 0) or 0
        best_short_strike = overall_best['best_spread']['short_strike']
        best_long_strike = overall_best['best_spread']['long_strike']
        best_short_premium = overall_best['best_spread']['short_price']
        best_long_premium = overall_best['best_spread']['long_price']
        _s = "Short (bid)" if not use_mid_price else "Short"
        _l = "Long (ask)" if not use_mid_price else "Long"
        summary_parts.append(
            f"BEST: {best_type} ${best_credit:.2f} @ {best_timestamp} "
            f"({best_price_diff_str}, {best_contracts} contracts) | "
            f"Spread: ${best_short_strike:.2f}/${best_long_strike:.2f} | "
            f"{_s}: ${best_short_premium:.2f} {_l}: ${best_long_premium:.2f}"
        )
    elif option_type == "both":
        # Only one type available
        best = overall_best_call or overall_best_put
        if best:
            btype = "CALL" if best == overall_best_call else "PUT"
            bcredit = max_credit_call if best == overall_best_call else max_credit_put
            bpd = best.get('price_diff_pct')
            bpd_str = f"{bpd:+.2f}%" if bpd is not None else "N/A"
            bts = format_timestamp(best['timestamp'], output_tz)
            bc = best['best_spread'].get('num_contracts', 0) or 0
            bss = best['best_spread']['short_strike']
            bls = best['best_spread']['long_strike']
            bsp = best['best_spread']['short_price']
            blp = best['best_spread']['long_price']
            _s = "Short (bid)" if not use_mid_price else "Short"
            _l = "Long (ask)" if not use_mid_price else "Long"
            summary_parts.append(
                f"BEST: {btype} ${bcredit:.2f} @ {bts} "
                f"({bpd_str}, {bc} contracts) | "
                f"Spread: ${bss:.2f}/${bls:.2f} | "
                f"{_s}: ${bsp:.2f} {_l}: ${blp:.2f}"
            )

    return summary_parts


def format_and_print_results(results, args, output_tz, csv_paths, percent_beyond, max_spread_width, original_results_count=None):
    """Format and print results based on args flags.

    This is a dispatcher that calls the appropriate printing logic
    based on --summary, --summary-only, or detailed view flags.

    Note: This function handles the output formatting that was inline in main().
    The actual printing logic remains in the main file since it's tightly
    coupled to the args namespace and control flow.
    """
    # This function exists as a hook for the continuous runner and
    # other callers that need formatted output. The actual implementation
    # of the complex output formatting remains in the main file's main()
    # function because it's deeply coupled to the args namespace.
    pass
