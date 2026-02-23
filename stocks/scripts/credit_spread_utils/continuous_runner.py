"""
Continuous analysis mode for real-time credit spread monitoring.

Provides market-hours-aware continuous execution with intelligent
interval management and market transition detection.
"""

import asyncio
import multiprocessing
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

from common.questdb_db import StockQuestDB
from common.logging_utils import get_logger
from common.market_hours import is_market_hours, compute_market_transition_times

from .interval_analyzer import analyze_interval
from .data_loader import load_data_cached, process_single_csv, process_single_csv_sync
from .metrics import filter_top_n_per_day
from .output_formatter import format_and_print_results
from .timezone_utils import format_timestamp
from .capital_utils import calculate_position_capital, filter_results_by_capital_limit
from .rate_limiter import SlidingWindowRateLimiter
from .time_block_rate_limiter import TimeBlockRateLimiter


# Constants for continuous mode intervals
RUN_INTERVAL_MARKET_OPEN = 300  # 5 minutes when market is open
RUN_INTERVAL_MARKET_CLOSED = 3600  # 1 hour when market is closed


async def run_continuous_analysis(args, csv_paths, percent_beyond, max_spread_width, option_types_to_analyze, output_tz, logger, min_premium_diff=None, delta_filter_config=None, close_predictor_gate=None, strategy=None):
    """
    Continuously run credit spread analysis with intelligent interval management.

    The function optimizes run intervals based on:
    - The wait time specified in --continuous (default: 10 seconds)
    - Market hours awareness (if enabled), with transition-aware scheduling
    - When market transitions from open to closed, runs one more time to capture final data
    """
    # Get wait time from --continuous (defaults to 10.0 seconds)
    wait_time = args.continuous if args.continuous is not None else 10.0

    print(f"Starting continuous credit spread analysis (wait time: {wait_time}s)...")
    print(f"Max runs: {args.continuous_max_runs if args.continuous_max_runs else 'unlimited'}")

    run_count = 0
    last_run_duration = 0  # Track how long the last run took
    was_market_open = None  # Track previous market state to detect transitions

    # Store the original main function logic in a callable
    async def run_single_analysis():
        """Run a single analysis iteration."""
        return await _run_single_analysis_iteration(
            args, csv_paths, percent_beyond, max_spread_width,
            option_types_to_analyze, output_tz, logger, min_premium_diff, delta_filter_config,
            close_predictor_gate=close_predictor_gate,
            strategy=strategy,
        )

    while True:
        run_count += 1
        start_time = time.time()

        if args.use_market_hours:
            is_market_open_start = is_market_hours()
            market_status = "MARKET OPEN" if is_market_open_start else "MARKET CLOSED"

            # Detect market transition from open to closed
            if was_market_open is True and not is_market_open_start:
                print(f"\n--- MARKET TRANSITION DETECTED: OPEN → CLOSED at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} ---")
                print(f"Performing final analysis after market close to capture EOD data...")

            print(f"\n--- Run #{run_count} at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} [{market_status}] ---")
        else:
            print(f"\n--- Run #{run_count} at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} ---")

        try:
            # Run the analysis
            await run_single_analysis()

            # Calculate how long this run took
            run_duration = time.time() - start_time
            last_run_duration = run_duration

            print(f"Run #{run_count} completed in {run_duration:.1f}s")

            # Check if we should stop
            if args.continuous_max_runs and run_count >= args.continuous_max_runs:
                print(f"Reached maximum runs ({args.continuous_max_runs}), stopping continuous analysis.")
                break

            # Calculate optimal sleep time
            # Use intelligent intervals based on market hours and run duration

            if args.use_market_hours:
                is_market_open = is_market_hours()
                now_utc = datetime.now(timezone.utc)
                seconds_to_open, seconds_to_close = compute_market_transition_times(now_utc, args.output_timezone)

                # Check if we just transitioned from open to closed
                # If so, we already did a post-close run, so now go into long sleep mode
                just_closed = (was_market_open is True and not is_market_open)

                if is_market_open:
                    # Market is open - use the specified wait time, but don't sleep past close
                    base_sleep = max(wait_time - run_duration, 1)
                    if seconds_to_close is not None:
                        sleep_time = max(min(base_sleep, seconds_to_close), 1)
                        print(f"Next run in {sleep_time:.1f}s (market open, {wait_time}s interval; {seconds_to_close:.1f}s until close) [MARKET OPEN]")
                    else:
                        sleep_time = base_sleep
                        print(f"Next run in {sleep_time:.1f}s (market open, {wait_time}s interval) [MARKET OPEN]")
                else:
                    # Market is closed - override continuous interval and wait until market opens
                    if just_closed:
                        # We just performed the post-close run, now sleep until next market open
                        print(f"Post-close analysis completed. Entering extended sleep until next market open.")

                    # When market is closed, override the continuous interval and wait until market opens
                    if seconds_to_open is not None:
                        # Sleep until market opens (completely override continuous interval)
                        sleep_time = seconds_to_open
                        hours_to_wait = sleep_time / 3600
                        print(f"Market is closed. Waiting {hours_to_wait:.2f} hours ({sleep_time:.0f} seconds) until market opens. [MARKET CLOSED→OPEN]")
                    else:
                        # Don't know when market opens - use default closed interval as fallback
                        base_sleep = max(RUN_INTERVAL_MARKET_CLOSED - run_duration, 60)
                        sleep_time = max(base_sleep, 1)
                        print(f"Next run in {sleep_time:.1f}s (markets closed, {RUN_INTERVAL_MARKET_CLOSED/60:.0f}min interval) [MARKET CLOSED]")

                # Update the market state tracker for next iteration
                was_market_open = is_market_open
            else:
                # Standard behavior - use the specified wait time
                base_sleep = max(wait_time - run_duration, 1)
                sleep_time = max(base_sleep, 1)
                print(f"Next run in {sleep_time:.1f}s ({wait_time}s interval)")

            # Sleep until next run
            await asyncio.sleep(sleep_time)

            # After waking up, check if market transitioned from open to closed during sleep
            # If so, perform one more run to capture EOD data before long sleep
            if args.use_market_hours and was_market_open is True:
                current_market_state = is_market_hours()
                if not current_market_state:
                    # Market closed while we were sleeping - run once more for EOD data
                    print(f"\n--- MARKET CLOSED DURING SLEEP - Performing post-close analysis ---")
                    run_count += 1
                    start_time_post_close = time.time()

                    try:
                        # Run the post-close analysis
                        await run_single_analysis()
                        run_duration_post_close = time.time() - start_time_post_close
                        print(f"Post-close analysis #{run_count} completed in {run_duration_post_close:.1f}s")

                        # Check if we should stop
                        if args.continuous_max_runs and run_count >= args.continuous_max_runs:
                            print(f"Reached maximum runs ({args.continuous_max_runs}), stopping continuous analysis.")
                            break

                        # Update market state tracker
                        was_market_open = False

                    except Exception as e:
                        print(f"Error during post-close analysis: {e}", file=sys.stderr)
                        was_market_open = False

        except KeyboardInterrupt:
            print(f"\nContinuous analysis interrupted by user after {run_count} runs.")
            break
        except Exception as e:
            print(f"Error in continuous analysis run #{run_count}: {e}")
            # Wait a bit before retrying to avoid rapid error loops
            await asyncio.sleep(10)

    print(f"Continuous analysis stopped after {run_count} runs.")


async def _run_single_analysis_iteration(args, csv_paths, percent_beyond, max_spread_width, option_types_to_analyze, output_tz, logger, min_premium_diff=None, delta_filter_config=None, close_predictor_gate=None, strategy=None):
    """
    Run a single iteration of the analysis.
    This extracts the main analysis logic so it can be called repeatedly in continuous mode.
    """
    # This function contains the core analysis logic from main()
    # We'll extract the relevant parts from the main function

    # Determine if we should use multiprocessing
    num_processes = args.processes
    use_multiprocessing = len(csv_paths) > 1 and num_processes != 1

    # Auto-detect CPU count if requested
    if num_processes == 0:
        num_processes = multiprocessing.cpu_count()
        logger.info(f"Auto-detected {num_processes} CPUs")

    results = []
    skip_normal_processing = False

    # Process CSV files
    if use_multiprocessing:
        logger.info(f"Processing {len(csv_paths)} files using {num_processes} parallel processes")

        # Prepare arguments for each CSV file
        process_args = []
        for csv_path in csv_paths:
            args_tuple = (
                csv_path,
                option_types_to_analyze,
                percent_beyond,
                args.risk_cap,
                args.min_spread_width,
                max_spread_width,
                args.use_mid_price,
                args.min_contract_price,
                args.underlying_ticker,
                args.db_path,
                args.no_cache,
                args.log_level,
                args.max_credit_width_ratio,
                args.max_strike_distance_pct,
                args.curr_price and args.continuous is not None,
                args.max_trading_hour,
                args.min_trading_hour,
                args.profit_target_pct,
                args.most_recent,
                output_tz,
                args.force_close_hour,
                args.cache_dir,
                args.no_data_cache,
                min_premium_diff,
                args.rate_limit_max,
                args.rate_limit_window,
                getattr(args, 'rate_limit_blocks', None),
            )
            process_args.append(args_tuple)

        # Process files in parallel
        with multiprocessing.Pool(processes=num_processes) as pool:
            results_list = pool.map(process_single_csv_sync, process_args)

        # Flatten results
        for file_results in results_list:
            results.extend(file_results)

        logger.info(f"Parallel processing complete. Total results: {len(results)}")

        # Apply close predictor gate filter (before capital filtering)
        if close_predictor_gate is not None and close_predictor_gate.config.enabled:
            results = close_predictor_gate.filter_results(results)

        # Apply capital limit filter (accounts for position lifecycle)
        if args.max_live_capital is not None:
            original_count = len(results)
            results = filter_results_by_capital_limit(
                results,
                args.max_live_capital,
                output_tz,
                logger
            )
            logger.info(
                f"Capital limit filter: {original_count} -> {len(results)} positions "
                f"(max ${args.max_live_capital:,.2f} per day)"
            )

            # Calculate and log final capital usage
            daily_capital_usage = {}
            for result in results:
                position_capital, calendar_date = calculate_position_capital(result, output_tz)
                daily_capital_usage[calendar_date] = daily_capital_usage.get(calendar_date, 0.0) + position_capital

            if daily_capital_usage:
                logger.info("Final daily capital usage:")
                for date, capital in sorted(daily_capital_usage.items()):
                    logger.info(f"  {date}: ${capital:,.2f} / ${args.max_live_capital:,.2f} ({(capital/args.max_live_capital*100):.1f}%)")

        # Skip to post-processing (we already have results)
        skip_normal_processing = True
    else:
        # Read CSV file(s) sequentially with binary cache support
        logger.info(f"Processing {len(csv_paths)} file(s) sequentially")

        try:
            df = load_data_cached(
                csv_paths,
                cache_dir=args.cache_dir,
                no_cache=args.no_data_cache,
                logger=logger
            )
        except ValueError as e:
            logger.error(str(e))
            return
        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")
            return

        skip_normal_processing = False

    # Normal processing (when not using multiprocessing)
    if not skip_normal_processing:

        # Initialize database
        logger.info("Initializing database connection...")
        # If db_path is None, check environment variables or use empty string
        # QuestDBConnection doesn't handle None properly (calls .startswith() on None)
        if args.db_path:
            db_config = args.db_path
        else:
            db_config = os.getenv('QUESTDB_CONNECTION_STRING', '') or os.getenv('QUESTDB_URL', '')
        db = StockQuestDB(
            db_config,
            enable_cache=not args.no_cache,
            logger=logger
        )

        try:
            # Group by 15-minute intervals
            intervals_grouped = df.groupby('interval')
            total_intervals_count = len(intervals_grouped)

            # If --most-recent is used, only analyze the most recent interval
            if args.most_recent:
                # Find the most recent interval
                max_interval = df['interval'].max()
                max_interval_df = df[df['interval'] == max_interval]
                intervals_to_process = [(max_interval, max_interval_df)]
                logger.info(f"Analyzing most recent interval only: {max_interval}")
            else:
                intervals_to_process = intervals_grouped
                logger.info(f"Analyzing {total_intervals_count} intervals...")

            results = []

            # Create rate limiter for sequential mode
            # Time-block rate limiter takes precedence over sliding window
            time_block_limiter = None
            sliding_limiter = None

            if hasattr(args, 'rate_limit_blocks') and args.rate_limit_blocks:
                time_block_limiter = TimeBlockRateLimiter.from_string(args.rate_limit_blocks, logger=logger)
                logger.info(f"Time-block rate limiting enabled: {args.rate_limit_blocks}")
            elif args.rate_limit_max > 0 and args.rate_limit_window > 0:
                sliding_limiter = SlidingWindowRateLimiter(
                    max_transactions=args.rate_limit_max,
                    window_seconds=args.rate_limit_window,
                    logger=logger
                )
                logger.info(f"Sliding window rate limiting enabled: {args.rate_limit_max} transactions per {args.rate_limit_window}s")

            # Collect all results first (without capital filtering)
            for interval_time, interval_df in intervals_to_process:
                for opt_type in option_types_to_analyze:
                    # Apply rate limiting before each interval analysis
                    if time_block_limiter:
                        await time_block_limiter.acquire()
                    elif sliding_limiter:
                        await sliding_limiter.acquire()
                    # Use current price if --curr-price is set and --continuous mode is active
                    use_current_price = args.curr_price and args.continuous is not None
                    result = await analyze_interval(
                        db,
                        interval_df,
                        opt_type,
                        percent_beyond,
                        args.risk_cap,
                        args.min_spread_width,
                        max_spread_width,
                        args.use_mid_price,
                        args.min_contract_price,
                        args.underlying_ticker,
                        logger,
                        args.max_credit_width_ratio,
                        args.max_strike_distance_pct,
                        use_current_price,
                        args.max_trading_hour,
                        args.min_trading_hour,
                        args.profit_target_pct,
                        output_tz,
                        args.force_close_hour,
                        min_premium_diff,
                        None,  # dynamic_width_config
                        delta_filter_config,
                    )
                    if result:
                        results.append(result)

            # Apply close predictor gate filter (before capital filtering)
            if close_predictor_gate is not None and close_predictor_gate.config.enabled:
                results = close_predictor_gate.filter_results(results)

            # Apply capital limit filter (accounts for position lifecycle) [single analysis iteration]
            if args.max_live_capital is not None:
                original_count = len(results)
                results = filter_results_by_capital_limit(
                    results,
                    args.max_live_capital,
                    output_tz,
                    logger
                )
                logger.info(
                    f"Capital limit filter: {original_count} -> {len(results)} positions "
                    f"(max ${args.max_live_capital:,.2f} per day)"
                )

                # Calculate and log final capital usage
                daily_capital_usage = {}
                for result in results:
                    position_capital, calendar_date = calculate_position_capital(result, output_tz)
                    daily_capital_usage[calendar_date] = daily_capital_usage.get(calendar_date, 0.0) + position_capital

                if daily_capital_usage:
                    logger.info("Final daily capital usage:")
                    for date, capital in sorted(daily_capital_usage.items()):
                        logger.info(f"  {date}: ${capital:,.2f} / ${args.max_live_capital:,.2f} ({(capital/args.max_live_capital*100):.1f}%)")

        finally:
            await db.close()

    # Post-processing (common for both multiprocessing and sequential)
    if skip_normal_processing:
        # For multiprocessing, we already have results
        # Set total_intervals_count for reporting
        total_intervals_count = len(results)

    # Apply top-N filtering if requested (before most-recent mode)
    original_results_count = len(results)
    if args.top_n and results:
        results = filter_top_n_per_day(results, args.top_n)
        logger.info(f"Applied top-{args.top_n} per day filter: {original_results_count} -> {len(results)} results")

    # Handle --most-recent mode
    if args.most_recent:
        if results:
            # Find the most recent timestamp from results
            max_timestamp = max(result['timestamp'] for result in results)
            # Filter to only results from the most recent timestamp
            # For each option type, keep only the best one
            most_recent_results = []
            call_results = [r for r in results if r['timestamp'] == max_timestamp and r.get('option_type', '').lower() == 'call']
            put_results = [r for r in results if r['timestamp'] == max_timestamp and r.get('option_type', '').lower() == 'put']

            best_call = None
            best_put = None

            if call_results:
                # Get best call by max credit
                best_call = max(call_results, key=lambda x: x['best_spread'].get('total_credit') or x['best_spread'].get('net_credit_per_contract', 0))

            if put_results:
                # Get best put by max credit
                best_put = max(put_results, key=lambda x: x['best_spread'].get('total_credit') or x['best_spread'].get('net_credit_per_contract', 0))

            # If --best-only is enabled, keep only the single best spread (call or put)
            if args.best_only:
                if best_call and best_put:
                    # Compare credits and keep only the best one
                    call_credit = best_call['best_spread'].get('total_credit') or best_call['best_spread'].get('net_credit_per_contract', 0)
                    put_credit = best_put['best_spread'].get('total_credit') or best_put['best_spread'].get('net_credit_per_contract', 0)
                    if call_credit > put_credit:
                        most_recent_results = [best_call]
                    else:
                        most_recent_results = [best_put]
                elif best_call:
                    most_recent_results = [best_call]
                elif best_put:
                    most_recent_results = [best_put]
            else:
                # Keep both best call and best put
                if best_call:
                    most_recent_results.append(best_call)
                if best_put:
                    most_recent_results.append(best_put)

            results = most_recent_results

            # If using --most-recent --best-only --continuous, show the best option or a clear message
            if args.best_only and args.continuous is not None:
                if results:
                    # Show the best option from most recent timestamp
                    best_result = results[0]
                    timestamp_str = format_timestamp(best_result['timestamp'], output_tz)
                    max_credit = best_result['best_spread'].get('total_credit')
                    if max_credit is None:
                        max_credit = best_result['best_spread'].get('net_credit_per_contract', 0)
                    num_contracts = best_result['best_spread'].get('num_contracts', 0)
                    if num_contracts is None:
                        num_contracts = 0
                    opt_type_upper = best_result.get('option_type', 'UNKNOWN').upper()
                    short_strike = best_result['best_spread']['short_strike']
                    long_strike = best_result['best_spread']['long_strike']
                    short_premium = best_result['best_spread']['short_price']
                    long_premium = best_result['best_spread']['long_price']
                    _short_lbl = "Short (bid)" if not getattr(args, 'use_mid_price', False) else "Short"
                    _long_lbl = "Long (ask)" if not getattr(args, 'use_mid_price', False) else "Long"
                    print(f"BEST CURRENT OPTION: {timestamp_str} | Type: {opt_type_upper} | Max Credit: ${max_credit:.2f} | Contracts: {num_contracts} | Spread: ${short_strike:.2f}/${long_strike:.2f} | {_short_lbl}: ${short_premium:.2f} {_long_lbl}: ${long_premium:.2f}")
                else:
                    # No results found - use most recent timestamp from dataframe if available
                    most_recent_ts = None
                    try:
                        if df is not None and len(df) > 0:
                            most_recent_ts = df['timestamp'].max()
                    except (NameError, UnboundLocalError):
                        # df not defined (e.g., when using multiprocessing)
                        pass
                    if most_recent_ts:
                        max_timestamp_str = format_timestamp(most_recent_ts, output_tz)
                        print(f"NO RESULTS: No valid spreads found at most recent timestamp {max_timestamp_str} that meet the criteria.")
                    else:
                        print("NO RESULTS: No valid spreads found.")
        else:
            # No results at all - show message with most recent timestamp from dataframe if available
            most_recent_ts = None
            try:
                if df is not None and len(df) > 0:
                    most_recent_ts = df['timestamp'].max()
            except (NameError, UnboundLocalError):
                # df not defined (e.g., when using multiprocessing)
                pass
            if most_recent_ts:
                most_recent_str = format_timestamp(most_recent_ts, output_tz)
                if args.best_only and args.continuous is not None:
                    print(f"NO RESULTS: No valid spreads found at most recent timestamp {most_recent_str} that meet the criteria.")
                else:
                    print(f"NO RESULTS: No valid spreads found. Most recent data timestamp: {most_recent_str}")
            else:
                print("NO RESULTS: No valid spreads found.")

    # Print summary results in continuous mode
    if args.summary or args.summary_only:
        if results:
            # Sort by date (timestamp)
            sorted_results = sorted(results, key=lambda x: x['timestamp'])
            overall_best_call = None
            overall_best_put = None
            max_credit_call = 0
            max_credit_put = 0
            total_options = len(results)

            for result in sorted_results:
                # Get max credit (total_credit if available, otherwise per-contract credit)
                max_credit = result['best_spread'].get('total_credit')
                if max_credit is None:
                    max_credit = result['best_spread'].get('net_credit_per_contract', 0)

                # Track overall best for calls and puts separately
                opt_type = result.get('option_type', 'UNKNOWN').lower()
                if opt_type == 'call':
                    if max_credit > max_credit_call:
                        max_credit_call = max_credit
                        overall_best_call = result
                elif opt_type == 'put':
                    if max_credit > max_credit_put:
                        max_credit_put = max_credit
                        overall_best_put = result

                # Only print individual lines if --summary is used (not --summary-only)
                # Skip if --best-only --continuous was used (we already printed it above)
                if args.summary and not args.summary_only and not (args.best_only and args.continuous is not None):
                    timestamp_str = format_timestamp(result['timestamp'], output_tz)
                    num_contracts = result['best_spread'].get('num_contracts', 0)
                    if num_contracts is None:
                        num_contracts = 0
                    opt_type_upper = result.get('option_type', 'UNKNOWN').upper()
                    short_strike = result['best_spread']['short_strike']
                    long_strike = result['best_spread']['long_strike']
                    print(f"{timestamp_str} | Type: {opt_type_upper} | Max Credit: ${max_credit:.2f} | Contracts: {num_contracts} | Spread: ${short_strike:.2f}/${long_strike:.2f}")

            # Print final one-line summary
            summary_parts = []
            if args.top_n:
                summary_parts.append(f"Total Options: {total_options} (Top-{args.top_n} per day)")
            else:
                summary_parts.append(f"Total Options: {total_options}")

            if overall_best_call:
                call_price_diff = overall_best_call.get('price_diff_pct')
                call_price_diff_str = f"{call_price_diff:+.2f}%" if call_price_diff is not None else "N/A"
                summary_parts.append(f"CALL Max Credit: ${max_credit_call:.2f} (Price Diff: {call_price_diff_str})")

            if overall_best_put:
                put_price_diff = overall_best_put.get('price_diff_pct')
                put_price_diff_str = f"{put_price_diff:+.2f}%" if put_price_diff is not None else "N/A"
                summary_parts.append(f"PUT Max Credit: ${max_credit_put:.2f} (Price Diff: {put_price_diff_str})")

            # If analyzing both types, show the overall best across both modes
            if args.option_type == "both" and overall_best_call and overall_best_put:
                # Determine which is better based on max credit
                if max_credit_call > max_credit_put:
                    overall_best = overall_best_call
                    best_type = "CALL"
                    best_credit = max_credit_call
                    best_price_diff = overall_best_call.get('price_diff_pct')
                else:
                    overall_best = overall_best_put
                    best_type = "PUT"
                    best_credit = max_credit_put
                    best_price_diff = overall_best_put.get('price_diff_pct')

                best_price_diff_str = f"{best_price_diff:+.2f}%" if best_price_diff is not None else "N/A"
                best_timestamp = format_timestamp(overall_best['timestamp'], output_tz)
                best_contracts = overall_best['best_spread'].get('num_contracts', 0)
                if best_contracts is None:
                    best_contracts = 0
                best_short_strike = overall_best['best_spread']['short_strike']
                best_long_strike = overall_best['best_spread']['long_strike']
                best_short_premium = overall_best['best_spread']['short_price']
                best_long_premium = overall_best['best_spread']['long_price']
                _s = "Short (bid)" if not getattr(args, 'use_mid_price', False) else "Short"
                _l = "Long (ask)" if not getattr(args, 'use_mid_price', False) else "Long"
                summary_parts.append(f"BEST: {best_type} ${best_credit:.2f} @ {best_timestamp} ({best_price_diff_str}, {best_contracts} contracts) | Spread: ${best_short_strike:.2f}/${best_long_strike:.2f} | {_s}: ${best_short_premium:.2f} {_l}: ${best_long_premium:.2f}")
            elif args.option_type == "both" and overall_best_call:
                # Only call available
                call_price_diff = overall_best_call.get('price_diff_pct')
                call_price_diff_str = f"{call_price_diff:+.2f}%" if call_price_diff is not None else "N/A"
                call_timestamp = format_timestamp(overall_best_call['timestamp'], output_tz)
                call_contracts = overall_best_call['best_spread'].get('num_contracts', 0)
                if call_contracts is None:
                    call_contracts = 0
                call_short_strike = overall_best_call['best_spread']['short_strike']
                call_long_strike = overall_best_call['best_spread']['long_strike']
                call_short_premium = overall_best_call['best_spread']['short_price']
                call_long_premium = overall_best_call['best_spread']['long_price']
                _s = "Short (bid)" if not getattr(args, 'use_mid_price', False) else "Short"
                _l = "Long (ask)" if not getattr(args, 'use_mid_price', False) else "Long"
                summary_parts.append(f"BEST: CALL ${max_credit_call:.2f} @ {call_timestamp} ({call_price_diff_str}, {call_contracts} contracts) | Spread: ${call_short_strike:.2f}/${call_long_strike:.2f} | {_s}: ${call_short_premium:.2f} {_l}: ${call_long_premium:.2f}")
            elif args.option_type == "both" and overall_best_put:
                # Only put available
                put_price_diff = overall_best_put.get('price_diff_pct')
                put_price_diff_str = f"{put_price_diff:+.2f}%" if put_price_diff is not None else "N/A"
                put_timestamp = format_timestamp(overall_best_put['timestamp'], output_tz)
                put_contracts = overall_best_put['best_spread'].get('num_contracts', 0)
                if put_contracts is None:
                    put_contracts = 0
                put_short_strike = overall_best_put['best_spread']['short_strike']
                put_long_strike = overall_best_put['best_spread']['long_strike']
                put_short_premium = overall_best_put['best_spread']['short_price']
                put_long_premium = overall_best_put['best_spread']['long_price']
                _s = "Short (bid)" if not getattr(args, 'use_mid_price', False) else "Short"
                _l = "Long (ask)" if not getattr(args, 'use_mid_price', False) else "Long"
                summary_parts.append(f"BEST: PUT ${max_credit_put:.2f} @ {put_timestamp} ({put_price_diff_str}, {put_contracts} contracts) | Spread: ${put_short_strike:.2f}/${put_long_strike:.2f} | {_s}: ${put_short_premium:.2f} {_l}: ${put_long_premium:.2f}")

            if summary_parts:
                print(f"SUMMARY: {' | '.join(summary_parts)}")
        elif not args.most_recent:
            # Only print summary "no results" if not using --most-recent (to avoid duplicate messages)
            print("SUMMARY: No valid spreads found.")

    # Execute strategy on this iteration's results (when --strategy is used in continuous mode)
    if strategy is not None and results:
        from scripts.analyze_credit_spread_intervals import _execute_strategy_on_results
        _execute_strategy_on_results(strategy, results, output_tz, logger, args)
