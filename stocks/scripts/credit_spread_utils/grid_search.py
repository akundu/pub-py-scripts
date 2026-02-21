"""
Grid search optimization engine for credit spread parameters.

Supports parallel execution, resume capability, and configurable
parameter sweeps across all spread configuration dimensions.
"""

import asyncio
import csv
import itertools
import json
import multiprocessing
import os
import time
from typing import Dict, List, Optional, Any

import pandas as pd

from common.questdb_db import StockQuestDB
from common.logging_utils import get_logger

from .interval_analyzer import analyze_interval
from .spread_builder import parse_percent_beyond, parse_max_spread_width, parse_min_premium_diff
from .metrics import compute_metrics, filter_top_n_per_day
from .data_loader import find_csv_files_in_dir, load_data_cached
from .timezone_utils import resolve_timezone
from .delta_utils import DeltaFilterConfig, parse_delta_range
from .strategies import StrategyRegistry


def _build_strategy_for_combo(strategy_config: dict, combo: dict, logger=None):
    """Build a strategy instance for a grid combo, applying feature flag overrides.

    Extracts any 'strategy.feature_flags.*' keys from the combo and merges
    them into the strategy config before creating the strategy.

    Args:
        strategy_config: Base strategy config dict from grid JSON
        combo: Grid parameter combination
        logger: Optional logger

    Returns:
        Configured strategy instance, or None if no strategy configured
    """
    if not strategy_config:
        return None

    config_dict = dict(strategy_config)
    feature_flags = dict(config_dict.get('feature_flags', {}))

    # Extract strategy.feature_flags.* from combo
    for key, value in list(combo.items()):
        if key.startswith('strategy.feature_flags.'):
            flag_name = key[len('strategy.feature_flags.'):]
            feature_flags[flag_name] = value

    config_dict['feature_flags'] = feature_flags

    # Extract strategy.config.* keys from combo and apply to the strategy's
    # underlying config file (for strategies like time_allocated_tiered that
    # load config from a JSON file). These overrides are applied after loading.
    config_overrides = {}
    for key, value in list(combo.items()):
        if key.startswith('strategy.config.'):
            config_key = key[len('strategy.config.'):]
            config_overrides[config_key] = value

    if config_overrides:
        config_dict['_config_overrides'] = config_overrides

    name = config_dict.get('name', config_dict.get('strategy', 'single_entry'))

    strategy = StrategyRegistry.create(name, config_dict, logger=logger)

    # Apply config overrides to the strategy's underlying config object
    if config_overrides and hasattr(strategy, 'ta_config') and strategy.ta_config:
        _apply_config_overrides(strategy.ta_config, config_overrides)
    elif config_overrides and hasattr(strategy, 'tiered_config') and strategy.tiered_config:
        _apply_config_overrides(strategy.tiered_config, config_overrides)

    # Validate budget percentages sum <= 1.0 when budget keys are overridden
    budget_keys_changed = any(k.startswith('window_') and k.endswith('_budget_pct')
                              for k in config_overrides)
    if budget_keys_changed and hasattr(strategy, 'ta_config') and strategy.ta_config:
        strategy.ta_config.validate()

    return strategy


def _apply_config_overrides(config_obj, overrides: dict):
    """Apply dotted key overrides to a config object.

    Supports flat keys (e.g., 'carry_forward_decay') and dotted keys
    (e.g., 'slope_detection.lookback_bars') for nested config objects.
    Window budget overrides use format 'window_LABEL_budget_pct'.
    """
    for key, value in overrides.items():
        # Handle window budget overrides: window_6am_budget_pct -> hourly_windows[label=6am].budget_pct
        if key.startswith('window_') and key.endswith('_budget_pct'):
            label = key[len('window_'):-len('_budget_pct')]
            if hasattr(config_obj, 'hourly_windows'):
                for w in config_obj.hourly_windows:
                    if w.label == label:
                        w.budget_pct = float(value)
                        break
            continue

        # Handle tier-indexed keys: tiers.{put|call}.{index}.{field}
        if key.startswith('tiers.'):
            parts = key.split('.')
            if len(parts) == 4:
                _, side, idx_str, field = parts
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                tier_list = getattr(config_obj, f'{side}_tiers', None)
                if tier_list is not None and 0 <= idx < len(tier_list):
                    tier = tier_list[idx]
                    if hasattr(tier, field):
                        current = getattr(tier, field)
                        if current is not None:
                            setattr(tier, field, type(current)(value))
                        else:
                            setattr(tier, field, value)
            continue

        # Handle dotted keys for nested objects
        if '.' in key:
            parts = key.split('.', 1)
            parent_attr = parts[0]
            child_key = parts[1]
            # Map JSON key names to Python dataclass field names
            attr_aliases = {'slope_detection': 'slope_config'}
            parent_attr = attr_aliases.get(parent_attr, parent_attr)
            if hasattr(config_obj, parent_attr):
                parent = getattr(config_obj, parent_attr)
                if hasattr(parent, child_key):
                    setattr(parent, child_key, type(getattr(parent, child_key))(value))
            continue

        # Handle flat keys
        if hasattr(config_obj, key):
            current = getattr(config_obj, key)
            if current is not None:
                setattr(config_obj, key, type(current)(value))
            else:
                setattr(config_obj, key, value)


def _float_range(start, stop, step):
    """Generate float values from start to stop (inclusive) with given step."""
    values = []
    current = start
    while current <= stop + step * 0.01:
        values.append(round(current, 6))
        current += step
    return values


def _expand_grid_param(name, spec):
    """Expand a grid parameter specification into a list of values."""
    if isinstance(spec, list):
        return spec
    if isinstance(spec, dict):
        if 'min' in spec and 'max' in spec and 'step' in spec:
            return _float_range(spec['min'], spec['max'], spec['step'])
        raise ValueError(f"Dict spec for '{name}' must have min, max, step keys")
    return [spec]


def _generate_combinations(grid_params: dict) -> List[dict]:
    """Generate all parameter combinations from the grid specification."""
    param_names = []
    param_values = []
    for name, spec in grid_params.items():
        param_names.append(name)
        param_values.append(_expand_grid_param(name, spec))

    combinations = []
    for values in itertools.product(*param_values):
        combinations.append(dict(zip(param_names, values)))
    return combinations


def _combo_to_key(combo: dict) -> tuple:
    """Create a hashable key from a parameter combination."""
    # Convert dict/list values to JSON strings for hashability
    hashable_items = []
    for k, v in sorted(combo.items()):
        if isinstance(v, (dict, list)):
            hashable_items.append((k, json.dumps(v, sort_keys=True)))
        else:
            hashable_items.append((k, v))
    return tuple(hashable_items)


def _load_existing_grid_results(csv_path: str) -> set:
    """Load existing grid results to support resume."""
    existing_keys = set()
    if not os.path.exists(csv_path):
        return existing_keys
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        param_cols = [
            'option_type', 'percent_beyond_put', 'percent_beyond_call',
            'max_spread_width', 'max_spread_width_put', 'max_spread_width_call',
            'min_contract_price', 'max_credit_width_ratio',
            'max_strike_distance_pct', 'min_trading_hour', 'max_trading_hour',
            'profit_target_pct', 'min_premium_diff', 'min_premium_diff_put', 'min_premium_diff_call',
            'max_short_delta', 'min_short_delta', 'max_long_delta', 'min_long_delta',
            'delta_range', 'require_delta', 'delta_default_iv', 'use_vix1d',
        ]
        # Also include any strategy.feature_flags.* and strategy.config.* columns
        if reader.fieldnames:
            for col in reader.fieldnames:
                if (col.startswith('strategy.feature_flags.') or col.startswith('strategy.config.')) and col not in param_cols:
                    param_cols.append(col)
        for row in reader:
            combo = {}
            for col in param_cols:
                if col in row and row[col]:
                    try:
                        val = float(row[col])
                        if val == int(val) and '.' not in row[col]:
                            val = int(val)
                        combo[col] = val
                    except (ValueError, TypeError):
                        combo[col] = row[col]
            existing_keys.add(_combo_to_key(combo))
    return existing_keys


async def run_backtest_with_params(
    interval_groups,
    db,
    params: dict,
    logger,
    strategy=None,
) -> dict:
    """Run a full backtest with given params, return metrics dict.

    Args:
        interval_groups: Pre-grouped interval DataFrames
        db: Database connection
        params: Parameter dictionary
        logger: Logger instance
        strategy: Optional strategy instance for strategy-aware grid search

    Returns:
        Metrics dictionary with performance statistics
    """
    results = []
    option_types = ['call', 'put'] if params.get('option_type', 'both') == 'both' else [params['option_type']]

    # Construct percent_beyond tuple from separate put/call keys or parse combined format
    percent_beyond_raw = params.get('percent_beyond')
    if percent_beyond_raw is not None:
        if isinstance(percent_beyond_raw, str):
            # Parse string format (e.g., "0.02" or "0.02:0.03")
            try:
                percent_beyond = parse_percent_beyond(percent_beyond_raw)
            except ValueError:
                percent_beyond = (0.02, 0.02)
        elif isinstance(percent_beyond_raw, (int, float)):
            # Single numeric value - use for both
            percent_beyond = (float(percent_beyond_raw), float(percent_beyond_raw))
        elif isinstance(percent_beyond_raw, (list, tuple)) and len(percent_beyond_raw) == 2:
            percent_beyond = (float(percent_beyond_raw[0]), float(percent_beyond_raw[1]))
        else:
            percent_beyond = (0.02, 0.02)
    else:
        # Fallback to separate put/call keys
        percent_beyond = (params.get('percent_beyond_put', 0.02), params.get('percent_beyond_call', 0.02))

    # Construct max_spread_width tuple from separate put/call keys or parse combined format
    max_spread_width_raw = params.get('max_spread_width', 200)
    if isinstance(max_spread_width_raw, str):
        # Parse string format (e.g., "30" or "30:40")
        try:
            max_spread_width = parse_max_spread_width(max_spread_width_raw)
        except ValueError:
            max_spread_width = (200, 200)
    elif isinstance(max_spread_width_raw, (int, float)):
        max_spread_width = (float(max_spread_width_raw), float(max_spread_width_raw))
    elif isinstance(max_spread_width_raw, (list, tuple)) and len(max_spread_width_raw) == 2:
        max_spread_width = (float(max_spread_width_raw[0]), float(max_spread_width_raw[1]))
    else:
        max_spread_width = (
            params.get('max_spread_width_put', 200),
            params.get('max_spread_width_call', 200)
        )

    # Parse dynamic_spread_width config if provided
    dynamic_width_config = None
    dynamic_width_raw = params.get('dynamic_spread_width')
    if dynamic_width_raw:
        from credit_spread_utils.dynamic_width_utils import DynamicWidthConfig
        if isinstance(dynamic_width_raw, dict):
            dynamic_width_config = DynamicWidthConfig.from_dict(dynamic_width_raw)
        elif isinstance(dynamic_width_raw, str):
            from credit_spread_utils.dynamic_width_utils import parse_dynamic_width_config
            dynamic_width_config = parse_dynamic_width_config(dynamic_width_raw)

    # Build delta filter config if any delta params are provided
    delta_filter_config = None
    if any(params.get(k) is not None for k in ['max_short_delta', 'min_short_delta', 'max_long_delta',
                                                 'min_long_delta', 'delta_range']) or params.get('require_delta'):
        # Parse delta_range if provided (overrides min/max_short_delta)
        min_short_delta = params.get('min_short_delta')
        max_short_delta = params.get('max_short_delta')
        if params.get('delta_range'):
            parsed_min, parsed_max = parse_delta_range(params['delta_range'])
            if parsed_min is not None:
                min_short_delta = parsed_min
            if parsed_max is not None:
                max_short_delta = parsed_max

        delta_filter_config = DeltaFilterConfig(
            max_short_delta=max_short_delta,
            min_short_delta=min_short_delta,
            max_long_delta=params.get('max_long_delta'),
            min_long_delta=params.get('min_long_delta'),
            require_delta=params.get('require_delta', False),
            default_iv=params.get('delta_default_iv', 0.20),
            use_vix1d=params.get('use_vix1d', False),
            vix1d_dir=params.get('vix1d_dir', '../equities_output/I:VIX1D'),
        )

    for interval_time, interval_df in interval_groups:
        for opt_type in option_types:
            # Construct min_premium_diff tuple from separate put/call keys or fallback to single value
            min_premium_diff_default = params.get('min_premium_diff')
            min_premium_diff = None
            if min_premium_diff_default is not None:
                if isinstance(min_premium_diff_default, str):
                    # Parse if it's a string (put:call format)
                    try:
                        min_premium_diff = parse_min_premium_diff(min_premium_diff_default)
                    except ValueError:
                        logger.warning(f"Invalid min_premium_diff format: {min_premium_diff_default}")
                        min_premium_diff = None
                elif isinstance(min_premium_diff_default, (int, float)):
                    # Single value - use for both puts and calls
                    min_premium_diff = (float(min_premium_diff_default), float(min_premium_diff_default))
                else:
                    # Already a tuple or dict with separate keys
                    if isinstance(min_premium_diff_default, dict):
                        put_val = min_premium_diff_default.get('put', min_premium_diff_default.get('default'))
                        call_val = min_premium_diff_default.get('call', min_premium_diff_default.get('default'))
                        if put_val is not None and call_val is not None:
                            min_premium_diff = (float(put_val), float(call_val))
                    elif isinstance(min_premium_diff_default, (list, tuple)) and len(min_premium_diff_default) == 2:
                        min_premium_diff = (float(min_premium_diff_default[0]), float(min_premium_diff_default[1]))

            # Fallback to separate put/call keys if available
            if min_premium_diff is None:
                min_premium_diff_put = params.get('min_premium_diff_put')
                min_premium_diff_call = params.get('min_premium_diff_call')
                if min_premium_diff_put is not None and min_premium_diff_call is not None:
                    min_premium_diff = (float(min_premium_diff_put), float(min_premium_diff_call))
                elif min_premium_diff_put is not None:
                    min_premium_diff = (float(min_premium_diff_put), float(min_premium_diff_put))
                elif min_premium_diff_call is not None:
                    min_premium_diff = (float(min_premium_diff_call), float(min_premium_diff_call))

            result = await analyze_interval(
                db,
                interval_df,
                opt_type,
                percent_beyond,
                params.get('risk_cap'),
                params.get('min_spread_width', 5),
                max_spread_width,
                params.get('use_mid_price', False),
                params.get('min_contract_price', 0),
                params.get('underlying_ticker'),
                logger,
                params.get('max_credit_width_ratio', 0.60),
                params.get('max_strike_distance_pct'),
                False,  # use_current_price
                params.get('max_trading_hour', 15),
                params.get('min_trading_hour'),
                params.get('profit_target_pct'),
                params.get('output_tz'),
                params.get('force_close_hour'),
                min_premium_diff,
                dynamic_width_config,
                delta_filter_config,
            )
            if result:
                results.append(result)

    # Apply top_n filter if specified
    if params.get('top_n') and results:
        results = filter_top_n_per_day(results, params['top_n'])

    # If a strategy is provided, run it on the aggregated results
    if strategy and results:
        from collections import defaultdict
        from datetime import datetime as _dt
        import pandas as _pd

        results_by_date = defaultdict(lambda: {'put': [], 'call': []})
        for result in results:
            ts = result['timestamp']
            if hasattr(ts, 'date'):
                trading_date = ts.date()
            else:
                trading_date = _pd.to_datetime(ts).date()
            opt_type = result.get('option_type', 'unknown').lower()
            if opt_type in ['put', 'call']:
                results_by_date[trading_date][opt_type].append(result)

        strategy_results = []
        for trading_date, type_results in sorted(results_by_date.items()):
            for opt_type in ['put', 'call']:
                day_results = type_results[opt_type]
                if not day_results:
                    continue
                first_result = day_results[0]
                prev_close = first_result.get('prev_close')
                current_close = first_result.get('current_close')
                if prev_close is None or current_close is None:
                    continue

                trading_datetime = _dt.combine(trading_date, _dt.min.time())
                try:
                    positions = strategy.select_entries(
                        day_results=day_results,
                        prev_close=prev_close,
                        option_type=opt_type,
                        trading_date=trading_datetime,
                    )
                    strat_result = strategy.calculate_pnl(
                        positions=positions,
                        close_price=current_close,
                        option_type=opt_type,
                        trading_date=trading_datetime,
                    )
                    strategy_results.append(strat_result)
                except Exception:
                    continue

        # Build metrics from strategy results
        if strategy_results:
            total_trades = len(strategy_results)
            wins = sum(1 for sr in strategy_results if sr.total_pnl and sr.total_pnl > 0)
            total_credits = sum(sr.total_credit for sr in strategy_results)
            gains = sum(sr.total_pnl for sr in strategy_results if sr.total_pnl and sr.total_pnl > 0)
            losses = sum(abs(sr.total_pnl) for sr in strategy_results if sr.total_pnl and sr.total_pnl < 0)
            net_pnl = sum(sr.total_pnl or 0 for sr in strategy_results)
            total_max_loss = sum(sr.total_max_loss for sr in strategy_results)
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            profit_factor = (gains / losses) if losses > 0 else float('inf')
            roi = (net_pnl / total_max_loss * 100) if total_max_loss > 0 else 0

            return {
                'total_trades': total_trades,
                'win_rate': round(win_rate, 1),
                'total_credits': round(total_credits, 2),
                'total_gains': round(gains, 2),
                'total_losses': round(losses, 2),
                'net_pnl': round(net_pnl, 2),
                'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else float('inf'),
                'roi': round(roi, 2),
            }

    return compute_metrics(results)


def _format_grid_top_results(results: List[dict], sort_by: str, top_n: int) -> str:
    """Format top grid search results for terminal display."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f" TOP {min(top_n, len(results))} RESULTS (sorted by {sort_by})")
    lines.append(f"{'='*80}")

    for i, r in enumerate(results[:top_n], 1):
        combo = r['combo']
        m = r['metrics']

        pf_str = f"{m['profit_factor']:.1f}" if m['profit_factor'] != float('inf') else 'inf'
        pb_put = combo.get('percent_beyond_put', '-')
        pb_call = combo.get('percent_beyond_call', '-')

        # Format max_spread_width - show put:call if separate values, else single value
        msw_put = combo.get('max_spread_width_put', combo.get('max_spread_width', '-'))
        msw_call = combo.get('max_spread_width_call', combo.get('max_spread_width', '-'))
        if msw_put == msw_call:
            msw_str = str(msw_put)
        else:
            msw_str = f"{msw_put}:{msw_call}"

        param_str = (
            f"type={combo.get('option_type', '-')} "
            f"pb={pb_put}:{pb_call} "
            f"msw={msw_str} "
            f"mcp={combo.get('min_contract_price', '-')} "
            f"mcr={combo.get('max_credit_width_ratio', '-')} "
            f"msd={combo.get('max_strike_distance_pct', '-')} "
            f"mih={combo.get('min_trading_hour', '-')} "
            f"mth={combo.get('max_trading_hour', '-')} "
            f"ptp={combo.get('profit_target_pct', '-')}"
        )

        lines.append(
            f"#{i:<3} Net P&L: ${m['net_pnl']:>10,.2f}  "
            f"PF: {pf_str:<5}  "
            f"WR: {m['win_rate']:.0f}%  "
            f"Trades: {m['total_trades']:<4}  "
            f"| {param_str}"
        )
    return '\n'.join(lines)


def _write_grid_results_csv(results: List[dict], output_path: str):
    """Write grid search results to CSV."""
    if not results:
        return

    all_param_keys = set()
    for r in results:
        all_param_keys.update(r['combo'].keys())
    param_cols = sorted(all_param_keys)
    metric_cols = ['total_trades', 'win_rate', 'total_credits', 'total_gains',
                   'total_losses', 'net_pnl', 'profit_factor', 'roi']
    fieldnames = ['rank'] + param_cols + metric_cols

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(results, 1):
            row = {'rank': i}
            row.update(r['combo'])
            row.update(r['metrics'])
            writer.writerow(row)

    print(f"\nResults written to: {output_path} ({len(results)} rows)")


def _run_grid_combo_worker(args_tuple):
    """Worker function to run a single grid combination in a separate process."""
    combo, fixed_params, csv_paths, cache_dir, no_cache, log_level = args_tuple[:6]
    strategy_config = args_tuple[6] if len(args_tuple) > 6 else None

    logger = get_logger(f"grid_worker_{os.getpid()}", level=log_level)

    try:
        # Load data from cache (fast)
        df = load_data_cached(csv_paths, cache_dir=cache_dir, no_cache=no_cache, logger=None)

        # Create DB connection for this worker
        db_path = fixed_params.get('db_path')
        if isinstance(db_path, str) and db_path.startswith('$'):
            db_path = os.environ.get(db_path[1:], None)

        db = StockQuestDB(
            db_path,
            enable_cache=not fixed_params.get('no_cache', False),
            logger=None  # Reduce noise
        )

        # Group intervals
        interval_groups = list(df.groupby('interval'))

        # Resolve output timezone
        output_tz = None
        if fixed_params.get('output_timezone'):
            try:
                output_tz = resolve_timezone(fixed_params['output_timezone'])
            except Exception:
                pass

        # Build params
        params = dict(fixed_params)
        params.update(combo)
        params['output_tz'] = output_tz

        # Build strategy for this combo if configured
        strat = _build_strategy_for_combo(strategy_config, combo, logger=logger) if strategy_config else None

        # Run backtest
        metrics = asyncio.run(_run_backtest_with_params_sync(interval_groups, db, params, logger, strategy=strat))

        # Close DB
        asyncio.run(db.close())

        return {'combo': combo, 'metrics': metrics, 'success': True}

    except Exception as e:
        return {'combo': combo, 'error': str(e), 'success': False}


async def _run_backtest_with_params_sync(interval_groups, db, params, logger, strategy=None):
    """Wrapper for run_backtest_with_params for use in worker processes."""
    return await run_backtest_with_params(interval_groups, db, params, logger, strategy=strategy)


async def _run_dte_comparison_grid_search(args, config, combinations, fixed_params, grid_params, logger):
    """Run grid search in DTE comparison mode.

    Each combination creates a pseudo-args namespace and calls analyze_dte_comparison().
    Collects results and writes CSV.
    """
    import argparse
    from .dte_comparison_utils import analyze_dte_comparison

    print(f"\nRunning DTE comparison grid search ({len(combinations)} combinations)...")
    print("-" * 80)

    successful_results = []
    failed_count = 0

    for idx, combo in enumerate(combinations, 1):
        # Build a namespace merging fixed_params + combo
        params = dict(fixed_params)
        params.update(combo)

        # Create a mock args namespace
        mock_args = argparse.Namespace(
            underlying_ticker=params.get('underlying_ticker'),
            multi_dte_dir=params.get('multi_dte_dir', 'options_csv_output_full'),
            dte_buckets=str(params.get('dte_buckets', '0,3,5,10')),
            dte_tolerance=int(params.get('dte_tolerance', 1)),
            exit_profit_pcts=str(params.get('exit_profit_pcts', '50,60,70,80,90')),
            min_volume=int(params.get('min_volume', 5)),
            hold_max_days=params.get('hold_max_days'),
            percent_beyond=str(params.get('percent_beyond', '0.015')),
            max_spread_width=str(params.get('max_spread_width', '200')),
            risk_cap=params.get('risk_cap'),
            min_spread_width=float(params.get('min_spread_width', 5.0)),
            max_credit_width_ratio=float(params.get('max_credit_width_ratio', 0.60)),
            min_contract_price=float(params.get('min_contract_price', 0.0)),
            option_type=params.get('option_type', 'both'),
            use_mid_price=params.get('use_mid_price', False),
            start_date=params.get('start_date'),
            end_date=params.get('end_date'),
            output_timezone=params.get('output_timezone', 'America/Los_Angeles'),
            db_path=params.get('db_path'),
            no_cache=params.get('no_cache', False),
            no_data_cache=params.get('no_data_cache', False),
            cache_dir=params.get('cache_dir', getattr(args, 'cache_dir', '.options_cache')),
            log_level=getattr(args, 'log_level', 'WARNING'),
            summary=True,
            summary_only=True,
        )

        try:
            # Capture the analysis by running it (output goes to stdout)
            # We need to collect metrics, so we call analyze_dte_comparison
            # and parse the results from the return
            from .dte_comparison_utils import analyze_dte_comparison as _dte_analysis
            from .data_loader import load_multi_dte_data
            from .spread_builder import parse_percent_beyond, parse_max_spread_width, build_credit_spreads, calculate_option_price
            from .interval_analyzer import parse_pst_timestamp, round_to_15_minutes
            from .timezone_utils import resolve_timezone
            from .price_utils import get_previous_close_price
            from common.questdb_db import StockQuestDB
            from collections import defaultdict

            # --- inline mini-analysis to get metrics ---
            ticker = mock_args.underlying_ticker
            dte_buckets_list = tuple(int(x.strip()) for x in mock_args.dte_buckets.split(','))
            exit_pcts = [float(x.strip()) for x in mock_args.exit_profit_pcts.split(',')]
            percent_beyond = parse_percent_beyond(mock_args.percent_beyond)
            max_spread_width = parse_max_spread_width(mock_args.max_spread_width)

            df = load_multi_dte_data(
                csv_dir=mock_args.multi_dte_dir,
                ticker=ticker,
                start_date=mock_args.start_date,
                end_date=mock_args.end_date,
                dte_buckets=dte_buckets_list,
                dte_tolerance=mock_args.dte_tolerance,
                cache_dir=mock_args.cache_dir,
                no_cache=mock_args.no_data_cache,
                logger=None,
            )

            db_path = mock_args.db_path
            if isinstance(db_path, str) and db_path.startswith('$'):
                db_path = os.environ.get(db_path[1:], None)

            db = StockQuestDB(db_path, enable_cache=not mock_args.no_cache, logger=None)

            option_types = ['put', 'call'] if mock_args.option_type == 'both' else [mock_args.option_type]
            df['trading_date'] = df['timestamp'].apply(
                lambda x: x.date() if hasattr(x, 'date') else pd.to_datetime(x).date()
            )
            trading_dates = sorted(df['trading_date'].unique())

            all_trades = []
            try:
                for trading_date in trading_dates:
                    day_df = df[df['trading_date'] == trading_date]
                    first_ts = day_df['timestamp'].min()
                    prev_close_result = await get_previous_close_price(db, ticker, first_ts, None)
                    if prev_close_result is None:
                        continue
                    prev_close, _ = prev_close_result

                    for dte_bucket in dte_buckets_list:
                        bucket_df = day_df[day_df['dte_bucket'] == dte_bucket]
                        if bucket_df.empty:
                            continue
                        hold_max = mock_args.hold_max_days if mock_args.hold_max_days else max(dte_bucket, 1)

                        for opt_type in option_types:
                            entry_intervals = sorted(bucket_df['interval'].unique())
                            if not entry_intervals:
                                continue

                            best_spread = None
                            best_entry_interval = None
                            for ei in entry_intervals:
                                idata = bucket_df[bucket_df['interval'] == ei]
                                spreads = build_credit_spreads(
                                    idata, opt_type, prev_close,
                                    percent_beyond, mock_args.min_spread_width,
                                    max_spread_width, mock_args.use_mid_price,
                                    min_contract_price=mock_args.min_contract_price,
                                    max_credit_width_ratio=mock_args.max_credit_width_ratio,
                                    min_volume=mock_args.min_volume,
                                )
                                if not spreads:
                                    continue
                                if mock_args.risk_cap is not None:
                                    spreads = [s for s in spreads if s['max_loss_per_contract'] > 0 and s['max_loss_per_contract'] <= mock_args.risk_cap]
                                if not spreads:
                                    continue
                                c = max(spreads, key=lambda x: x['net_credit'])
                                if best_spread is None or c['net_credit'] > best_spread['net_credit']:
                                    best_spread = c
                                    best_entry_interval = ei

                            if best_spread is None:
                                continue

                            best_spread['option_type'] = opt_type
                            num_contracts = 1
                            if mock_args.risk_cap and best_spread['max_loss_per_contract'] > 0:
                                num_contracts = int(mock_args.risk_cap / best_spread['max_loss_per_contract'])
                                if num_contracts < 1:
                                    continue

                            for pct in exit_pcts:
                                from .dte_comparison_utils import find_intraday_exit, _compute_eod_pnl, track_held_position_inmem
                                exited, _, exit_pnl = find_intraday_exit(
                                    best_spread, bucket_df, best_entry_interval, pct, mock_args.use_mid_price
                                )
                                if exited:
                                    final_pnl = exit_pnl
                                else:
                                    if dte_bucket == 0:
                                        eod = _compute_eod_pnl(best_spread, bucket_df, mock_args.use_mid_price)
                                        final_pnl = eod if eod is not None else best_spread['net_credit']
                                    else:
                                        eod = _compute_eod_pnl(best_spread, bucket_df, mock_args.use_mid_price)
                                        if eod is not None and eod > 0:
                                            final_pnl = eod
                                        else:
                                            _, tp, _ = track_held_position_inmem(
                                                best_spread, trading_date, df,
                                                hold_max, pct, mock_args.use_mid_price
                                            )
                                            if tp is not None:
                                                final_pnl = tp
                                            elif eod is not None:
                                                final_pnl = eod
                                            else:
                                                final_pnl = best_spread['net_credit']

                                total_pnl = final_pnl * num_contracts * 100
                                all_trades.append({
                                    'dte_bucket': dte_bucket,
                                    'profit_target_pct': pct,
                                    'total_pnl': total_pnl,
                                    'win': final_pnl > 0,
                                })
            finally:
                await db.close()

            # Compute aggregate metrics
            if all_trades:
                total_trades = len(all_trades)
                wins = sum(1 for t in all_trades if t['win'])
                net_pnl = sum(t['total_pnl'] for t in all_trades)
                gains = sum(t['total_pnl'] for t in all_trades if t['total_pnl'] > 0)
                losses_val = sum(abs(t['total_pnl']) for t in all_trades if t['total_pnl'] < 0)
                win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
                pf = (gains / losses_val) if losses_val > 0 else float('inf')

                metrics = {
                    'total_trades': total_trades,
                    'win_rate': round(win_rate, 1),
                    'total_credits': 0,
                    'total_gains': round(gains, 2),
                    'total_losses': round(losses_val, 2),
                    'net_pnl': round(net_pnl, 2),
                    'profit_factor': round(pf, 2) if pf != float('inf') else float('inf'),
                    'roi': 0,
                }
                successful_results.append({'combo': combo, 'metrics': metrics})
                print(
                    f"  [{idx}/{len(combinations)}] OK  "
                    f"Net P&L: ${net_pnl:>10,.2f}  "
                    f"PF: {pf:.2f}  "
                    f"WR: {win_rate:.0f}%  "
                    f"Trades: {total_trades}"
                )
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
            if idx <= 5 or idx % 50 == 0:
                print(f"  [{idx}/{len(combinations)}] FAIL: {str(e)[:80]}")

    print("-" * 80)
    print(f"Completed | Successful: {len(successful_results)} | Failed/Skipped: {failed_count}")

    if not successful_results:
        print("\nNo successful results.")
        return 0

    # Sort and display
    sort_key_name = getattr(args, 'grid_sort', 'net_pnl')
    successful_results.sort(
        key=lambda r: r['metrics'].get(sort_key_name, 0)
        if r['metrics'].get(sort_key_name, 0) != float('inf') else float('inf'),
        reverse=True,
    )

    top_n = getattr(args, 'grid_top_n', 20)
    print(_format_grid_top_results(successful_results, sort_key_name, top_n))
    _write_grid_results_csv(successful_results, args.grid_output)

    return 0


async def run_grid_search(args):
    """Run the grid search optimization mode."""
    logger = get_logger("grid_search", level=getattr(args, 'log_level', 'INFO'))

    # Load grid config
    with open(args.grid_config, 'r') as f:
        config = json.load(f)

    fixed_params = config.get('fixed_params', {})
    grid_params = config.get('grid_params', {})
    strategy_config = config.get('strategy', None)
    linked_params = config.get('linked_params', {})

    if not grid_params:
        print("Error: grid_params section is empty in config")
        return 1

    # Remove linked param targets from grid_params (they mirror their source)
    sweep_params = {k: v for k, v in grid_params.items() if k not in linked_params}

    # Generate combinations from independent params only
    combinations = _generate_combinations(sweep_params)

    # Apply linked params: copy source value to each linked target
    if linked_params:
        for combo in combinations:
            for target_key, source_key in linked_params.items():
                if source_key in combo:
                    combo[target_key] = combo[source_key]

    total_combos = len(combinations)

    # Show grid info
    print(f"Grid Search Configuration:")
    print(f"  Config file: {args.grid_config}")
    if strategy_config:
        strategy_name = strategy_config.get('name', strategy_config.get('strategy', 'unknown'))
        print(f"  Strategy: {strategy_name}")
        if strategy_config.get('config_file'):
            print(f"  Strategy config: {strategy_config['config_file']}")
        if strategy_config.get('feature_flags'):
            print(f"  Base feature flags: {strategy_config['feature_flags']}")
    print(f"  Parameters: {len(grid_params)}")
    for name, spec in grid_params.items():
        values = _expand_grid_param(name, spec)
        if len(values) <= 5:
            print(f"    {name}: {len(values)} values {values}")
        else:
            print(f"    {name}: {len(values)} values [{values[0]} ... {values[-1]}]")
    if linked_params:
        print(f"  Linked parameters: {len(linked_params)}")
        for target, source in linked_params.items():
            print(f"    {target} <- {source}")
    print(f"  Total combinations: {total_combos:,}")
    print(f"  Sort by: {args.grid_sort}")
    print(f"  Output: {args.grid_output}")

    if args.grid_dry_run:
        print(f"\n--grid-dry-run: Would run {total_combos:,} backtests. Exiting.")
        return 0

    # Dispatch to DTE comparison grid search if mode is dte-comparison
    mode = fixed_params.get('mode', getattr(args, 'mode', 'credit-spread'))
    if mode == 'dte-comparison':
        two_phase = fixed_params.get('two_phase', False)
        if two_phase:
            return await _run_dte_comparison_grid_search_v2(
                args, config, combinations, fixed_params, grid_params, logger
            )
        return await _run_dte_comparison_grid_search(
            args, config, combinations, fixed_params, grid_params, logger
        )

    # Resolve CSV paths from fixed_params
    csv_paths = []
    if 'csv_path' in fixed_params:
        import glob as glob_module
        raw_paths = fixed_params['csv_path']
        if isinstance(raw_paths, str):
            raw_paths = [raw_paths]
        for p in raw_paths:
            p = os.path.expandvars(p)
            expanded = glob_module.glob(p)
            if expanded:
                csv_paths.extend(sorted(expanded))
            else:
                csv_paths.append(p)
    elif 'csv_dir' in fixed_params and fixed_params.get('underlying_ticker'):
        ticker = fixed_params['underlying_ticker']
        found = find_csv_files_in_dir(
            fixed_params['csv_dir'], ticker,
            fixed_params.get('start_date'), fixed_params.get('end_date'),
            logger
        )
        csv_paths = [str(p) for p in found]

    if not csv_paths:
        print("Error: No CSV files resolved from fixed_params")
        return 1

    print(f"  CSV files: {len(csv_paths)}")

    # Load data (ONCE) using cache
    cache_dir = getattr(args, 'cache_dir', '.options_cache')
    no_cache = getattr(args, 'no_data_cache', False)
    try:
        df = load_data_cached(csv_paths, cache_dir=cache_dir, no_cache=no_cache, logger=logger)
    except (ValueError, Exception) as e:
        print(f"Error loading data: {e}")
        return 1

    print(f"  Loaded {len(df):,} rows, {df['interval'].nunique()} intervals")

    # Initialize DB (ONCE)
    db_path = fixed_params.get('db_path')
    if isinstance(db_path, str) and db_path.startswith('$'):
        db_path = os.environ.get(db_path[1:], None)
    db = StockQuestDB(
        db_path,
        enable_cache=not fixed_params.get('no_cache', False),
        logger=logger
    )

    # Pre-group intervals (ONCE)
    interval_groups = list(df.groupby('interval'))
    print(f"  Interval groups: {len(interval_groups)}")

    # Resolve output timezone
    output_tz = None
    if fixed_params.get('output_timezone'):
        try:
            output_tz = resolve_timezone(fixed_params['output_timezone'])
        except Exception:
            pass

    # Resume support
    existing_keys = set()
    if args.grid_resume:
        existing_keys = _load_existing_grid_results(args.grid_output)
        if existing_keys:
            print(f"  Resuming: {len(existing_keys)} existing results found, skipping those.")

    # Filter pending combos
    pending_combos = []
    for combo in combinations:
        if _combo_to_key(combo) not in existing_keys:
            pending_combos.append(combo)

    if not pending_combos:
        print("\nAll combinations already completed. Nothing to run.")
        await db.close()
        return 0

    # Determine number of processes
    num_processes = getattr(args, 'processes', 1) or 1
    if num_processes == 0:
        num_processes = multiprocessing.cpu_count()
    use_parallel = num_processes > 1

    print(f"  Combinations to run: {len(pending_combos):,}")
    print(f"  Parallel processes: {num_processes}")
    print(f"\nStarting grid search...")
    print("-" * 80)

    # Run grid search
    successful_results = []
    failed_count = 0
    start_time = time.time()

    if use_parallel:
        # Parallel execution using multiprocessing
        await db.close()  # Close DB - workers will create their own

        # Prepare worker args (strategy_config is JSON-serializable, no pickling issues)
        log_level = getattr(args, 'log_level', 'WARNING')
        worker_args = [
            (combo, fixed_params, csv_paths, cache_dir, no_cache, log_level, strategy_config)
            for combo in pending_combos
        ]

        completed = 0
        with multiprocessing.Pool(processes=num_processes) as pool:
            for result in pool.imap_unordered(_run_grid_combo_worker, worker_args):
                completed += 1
                if result['success']:
                    metrics = result['metrics']
                    if metrics['total_trades'] > 0:
                        successful_results.append({'combo': result['combo'], 'metrics': metrics})
                        print(
                            f"  [{completed}/{len(pending_combos)}] OK  "
                            f"Net P&L: ${metrics['net_pnl']:>10,.2f}  "
                            f"PF: {metrics['profit_factor']:.2f}  "
                            f"WR: {metrics['win_rate']:.0f}%  "
                            f"Trades: {metrics['total_trades']}"
                        )
                    else:
                        failed_count += 1
                else:
                    failed_count += 1
                    if completed <= 5 or completed % 100 == 0:
                        print(f"  [{completed}/{len(pending_combos)}] FAIL: {result.get('error', 'Unknown')[:80]}")
    else:
        # Sequential execution
        try:
            for idx, combo in enumerate(pending_combos, 1):
                # Build params from combo + fixed_params
                params = dict(fixed_params)
                params.update(combo)
                params['output_tz'] = output_tz

                # Build strategy for this combo if configured
                strat = _build_strategy_for_combo(strategy_config, combo, logger=logger) if strategy_config else None

                try:
                    metrics = await run_backtest_with_params(interval_groups, db, params, logger, strategy=strat)
                    if metrics['total_trades'] > 0:
                        successful_results.append({'combo': combo, 'metrics': metrics})
                        print(
                            f"  [{idx}/{len(pending_combos)}] OK  "
                            f"Net P&L: ${metrics['net_pnl']:>10,.2f}  "
                            f"PF: {metrics['profit_factor']:.2f}  "
                            f"WR: {metrics['win_rate']:.0f}%  "
                            f"Trades: {metrics['total_trades']}"
                        )
                    else:
                        failed_count += 1
                        if idx <= 5 or idx % 100 == 0:
                            print(f"  [{idx}/{len(pending_combos)}] SKIP: No trades")
                except Exception as e:
                    failed_count += 1
                    if idx <= 5 or idx % 100 == 0:
                        print(f"  [{idx}/{len(pending_combos)}] FAIL: {str(e)[:80]}")

        finally:
            await db.close()

    elapsed = time.time() - start_time
    print("-" * 80)
    print(f"Completed in {elapsed:.1f}s | Successful: {len(successful_results)} | Failed/Skipped: {failed_count}")

    if not successful_results:
        print("\nNo successful results to report.")
        return 0

    # Sort results
    def sort_key(r):
        val = r['metrics'].get(args.grid_sort, 0)
        if val == float('inf'):
            return float('inf')
        return val

    successful_results.sort(key=sort_key, reverse=True)

    # Display top results
    print(_format_grid_top_results(successful_results, args.grid_sort, args.grid_top_n))

    # Write CSV
    _write_grid_results_csv(successful_results, args.grid_output)

    return 0


async def _run_dte_comparison_grid_search_v2(args, config, combinations, fixed_params, grid_params, logger):
    """Two-phase grid search for DTE comparison mode.

    Phase A: For each unique percent_beyond_percentile, run build_raw_trades() once.
    Phase B: For each combination of exit strategy params, evaluate post-hoc.
    This is orders of magnitude faster than re-running Phase A for every combo.
    """
    from .dte_comparison_utils import (
        precompute_adaptive_percent_beyond,
        build_raw_trades,
        evaluate_exit_strategies,
        ExitStrategyConfig,
        RawTradeStore,
        _summarize_trades_v2,
    )
    from .data_loader import (
        load_multi_dte_data,
        preload_vix1d_series,
        preload_underlying_close_series,
    )
    from .spread_builder import parse_max_spread_width
    from .price_utils import clear_price_cache
    from .timezone_utils import resolve_timezone
    from common.questdb_db import StockQuestDB
    import argparse

    print(f"\nRunning TWO-PHASE DTE comparison grid search...")
    print("-" * 80)

    ticker = fixed_params.get('underlying_ticker')
    multi_dte_dir = fixed_params.get('multi_dte_dir', 'options_csv_output_full')
    dte_buckets_str = str(fixed_params.get('dte_buckets', '0,3,5,10'))
    dte_tolerance = int(fixed_params.get('dte_tolerance', 1))
    lookback_days = int(fixed_params.get('percentile_lookback', 180))
    cache_dir = getattr(args, 'cache_dir', fixed_params.get('cache_dir', '.options_cache'))

    dte_buckets = tuple(int(x.strip()) for x in dte_buckets_str.split(','))

    # Separate Phase A params (require re-run) from Phase B params (fast sweep)
    percentile_values = grid_params.get('percent_beyond_percentile', [95])
    if not isinstance(percentile_values, list):
        percentile_values = [percentile_values]
    percentile_values = [int(p) for p in percentile_values]

    exit_profit_pcts_grid = grid_params.get('exit_profit_pcts', [50, 70, 90])
    if not isinstance(exit_profit_pcts_grid, list):
        exit_profit_pcts_grid = [exit_profit_pcts_grid]

    exit_dte_grid = grid_params.get('exit_dte', [None])
    if not isinstance(exit_dte_grid, list):
        exit_dte_grid = [exit_dte_grid]

    min_vix1d_grid = grid_params.get('min_vix1d_entry', [None])
    if not isinstance(min_vix1d_grid, list):
        min_vix1d_grid = [min_vix1d_grid]

    max_vix1d_grid = grid_params.get('max_vix1d_entry', [None])
    if not isinstance(max_vix1d_grid, list):
        max_vix1d_grid = [max_vix1d_grid]

    stop_loss_grid = grid_params.get('stop_loss_multiple', [None])
    if not isinstance(stop_loss_grid, list):
        stop_loss_grid = [stop_loss_grid]

    flow_filter_grid = grid_params.get('flow_filter', [None])
    if not isinstance(flow_filter_grid, list):
        flow_filter_grid = [flow_filter_grid]

    # Count Phase B combinations
    phase_b_combos = (len(exit_profit_pcts_grid) * len(exit_dte_grid) *
                      len(min_vix1d_grid) * len(max_vix1d_grid) *
                      len(stop_loss_grid) * len(flow_filter_grid))
    total_combos = len(percentile_values) * phase_b_combos
    print(f"  Phase A runs: {len(percentile_values)} (one per percentile)")
    print(f"  Phase B evaluations per Phase A: {phase_b_combos}")
    print(f"  Total combinations: {total_combos}")

    # Load multi-DTE data ONCE
    t_start = time.time()
    print(f"\nLoading multi-DTE data from {multi_dte_dir}/{ticker}...", flush=True)
    df = load_multi_dte_data(
        csv_dir=multi_dte_dir,
        ticker=ticker,
        start_date=fixed_params.get('start_date'),
        end_date=fixed_params.get('end_date'),
        dte_buckets=dte_buckets,
        dte_tolerance=dte_tolerance,
        cache_dir=cache_dir,
        no_cache=fixed_params.get('no_data_cache', False),
        logger=None,
    )
    df['trading_date'] = df['timestamp'].apply(
        lambda x: x.date() if hasattr(x, 'date') else pd.to_datetime(x).date()
    )
    trading_dates = sorted(df['trading_date'].unique())
    print(f"Loaded {len(df):,} rows, {len(trading_dates)} trading days", flush=True)

    # Initialize DB
    db_path = fixed_params.get('db_path')
    if isinstance(db_path, str) and db_path.startswith('$'):
        db_path = os.environ.get(db_path[1:], None)
    db = StockQuestDB(db_path, enable_cache=True, logger=None)

    output_tz = None
    if fixed_params.get('output_timezone'):
        try:
            output_tz = resolve_timezone(fixed_params['output_timezone'])
        except Exception:
            pass

    try:
        # Preload data
        from datetime import timedelta as _td
        start_str = (min(trading_dates) - _td(days=lookback_days + 30)).isoformat()
        end_str = max(trading_dates).isoformat()

        print("Preloading VIX1D and underlying close series...", flush=True)
        vix1d_series = await preload_vix1d_series(db, start_str, end_str, None)
        underlying_series = await preload_underlying_close_series(db, ticker, start_str, end_str, None)
        print(f"  VIX1D: {len(vix1d_series)} days, Underlying: {len(underlying_series)} days", flush=True)

        # Precompute adaptive percent_beyond
        print("Precomputing adaptive percent_beyond...", flush=True)
        adaptive_lookup = await precompute_adaptive_percent_beyond(
            db, ticker, trading_dates, dte_buckets, lookback_days, percentile_values, None,
        )
        print(f"  Computed {len(adaptive_lookup)} adaptive entries", flush=True)

        # Phase A: Build raw trades for each unique percentile
        raw_stores = {}
        for pctl in percentile_values:
            strike_sel = fixed_params.get('strike_selection', 'max_credit')
            ic = fixed_params.get('iron_condor', False)
            p_hash = RawTradeStore._make_params_hash(
                ticker=ticker, percentile=pctl,
                lookback_days=lookback_days,
                min_spread_width=float(fixed_params.get('min_spread_width', 5.0)),
                max_spread_width=str(fixed_params.get('max_spread_width', '200')),
                risk_cap=fixed_params.get('risk_cap'),
                max_credit_width_ratio=float(fixed_params.get('max_credit_width_ratio', 0.60)),
                start_date=fixed_params.get('start_date', ''),
                end_date=fixed_params.get('end_date', ''),
                strike_selection=strike_sel,
                iron_condor=ic,
            )
            cached_path = RawTradeStore.find_cache(ticker, pctl, p_hash, cache_dir)
            if cached_path and not fixed_params.get('no_data_cache', False):
                print(f"\n  Loading cached raw trades for p{pctl}: {cached_path}", flush=True)
                raw_store = RawTradeStore.load(cached_path)
                print(f"  Loaded {len(raw_store.trades)} cached trades", flush=True)
            else:
                # Build mock args for Phase A
                mock_args = argparse.Namespace(
                    underlying_ticker=ticker,
                    multi_dte_dir=multi_dte_dir,
                    use_mid_price=fixed_params.get('use_mid_price', False),
                    risk_cap=fixed_params.get('risk_cap'),
                    min_spread_width=float(fixed_params.get('min_spread_width', 5.0)),
                    min_volume=int(fixed_params.get('min_volume', 5)),
                    max_credit_width_ratio=float(fixed_params.get('max_credit_width_ratio', 0.60)),
                    max_spread_width=str(fixed_params.get('max_spread_width', '200')),
                    min_contract_price=float(fixed_params.get('min_contract_price', 0.0)),
                    option_type=fixed_params.get('option_type', 'both'),
                    percentile_lookback=lookback_days,
                    no_data_cache=fixed_params.get('no_data_cache', False),
                    strike_selection=strike_sel,
                    iron_condor=ic,
                    start_date=fixed_params.get('start_date', ''),
                    end_date=fixed_params.get('end_date', ''),
                )

                raw_store = await build_raw_trades(
                    mock_args, logger, adaptive_lookup, pctl,
                    df, db, dte_buckets, vix1d_series, underlying_series, output_tz,
                )
                raw_store.save(cache_dir)

            raw_stores[pctl] = raw_store

        # Phase B: Sweep exit strategies
        print(f"\nPhase B: Evaluating {phase_b_combos} exit strategy combos per percentile...", flush=True)
        t_phase_b = time.time()

        successful_results = []

        for pctl in percentile_values:
            raw_store = raw_stores[pctl]

            for pt in exit_profit_pcts_grid:
                for ed in exit_dte_grid:
                    for min_v in min_vix1d_grid:
                        for max_v in max_vix1d_grid:
                            for sl in stop_loss_grid:
                                for ff in flow_filter_grid:
                                    exit_cfg = ExitStrategyConfig(
                                        profit_target_pct=float(pt) if pt is not None else None,
                                        exit_dte=int(ed) if ed is not None else None,
                                        min_vix1d_entry=float(min_v) if min_v is not None else None,
                                        max_vix1d_entry=float(max_v) if max_v is not None else None,
                                        stop_loss_multiple=float(sl) if sl is not None else None,
                                        flow_filter=ff,
                                    )

                                    evaluated = evaluate_exit_strategies(raw_store, [exit_cfg], dte_buckets)
                                    trades = evaluated.get(exit_cfg.label(), [])

                                    if not trades:
                                        continue

                                    total_trades = len(trades)
                                    wins = sum(1 for t in trades if t['win'])
                                    net_pnl = sum(t['total_pnl'] for t in trades)
                                    gains = sum(t['total_pnl'] for t in trades if t['total_pnl'] > 0)
                                    losses_val = sum(abs(t['total_pnl']) for t in trades if t['total_pnl'] < 0)
                                    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
                                    pf = (gains / losses_val) if losses_val > 0 else float('inf')
                                    avg_hold = sum(t.get('hold_days', 0) for t in trades) / total_trades
                                    avg_ann_roi = sum(t.get('annualized_roi', 0) for t in trades) / total_trades

                                    combo = {
                                        'percent_beyond_percentile': pctl,
                                        'exit_profit_pcts': pt,
                                        'exit_dte': ed,
                                        'min_vix1d_entry': min_v,
                                        'max_vix1d_entry': max_v,
                                        'stop_loss_multiple': sl,
                                        'flow_filter': ff,
                                    }
                                    metrics = {
                                        'total_trades': total_trades,
                                        'win_rate': round(win_rate, 1),
                                        'total_credits': 0,
                                        'total_gains': round(gains, 2),
                                        'total_losses': round(losses_val, 2),
                                        'net_pnl': round(net_pnl, 2),
                                        'profit_factor': round(pf, 2) if pf != float('inf') else float('inf'),
                                        'roi': 0,
                                        'avg_hold_days': round(avg_hold, 1),
                                        'annualized_roi': round(avg_ann_roi, 1),
                                    }
                                    successful_results.append({'combo': combo, 'metrics': metrics})

        phase_b_elapsed = time.time() - t_phase_b
        print(f"Phase B complete: {len(successful_results)} results in {phase_b_elapsed:.2f}s", flush=True)

    finally:
        await db.close()

    if not successful_results:
        print("\nNo successful results.")
        return 0

    # Sort and display
    sort_key_name = getattr(args, 'grid_sort', 'net_pnl')

    def sort_key_fn(r):
        val = r['metrics'].get(sort_key_name, 0)
        if val == float('inf'):
            return float('inf')
        return val

    successful_results.sort(key=sort_key_fn, reverse=True)

    top_n = getattr(args, 'grid_top_n', 20)

    # Custom display for two-phase results
    print(f"\n{'=' * 120}")
    print(f" TOP {min(top_n, len(successful_results))} RESULTS (sorted by {sort_key_name})")
    print(f"{'=' * 120}")

    for i, r in enumerate(successful_results[:top_n], 1):
        combo = r['combo']
        m = r['metrics']
        pf_str = f"{m['profit_factor']:.1f}" if m['profit_factor'] != float('inf') else 'inf'
        print(
            f"#{i:<3} Net P&L: ${m['net_pnl']:>12,.2f}  "
            f"PF: {pf_str:<5}  WR: {m['win_rate']:.0f}%  "
            f"Trades: {m['total_trades']:<4}  "
            f"Hold: {m.get('avg_hold_days', 0):.1f}d  "
            f"AnnROI: {m.get('annualized_roi', 0):>+.1f}%  "
            f"| p{combo['percent_beyond_percentile']} "
            f"PT={combo['exit_profit_pcts']} "
            f"SL={combo.get('stop_loss_multiple', '-')} "
            f"Flow={combo.get('flow_filter', '-')} "
            f"ExDTE={combo.get('exit_dte', '-')} "
            f"VIX>={combo.get('min_vix1d_entry', '-')} "
            f"VIX<={combo.get('max_vix1d_entry', '-')}"
        )

    # Write CSV
    _write_grid_results_csv(successful_results, args.grid_output)

    total_elapsed = time.time() - t_start
    print(f"\nTotal grid search time: {total_elapsed:.1f}s")

    return 0
