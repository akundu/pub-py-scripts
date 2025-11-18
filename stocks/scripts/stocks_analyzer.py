#!/usr/bin/env python3
"""
Stocks Analyzer - Intraday Deviation vs Previous Close and Today's Open

This tool analyzes a list of tickers and shows how far the current market price
is from the previous close and today's open, both in absolute and percentage terms.

It mirrors the CLI style of scripts/options_analyzer.py, but focuses on stocks.

Usage:
    export POLYGON_API_KEY=YOUR_API_KEY  # Optional, for symbol lists
    python scripts/stocks_analyzer.py --db-conn questdb://user:pass@host:8812/db
    python scripts/stocks_analyzer.py --symbols AAPL MSFT GOOGL --sort diff_close_pct
    python scripts/stocks_analyzer.py --types sp-500 --output csv
    python scripts/stocks_analyzer.py --output results.csv --csv-columns "ticker,current_price,diff_close_pct,diff_open_pct"
"""

import os
import sys
import argparse
import asyncio
import pandas as pd
from typing import Dict, List, Any, Optional
from tabulate import tabulate
from pathlib import Path

# Ensure project root is on sys.path so `common` can be imported when running from any cwd
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.symbol_loader import add_symbol_arguments, fetch_lists_data
from common.stock_db import get_stock_db
from common.common_strategies import compute_rsi_series


class StocksAnalyzer:
    """Analyzes deviations of current stock price vs previous close and today's open."""

    def __init__(self, db_conn: str, quiet: bool = False, enable_cache: bool = True, redis_url: str | None = None):
        self.db_conn = db_conn
        self.quiet = quiet
        self.enable_cache = enable_cache
        self.redis_url = redis_url
        self.db = None

    async def initialize(self):
        try:
            self.db = get_stock_db('questdb', db_config=self.db_conn, enable_cache=self.enable_cache, redis_url=self.redis_url)
            # Ensure database is initialized (this also initializes the cache)
            if hasattr(self.db, '_init_db'):
                await self.db._init_db()
            if not self.quiet:
                print("Database connection established successfully.", file=sys.stderr)
        except Exception as e:
            print(f"Error connecting to database: {e}", file=sys.stderr)
            sys.exit(1)

    async def get_financial_info(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch latest available financial info per ticker (pe_ratio, market_cap, price)."""
        results: Dict[str, Dict[str, Any]] = {}
        for t in tickers:
            try:
                # StockQuestDB.get_financial_info returns latest by date
                df = await self.db.get_financial_info(t)
                if not df.empty:
                    row = df.reset_index().iloc[-1]
                    results[t] = {
                        'pe_ratio': row.get('price_to_earnings'),
                        'market_cap': row.get('market_cap'),
                        'price': row.get('price'),
                    }
                else:
                    results[t] = {'pe_ratio': None, 'market_cap': None, 'price': None}
            except Exception:
                results[t] = {'pe_ratio': None, 'market_cap': None, 'price': None}
        return results

    async def analyze(
        self,
        tickers: List[str],
        use_market_time: bool = True,
        rsi_periods: Optional[List[int]] = None,
        debug: bool = False,
    ) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        tickers_upper = [t.upper() for t in tickers]

        # Fetch data concurrently where possible
        latest_prices_task = asyncio.create_task(self.db.get_latest_prices(tickers_upper, use_market_time=use_market_time))
        prev_close_task = asyncio.create_task(self.db.get_previous_close_prices(tickers_upper))
        today_open_task = asyncio.create_task(self.db.get_today_opening_prices(tickers_upper))
        financial_task = asyncio.create_task(self.get_financial_info(tickers_upper))

        latest_prices, prev_close, today_open, fin = await asyncio.gather(
            latest_prices_task, prev_close_task, today_open_task, financial_task
        )

        # Fetch historical closes for RSI if requested
        periods = [14]
        if rsi_periods and len(rsi_periods) > 0:
            periods = sorted(set(int(p) for p in rsi_periods if int(p) > 0))

        need_rsi = bool(periods)
        hist_df = pd.DataFrame()
        if need_rsi:
            max_period = max(periods) if periods else 14
            # Fetch a little extra history to be safe
            lookback = max_period + 5
            # QuestDB may choke on very large IN lists; fetch in batches
            batch_size = 150
            hist_parts: List[pd.DataFrame] = []
            for i in range(0, len(tickers_upper), batch_size):
                batch = tickers_upper[i:i+batch_size]
                in_list = ",".join([f"'{t}'" for t in batch])
                sql = (
                    f"SELECT ticker, date, close FROM daily_prices "
                    f"WHERE ticker IN ({in_list}) "
                    f"ORDER BY ticker, date ASC"
                )
                try:
                    part = await self.db.execute_select_sql(sql)
                    if part is not None and not part.empty:
                        # Normalize columns per part to be resilient to unnamed columns
                        cols = [str(c).lower() for c in part.columns]
                        part.columns = cols
                        # If expected names missing but exactly 3 columns, assign by position
                        if not any(name in part.columns for name in ('ticker', 'symbol')) or 'close' not in part.columns:
                            if part.shape[1] >= 3:
                                part = part.copy()
                                part.rename(columns={cols[0]: 'ticker', cols[1]: 'date', cols[2]: 'close'}, inplace=True)
                        hist_parts.append(part)
                except Exception:
                    continue

            if hist_parts:
                hist_df = pd.concat(hist_parts, ignore_index=True)
            else:
                hist_df = pd.DataFrame()

            if not hist_df.empty:
                # Normalize column names to lowercase and reconcile timestamp/date names
                hist_df.columns = [str(c).lower() for c in hist_df.columns]
                # Handle unnamed numeric columns from concatenation if any
                if not {'ticker','date','close'}.issubset(set(hist_df.columns)) and hist_df.shape[1] >= 3:
                    # Attempt to map first three columns to expected names
                    current_cols = list(hist_df.columns)
                    rename_map = {current_cols[0]: 'ticker', current_cols[1]: 'date', current_cols[2]: 'close'}
                    hist_df = hist_df.rename(columns=rename_map)
            if 'date' not in hist_df.columns:
                # Try common alternatives
                if 'timestamp' in hist_df.columns:
                    hist_df['date'] = hist_df['timestamp']
                elif 'ts' in hist_df.columns:
                    hist_df['date'] = hist_df['ts']
            if 'close' not in hist_df.columns:
                # Try alternative casing
                for alt in ['closing_price', 'adj_close', 'price']:
                    if alt in hist_df.columns:
                        hist_df['close'] = hist_df[alt]
                        break
            if 'ticker' not in hist_df.columns and 'symbol' in hist_df.columns:
                hist_df['ticker'] = hist_df['symbol']

            # Ensure required columns exist before proceeding
            required_cols = {'ticker', 'date', 'close'}
            if required_cols.issubset(set(hist_df.columns)):
                hist_df['date'] = pd.to_datetime(hist_df['date'])
                # Keep the last N rows per ticker (avoid deprecated apply)
                hist_df = hist_df.sort_values(['ticker', 'date']).groupby('ticker', group_keys=False).tail(lookback)
            else:
                # If columns are missing, skip RSI computation
                if not self.quiet:
                    print("Skipping RSI: historical data missing required columns", file=sys.stderr)
                hist_df = pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        for t in tickers_upper:
            cur = latest_prices.get(t)
            pc = prev_close.get(t)
            op = today_open.get(t)

            # Compute deviations (guard against None)
            def pct_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
                if a is None or b is None or b == 0:
                    return None
                return round(((a - b) / b) * 100.0, 2)

            def abs_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
                if a is None or b is None:
                    return None
                return round(a - b, 2)

            diff_close = abs_diff(cur, pc)
            diff_close_pct = pct_diff(cur, pc)
            diff_open = abs_diff(cur, op)
            diff_open_pct = pct_diff(cur, op)

            pe_ratio = fin.get(t, {}).get('pe_ratio')
            market_cap = fin.get(t, {}).get('market_cap')
            market_cap_b = round(market_cap / 1e9, 2) if market_cap is not None and pd.notna(market_cap) else None

            base_row: Dict[str, Any] = {
                'ticker': t,
                'current_price': cur,
                'prev_close': pc,
                'today_open': op,
                'diff_close': diff_close,
                'diff_close_pct': diff_close_pct,
                'diff_open': diff_open,
                'diff_open_pct': diff_open_pct,
                'pe_ratio': pe_ratio,
                'market_cap': market_cap,
                'market_cap_b': market_cap_b,
            }

            if need_rsi:
                for p in periods:
                    base_row.setdefault(f'rsi_{p}', None)

            # Compute current RSI values for requested periods
            if need_rsi and not hist_df.empty:
                sub = hist_df[hist_df['ticker'] == t]
                if not sub.empty:
                    series = sub['close'].astype(float)
                    for p in periods:
                        # Compute RSI and optionally print debug details
                        rsi_series = compute_rsi_series(series, window=p)
                        val = rsi_series.iloc[-1] if not rsi_series.empty else None
                        base_row[f'rsi_{p}'] = round(float(val), 2) if pd.notna(val) else None

                        if debug:
                            # Derive intermediate values for transparency
                            delta = series.diff()
                            gain = delta.where(delta > 0, 0.0).rolling(window=p, min_periods=p).mean()
                            loss = (-delta.where(delta < 0, 0.0)).rolling(window=p, min_periods=p).mean()
                            rs = gain / loss
                            # Build compact debug snapshot
                            closes_dbg = series.tail(p + 1).round(4).tolist()
                            deltas_dbg = delta.tail(p + 1).round(4).tolist()
                            avg_gain = float(gain.iloc[-1]) if pd.notna(gain.iloc[-1]) else None
                            avg_loss = float(loss.iloc[-1]) if pd.notna(loss.iloc[-1]) else None
                            rs_last = float(rs.iloc[-1]) if pd.notna(rs.iloc[-1]) else None
                            rsi_last = float(val) if pd.notna(val) else None
                            print(
                                f"[DEBUG][{t}] RSI_{p}: closes={closes_dbg} deltas={deltas_dbg} "
                                f"avg_gain={avg_gain} avg_loss={avg_loss} rs={rs_last} rsi={rsi_last}",
                                file=sys.stderr,
                            )

            rows.append(base_row)

        df = pd.DataFrame(rows)
        return df

    def _format_csv_output(
        self,
        df: pd.DataFrame,
        delimiter: str = ',',
        quoting: str = 'minimal',
        output_file: Optional[str] = None,
        csv_columns: Optional[List[str]] = None,
    ) -> str:
        import csv

        quoting_map = {
            'minimal': csv.QUOTE_MINIMAL,
            'all': csv.QUOTE_ALL,
            'none': csv.QUOTE_NONE,
            'nonnumeric': csv.QUOTE_NONNUMERIC,
        }
        csv_quoting = quoting_map.get(quoting, csv.QUOTE_MINIMAL)

        df_csv = df.copy()
        if csv_columns:
            cols = [c for c in csv_columns if c in df_csv.columns]
            if cols:
                df_csv = df_csv[cols]

        csv_content = df_csv.to_csv(index=False, sep=delimiter, quoting=csv_quoting, na_rep='', float_format='%.2f')
        if output_file:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                f.write(csv_content)
            if not self.quiet:
                print(f"CSV results saved to {output_file}")
        return csv_content

    def format_output(
        self,
        df: pd.DataFrame,
        output_format: str = 'table',
        output_file: Optional[str] = None,
        sort_by: Optional[str] = 'diff_close_pct',
        csv_delimiter: str = ',',
        csv_quoting: str = 'minimal',
        csv_columns: Optional[List[str]] = None,
    ) -> str:
        if df.empty:
            return "No stock data available."

        # Sorting (support substring match)
        # Map generic 'rsi' sort to default 'rsi_14'
        if sort_by == 'rsi' and 'rsi_14' in df.columns:
            sort_by = 'rsi_14'
        if sort_by and sort_by not in df.columns:
            candidates = [c for c in df.columns if sort_by.lower() in str(c).lower()]
            if len(candidates) == 1:
                sort_by = candidates[0]
        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=False, na_position='last')

        # Display columns order
        display_columns = [
            'ticker', 'pe_ratio', 'market_cap_b', 'current_price', 'prev_close', 'today_open',
            'diff_close', 'diff_close_pct', 'diff_open', 'diff_open_pct',
        ]
        # Include RSI columns if present
        rsi_cols = [c for c in df.columns if str(c).startswith('rsi_')]
        display_columns = ['ticker'] + rsi_cols + [c for c in display_columns if c != 'ticker']
        cols = [c for c in display_columns if c in df.columns]
        df_display = df[cols].copy()

        if output_format == 'csv':
            return self._format_csv_output(df_display, csv_delimiter, csv_quoting, output_file, csv_columns)

        # Pretty table formatting
        df_fmt = df_display.copy()
        for col in ['current_price', 'prev_close', 'today_open', 'diff_close', 'diff_open']:
            if col in df_fmt.columns:
                df_fmt[col] = df_fmt[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
        if 'pe_ratio' in df_fmt.columns:
            df_fmt['pe_ratio'] = df_fmt['pe_ratio'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        if 'market_cap_b' in df_fmt.columns:
            df_fmt['market_cap_b'] = df_fmt['market_cap_b'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        for col in ['diff_close_pct', 'diff_open_pct']:
            if col in df_fmt.columns:
                df_fmt[col] = df_fmt[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        for col in [c for c in df_fmt.columns if str(c).startswith('rsi_')]:
            df_fmt[col] = df_fmt[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

        table = tabulate(df_fmt, headers='keys', tablefmt='grid', showindex=False)
        if output_file and output_format != 'csv':
            with open(output_file, 'w') as f:
                f.write(table)
            if not self.quiet:
                print(f"Results saved to {output_file}")
        return table


async def main():
    parser = argparse.ArgumentParser(
        description="Analyze current stock performance vs previous close and today's open.",
        epilog="""
Examples:
  python scripts/stocks_analyzer.py --db-conn questdb://user:pass@host:8812/db
  python scripts/stocks_analyzer.py --symbols AAPL MSFT --sort diff_open_pct
  python scripts/stocks_analyzer.py --types sp-500 --output csv
  python scripts/stocks_analyzer.py --output results.csv --csv-columns "ticker,current_price,diff_close_pct,diff_open_pct"
""",
    )

    # Symbol inputs
    add_symbol_arguments(parser, required=False, allow_positional=False)

    # Database connection
    parser.add_argument(
        '--db-conn',
        type=str,
        required=True,
        help="QuestDB connection string (e.g. questdb://user:pass@host:8812/db).",
    )

    # Data directory for list files (expected by fetch_lists_data)
    parser.add_argument(
        '--data-dir',
        default='data',
        help="Directory to store/read data files (default: data).",
    )

    parser.add_argument(
        '--no-market-time',
        action='store_true',
        help="Disable market-hours logic (gets latest from any source regardless of market open/closed).",
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help="Disable Redis caching for QuestDB operations (default: cache enabled)"
    )

    parser.add_argument(
        '--output',
        type=str,
        default='table',
        help="Output format: 'table' 'csv' or filename (e.g. 'results.csv').",
    )
    parser.add_argument(
        '--sort',
        type=str,
        default='diff_close_pct',
        help="Sort by any displayed column (e.g., diff_close_pct diff_open_pct ticker current_price).",
    )
    # RSI options
    parser.add_argument(
        '--rsi-periods',
        type=str,
        default='14',
        help="Comma or space separated RSI periods to compute (default: 14)",
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Suppress progress output.",
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Enable debug output for RSI computations (prints to stderr).",
    )

    # CSV formatting options
    parser.add_argument('--csv-delimiter', type=str, default=',', help="CSV delimiter (default ',')")
    parser.add_argument(
        '--csv-quoting',
        choices=['minimal', 'all', 'none', 'nonnumeric'],
        default='minimal',
        help="CSV quoting style.",
    )
    parser.add_argument('--csv-columns', type=str, help="Comma-separated list of columns for CSV output.")

    # Early help exit
    if any(flag in sys.argv for flag in ("-h", "--help")):
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    # Determine cache settings
    enable_cache = not args.no_cache
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None

    analyzer = StocksAnalyzer(args.db_conn, args.quiet, enable_cache=enable_cache, redis_url=redis_url)
    await analyzer.initialize()

    # Resolve symbols
    symbols_list = await fetch_lists_data(args, args.quiet)
    if not symbols_list:
        print("No symbols specified or found. Exiting.", file=sys.stderr)
        sys.exit(1)
    if not args.quiet:
        print(f"Analyzing {len(symbols_list)} tickers...")

    # Parse RSI periods
    rsi_periods: List[int] = []
    if getattr(args, 'rsi_periods', None):
        raw = args.rsi_periods.replace(',', ' ').split()
        try:
            rsi_periods = [int(x) for x in raw if int(x) > 0]
        except Exception:
            rsi_periods = [14]

    df = await analyzer.analyze(
        symbols_list,
        use_market_time=not args.no_market_time,
        rsi_periods=rsi_periods,
        debug=bool(getattr(args, 'debug', False)),
    )
    if df.empty:
        print("No stock data available.")
        return

    output_format = 'table'
    output_file = None
    if args.output.lower() == 'csv':
        output_format = 'csv'
    elif args.output.lower() != 'table':
        output_file = args.output
        output_format = 'csv' if args.output.endswith('.csv') else 'table'

    csv_columns = None
    if getattr(args, 'csv_columns', None):
        csv_columns = [c.strip() for c in args.csv_columns.split(',')]

    # Normalize sort input by stripping whitespace
    import re as _re
    sort_arg = _re.sub(r"\s+", "", args.sort) if getattr(args, 'sort', None) else None

    result = analyzer.format_output(
        df=df,
        output_format=output_format,
        output_file=output_file,
        sort_by=sort_arg,
        csv_delimiter=args.csv_delimiter,
        csv_quoting=args.csv_quoting,
        csv_columns=csv_columns,
    )

    if not args.quiet or output_file is None:
        print(result)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


