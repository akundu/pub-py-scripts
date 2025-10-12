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


class StocksAnalyzer:
    """Analyzes deviations of current stock price vs previous close and today's open."""

    def __init__(self, db_conn: str, quiet: bool = False):
        self.db_conn = db_conn
        self.quiet = quiet
        self.db = None

    async def initialize(self):
        try:
            self.db = get_stock_db('questdb', db_config=self.db_conn)
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
            market_cap_b = round(market_cap / 1e9, 2) if isinstance(market_cap, (int, float)) and market_cap is not None else None

            rows.append({
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
            })

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
    add_symbol_arguments(parser, required=False)

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
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Suppress progress output.",
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

    analyzer = StocksAnalyzer(args.db_conn, args.quiet)
    await analyzer.initialize()

    # Resolve symbols
    symbols_list = await fetch_lists_data(args, args.quiet)
    if not symbols_list:
        print("No symbols specified or found. Exiting.", file=sys.stderr)
        sys.exit(1)
    if not args.quiet:
        print(f"Analyzing {len(symbols_list)} tickers...")

    df = await analyzer.analyze(symbols_list, use_market_time=not args.no_market_time)
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


