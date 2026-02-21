#!/usr/bin/env python3
"""
Check if QuestDB has the required data for closing price predictions.

This script verifies that the database contains sufficient historical data
for the prediction system to work properly.

Usage:
    python scripts/check_prediction_data.py
    python scripts/check_prediction_data.py --db-config questdb://admin:quest@localhost:8812/qdb
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, '.')

from common.questdb_db import StockQuestDB


# Required tickers for prediction
REQUIRED_TICKERS = ['I:NDX', 'I:SPX', 'I:VIX1D']

# Minimum data requirements
MIN_HISTORICAL_DAYS = 30  # Minimum for basic predictions
RECOMMENDED_HISTORICAL_DAYS = 365  # Recommended for good predictions
MIN_HOURLY_BARS_TODAY = 1  # At least 1 hourly bar for today


class DataCheck:
    """Result of a data check."""
    def __init__(self, name: str, passed: bool, message: str,
                 is_critical: bool = False, details: Dict = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.is_critical = is_critical
        self.details = details or {}


async def check_table_exists(db: StockQuestDB, table_name: str) -> DataCheck:
    """Check if a table exists in QuestDB."""
    try:
        async with db.connection.get_connection() as conn:
            # Try to query the table - QuestDB will error if it doesn't exist
            result = await conn.fetchval(f"""
                SELECT COUNT(*) FROM {table_name} LIMIT 1
            """)

            return DataCheck(
                name=f"Table {table_name}",
                passed=True,
                message=f"Table '{table_name}' exists ({result:,} rows)",
                is_critical=True
            )
    except Exception as e:
        error_msg = str(e).lower()
        if 'does not exist' in error_msg or 'not found' in error_msg or 'undefined' in error_msg:
            return DataCheck(
                name=f"Table {table_name}",
                passed=False,
                message=f"Table '{table_name}' does not exist",
                is_critical=True
            )
        else:
            return DataCheck(
                name=f"Table {table_name}",
                passed=False,
                message=f"Error checking table: {str(e)}",
                is_critical=True
            )


async def check_ticker_data(db: StockQuestDB, table: str, ticker: str,
                           min_days: int) -> DataCheck:
    """Check if a ticker has sufficient historical data."""
    try:
        async with db.connection.get_connection() as conn:
            # Get row count and date range
            if table == 'daily_prices':
                date_col = 'date'
            else:
                date_col = 'datetime'

            result = await conn.fetchrow(f"""
                SELECT
                    COUNT(*) as cnt,
                    MIN({date_col}) as earliest,
                    MAX({date_col}) as latest
                FROM {table}
                WHERE ticker = $1
            """, ticker)

            if result is None or result['cnt'] == 0:
                return DataCheck(
                    name=f"{ticker} in {table}",
                    passed=False,
                    message=f"No data found for {ticker} in {table}",
                    is_critical=True,
                    details={'count': 0}
                )

            count = result['cnt']
            earliest = result['earliest']
            latest = result['latest']

            # Calculate days of data
            if earliest and latest:
                days_of_data = (latest.date() - earliest.date()).days
            else:
                days_of_data = 0

            passed = days_of_data >= min_days

            if passed:
                return DataCheck(
                    name=f"{ticker} in {table}",
                    passed=True,
                    message=f"{count:,} rows, {days_of_data} days ({earliest.date()} to {latest.date()})",
                    details={
                        'count': count,
                        'days': days_of_data,
                        'earliest': str(earliest.date()),
                        'latest': str(latest.date())
                    }
                )
            else:
                return DataCheck(
                    name=f"{ticker} in {table}",
                    passed=False,
                    message=f"Only {days_of_data} days of data (need {min_days}+). Range: {earliest.date()} to {latest.date()}",
                    is_critical=True,
                    details={
                        'count': count,
                        'days': days_of_data,
                        'earliest': str(earliest.date()),
                        'latest': str(latest.date())
                    }
                )

    except Exception as e:
        return DataCheck(
            name=f"{ticker} in {table}",
            passed=False,
            message=f"Error checking data: {str(e)}",
            is_critical=True
        )


async def check_today_data(db: StockQuestDB, ticker: str) -> DataCheck:
    """Check if there's data for today (needed for live predictions)."""
    try:
        today = datetime.now().date()

        async with db.connection.get_connection() as conn:
            # Check daily_prices for today's open
            daily_result = await conn.fetchrow("""
                SELECT open, close
                FROM daily_prices
                WHERE ticker = $1 AND date::date = $2
            """, ticker, today)

            # Check hourly_prices for today's bars
            hourly_result = await conn.fetchval("""
                SELECT COUNT(*)
                FROM hourly_prices
                WHERE ticker = $1 AND datetime::date = $2
            """, ticker, today)

            has_daily = daily_result is not None
            hourly_count = hourly_result or 0

            if has_daily and hourly_count > 0:
                return DataCheck(
                    name=f"{ticker} today's data",
                    passed=True,
                    message=f"Today's data available: daily open={daily_result['open']:.2f}, {hourly_count} hourly bars",
                    details={
                        'has_daily': True,
                        'hourly_count': hourly_count,
                        'open': daily_result['open']
                    }
                )
            elif has_daily:
                return DataCheck(
                    name=f"{ticker} today's data",
                    passed=False,
                    message=f"Daily data exists but no hourly bars for today yet",
                    is_critical=False,  # Not critical if market hasn't opened
                    details={
                        'has_daily': True,
                        'hourly_count': 0
                    }
                )
            else:
                return DataCheck(
                    name=f"{ticker} today's data",
                    passed=False,
                    message=f"No data for today ({today})",
                    is_critical=False,  # Not critical for historical analysis
                    details={
                        'has_daily': False,
                        'hourly_count': 0
                    }
                )

    except Exception as e:
        return DataCheck(
            name=f"{ticker} today's data",
            passed=False,
            message=f"Error checking today's data: {str(e)}",
            is_critical=False
        )


async def run_all_checks(db_config: str) -> Tuple[List[DataCheck], bool]:
    """Run all data checks and return results."""
    checks = []
    all_critical_passed = True

    try:
        db = StockQuestDB(db_config=db_config)
        await db._init_db()

        # Check tables exist
        for table in ['daily_prices', 'hourly_prices']:
            check = await check_table_exists(db, table)
            checks.append(check)
            if not check.passed and check.is_critical:
                all_critical_passed = False

        # If tables don't exist, skip further checks
        if not all_critical_passed:
            await db.close()
            return checks, all_critical_passed

        # Check each ticker in both tables
        for ticker in REQUIRED_TICKERS:
            # Check daily_prices
            check = await check_ticker_data(db, 'daily_prices', ticker, MIN_HISTORICAL_DAYS)
            checks.append(check)
            if not check.passed and check.is_critical:
                all_critical_passed = False

            # Check hourly_prices
            check = await check_ticker_data(db, 'hourly_prices', ticker, MIN_HISTORICAL_DAYS)
            checks.append(check)
            if not check.passed and check.is_critical:
                all_critical_passed = False

        # Check today's data for prediction-relevant tickers
        for ticker in ['I:NDX', 'I:SPX']:
            check = await check_today_data(db, ticker)
            checks.append(check)
            # Today's data check is not critical

        await db.close()

    except Exception as e:
        checks.append(DataCheck(
            name="Database Connection",
            passed=False,
            message=f"Could not connect to database: {str(e)}",
            is_critical=True
        ))
        all_critical_passed = False

    return checks, all_critical_passed


def print_results(checks: List[DataCheck], all_passed: bool):
    """Print check results in a formatted way."""
    print("=" * 70)
    print(" PREDICTION DATA VERIFICATION REPORT")
    print("=" * 70)
    print()

    # Group by status
    passed_checks = [c for c in checks if c.passed]
    failed_critical = [c for c in checks if not c.passed and c.is_critical]
    failed_warnings = [c for c in checks if not c.passed and not c.is_critical]

    # Print passed checks
    if passed_checks:
        print("PASSED:")
        for check in passed_checks:
            print(f"  [OK] {check.name}: {check.message}")
        print()

    # Print warnings
    if failed_warnings:
        print("WARNINGS (non-critical):")
        for check in failed_warnings:
            print(f"  [WARN] {check.name}: {check.message}")
        print()

    # Print failures
    if failed_critical:
        print("FAILED (critical):")
        for check in failed_critical:
            print(f"  [FAIL] {check.name}: {check.message}")
        print()

    # Summary
    print("=" * 70)
    if all_passed:
        print(" STATUS: READY FOR PREDICTIONS")
        print(" All required data is available.")
    else:
        print(" STATUS: NOT READY")
        print(" Some required data is missing. Run --load-historical first.")
    print("=" * 70)

    # Recommendations
    if failed_critical:
        print()
        print("RECOMMENDED ACTIONS:")
        print("  1. Set your Polygon API key:")
        print("     export POLYGON_API_KEY=your_key_here")
        print()
        print("  2. Load historical data:")
        print("     python fetch_all_data.py \\")
        print("       --symbols I:NDX I:SPX I:VIX1D \\")
        print("       --fetch-market-data \\")
        print("       --days-back 365 \\")
        print("       --db-path questdb://admin:quest@localhost:8812/qdb")
        print()
        print("  Or use the convenience script:")
        print("     ./run_scripts/prediction_setup.sh --load-historical")


def main():
    parser = argparse.ArgumentParser(
        description='Check if QuestDB has required data for predictions'
    )
    parser.add_argument(
        '--db-config',
        type=str,
        default='questdb://admin:quest@localhost:8812/qdb',
        help='QuestDB connection string'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )

    args = parser.parse_args()

    # Run checks
    checks, all_passed = asyncio.run(run_all_checks(args.db_config))

    if args.json:
        import json
        result = {
            'ready': all_passed,
            'checks': [
                {
                    'name': c.name,
                    'passed': c.passed,
                    'message': c.message,
                    'critical': c.is_critical,
                    'details': c.details
                }
                for c in checks
            ]
        }
        print(json.dumps(result, indent=2))
    else:
        print_results(checks, all_passed)

    # Exit code
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
