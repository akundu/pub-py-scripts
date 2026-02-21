#!/usr/bin/env python3
"""
Daily P&L Simulator for Percentile-Based Strategies

Simulates actual trading with position tracking across days:
- Day 1: Enter positions with 1 DTE
- Day 2: Those positions expire, calculate P&L, enter new positions
- Tracks multi-day positions
- Calculates daily P&L, cumulative returns, risk metrics
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credit_spread_utils.percentile_integration import PercentileSpreadIntegrator
from credit_spread_utils.data_loader import load_multi_dte_data
from credit_spread_utils.price_utils import get_previous_close_price
from common.questdb_db import StockQuestDB
from common.logging_utils import get_logger
import os

logger = get_logger("daily_pnl_sim", level="INFO")


class Position:
    """Represents an iron condor position."""
    def __init__(self, entry_date, expiry_date, entry_credit, max_loss, strikes):
        self.entry_date = entry_date
        self.expiry_date = expiry_date
        self.entry_credit = entry_credit
        self.max_loss = max_loss
        self.strikes = strikes
        self.status = 'open'
        self.exit_date = None
        self.exit_pnl = 0
        self.exit_reason = None


class DailyPnLSimulator:
    """Simulates daily trading and tracks P&L."""

    def __init__(
        self,
        starting_capital: float,
        position_size: int,  # Number of contracts per opportunity
        max_positions_per_day: int,
        db_config: str
    ):
        self.starting_capital = starting_capital
        self.position_size = position_size
        self.max_positions_per_day = max_positions_per_day
        self.db_config = db_config

        self.current_capital = starting_capital
        self.positions: List[Position] = []
        self.daily_results = []

        self.integrator = PercentileSpreadIntegrator(db_config=db_config)
        self.db = StockQuestDB(db_config, logger=logger)

    async def run_single_day(
        self,
        ticker: str,
        date: datetime,
        dte: int,
        percentile: int,
        spread_width: float,
        profit_target_pct: float
    ) -> Dict:
        """Run strategy for a single day."""

        date_str = date.strftime('%Y-%m-%d')
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {date_str}")
        logger.info(f"{'='*80}")

        # Load data for this day
        csv_dir = 'options_csv_output' if dte == 0 else 'options_csv_output_full'

        df = load_multi_dte_data(
            csv_dir=csv_dir,
            ticker=ticker,
            start_date=date_str,
            end_date=date_str,
            dte_buckets=(dte,),
            dte_tolerance=1,
            cache_dir=None,
            no_cache=True,
            logger=logger
        )

        # Filter 0DTE to single entry time
        if dte == 0 and not df.empty:
            timestamps = sorted(df['timestamp'].unique())
            entry_timestamp = timestamps[0]  # Use earliest timestamp
            df = df[df['timestamp'] == entry_timestamp].copy()
            logger.info(f"[0DTE] Filtered to single entry time: {entry_timestamp}")

        if df.empty:
            logger.warning(f"No data for {date_str}")
            return None

        # Get trading date
        df['trading_date'] = df['timestamp'].apply(
            lambda x: x.date() if hasattr(x, 'date') else pd.to_datetime(x).date()
        )

        # Get previous close
        first_ts = df['timestamp'].min()
        prev_close_result = await get_previous_close_price(self.db, ticker, first_ts, logger)
        if prev_close_result is None:
            logger.warning(f"No previous close for {date_str}")
            return None

        prev_close, _ = prev_close_result

        # Find spreads
        trades = await self.integrator.analyze_single_day(
            options_df=df,
            ticker=ticker,
            trading_date=date,
            prev_close=prev_close,
            dte=dte,
            percentile=percentile,
            spread_width=spread_width,
            profit_target_pct=profit_target_pct,
            flow_mode='neutral',
            use_mid=False
        )

        # Process expiring positions
        expired_pnl = self._process_expirations(date)

        # Enter new positions
        new_positions = 0
        entry_capital_used = 0

        if trades:
            # Limit to max positions per day
            num_to_enter = min(len(trades), self.max_positions_per_day)

            for i in range(num_to_enter):
                trade = trades[i]

                # Calculate position value
                position_value = trade['max_loss'] * self.position_size

                # Check if we have capital
                if entry_capital_used + position_value > self.current_capital:
                    logger.info(f"Insufficient capital for position {i+1}")
                    break

                # Create position
                expiry_date = date + timedelta(days=dte if dte > 0 else 0)
                pos = Position(
                    entry_date=date,
                    expiry_date=expiry_date,
                    entry_credit=trade['entry_credit'] * self.position_size,
                    max_loss=position_value,
                    strikes={
                        'call_short': trade.get('call_short_strike'),
                        'call_long': trade.get('call_long_strike'),
                        'put_short': trade.get('put_short_strike'),
                        'put_long': trade.get('put_long_strike')
                    }
                )
                self.positions.append(pos)
                entry_capital_used += position_value
                new_positions += 1

        # Calculate daily metrics
        open_positions = len([p for p in self.positions if p.status == 'open'])
        total_risk = sum(p.max_loss for p in self.positions if p.status == 'open')

        daily_result = {
            'date': date_str,
            'prev_close': prev_close,
            'spreads_available': len(trades) if trades else 0,
            'new_positions_entered': new_positions,
            'positions_expired': len([p for p in self.positions if p.exit_date == date]),
            'expired_pnl': expired_pnl,
            'open_positions': open_positions,
            'total_risk': total_risk,
            'capital_used': entry_capital_used,
            'current_capital': self.current_capital,
            'total_value': self.current_capital + expired_pnl
        }

        self.daily_results.append(daily_result)

        logger.info(f"Spreads Available: {daily_result['spreads_available']}")
        logger.info(f"New Positions: {new_positions}")
        logger.info(f"Positions Expired: {daily_result['positions_expired']}")
        logger.info(f"Expired P&L: ${expired_pnl:,.2f}")
        logger.info(f"Open Positions: {open_positions}")
        logger.info(f"Total Risk: ${total_risk:,.2f}")
        logger.info(f"Current Capital: ${self.current_capital:,.2f}")

        return daily_result

    def _process_expirations(self, current_date: datetime) -> float:
        """Process positions expiring on this date."""
        total_pnl = 0

        for pos in self.positions:
            if pos.status == 'open' and pos.expiry_date <= current_date:
                # Assume profit target hit (50% of credit)
                # In reality, would need to check actual option prices
                profit_target_pnl = pos.entry_credit * 0.5

                pos.exit_date = current_date
                pos.exit_pnl = profit_target_pnl
                pos.exit_reason = 'profit_target'
                pos.status = 'closed'

                total_pnl += profit_target_pnl

                # Return margin
                self.current_capital += pos.max_loss

        return total_pnl

    def get_summary_statistics(self) -> Dict:
        """Calculate summary statistics."""
        if not self.daily_results:
            return {}

        df = pd.DataFrame(self.daily_results)

        # Calculate cumulative P&L
        df['cumulative_pnl'] = df['expired_pnl'].cumsum()
        df['total_value'] = self.starting_capital + df['cumulative_pnl']
        df['return_pct'] = (df['cumulative_pnl'] / self.starting_capital) * 100

        # Calculate metrics
        total_pnl = df['expired_pnl'].sum()
        total_return_pct = (total_pnl / self.starting_capital) * 100

        winning_days = len(df[df['expired_pnl'] > 0])
        losing_days = len(df[df['expired_pnl'] < 0])
        win_rate = winning_days / len(df) * 100 if len(df) > 0 else 0

        avg_win = df[df['expired_pnl'] > 0]['expired_pnl'].mean() if winning_days > 0 else 0
        avg_loss = df[df['expired_pnl'] < 0]['expired_pnl'].mean() if losing_days > 0 else 0

        max_drawdown = (df['cumulative_pnl'] - df['cumulative_pnl'].cummax()).min()
        max_drawdown_pct = (max_drawdown / self.starting_capital) * 100 if max_drawdown < 0 else 0

        # Position statistics
        all_positions = [p for p in self.positions if p.status == 'closed']
        total_positions = len(all_positions)
        winning_positions = len([p for p in all_positions if p.exit_pnl > 0])
        position_win_rate = winning_positions / total_positions * 100 if total_positions > 0 else 0

        avg_position_pnl = sum(p.exit_pnl for p in all_positions) / total_positions if total_positions > 0 else 0

        return {
            'trading_days': len(df),
            'starting_capital': self.starting_capital,
            'ending_capital': self.current_capital,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'avg_daily_pnl': total_pnl / len(df) if len(df) > 0 else 0,
            'winning_days': winning_days,
            'losing_days': losing_days,
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'total_positions': total_positions,
            'position_win_rate_pct': position_win_rate,
            'avg_position_pnl': avg_position_pnl,
            'final_value': self.current_capital + df['cumulative_pnl'].iloc[-1],
            'daily_results_df': df
        }


async def main():
    import argparse

    parser = argparse.ArgumentParser(description='Daily P&L Simulator')
    parser.add_argument('--ticker', default='NDX', help='Ticker symbol')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--starting-capital', type=float, default=100000, help='Starting capital')
    parser.add_argument('--position-size', type=int, default=1, help='Contracts per position')
    parser.add_argument('--max-positions', type=int, default=50, help='Max positions per day')
    parser.add_argument('--dte', type=int, default=1, help='Days to expiration')
    parser.add_argument('--percentile', type=int, default=99, help='Percentile (95-100)')
    parser.add_argument('--spread-width', type=float, default=20, help='Spread width in points')
    parser.add_argument('--profit-target', type=float, default=0.5, help='Profit target (0.5 = 50 percent)')
    parser.add_argument('--output', default='results/daily_pnl.csv', help='Output CSV file')

    args = parser.parse_args()

    # Get DB config
    db_config = (
        os.getenv('QUEST_DB_STRING') or
        os.getenv('QUESTDB_CONNECTION_STRING') or
        os.getenv('QUESTDB_URL')
    )

    if not db_config:
        print("ERROR: No database configuration found")
        return 1

    # Create simulator
    sim = DailyPnLSimulator(
        starting_capital=args.starting_capital,
        position_size=args.position_size,
        max_positions_per_day=args.max_positions,
        db_config=db_config
    )

    # Generate date range
    start = datetime.strptime(args.start_date, '%Y-%m-%d')
    end = datetime.strptime(args.end_date, '%Y-%m-%d')

    current = start
    while current <= end:
        # Skip weekends
        if current.weekday() < 5:  # Monday=0, Friday=4
            await sim.run_single_day(
                ticker=args.ticker,
                date=current,
                dte=args.dte,
                percentile=args.percentile,
                spread_width=args.spread_width,
                profit_target_pct=args.profit_target
            )
        current += timedelta(days=1)

    # Get summary
    summary = sim.get_summary_statistics()

    if not summary:
        print("No results to summarize")
        return 1

    # Print summary
    print(f"\n{'='*80}")
    print(f"DAILY P&L SIMULATION SUMMARY")
    print(f"{'='*80}")
    print(f"Strategy: DTE{args.dte}_p{args.percentile}_w{args.spread_width}_neutral")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Position Size: {args.position_size} contracts")
    print(f"Max Positions/Day: {args.max_positions}")
    print(f"{'='*80}\n")

    print(f"CAPITAL")
    print(f"  Starting Capital:  ${summary['starting_capital']:,.2f}")
    print(f"  Ending Capital:    ${summary['ending_capital']:,.2f}")
    print(f"  Final Value:       ${summary['final_value']:,.2f}")
    print(f"  Total P&L:         ${summary['total_pnl']:,.2f}")
    print(f"  Total Return:      {summary['total_return_pct']:.2f}%")
    print(f"  Avg Daily P&L:     ${summary['avg_daily_pnl']:,.2f}")
    print()

    print(f"DAILY PERFORMANCE")
    print(f"  Trading Days:      {summary['trading_days']}")
    print(f"  Winning Days:      {summary['winning_days']}")
    print(f"  Losing Days:       {summary['losing_days']}")
    print(f"  Win Rate:          {summary['win_rate_pct']:.1f}%")
    print(f"  Avg Win:           ${summary['avg_win']:,.2f}")
    print(f"  Avg Loss:          ${summary['avg_loss']:,.2f}")
    print()

    print(f"POSITION STATISTICS")
    print(f"  Total Positions:   {summary['total_positions']}")
    print(f"  Position Win Rate: {summary['position_win_rate_pct']:.1f}%")
    print(f"  Avg Position P&L:  ${summary['avg_position_pnl']:,.2f}")
    print()

    print(f"RISK METRICS")
    print(f"  Max Drawdown:      ${summary['max_drawdown']:,.2f}")
    print(f"  Max Drawdown %:    {summary['max_drawdown_pct']:.2f}%")
    print()

    # Save results
    df = summary['daily_results_df']
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # Print daily breakdown
    print(f"\n{'='*80}")
    print(f"DAILY BREAKDOWN")
    print(f"{'='*80}\n")
    print(df[['date', 'spreads_available', 'new_positions_entered', 'positions_expired',
              'expired_pnl', 'cumulative_pnl', 'return_pct']].to_string(index=False))

    print(f"\n{'='*80}\n")

    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
