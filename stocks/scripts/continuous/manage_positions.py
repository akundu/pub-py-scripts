#!/usr/bin/env python3
"""
Position Management CLI

Manually add, update, and close positions.

Usage:
    # Add a position
    python scripts/continuous/manage_positions.py add \
        --dte 3 --band P98 --spread iron_condor \
        --credit 285 --risk 1715 --contracts 2

    # Update P&L
    python scripts/continuous/manage_positions.py update <position_id> --pnl 142.50

    # Close position
    python scripts/continuous/manage_positions.py close <position_id> --pnl 142.50 --note "Profit target hit"

    # List positions
    python scripts/continuous/manage_positions.py list

    # Show summary
    python scripts/continuous/manage_positions.py summary
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.continuous.position_tracker import PositionTracker


def add_position(args):
    """Add a new position."""
    tracker = PositionTracker()

    # Calculate expiration date
    exp_date = (datetime.now() + timedelta(days=args.dte)).strftime('%Y-%m-%d')

    # Parse strikes
    strikes = {}
    if args.spread in ['iron_condor']:
        if not all([args.short_call, args.long_call, args.short_put, args.long_put]):
            print("Error: Iron condor requires all 4 strikes")
            return
        strikes = {
            'short_call': args.short_call,
            'long_call': args.long_call,
            'short_put': args.short_put,
            'long_put': args.long_put,
        }
    elif args.spread == 'call_spread':
        if not all([args.short_call, args.long_call]):
            print("Error: Call spread requires short_call and long_call")
            return
        strikes = {
            'short_call': args.short_call,
            'long_call': args.long_call,
        }
    elif args.spread == 'put_spread':
        if not all([args.short_put, args.long_put]):
            print("Error: Put spread requires short_put and long_put")
            return
        strikes = {
            'short_put': args.short_put,
            'long_put': args.long_put,
        }

    position = tracker.add_position(
        ticker=args.ticker,
        dte=args.dte,
        expiration_date=exp_date,
        band=args.band,
        spread_type=args.spread,
        flow_mode=args.flow_mode,
        strikes=strikes,
        n_contracts=args.contracts,
        credit_received=args.credit,
        max_risk=args.risk,
        notes=args.note or '',
    )

    print(f"\n✓ Position added: {position.position_id}")
    print(f"  DTE: {args.dte}")
    print(f"  Band: {args.band}")
    print(f"  Spread: {args.spread}")
    print(f"  Credit: ${args.credit:.2f}")
    print(f"  Max Risk: ${args.risk:.2f}")
    print(f"  Profit Target: ${position.profit_target:.2f}")
    print(f"  Stop Loss: ${position.stop_loss:.2f}")


def update_position(args):
    """Update position P&L."""
    tracker = PositionTracker()

    if tracker.update_pnl(args.position_id, args.pnl):
        print(f"\n✓ Position {args.position_id} updated")
        print(f"  Current P&L: ${args.pnl:.2f}")

        # Check exit conditions
        for pos in tracker.get_open_positions():
            if pos.position_id == args.position_id:
                reasons = pos.check_exit_conditions()
                if reasons:
                    print(f"\n⚠️  EXIT SIGNALS:")
                    for reason in reasons:
                        print(f"    - {reason}")
                break
    else:
        print(f"\n✗ Position {args.position_id} not found")


def close_position(args):
    """Close a position."""
    tracker = PositionTracker()

    if tracker.close_position(args.position_id, args.pnl, args.note or ''):
        print(f"\n✓ Position {args.position_id} closed")
        print(f"  Final P&L: ${args.pnl:.2f}")
    else:
        print(f"\n✗ Position {args.position_id} not found")


def list_positions(args):
    """List all positions."""
    tracker = PositionTracker()

    open_positions = tracker.get_open_positions()
    closed_positions = [p for p in tracker.positions if p.status == 'closed']

    print("\n" + "=" * 100)
    print("OPEN POSITIONS")
    print("=" * 100)

    if open_positions:
        for pos in open_positions:
            pnl_pct = (pos.current_pnl / pos.credit_received * 100) if pos.credit_received > 0 else 0
            print(f"\n{pos.position_id} | {pos.dte}DTE {pos.band} {pos.spread_type.upper()} ({pos.flow_mode})")
            print(f"  Entered: {pos.entry_time}")
            print(f"  Expires: {pos.expiration_date}")
            print(f"  Credit: ${pos.credit_received:.2f} | Max Risk: ${pos.max_risk:.2f}")
            print(f"  P&L: ${pos.current_pnl:.2f} ({pnl_pct:+.1f}%)")
            print(f"  Profit Target: ${pos.profit_target:.2f} | Stop Loss: ${pos.stop_loss:.2f}")

            if pos.notes:
                print(f"  Notes: {pos.notes}")

            # Check exit conditions
            reasons = pos.check_exit_conditions()
            if reasons:
                print(f"  ⚠️  EXIT SIGNALS: {', '.join(reasons)}")
    else:
        print("No open positions")

    if closed_positions and args.include_closed:
        print("\n" + "=" * 100)
        print("CLOSED POSITIONS")
        print("=" * 100)

        for pos in closed_positions[-10:]:  # Last 10
            pnl_pct = (pos.current_pnl / pos.credit_received * 100) if pos.credit_received > 0 else 0
            print(f"\n{pos.position_id} | {pos.dte}DTE {pos.band} {pos.spread_type.upper()}")
            print(f"  Entered: {pos.entry_time}")
            print(f"  P&L: ${pos.current_pnl:.2f} ({pnl_pct:+.1f}%)")


def show_summary(args):
    """Show portfolio summary."""
    tracker = PositionTracker()
    summary = tracker.get_summary()

    print("\n" + "=" * 60)
    print("PORTFOLIO SUMMARY")
    print("=" * 60)

    print(f"\nPositions:")
    print(f"  Total: {summary['total_positions']}")
    print(f"  Open: {summary['open_positions']}")
    print(f"  Closed: {summary['closed_positions']}")

    print(f"\nCapital:")
    print(f"  Total Risk: ${summary['total_risk']:,.2f}")

    print(f"\nP&L:")
    print(f"  Unrealized: ${summary['unrealized_pnl']:,.2f}")
    print(f"  Realized: ${summary['realized_pnl']:,.2f}")
    print(f"  Total: ${summary['total_pnl']:,.2f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Manage trading positions')
    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Add position
    add_parser = subparsers.add_parser('add', help='Add a new position')
    add_parser.add_argument('--ticker', type=str, default='NDX', help='Ticker')
    add_parser.add_argument('--dte', type=int, required=True, help='Days to expiration')
    add_parser.add_argument('--band', type=str, required=True, help='Band (P95, P97, P98, etc.)')
    add_parser.add_argument('--spread', type=str, required=True,
                            choices=['put_spread', 'call_spread', 'iron_condor'],
                            help='Spread type')
    add_parser.add_argument('--flow-mode', type=str, default='with_flow',
                            choices=['with_flow', 'against_flow', 'neutral'],
                            help='Flow mode')
    add_parser.add_argument('--contracts', type=int, default=1, help='Number of contracts')
    add_parser.add_argument('--credit', type=float, required=True, help='Credit received')
    add_parser.add_argument('--risk', type=float, required=True, help='Max risk')
    add_parser.add_argument('--short-call', type=float, help='Short call strike')
    add_parser.add_argument('--long-call', type=float, help='Long call strike')
    add_parser.add_argument('--short-put', type=float, help='Short put strike')
    add_parser.add_argument('--long-put', type=float, help='Long put strike')
    add_parser.add_argument('--note', type=str, help='Notes')

    # Update position
    update_parser = subparsers.add_parser('update', help='Update position P&L')
    update_parser.add_argument('position_id', type=str, help='Position ID')
    update_parser.add_argument('--pnl', type=float, required=True, help='Current P&L')

    # Close position
    close_parser = subparsers.add_parser('close', help='Close a position')
    close_parser.add_argument('position_id', type=str, help='Position ID')
    close_parser.add_argument('--pnl', type=float, required=True, help='Final P&L')
    close_parser.add_argument('--note', type=str, help='Closing note')

    # List positions
    list_parser = subparsers.add_parser('list', help='List all positions')
    list_parser.add_argument('--include-closed', action='store_true',
                             help='Include closed positions')

    # Summary
    subparsers.add_parser('summary', help='Show portfolio summary')

    args = parser.parse_args()

    if args.command == 'add':
        add_position(args)
    elif args.command == 'update':
        update_position(args)
    elif args.command == 'close':
        close_position(args)
    elif args.command == 'list':
        list_positions(args)
    elif args.command == 'summary':
        show_summary(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
