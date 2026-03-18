#!/usr/bin/env python3
"""
UTP Demo Walkthrough — Simulates a realistic trading session with fake data.

Shows what UTP looks like with real positions, P&L, and trade history.
Run this without any broker connection:

    python demo_walkthrough.py

This creates positions in a temporary data directory (cleaned up on exit).
"""

import asyncio
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

from app.models import (
    Broker,
    EquityOrder,
    MultiLegOrder,
    OptionAction,
    OptionLeg,
    OptionType,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    TradeRequest,
)


def _color(text, code):
    return f"\033[{code}m{text}\033[0m"


def _header(title):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}\n")


def _section(title):
    print(f"\n{'─' * 72}")
    print(f"  {title}")
    print(f"{'─' * 72}")


async def run_demo():
    # Create temp data directory
    tmp_dir = Path(tempfile.mkdtemp(prefix="utp_demo_"))
    print(f"  Demo data directory: {tmp_dir}\n")

    try:
        # Initialize services
        from app.core.provider import ProviderRegistry
        from app.core.providers.etrade import EtradeProvider
        from app.core.providers.ibkr import IBKRProvider
        from app.core.providers.robinhood import RobinhoodProvider
        from app.services.ledger import init_ledger, reset_ledger
        from app.services.metrics import compute_metrics
        from app.services.position_store import (
            init_position_store,
            reset_position_store,
        )

        init_ledger(tmp_dir)
        store = init_position_store(tmp_dir)

        for p in [RobinhoodProvider(), EtradeProvider(), IBKRProvider()]:
            ProviderRegistry.register(p)
            await p.connect()

        # ================================================================
        _header("UTP Demo Walkthrough -- Simulated Trading Session")
        # ================================================================

        print("  This demo simulates a week of trading to show UTP's full")
        print("  portfolio management, P&L tracking, and reporting capabilities.")
        print("  No broker connection needed -- using stub providers.\n")

        # ────────────────────────────────────────────────────────────────
        _section("Step 1: Opening Positions")
        # ────────────────────────────────────────────────────────────────

        # Trade 1: SPX Put Credit Spread (winner)
        req1 = TradeRequest(multi_leg_order=MultiLegOrder(
            broker=Broker.IBKR,
            legs=[
                OptionLeg(symbol="SPX", expiration="2026-03-20", strike=5450.0,
                          option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN, quantity=1),
                OptionLeg(symbol="SPX", expiration="2026-03-20", strike=5425.0,
                          option_type=OptionType.PUT, action=OptionAction.BUY_TO_OPEN, quantity=1),
            ],
            net_price=3.80, quantity=2,
        ))
        res1 = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=3.80)
        pid1 = store.add_position(req1, res1)
        print(f"  [1] SPX 5450/5425 Put Credit Spread  x2  @$3.80 credit")
        print(f"      Position ID: {pid1[:12]}...")

        # Trade 2: NDX Put Credit Spread (winner)
        req2 = TradeRequest(multi_leg_order=MultiLegOrder(
            broker=Broker.IBKR,
            legs=[
                OptionLeg(symbol="NDX", expiration="2026-03-20", strike=20000.0,
                          option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN, quantity=1),
                OptionLeg(symbol="NDX", expiration="2026-03-20", strike=19900.0,
                          option_type=OptionType.PUT, action=OptionAction.BUY_TO_OPEN, quantity=1),
            ],
            net_price=5.20, quantity=3,
        ))
        res2 = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=5.20)
        pid2 = store.add_position(req2, res2)
        print(f"  [2] NDX 20000/19900 Put Credit Spread  x3  @$5.20 credit")
        print(f"      Position ID: {pid2[:12]}...")

        # Trade 3: SPX Call Credit Spread (loser)
        req3 = TradeRequest(multi_leg_order=MultiLegOrder(
            broker=Broker.IBKR,
            legs=[
                OptionLeg(symbol="SPX", expiration="2026-03-18", strike=5700.0,
                          option_type=OptionType.CALL, action=OptionAction.SELL_TO_OPEN, quantity=1),
                OptionLeg(symbol="SPX", expiration="2026-03-18", strike=5725.0,
                          option_type=OptionType.CALL, action=OptionAction.BUY_TO_OPEN, quantity=1),
            ],
            net_price=2.10, quantity=1,
        ))
        res3 = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=2.10)
        pid3 = store.add_position(req3, res3)
        print(f"  [3] SPX 5700/5725 Call Credit Spread  x1  @$2.10 credit")
        print(f"      Position ID: {pid3[:12]}...")

        # Trade 4: SPX Iron Condor (winner)
        req4 = TradeRequest(multi_leg_order=MultiLegOrder(
            broker=Broker.IBKR,
            legs=[
                OptionLeg(symbol="SPX", expiration="2026-03-21", strike=5400.0,
                          option_type=OptionType.PUT, action=OptionAction.BUY_TO_OPEN, quantity=1),
                OptionLeg(symbol="SPX", expiration="2026-03-21", strike=5425.0,
                          option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN, quantity=1),
                OptionLeg(symbol="SPX", expiration="2026-03-21", strike=5750.0,
                          option_type=OptionType.CALL, action=OptionAction.SELL_TO_OPEN, quantity=1),
                OptionLeg(symbol="SPX", expiration="2026-03-21", strike=5775.0,
                          option_type=OptionType.CALL, action=OptionAction.BUY_TO_OPEN, quantity=1),
            ],
            net_price=4.50, quantity=2,
        ))
        res4 = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=4.50)
        pid4 = store.add_position(req4, res4)
        print(f"  [4] SPX 5425/5400p -- 5750/5775c Iron Condor  x2  @$4.50 credit")
        print(f"      Position ID: {pid4[:12]}...")

        # Trade 5: Equity buy (SPY)
        req5 = TradeRequest(equity_order=EquityOrder(
            broker=Broker.IBKR, symbol="SPY", side=OrderSide.BUY,
            quantity=50, order_type=OrderType.LIMIT, limit_price=565.20,
        ))
        res5 = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=565.20)
        pid5 = store.add_position(req5, res5)
        print(f"  [5] BUY 50 SPY @$565.20")
        print(f"      Position ID: {pid5[:12]}...")

        # Trade 6: RUT Put Credit Spread (winner, closed early)
        req6 = TradeRequest(multi_leg_order=MultiLegOrder(
            broker=Broker.IBKR,
            legs=[
                OptionLeg(symbol="RUT", expiration="2026-03-19", strike=2200.0,
                          option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN, quantity=1),
                OptionLeg(symbol="RUT", expiration="2026-03-19", strike=2195.0,
                          option_type=OptionType.PUT, action=OptionAction.BUY_TO_OPEN, quantity=1),
            ],
            net_price=1.75, quantity=5,
        ))
        res6 = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=1.75)
        pid6 = store.add_position(req6, res6)
        print(f"  [6] RUT 2200/2195 Put Credit Spread  x5  @$1.75 credit")
        print(f"      Position ID: {pid6[:12]}...")

        print(f"\n  {_color('6 positions opened', '92')}")

        # ────────────────────────────────────────────────────────────────
        _section("Step 2: Closing Some Positions (Simulating P&L)")
        # ────────────────────────────────────────────────────────────────

        # Close Trade 1: SPX put spread -- full profit (expired worthless)
        store.close_position(pid1, 0.0, "expired")
        pnl1 = 3.80 * 2 * 100
        print(f"  [1] SPX 5450/5425 Put Spread: {_color(f'EXPIRED WORTHLESS  P&L: +${pnl1:.0f}', '92')}")

        # Close Trade 3: SPX call spread -- loss (breached)
        store.close_position(pid3, 18.50, "stop_loss")
        pnl3 = (2.10 - 18.50) * 1 * 100
        print(f"  [3] SPX 5700/5725 Call Spread: {_color(f'STOP LOSS  P&L: ${pnl3:.0f}', '91')}")

        # Close Trade 6: RUT put spread -- partial profit (closed early)
        store.close_position(pid6, 0.35, "profit_target")
        pnl6 = (1.75 - 0.35) * 5 * 100
        print(f"  [6] RUT 2200/2195 Put Spread: {_color(f'PROFIT TARGET  P&L: +${pnl6:.0f}', '92')}")

        # Update marks on open positions
        store.update_mark(pid2, 2.10)    # NDX spread mark dropped from 5.20 to 2.10 (profit)
        store.update_mark(pid4, 1.80)    # Iron condor mark dropped from 4.50 to 1.80 (profit)
        store.update_mark(pid5, 571.50)  # SPY up from 565.20

        print(f"\n  3 positions closed, 3 still open with updated marks")

        # ────────────────────────────────────────────────────────────────
        _section("Step 3: Portfolio View")
        # ────────────────────────────────────────────────────────────────

        open_positions = store.get_open_positions()
        closed_positions = store.get_closed_positions()

        print(f"\n  {_color('OPEN POSITIONS', '96')}")
        print(f"  {'Symbol':<8} {'Type':<12} {'Qty':>5} {'Entry':>10} {'Mark':>10} {'Unreal P&L':>12}")
        print(f"  {'─' * 8} {'─' * 12} {'─' * 5} {'─' * 10} {'─' * 10} {'─' * 12}")

        total_unrealized = 0.0
        for p in open_positions:
            sym = p.get("symbol", "?")
            otype = p.get("order_type", "?")
            qty = p.get("quantity", 0)
            qty_str = str(int(qty)) if qty == int(qty) else str(qty)
            entry = p.get("entry_price", 0)
            mark = p.get("current_mark", 0) or 0
            upnl = p.get("unrealized_pnl", 0) or 0
            total_unrealized += upnl
            color = "92" if upnl >= 0 else "91"
            print(f"  {sym:<8} {otype:<12} {qty_str:>5} {f'${entry:.2f}':>10} {f'${mark:.2f}':>10} "
                  f"{_color(f'${upnl:>+,.2f}', color)}")

        u_color = "92" if total_unrealized >= 0 else "91"
        print(f"  {'':>47} {_color(f'${total_unrealized:>+,.2f}', u_color)}")

        print(f"\n  {_color('CLOSED POSITIONS', '96')}")
        print(f"  {'Symbol':<8} {'Type':<12} {'Qty':>5} {'Entry':>10} {'Exit':>10} {'P&L':>12} {'Reason'}")
        print(f"  {'─' * 8} {'─' * 12} {'─' * 5} {'─' * 10} {'─' * 10} {'─' * 12} {'─' * 14}")

        total_realized = 0.0
        for p in closed_positions:
            sym = p.get("symbol", "?")
            otype = p.get("order_type", "?")
            qty = p.get("quantity", 0)
            qty_str = str(int(qty)) if qty == int(qty) else str(qty)
            entry = p.get("entry_price", 0)
            exit_p = p.get("exit_price", 0)
            pnl = p.get("pnl", 0)
            total_realized += pnl
            reason = p.get("exit_reason", "?")
            color = "92" if pnl >= 0 else "91"
            print(f"  {sym:<8} {otype:<12} {qty_str:>5} {f'${entry:.2f}':>10} {f'${exit_p:.2f}':>10} "
                  f"{_color(f'${pnl:>+,.2f}', color)} {reason}")

        r_color = "92" if total_realized >= 0 else "91"
        print(f"  {'':>59} {_color(f'${total_realized:>+,.2f}', r_color)}")

        # ────────────────────────────────────────────────────────────────
        _section("Step 4: Account Summary")
        # ────────────────────────────────────────────────────────────────

        summary = store.get_account_summary()
        realized = summary["realized_pnl"]
        unrealized = summary["unrealized_pnl"]
        total = summary["total_pnl"]

        r_color = "92" if realized >= 0 else "91"
        u_color = "92" if unrealized >= 0 else "91"
        t_color = "92" if total >= 0 else "91"

        print(f"\n  Open positions:    {summary['open_count']}")
        print(f"  Closed positions:  {summary['closed_count']}")
        print(f"  Cash deployed:     ${summary['cash_deployed']:>10,.2f}")
        print(f"  Realized P&L:      {_color(f'${realized:>+10,.2f}', r_color)}")
        print(f"  Unrealized P&L:    {_color(f'${unrealized:>+10,.2f}', u_color)}")
        print(f"  Total P&L:         {_color(f'${total:>+10,.2f}', t_color)}")

        # ────────────────────────────────────────────────────────────────
        _section("Step 5: Performance Metrics (Closed Trades)")
        # ────────────────────────────────────────────────────────────────

        results = store.export_results()
        metrics = compute_metrics(results)

        print(f"\n  Total trades:    {metrics['total_trades']}")
        print(f"  Wins:            {metrics['wins']}")
        print(f"  Losses:          {metrics['losses']}")
        win_rate = metrics['win_rate']
        print(f"  Win rate:        {_color(f'{win_rate:.1%}', '92' if win_rate >= 0.5 else '91')}")
        net = metrics['net_pnl']
        print(f"  Net P&L:         {_color(f'${net:>+,.2f}', '92' if net >= 0 else '91')}")
        print(f"  Avg P&L/trade:   ${metrics['avg_pnl']:>+,.2f}")
        pf = metrics['profit_factor']
        print(f"  Profit factor:   {pf:.2f}")
        print(f"  Sharpe ratio:    {metrics['sharpe']:.2f}")
        print(f"  Max drawdown:    ${metrics['max_drawdown']:>,.2f}")

        # ────────────────────────────────────────────────────────────────
        _section("Step 6: What Commands to Run")
        # ────────────────────────────────────────────────────────────────

        print("""
  To use UTP for real trading, here is the workflow:

  1. START THE DAEMON
     python utp.py daemon --paper          # Paper trading
     python utp.py daemon --live           # Live IBKR

  2. CHECK PORTFOLIO & QUOTES
     python utp.py portfolio               # Positions & P&L
     python utp.py quote SPX NDX           # Real-time quotes
     python utp.py options SPX --type PUT  # Option chain

  3. TRADE
     python utp.py trade credit-spread --symbol SPX \\
       --short-strike 5500 --long-strike 5475 \\
       --option-type PUT --expiration 2026-03-20 \\
       --quantity 1 --net-price 3.50 --paper

  4. MONITOR
     python utp.py trades --paper          # Today's trades
     python utp.py status                  # Full dashboard

  5. CLOSE POSITIONS
     python utp.py close <id> --paper      # Close by position ID

  6. INTERACTIVE REPL
     python utp.py repl                    # Auto-detect daemon
""")

        # ────────────────────────────────────────────────────────────────
        _section("Demo Complete")
        # ────────────────────────────────────────────────────────────────
        print(f"\n  {_color('All demo data was in a temp directory and has been cleaned up.', '93')}\n")

    finally:
        # Cleanup
        reset_ledger()
        reset_position_store()
        ProviderRegistry.clear()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
