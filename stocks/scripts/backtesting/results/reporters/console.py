"""Console reporter -- prints results to stdout."""

from typing import Any, Dict

from .base import ReportGenerator


class ConsoleReporter(ReportGenerator):
    """Prints formatted backtest results to the console."""

    def generate(self, summary: Dict[str, Any], config: Any) -> None:
        metrics = summary.get("metrics", {})
        total = summary.get("total_trades", 0)

        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)

        if hasattr(config, "infra"):
            print(f"  Ticker: {config.infra.ticker}")
            if config.infra.start_date:
                print(f"  Period: {config.infra.start_date} to {config.infra.end_date or 'present'}")
        if hasattr(config, "strategy"):
            print(f"  Strategy: {config.strategy.name}")

        print(f"\n  Total Trades: {total}")
        print(f"  Wins: {metrics.get('wins', 0)}")
        print(f"  Losses: {metrics.get('losses', 0)}")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.1f}%")

        print(f"\n  Total Credits: ${metrics.get('total_credits', 0):,.2f}")
        print(f"  Total Gains:   ${metrics.get('total_gains', 0):,.2f}")
        print(f"  Total Losses:  ${metrics.get('total_losses', 0):,.2f}")
        net_pnl = metrics.get("net_pnl", 0)
        marker = "+" if net_pnl >= 0 else ""
        print(f"  Net P&L:       {marker}${net_pnl:,.2f}")

        print(f"\n  ROI:            {metrics.get('roi', 0):+.2f}%")
        print(f"  Profit Factor:  {metrics.get('profit_factor', 0):.4f}")
        print(f"  Sharpe Ratio:   {metrics.get('sharpe', 0):.4f}")
        print(f"  Max Drawdown:   ${metrics.get('max_drawdown', 0):,.2f}")
        print(f"  Avg P&L/Trade:  ${metrics.get('avg_pnl', 0):,.2f}")

        # Daily breakdown
        daily = summary.get("daily_breakdown", {})
        if daily:
            print(f"\n  Daily Performance ({len(daily)} days):")
            print(f"  {'Date':<12} {'Trades':<8} {'Win%':<8} {'Net P&L':<12}")
            print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*12}")
            for d, dm in list(daily.items())[:10]:
                print(
                    f"  {d:<12} {dm.get('total_trades', 0):<8} "
                    f"{dm.get('win_rate', 0):>5.1f}%  "
                    f"${dm.get('net_pnl', 0):>10,.2f}"
                )
            if len(daily) > 10:
                print(f"  ... and {len(daily) - 10} more days")

        print("=" * 80)
