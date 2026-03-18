"""ANSI terminal rendering for dashboard output."""

from __future__ import annotations

from datetime import UTC, datetime

from app.models import DashboardSummary, PerformanceMetrics


class TerminalRenderer:
    """Renders dashboard data as ANSI-colored text for terminal display."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    DIM = "\033[2m"

    @classmethod
    def render(cls, summary: DashboardSummary, metrics: PerformanceMetrics) -> str:
        lines: list[str] = []

        # Header
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines.append(f"{cls.BOLD}{cls.CYAN}═══ Universal Trade Platform Dashboard ═══{cls.RESET}")
        lines.append(f"{cls.DIM}{now}{cls.RESET}")
        lines.append("")

        # KPI strip
        lines.append(f"{cls.BOLD}── KPIs ──{cls.RESET}")
        lines.append(
            f"  Positions: {len(summary.active_positions)}  "
            f"Deployed: {cls._fmt_money(summary.cash_deployed)}  "
            f"Total P&L: {cls._color(summary.total_pnl)}  "
            f"Win Rate: {metrics.win_rate:.0%}  "
            f"Sharpe: {metrics.sharpe:.2f}"
        )
        lines.append("")

        # Positions table
        if summary.active_positions:
            lines.append(f"{cls.BOLD}── Open Positions ──{cls.RESET}")
            header = f"  {'POS_ID':>8}  {'BROKER':<12} {'SYMBOL':<8} {'QTY':>5} {'AVG':>10} {'UNRL P&L':>10} {'SOURCE':<15}"
            lines.append(header)
            lines.append(f"  {'─' * 78}")
            for pos in summary.active_positions:
                unrl = pos.unrealized_pnl or 0
                lines.append(
                    f"  {pos.position_id[:8]:>8}  {pos.broker.value:<12} "
                    f"{pos.symbol:<8} {pos.quantity:>5.0f} "
                    f"{cls._fmt_money(pos.entry_price):>10} "
                    f"{cls._color(unrl):>10} "
                    f"{pos.source.value:<15}"
                )
        else:
            lines.append(f"  {cls.DIM}No open positions{cls.RESET}")
        lines.append("")

        # P&L totals
        lines.append(f"{cls.BOLD}── P&L Summary ──{cls.RESET}")
        lines.append(f"  Realized:   {cls._color(summary.realized_pnl)}")
        lines.append(f"  Unrealized: {cls._color(summary.unrealized_pnl)}")
        lines.append(f"  Total:      {cls._color(summary.total_pnl)}")
        lines.append("")

        # Performance
        if metrics.total_trades > 0:
            lines.append(f"{cls.BOLD}── Performance ──{cls.RESET}")
            lines.append(f"  Trades: {metrics.total_trades}  W/L: {metrics.wins}/{metrics.losses}  "
                         f"Avg P&L: {cls._fmt_money(metrics.avg_pnl)}  "
                         f"Max DD: {cls._fmt_money(metrics.max_drawdown)}")
            lines.append("")

        # Source breakdown
        if summary.positions_by_source:
            lines.append(f"{cls.BOLD}── By Source ──{cls.RESET}")
            for src, count in summary.positions_by_source.items():
                lines.append(f"  {src:<20} {count}")
            lines.append("")

        lines.append(f"{cls.CYAN}{'═' * 50}{cls.RESET}")
        return "\n".join(lines)

    @staticmethod
    def _fmt_money(value: float) -> str:
        return f"${value:,.2f}"

    @classmethod
    def _color(cls, value: float) -> str:
        formatted = f"${value:+,.2f}"
        if value > 0:
            return f"{cls.GREEN}{formatted}{cls.RESET}"
        elif value < 0:
            return f"{cls.RED}{formatted}{cls.RESET}"
        return formatted
