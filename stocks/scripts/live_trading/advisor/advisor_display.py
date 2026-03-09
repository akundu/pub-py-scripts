"""Terminal display for the live advisor with ANSI colors.

Renders entry/exit recommendations, tracked positions, and status in a
readable terminal format with color coding. Reads all config from the
AdvisorProfile — no hardcoded constants.
"""

import os
from datetime import datetime, time, timezone
from typing import Dict, List, Optional

from .position_tracker import PositionTracker, TrackedPosition
from .profile_loader import AdvisorProfile
from .tier_evaluator import Recommendation


# ---------------------------------------------------------------------------
# ANSI color codes
# ---------------------------------------------------------------------------
class C:
    """ANSI color helpers."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"

    @classmethod
    def disable(cls):
        """Disable colors (for non-interactive/pipe mode)."""
        for attr in dir(cls):
            if attr.isupper() and not attr.startswith("_"):
                setattr(cls, attr, "")


def _dir_color(direction: str) -> str:
    """Green for puts (bullish), red for calls (bearish)."""
    return C.GREEN if direction == "put" else C.RED


def _action_color(action: str) -> str:
    if action == "ENTER":
        return C.CYAN
    elif action == "EXIT":
        return C.RED + C.BOLD
    elif action == "ROLL":
        return C.YELLOW + C.BOLD
    return C.WHITE


def _fmt_money(value: float) -> str:
    """Format dollar amount with commas."""
    if abs(value) >= 1000:
        return f"${value:,.0f}"
    return f"${value:.2f}"


def _fmt_price(value: float) -> str:
    """Format price with commas."""
    return f"{value:,.2f}"


def _utc_to_et(utc_dt: datetime) -> str:
    """Format UTC datetime as ET string (UTC-4 during EDT)."""
    et_hour = (utc_dt.hour - 4) % 24
    return f"{et_hour:02d}:{utc_dt.minute:02d}:{utc_dt.second:02d} ET"


class AdvisorDisplay:
    """Renders the advisor terminal UI."""

    def __init__(self, profile: AdvisorProfile, interactive: bool = True):
        self._profile = profile
        self._ticker = profile.ticker
        self._interactive = interactive
        try:
            self._width = min(os.get_terminal_size().columns, 100) if interactive else 100
        except OSError:
            self._width = 100

    def clear(self) -> None:
        """Clear terminal screen."""
        if self._interactive:
            print("\033[2J\033[H", end="")

    def refresh(
        self,
        current_price: Optional[float],
        prev_close: Optional[float],
        entries: List[Recommendation],
        exits: List[Recommendation],
        tracker: PositionTracker,
        now: datetime,
    ) -> None:
        """Full screen refresh with all sections."""
        self.clear()
        self._print_header(current_price, prev_close, now)

        if exits:
            self._print_exits(exits)

        if entries:
            self._print_entries(entries)

        self._print_positions(tracker.get_open_positions(), current_price)
        self._print_footer(tracker, now)

        if self._interactive and (entries or exits):
            self._print_commands()

    def _print_header(
        self,
        current_price: Optional[float],
        prev_close: Optional[float],
        now: datetime,
    ) -> None:
        bar = "=" * self._width
        print(f"\n{C.BOLD}{C.CYAN}{bar}{C.RESET}")

        # Title line
        title = f" LIVE ADVISOR | {self._profile.name} | {self._ticker}"
        print(f"{C.BOLD}{C.WHITE}{title}{C.RESET}")

        # Price + day change
        price_str = _fmt_price(current_price) if current_price else "---"
        et_str = _utc_to_et(now)

        if current_price and prev_close and prev_close > 0:
            day_pct = (current_price - prev_close) / prev_close
            day_color = C.GREEN if day_pct >= 0 else C.RED
            direction = "CALL" if day_pct > 0 else "PUT"
            dir_color = C.RED if day_pct > 0 else C.GREEN
            print(
                f" {et_str} | Price: {C.BOLD}{price_str}{C.RESET} | "
                f"Day: {day_color}{day_pct:+.2%}{C.RESET} | "
                f"Pursuit: {dir_color}{C.BOLD}{direction}{C.RESET}"
            )
        else:
            print(f" {et_str} | Price: {price_str}")

        print(f"{C.CYAN}{bar}{C.RESET}")

    def _print_entries(self, entries: List[Recommendation]) -> None:
        print(f"\n{C.BOLD}{C.CYAN}>>> NEW ENTRY RECOMMENDATIONS{C.RESET}\n")

        # Header
        hdr = (
            f"  {'PRI':>3}  {'TIER':<14} {'DIR':>4}  {'SHORT':>7}  {'LONG':>7}  "
            f"{'CREDIT':>7}  {'CTR':>3}  {'MAX LOSS':>9}  {'DTE':>3}"
        )
        print(f"{C.DIM}{hdr}{C.RESET}")

        for rec in entries:
            dc = _dir_color(rec.direction)
            print(
                f"  {C.BOLD}{rec.priority:>3}{C.RESET}  "
                f"{dc}{rec.tier_label:<14}{C.RESET} "
                f"{dc}{rec.direction.upper():>4}{C.RESET}  "
                f"{rec.short_strike:>7.0f}  {rec.long_strike:>7.0f}  "
                f"{C.GREEN}{_fmt_money(rec.credit):>7}{C.RESET}  "
                f"{rec.num_contracts:>3}  "
                f"{C.YELLOW}{_fmt_money(rec.max_loss):>9}{C.RESET}  "
                f"{rec.dte:>3}"
            )
            print(f"         {C.DIM}{rec.reason}{C.RESET}")

    def _print_exits(self, exits: List[Recommendation]) -> None:
        for rec in exits:
            ac = _action_color(rec.action)
            dc = _dir_color(rec.direction)
            print(
                f"\n{ac}!!! {rec.action} ALERT: "
                f"{rec.position_id} ({rec.tier_label} {rec.direction.upper()} "
                f"{rec.short_strike:.0f}/{rec.long_strike:.0f}){C.RESET}"
            )
            print(f"    {C.YELLOW}{rec.reason}{C.RESET}")

    def _print_positions(
        self, positions: List[TrackedPosition], current_price: Optional[float]
    ) -> None:
        print(f"\n{C.DIM}--- TRACKED POSITIONS ({len(positions)}) "
              f"{'-' * max(0, self._width - 30)}{C.RESET}")

        if not positions:
            print(f"  {C.DIM}No open positions{C.RESET}")
            return

        hdr = (
            f"  {'ID':<8} {'TIER':<14} {'DIR':>4}  {'SHORT/LONG':>13}  "
            f"{'CREDIT':>7}  {'CTR':>3}  {'DTE':>3}  STATUS"
        )
        print(f"{C.DIM}{hdr}{C.RESET}")

        for pos in positions:
            dc = _dir_color(pos.direction)

            # Status
            if current_price and current_price > 0:
                if pos.direction == "put":
                    otm_pct = (current_price - pos.short_strike) / current_price
                else:
                    otm_pct = (pos.short_strike - current_price) / current_price

                if otm_pct < 0:
                    status = f"{C.RED}{C.BOLD}ITM ({abs(otm_pct):.1%}){C.RESET}"
                elif otm_pct < 0.005:
                    status = f"{C.YELLOW}CLOSE ({otm_pct:.1%} OTM){C.RESET}"
                else:
                    status = f"{C.GREEN}OK ({otm_pct:.1%} OTM){C.RESET}"
            else:
                status = "---"

            print(
                f"  {pos.pos_id:<8} "
                f"{dc}{pos.tier_label:<14}{C.RESET} "
                f"{dc}{pos.direction.upper():>4}{C.RESET}  "
                f"{pos.short_strike:>6.0f}/{pos.long_strike:<6.0f}  "
                f"{_fmt_money(pos.credit):>7}  "
                f"{pos.num_contracts:>3}  "
                f"{pos.dte:>3}  "
                f"{status}"
            )

    def _print_footer(self, tracker: PositionTracker, now: datetime) -> None:
        risk = self._profile.risk
        budget_used = tracker.get_daily_budget_used()
        trades_today = tracker.get_daily_trade_count()
        rate_remaining = tracker.check_rate_limit(
            risk.trade_window_minutes, risk.max_trades_per_window
        )

        print(
            f"\n{C.DIM}--- Budget: {_fmt_money(budget_used)}/{_fmt_money(risk.daily_budget)} | "
            f"Trades today: {trades_today} | "
            f"Rate: {rate_remaining}/{risk.max_trades_per_window} remaining | "
            f"Next: {_utc_to_et(now)}"
        )
        print("=" * self._width + C.RESET)

    def _print_commands(self) -> None:
        print(
            f"\n{C.DIM}  Commands: "
            f"'y <pri>' confirm entry | "
            f"'x <id> [price]' close | "
            f"'r <id>' confirm roll | "
            f"'p' positions | "
            f"'s' summary | "
            f"'q' quit{C.RESET}"
        )

    def print_waiting(self, msg: str) -> None:
        """Show waiting message outside market hours."""
        print(f"\r{C.DIM}{msg}{C.RESET}", end="", flush=True)

    def print_info(self, msg: str) -> None:
        print(f"{C.CYAN}{msg}{C.RESET}")

    def print_success(self, msg: str) -> None:
        print(f"{C.GREEN}{C.BOLD}{msg}{C.RESET}")

    def print_error(self, msg: str) -> None:
        print(f"{C.RED}{msg}{C.RESET}")

    def print_summary(self, tracker: PositionTracker) -> None:
        """Print daily performance summary."""
        summary = tracker.get_daily_summary()
        print(f"\n{C.BOLD}{C.CYAN}=== Daily Summary ==={C.RESET}")
        print(f"  Profile:       {self._profile.name}")
        print(f"  Ticker:        {self._ticker}")
        print(f"  Trades today:  {summary['trades_today']}")
        print(f"  Open:          {summary['open_count']}")
        print(f"  Closed:        {summary['closed_today']}")
        print(f"  Budget used:   {_fmt_money(summary['budget_used'])}")
        print(f"  Realized P&L:  {_fmt_money(summary['realized_pnl'])}")
        print(f"  Total credit:  {_fmt_money(summary['total_credit'])}")

    def print_positions_detail(self, tracker: PositionTracker, current_price: Optional[float]) -> None:
        """Print detailed position view."""
        positions = tracker.get_open_positions()
        self._print_positions(positions, current_price)

    def print_dry_run_config(self) -> None:
        """Print configuration in dry-run mode."""
        risk = self._profile.risk
        print(f"\n{C.BOLD}{C.CYAN}=== Live Advisor — {self._profile.name} — Dry Run ==={C.RESET}")
        print(f"\n  Ticker: {self._ticker}")
        print(f"  Daily budget: {_fmt_money(risk.daily_budget)}")
        print(f"  Max risk/trade: {_fmt_money(risk.max_risk_per_trade)}")
        print(f"  Rate limit: {risk.max_trades_per_window} per {risk.trade_window_minutes}min")
        print(f"\n{C.BOLD}  Tiers ({len(self._profile.tiers)}):{C.RESET}")
        for t in self._profile.tiers:
            dc = C.CYAN if t.directional == "pursuit" else C.MAGENTA
            dte_str = f"DTE={t.dte:<2}" if t.dte is not None else "DTE=*"
            pct_str = f"P{t.percentile:<2}" if t.percentile is not None else ""
            width_str = f"W={t.spread_width}pt" if t.spread_width is not None else ""
            print(
                f"    P{t.priority} {dc}{t.label:<16}{C.RESET} "
                f"{dte_str} {pct_str} "
                f"{width_str} "
                f"{t.entry_start}-{t.entry_end} UTC"
            )
        print()
