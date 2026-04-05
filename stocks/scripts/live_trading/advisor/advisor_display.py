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


def _utc_to_local(utc_dt) -> str:
    """Format UTC datetime as local time string. Handles both stdlib and pandas Timestamps."""
    try:
        # Convert to stdlib datetime first if it's a pandas Timestamp
        if hasattr(utc_dt, 'to_pydatetime'):
            utc_dt = utc_dt.to_pydatetime()
        local_dt = utc_dt.astimezone() if utc_dt.tzinfo else utc_dt
        return local_dt.strftime("%H:%M:%S %Z")
    except Exception:
        return str(utc_dt)


def _utc_time_to_local(utc_time_str: str) -> str:
    """Convert a UTC time string like '14:30' to local time string."""
    from datetime import date as _date
    try:
        parts = utc_time_str.split(":")
        h, m = int(parts[0]), int(parts[1])
        utc_dt = datetime(2026, 1, 1, h, m, tzinfo=timezone.utc)
        local_dt = utc_dt.astimezone()
        return local_dt.strftime("%H:%M")
    except Exception:
        return utc_time_str


class AdvisorDisplay:
    """Renders the advisor terminal UI."""

    def __init__(self, profile: AdvisorProfile, interactive: bool = True):
        self._profile = profile
        self._ticker = profile.ticker
        self._interactive = interactive
        self._status_messages: List[str] = []  # persist across screen clears
        self._last_fingerprint: str = ""  # skip refresh if nothing changed
        self._input_log: List[str] = []  # last N commands received
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
        *,
        ticker_prices: Optional[Dict[str, dict]] = None,
        all_entries: Optional[List[Recommendation]] = None,
    ) -> None:
        """Full screen refresh with all sections.

        Args:
            ticker_prices: {ticker: {"price": float, "prev_close": float}} for multi-ticker header
            all_entries: all candidates before filtering to top picks (shown as rejected below the line)
        """
        # Skip refresh if nothing changed (unless there are status messages)
        fingerprint = (
            f"{current_price:.2f}_{len(entries)}_{len(exits)}_"
            f"{len(tracker.get_open_positions())}_{now.minute}"
        )
        has_status = bool(self._status_messages)
        if fingerprint == self._last_fingerprint and not has_status:
            return  # nothing changed, don't flicker the screen
        self._last_fingerprint = fingerprint

        self.clear()
        self._print_header(current_price, prev_close, now, ticker_prices=ticker_prices)

        # Show recent status messages (confirmations, errors) that survive clear
        if self._status_messages:
            for msg in self._status_messages[-3:]:
                print(f"  {msg}")
            self._status_messages.clear()

        if exits:
            self._print_exits(exits)

        if entries:
            self._print_entries(entries)

        # Always show top 5 scanned trades (even if none nominated)
        if all_entries:
            picked_ids = {(r.tier_label, r.direction, r.short_strike) for r in entries} if entries else set()
            others = [r for r in all_entries
                      if (r.tier_label, r.direction, r.short_strike) not in picked_ids]
            self._print_rejected(others)
        elif not entries:
            print(f"\n{C.DIM}  No trades found this interval{C.RESET}")

        self._print_positions(tracker.get_open_positions(), current_price,
                              ticker_prices=ticker_prices)
        self._print_footer(tracker, now)

        if self._interactive:
            self._print_commands()

    def _print_header(
        self,
        current_price: Optional[float],
        prev_close: Optional[float],
        now: datetime,
        *,
        ticker_prices: Optional[Dict[str, dict]] = None,
    ) -> None:
        bar = "=" * self._width
        print(f"\n{C.BOLD}{C.CYAN}{bar}{C.RESET}")

        # Title line
        title = f" LIVE ADVISOR | {self._profile.name}"
        print(f"{C.BOLD}{C.WHITE}{title}{C.RESET}")

        et_str = _utc_to_local(now)
        print(f" {et_str}")

        # Multi-ticker price line
        if ticker_prices:
            parts = []
            for ticker, info in ticker_prices.items():
                p = info.get("price")
                pc = info.get("prev_close")
                qt = info.get("quote_ts")
                # Show age of quote
                age_str = ""
                if qt is not None:
                    try:
                        if hasattr(qt, 'to_pydatetime'):
                            qt = qt.to_pydatetime()
                        if qt.tzinfo:
                            age_secs = (datetime.now(qt.tzinfo) - qt).total_seconds()
                        else:
                            age_secs = 0
                        if age_secs > 60:
                            age_str = f" {C.YELLOW}({int(age_secs//60)}m ago){C.RESET}"
                        elif age_secs > 10:
                            age_str = f" {C.DIM}({int(age_secs)}s){C.RESET}"
                    except Exception:
                        pass

                if p and pc and pc > 0:
                    pct = (p - pc) / pc
                    clr = C.GREEN if pct >= 0 else C.RED
                    parts.append(
                        f"{C.BOLD}{ticker}{C.RESET} {_fmt_price(p)} "
                        f"{clr}{pct:+.2%}{C.RESET}{age_str}"
                    )
                elif p:
                    parts.append(f"{C.BOLD}{ticker}{C.RESET} {_fmt_price(p)}{age_str}")
                else:
                    parts.append(f"{C.BOLD}{ticker}{C.RESET} {C.RED}no quote{C.RESET}")
            if parts:
                print(f" {' | '.join(parts)}")
        elif current_price:
            price_str = _fmt_price(current_price)
            if prev_close and prev_close > 0:
                day_pct = (current_price - prev_close) / prev_close
                day_color = C.GREEN if day_pct >= 0 else C.RED
                direction = "CALL" if day_pct > 0 else "PUT"
                dir_color = C.RED if day_pct > 0 else C.GREEN
                print(
                    f" {self._ticker}: {C.BOLD}{price_str}{C.RESET} | "
                    f"Day: {day_color}{day_pct:+.2%}{C.RESET} | "
                    f"Pursuit: {dir_color}{C.BOLD}{direction}{C.RESET}"
                )
            else:
                print(f" {self._ticker}: {price_str}")

        print(f"{C.CYAN}{bar}{C.RESET}")

    def _print_entries(self, entries: List[Recommendation]) -> None:
        print(f"\n{C.BOLD}{C.CYAN}>>> ENTRY RECOMMENDATIONS{C.RESET}\n")

        # Get ROI tier thresholds for multiplier display
        ab = self._profile.adaptive_budget
        roi_thresholds = ab.roi_thresholds if ab and ab.enabled else [6.0, 9.0]
        roi_multipliers = ab.roi_multipliers if ab and ab.enabled else [1.0, 2.0, 4.0]

        # Header
        hdr = (
            f"  {'ORDER ID':<22} {'TKR':<5} {'TIER':<14} {'DIR':>4}  "
            f"{'SHORT':>7} {'SELL$':>6}  {'LONG':>7} {'BUY$':>6}  "
            f"{'CR/SH':>6} {'CTR':>4} {'TOT CREDIT':>10} {'MAX LOSS':>9}  {'nROI/ROI':>10} {'MULT':>4}"
        )
        print(f"{C.DIM}{hdr}{C.RESET}")

        for rec in entries:
            dc = _dir_color(rec.direction)
            tkr = rec.ticker or self._ticker
            sell_str = f"${rec.short_price:.2f}" if rec.short_price else "—"
            buy_str = f"${rec.long_price:.2f}" if rec.long_price else "—"
            oid = rec.order_id or "—"

            w = rec.spread_width or 1
            ml = w - rec.credit
            raw_roi = (rec.credit / ml * 100) if ml > 0 else 0
            norm_roi = raw_roi / (rec.dte + 1)
            roi_str = f"{norm_roi:.1f}/{raw_roi:.1f}%"

            # Compute multiplier from ROI tiers
            mult = roi_multipliers[0]
            for i in range(len(roi_thresholds) - 1, -1, -1):
                if norm_roi >= roi_thresholds[i]:
                    mult = roi_multipliers[i + 1]
                    break
            mult_color = C.GREEN if mult >= 4 else C.YELLOW if mult >= 2 else C.DIM
            mult_str = f"{mult:.0f}x"

            print(
                f"  {C.CYAN}{oid:<22}{C.RESET} "
                f"{C.BOLD}{tkr:<5}{C.RESET} "
                f"{dc}{rec.tier_label:<14}{C.RESET} "
                f"{dc}{rec.direction.upper():>4}{C.RESET}  "
                f"{rec.short_strike:>7.0f} {C.GREEN}{sell_str:>6}{C.RESET}  "
                f"{rec.long_strike:>7.0f} {C.RED}{buy_str:>6}{C.RESET}  "
                f"${rec.credit:>5.2f} {rec.num_contracts:>4}  "
                f"{C.GREEN}${rec.total_credit:>8,.0f}{C.RESET} "
                f"{C.YELLOW}{_fmt_money(rec.max_loss):>9}{C.RESET}  "
                f"{C.CYAN}{roi_str:>10}{C.RESET} "
                f"{mult_color}{mult_str:>4}{C.RESET}"
            )
            for part in rec.reason.split(" | "):
                print(f"                          {C.DIM}{part}{C.RESET}")

    def _print_rejected(self, rejected: List[Recommendation], rank_start: int = 4) -> None:
        """Show alternatives that weren't picked (up to 5), with total credit and reason."""
        print(f"\n{C.DIM}--- OTHER CANDIDATES ({len(rejected)}) "
              f"{'-' * max(0, self._width - 30)}{C.RESET}")
        for i, rec in enumerate(rejected[:5]):
            dc = _dir_color(rec.direction)
            oid = rec.order_id or "—"
            tkr = rec.ticker or "?"
            sell_str = f"s@${rec.short_price:.2f}" if rec.short_price else ""
            buy_str = f"b@${rec.long_price:.2f}" if rec.long_price else ""
            leg_str = f" {sell_str} {buy_str}" if sell_str else ""

            w = rec.spread_width or 1
            ml = w - rec.credit
            raw_roi = (rec.credit / ml * 100) if ml > 0 else 0
            norm_roi = raw_roi / (rec.dte + 1)

            print(
                f"  {C.DIM}{oid:<22} {tkr:<4} "
                f"{dc}{rec.direction.upper():>4}{C.RESET}{C.DIM} "
                f"{rec.short_strike:.0f}/{rec.long_strike:.0f}{leg_str} "
                f"cr=${rec.credit:.2f} x{rec.num_contracts} "
                f"{C.RESET}{C.DIM}total=${rec.total_credit:>7,.0f}{C.RESET}{C.DIM} "
                f"DTE={rec.dte} nROI/ROI={norm_roi:.1f}/{raw_roi:.1f}% (#{rank_start + i}){C.RESET}"
            )
        if len(rejected) > 5:
            print(f"  {C.DIM}... and {len(rejected)-5} more{C.RESET}")

    def _print_exits(self, exits: List[Recommendation]) -> None:
        for rec in exits:
            ac = _action_color(rec.action)
            oid = rec.order_id or rec.position_id or "—"
            print(
                f"\n{ac}!!! {rec.action} [{oid}]: "
                f"{rec.tier_label} {rec.direction.upper()} "
                f"{rec.short_strike:.0f}/{rec.long_strike:.0f}{C.RESET}"
            )
            for part in rec.reason.split(" | "):
                print(f"    {C.YELLOW}{part}{C.RESET}")

    def _print_positions(
        self, positions: List[TrackedPosition], current_price: Optional[float],
        ticker_prices: Optional[Dict[str, dict]] = None,
    ) -> None:
        print(f"\n{C.DIM}--- TRACKED POSITIONS ({len(positions)}) "
              f"{'-' * max(0, self._width - 30)}{C.RESET}")

        if not positions:
            print(f"  {C.DIM}No open positions{C.RESET}")
            return

        hdr = (
            f"  {'ID':<8} {'TKR':<5} {'TIER':<14} {'DIR':>4}  {'SHORT/LONG':>13}  "
            f"{'CREDIT':>7}  {'CTR':>3}  {'DTE':>3}  STATUS"
        )
        print(f"{C.DIM}{hdr}{C.RESET}")

        for pos in positions:
            dc = _dir_color(pos.direction)

            # Use the right price for this position's ticker
            pos_price = current_price
            if ticker_prices and pos.ticker and pos.ticker in ticker_prices:
                pos_price = ticker_prices[pos.ticker].get("price", current_price)

            # Status
            if pos_price and pos_price > 0:
                if pos.direction == "put":
                    otm_pct = (pos_price - pos.short_strike) / pos_price
                else:
                    otm_pct = (pos.short_strike - pos_price) / pos_price

                if otm_pct < 0:
                    status = f"{C.RED}{C.BOLD}ITM ({abs(otm_pct):.1%}){C.RESET}"
                elif otm_pct < 0.005:
                    status = f"{C.YELLOW}CLOSE ({otm_pct:.1%} OTM){C.RESET}"
                else:
                    status = f"{C.GREEN}OK ({otm_pct:.1%} OTM){C.RESET}"
            else:
                status = "---"

            tkr = pos.ticker or "?"
            print(
                f"  {pos.pos_id:<8} "
                f"{C.BOLD}{tkr:<5}{C.RESET} "
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
            f"Next: {_utc_to_local(now)}"
        )

        # Config summary
        ab = self._profile.adaptive_budget
        if ab and ab.enabled:
            min_cr_val = self._profile.strategy_defaults.get("min_credit", 0)
            mode = getattr(ab, "roi_mode", "?")
            if mode == "fixed_roi":
                thresh = getattr(ab, "roi_thresholds", [])
                mults = getattr(ab, "roi_multipliers", [])
                tier_str = " / ".join(
                    f"<{thresh[i]}%→{mults[i]:.0f}x" if i < len(thresh)
                    else f">{thresh[-1]}%→{mults[-1]:.0f}x"
                    for i in range(len(mults))
                )
            else:
                tier_str = f"mode={mode}"
            min_tot = getattr(ab, "min_total_credit", 0)
            print(
                f"--- Allocation: {tier_str} | "
                f"min_credit: ${min_cr_val:.2f}/sh | "
                f"min_total: ${min_tot:,.0f} | "
                f"max_risk: ${risk.max_risk_per_trade:,.0f} | "
                f"use_mid: {self._profile.strategy_defaults.get('use_mid', '?')}"
            )

        # Tier constraints: show each tier's percentile + DTE + entry window
        tiers = self._profile.tiers
        if tiers:
            tier_parts = []
            for t in tiers:
                pct = f"P{t.percentile}" if t.percentile else "—"
                dte = f"D{t.dte}" if t.dte is not None else "—"
                start_l = _utc_time_to_local(t.entry_start)
                end_l = _utc_time_to_local(t.entry_end)
                tier_parts.append(f"{t.label}({pct} {dte} {start_l}-{end_l})")
            print(f"--- Tiers: {' | '.join(tier_parts)}")

        # Exit rules summary
        er = self._profile.exit_rules
        exit_parts = []
        if er.profit_target_pct is not None:
            exit_parts.append(f"profit={er.profit_target_pct*100:.0f}%")
        if er.stop_loss_pct is not None:
            exit_parts.append(f"stop={er.stop_loss_pct:.0f}x")
        if er.time_exit_utc:
            exit_parts.append(f"time_exit={_utc_time_to_local(er.time_exit_utc)}")
        if er.roll_enabled:
            exit_parts.append(f"roll@{_utc_time_to_local(er.roll_check_start_utc)}")
        if exit_parts:
            print(f"--- Exits: {' | '.join(exit_parts)}")

        print("=" * self._width + C.RESET)

    def log_input(self, raw_input: str, result: str) -> None:
        """Log a received command and its result for display."""
        from datetime import datetime as _dt
        ts = _dt.now().strftime("%H:%M:%S")
        entry = f"{ts} > {raw_input!r} → {result}"
        self._input_log.append(entry)
        if len(self._input_log) > 5:
            self._input_log = self._input_log[-5:]

    def _print_commands(self) -> None:
        # Show recent input log
        if self._input_log:
            print(f"\n{C.DIM}  Recent commands:{C.RESET}")
            for entry in self._input_log:
                print(f"  {C.DIM}  {entry}{C.RESET}")

        print(
            f"\n{C.DIM}  Commands: "
            f"'buy <order_id>' confirm trade | "
            f"'close <id> [price]' close | "
            f"'flush' close all | "
            f"'r <id>' roll | "
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
        self._status_messages.append(f"{C.GREEN}{C.BOLD}{msg}{C.RESET}")

    def print_error(self, msg: str) -> None:
        print(f"{C.RED}{msg}{C.RESET}")
        self._status_messages.append(f"{C.RED}{msg}{C.RESET}")

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
                f"{_utc_time_to_local(t.entry_start)}-{_utc_time_to_local(t.entry_end)}"
            )

        er = self._profile.exit_rules
        print(f"\n{C.BOLD}  Exit Rules:{C.RESET}")
        if er.profit_target_pct is not None:
            print(f"    Profit target: {er.profit_target_pct:.0%}")
        if er.stop_loss_pct is not None:
            sl_gate = f" (after {_utc_time_to_local(er.stop_loss_start_utc)})" if er.stop_loss_start_utc else ""
            print(f"    Stop loss: {er.stop_loss_pct:.0f}x credit{sl_gate}")
        if er.time_exit_utc:
            print(f"    Time exit: {_utc_time_to_local(er.time_exit_utc)}")
        if er.roll_enabled:
            print(f"    Rolling: enabled at {_utc_time_to_local(er.roll_check_start_utc)}")
            print(f"      Proximity: {er.roll_proximity_pct:.1%} | "
                  f"Roll target: P{er.roll_percentile} at DTE {er.roll_min_dte}-{er.roll_max_dte}")
            print(f"      Max rolls: {er.max_rolls} | Width expansion: up to {er.max_width_multiplier}x")
        print()
