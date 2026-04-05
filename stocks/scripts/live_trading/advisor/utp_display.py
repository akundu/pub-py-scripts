"""Phase 1 UTP display — raw market data snapshot for the live advisor.

Fetches equity prices and option chains from UTP and displays them in a
formatted terminal table. Supports multiple tickers in a single view.
No trade recommendations — display only.
"""

import time as time_mod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from .advisor_display import C, _utc_to_et


def _fmt_price(value: float) -> str:
    """Format price with commas."""
    if value >= 100:
        return f"{value:,.2f}"
    return f"{value:.2f}"


@dataclass
class TickerSnapshot:
    """Data fetched for a single ticker in one cycle."""
    ticker: str
    price: Optional[float]
    prev_close: Optional[float]
    options_df: Optional[pd.DataFrame]
    quote_cache_age: Optional[float] = None  # seconds since last fetch, None = fresh


class UtpDataDisplay:
    """Renders UTP market data snapshots in the terminal."""

    def __init__(self, profile_name: str, tickers: List[str], width: int = 100):
        self._profile_name = profile_name
        self._tickers = tickers
        self._width = width

    def print_multi_ticker_snapshot(
        self,
        snapshots: List[TickerSnapshot],
        now: datetime,
        cache_stats: Optional[Dict[str, Dict]] = None,
        next_refresh_secs: int = 0,
    ) -> None:
        """Print market data for all tickers in one screen."""
        # Clear screen
        print("\033[2J\033[H", end="")

        self._print_banner(now)
        self._print_price_summary(snapshots, now)

        for snap in snapshots:
            self._print_ticker_section(snap)

        self._print_footer(cache_stats, next_refresh_secs, now)

    def _print_banner(self, now: datetime) -> None:
        bar = "=" * self._width
        print(f"\n{C.BOLD}{C.CYAN}{bar}{C.RESET}")

        tickers_str = ", ".join(self._tickers)
        title = f" LIVE ADVISOR | {self._profile_name} | {tickers_str} (via UTP/IBKR)"
        print(f"{C.BOLD}{C.WHITE}{title}{C.RESET}")

        et_str = _utc_to_et(now)
        print(f" {et_str}")
        print(f"{C.CYAN}{bar}{C.RESET}")

    def _print_price_summary(
        self, snapshots: List[TickerSnapshot], now: datetime
    ) -> None:
        """Compact one-line-per-ticker price strip at the top."""
        print(f"\n{C.BOLD}{C.WHITE}  {'TICKER':<8} {'PRICE':>12} {'DAY CHG':>10} {'PURSUIT':>8}{C.RESET}")

        for snap in snapshots:
            if snap.price is None:
                print(f"  {snap.ticker:<8} {C.DIM}{'---':>12}{C.RESET}")
                continue

            price_str = _fmt_price(snap.price)
            if snap.prev_close and snap.prev_close > 0:
                day_pct = (snap.price - snap.prev_close) / snap.prev_close
                day_color = C.GREEN if day_pct >= 0 else C.RED
                direction = "CALL" if day_pct > 0 else "PUT"
                dir_color = C.RED if day_pct > 0 else C.GREEN
                print(
                    f"  {C.BOLD}{snap.ticker:<8}{C.RESET} "
                    f"{price_str:>12} "
                    f"{day_color}{day_pct:>+9.2%}{C.RESET} "
                    f"{dir_color}{C.BOLD}{direction:>8}{C.RESET}"
                )
            else:
                print(f"  {C.BOLD}{snap.ticker:<8}{C.RESET} {price_str:>12}")

    def _print_ticker_section(self, snap: TickerSnapshot) -> None:
        """Print options chain for a single ticker."""
        divider = "-" * self._width
        print(f"\n{C.BOLD}{C.YELLOW}{divider}{C.RESET}")
        price_str = _fmt_price(snap.price) if snap.price else "---"
        print(f"{C.BOLD}{C.YELLOW} {snap.ticker}{C.RESET} | Price: {C.BOLD}{price_str}{C.RESET}")

        if snap.options_df is not None and not snap.options_df.empty:
            self._print_options_chain(snap.options_df, snap.price)
        else:
            print(f"  {C.DIM}No options data available{C.RESET}")

    def _print_options_chain(
        self,
        options_df: pd.DataFrame,
        current_price: Optional[float],
    ) -> None:
        """Print options grouped by DTE, showing strikes near current price."""
        if "dte" not in options_df.columns:
            return

        sorted_df = options_df.sort_values(["dte", "strike"])

        for dte_val, group in sorted_df.groupby("dte", sort=True):
            expiration = ""
            if "expiration" in group.columns:
                exp_vals = group["expiration"].dropna().unique()
                if len(exp_vals) > 0:
                    expiration = str(exp_vals[0])

            dte_label = f"{int(dte_val)}DTE"
            header = f"  --- {dte_label}: {expiration} "
            header += "-" * max(0, self._width - len(header) - 2)
            print(f"\n{C.WHITE}{header}{C.RESET}")

            # Table header
            hdr = (
                f"  {'Strike':>10}  {'Type':>6}  {'Bid':>8}  {'Ask':>8}  "
                f"{'Mid':>8}  {'Vol':>6}  {'OI':>8}  {'DTE':>3}"
            )
            print(f"{C.DIM}{hdr}{C.RESET}")

            # Filter to ~10 strikes above and below current price
            display_group = group
            if current_price and "strike" in group.columns:
                strikes = group["strike"].unique()
                below = strikes[strikes <= current_price]
                above = strikes[strikes > current_price]
                near_below = below[-10:] if len(below) > 10 else below
                near_above = above[:10] if len(above) > 10 else above
                near_strikes = set(near_below) | set(near_above)
                if near_strikes:
                    display_group = group[group["strike"].isin(near_strikes)]

            for _, row in display_group.iterrows():
                strike = float(row.get("strike", 0))
                opt_type = str(row.get("type", "")).upper()
                bid = float(row.get("bid", 0))
                ask = float(row.get("ask", 0))
                mid = float(row.get("mid", (bid + ask) / 2 if bid and ask else 0))
                vol = int(row.get("volume", 0))
                oi = int(row.get("open_interest", 0))
                dte = int(row.get("dte", 0))

                strike_color = C.WHITE
                if current_price:
                    if opt_type == "PUT" and strike > current_price:
                        strike_color = C.RED
                    elif opt_type == "CALL" and strike < current_price:
                        strike_color = C.RED

                type_color = C.GREEN if opt_type == "PUT" else C.RED

                print(
                    f"  {strike_color}{strike:>10,.0f}{C.RESET}  "
                    f"{type_color}{opt_type:>6}{C.RESET}  "
                    f"{bid:>8.2f}  {ask:>8.2f}  "
                    f"{C.CYAN}{mid:>8.2f}{C.RESET}  "
                    f"{vol:>6,}  {oi:>8,}  {dte:>3}"
                )

    def _print_footer(
        self,
        cache_stats: Optional[Dict[str, Dict]],
        next_refresh_secs: int,
        now: datetime,
    ) -> None:
        bar = "=" * self._width

        parts = []
        if cache_stats:
            for name, stats in cache_stats.items():
                age = stats.get("age")
                if age is not None:
                    parts.append(f"{name}=HIT (age {int(age)}s)")
                else:
                    parts.append(f"{name}=FRESH")

        cache_str = " | ".join(parts) if parts else "no cache info"
        refresh_str = f"next refresh: {next_refresh_secs}s" if next_refresh_secs > 0 else "refreshing..."

        print(
            f"\n{C.DIM}--- Cache: {cache_str} | {refresh_str}"
        )
        print(f"{bar}{C.RESET}")
        print(f"{C.DIM}  Commands: 'q' quit{C.RESET}")

    # Keep the single-ticker method for backwards compatibility / tests
    def print_market_snapshot(
        self,
        price: Optional[float],
        prev_close: Optional[float],
        options_df: Optional[pd.DataFrame],
        now: datetime,
        cache_stats: Optional[Dict[str, Dict]] = None,
        next_refresh_secs: int = 0,
    ) -> None:
        """Single-ticker snapshot (backwards compat)."""
        ticker = self._tickers[0] if self._tickers else "???"
        snap = TickerSnapshot(
            ticker=ticker, price=price, prev_close=prev_close,
            options_df=options_df,
        )
        self.print_multi_ticker_snapshot(
            [snap], now, cache_stats, next_refresh_secs,
        )
