"""Tests for `_inject_range_percentiles_ws_script` — the JS payload
embedded into /range_percentiles HTML for live price updates over
WebSocket.

These are deliberately string-level smoke tests. The script's behavior
ultimately runs in a browser DOM, but full DOM testing is heavy for a
small surface area. The smoke tests pin the structural contracts that
the auto-live-prices feature depends on so they can't silently rot.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db_server import _inject_range_percentiles_ws_script


def _inject(tickers: list[str]) -> str:
    """Return the injected document for a minimal HTML shell. The
    function looks for `</body>` and inserts the script ahead of it,
    so we provide one for the replace to land on."""
    return _inject_range_percentiles_ws_script(
        "<html><body>placeholder</body></html>", tickers
    )


def test_injected_script_contains_market_open_check_and_polling():
    """The script must include both the market-open guard and the
    pre-market poll loop — the auto-flip story depends on the WS
    actually getting connected once 9:30 ET arrives."""
    out = _inject(["NDX", "SPX"])
    assert "isMarketOpen" in out
    assert "tryConnectAll" in out
    # The pre-market poller — every 30s. If this disappears, pages
    # loaded pre-market will never auto-connect.
    assert "setInterval" in out
    assert "30000" in out


def test_injected_script_does_not_auto_flip_main_section():
    """Main (overall percentile bands) section MUST stay anchored on
    previous close until the user clicks "Use Live Prices". Only the
    intraday section auto-flips at market open. Locks the contract:
    no `=== undefined` sentinel that would steer main into live on
    its own. Different mental model: intraday tables show today's
    move; main shows percentile what-ifs anchored on a stable
    reference."""
    out = _inject(["NDX"])
    assert "liveMain[ticker] === undefined" not in out
    # The gate is the plain truthy check on the user-set toggle.
    assert "if (liveMain[ticker])" in out


def test_injected_script_keeps_hourly_section_unconditional():
    """Hourly (intraday) section auto-updates on every WS message —
    no toggle gate. Combined with the pre-market poller that only
    connects WS once isMarketOpen() is true, this IS the "automatic
    switch to live at market open" behavior for intraday data."""
    out = _inject(["NDX"])
    assert "applySectionPrices(ticker, price, 'hourly', true)" in out


def test_injected_script_button_sync_helper_handles_both_states():
    """`_syncLiveToggleButton(ticker, on)` must drive both the on and
    off visual states so the manual click and the auto-flip render
    the same button."""
    out = _inject(["NDX"])
    assert "function _syncLiveToggleButton" in out
    # Both UI states: live (active green) and previous-close (default).
    assert "Use Previous Close" in out
    assert "Use Live Prices" in out


def test_injected_script_toggle_uses_shared_button_sync():
    """`toggleLiveMain` flips the boolean and routes UI updates through
    `_syncLiveToggleButton` so the visual state stays consistent.
    The helper is shared with any future auto-flip path — keeps style
    drift away from the manual click handler."""
    out = _inject(["NDX"])
    assert "liveMain[ticker] = !liveMain[ticker]" in out
    assert "_syncLiveToggleButton(ticker, !!liveMain[ticker])" in out


def test_injected_script_embeds_tickers_list():
    """The ticker list must end up in the JS as a JSON array so the
    poller knows which symbols to subscribe to."""
    out = _inject(["NDX", "SPX", "RUT"])
    assert '"NDX"' in out
    assert '"SPX"' in out
    assert '"RUT"' in out


def test_injected_script_inserted_before_body_close():
    """The injection point matters — script must sit inside <body>
    so DOMContentLoaded helpers (`paintInitialPrices`) run after the
    page is parsed."""
    out = _inject(["NDX"])
    # Script tag appears before the </body> we provided.
    assert out.index("<script>") < out.index("</body>")
