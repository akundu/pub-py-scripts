"""Tests for common/range_percentiles_formatter.py — focused on the
`data-prev-close` attribute used by the LIVE-price diff badge."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from common.range_percentiles_formatter import _raw_prev_close


def test_raw_prev_close_float():
    assert _raw_prev_close(27222.39) == "27222.3900"


def test_raw_prev_close_int():
    assert _raw_prev_close(27100) == "27100.0000"


def test_raw_prev_close_string_numeric():
    assert _raw_prev_close("27100.5") == "27100.5000"


def test_raw_prev_close_invalid_returns_empty():
    assert _raw_prev_close("n/a") == ""
    assert _raw_prev_close(None) == ""


def test_data_prev_close_attribute_in_formatter_source():
    """Every ref-close span the formatter emits must carry data-prev-close,
    otherwise the JS in db_server.py can't render the LIVE diff badge."""
    from common import range_percentiles_formatter as fmt
    src = Path(fmt.__file__).read_text()
    ref_spans = src.count('class="ref-close"')
    with_attr = sum(
        1 for line in src.splitlines()
        if 'class="ref-close"' in line
        and 'data-prev-close="{_raw_prev_close(prev_close)}"' in line
    )
    assert ref_spans > 0, "no ref-close spans found"
    assert with_attr == ref_spans, (
        f"mismatch: {ref_spans} ref-close spans but only {with_attr} have "
        f"data-prev-close — every ref-close must expose the attribute"
    )


def test_js_reads_data_prev_close_and_renders_diff():
    """db_server.py JS must consume the attribute we emit."""
    src = (_REPO / "db_server.py").read_text()
    assert "el.dataset.prevClose" in src, \
        "applySectionPrices must read el.dataset.prevClose to compute the diff"
    assert "vs prev" in src, "diff badge text must include 'vs prev'"
    assert "diffPct" in src, "diff pct variable must be used"


# ----- buffer addendum tests --------------------------------------------------

from common.range_percentiles_formatter import (
    _buffered_pct_value,
    _buffer_pct_html,
    _buffer_price_html,
    _buffer_data_attr,
    format_as_html,
    format_multi_window_as_html,
    format_hourly_moves_as_html,
)


def test_buffered_pct_value_zero_or_missing_returns_none():
    assert _buffered_pct_value(-1.5, 0) is None
    assert _buffered_pct_value(-1.5, None) is None  # type: ignore[arg-type]
    assert _buffered_pct_value(-1.5, -0.5) is None


def test_buffered_pct_value_direction_down_subtracts():
    # DOWN: pct=-1.5 with 0.5 buffer => -2.0 (further negative)
    assert _buffered_pct_value(-1.5, 0.5, direction="down") == -2.0


def test_buffered_pct_value_direction_up_adds():
    assert _buffered_pct_value(1.5, 0.5, direction="up") == 2.0


def test_buffered_pct_value_no_direction_uses_sign():
    assert _buffered_pct_value(-1.0, 0.5) == -1.5
    assert _buffered_pct_value(1.0, 0.5) == 1.5
    # zero pct treated as up
    assert _buffered_pct_value(0.0, 0.5) == 0.5


def test_buffer_pct_html_empty_when_inactive():
    assert _buffer_pct_html(-1.5, 0) == ""
    assert _buffer_pct_html(-1.5, 0.0) == ""


def test_buffer_pct_html_has_signed_text_and_class():
    out = _buffer_pct_html(-1.5, 0.5, direction="down")
    assert "buffer-addendum" in out
    assert "-2.00%" in out
    out_up = _buffer_pct_html(1.5, 0.5, direction="up")
    assert "+2.00%" in out_up


def test_buffer_price_html_uses_buffered_pct_and_prev_close():
    # DOWN: prev=100, pct=-1.5, buf=0.5 => buffered_pct=-2.0, price=98.00
    out = _buffer_price_html(-1.5, 100, 0.5, direction="down")
    assert "$98.00" in out
    # UP: prev=100, pct=1.5, buf=0.5 => buffered_pct=2.0, price=102.00
    out_up = _buffer_price_html(1.5, 100, 0.5, direction="up")
    assert "$102.00" in out_up


def test_buffer_data_attr_active_emits_dataset_attrs():
    attr = _buffer_data_attr(-1.5, 0.5, direction="down")
    assert 'data-buffer="0.5000"' in attr
    assert 'data-buffered-pct="-2.0000"' in attr


def test_buffer_data_attr_inactive_empty():
    assert _buffer_data_attr(-1.5, 0) == ""


def _make_single_window_result() -> dict:
    return {
        "ticker": "NDX",
        "last_trading_day": "2026-04-24",
        "previous_close": 20000.00,
        "lookback_trading_days": 60,
        "lookback_days": 60,
        "percentiles": [75, 90],
        "window": 1,
        "min_direction_days": 5,
        "when_up_day_count": 30,
        "when_down_day_count": 30,
        "when_up": {
            "day_count": 30,
            "pct": {"p75": 0.50, "p90": 1.00},
            "price": {"p75": 20100.00, "p90": 20200.00},
        },
        "when_down": {
            "day_count": 30,
            "pct": {"p75": -0.50, "p90": -1.00},
            "price": {"p75": 19900.00, "p90": 19800.00},
        },
    }


def test_format_as_html_no_buffer_omits_addendum():
    html = format_as_html([_make_single_window_result()])
    assert "buffer-addendum" not in html
    assert 'data-buffer="' not in html


def test_format_as_html_with_buffer_emits_addendum_and_data_attrs():
    html = format_as_html([_make_single_window_result()], buffer=0.5)
    assert "buffer-addendum" in html
    # DOWN cell: pct=-0.50 + buffer 0.5 → -1.00%, price=20000*(1-0.01)=19800.00
    assert "-1.00%" in html
    assert "$19,800.00" in html
    # UP cell: pct=+0.50 + 0.5 → +1.00%, price=20000*(1+0.01)=20200.00
    assert "+1.00%" in html
    assert "$20,200.00" in html
    # data attributes for live-price JS reapplication
    assert 'data-buffer="0.5000"' in html
    assert 'data-buffered-pct="-1.0000"' in html
    assert 'data-buffered-pct="1.0000"' in html


def _make_multi_window_result() -> dict:
    return {
        "ticker": "NDX",
        "metadata": {
            "last_trading_day": "2026-04-24",
            "previous_close": 20000.00,
            "lookback_trading_days": 60,
            "lookback_days": 60,
            "percentiles": [75, 90],
            "window_list": [1],
            "skipped_windows": [],
        },
        "windows": {
            "1": {
                "when_up_day_count": 30,
                "when_down_day_count": 30,
                "when_up": {
                    "day_count": 30,
                    "pct": {"p75": 0.50, "p90": 1.00},
                    "price": {"p75": 20100.00, "p90": 20200.00},
                },
                "when_down": {
                    "day_count": 30,
                    "pct": {"p75": -0.50, "p90": -1.00},
                    "price": {"p75": 19900.00, "p90": 19800.00},
                },
            }
        },
        "recommended": {},
    }


def test_format_multi_window_as_html_with_buffer_emits_addendum():
    html = format_multi_window_as_html(_make_multi_window_result(), buffer=0.5)
    assert "buffer-addendum" in html
    assert "-1.00%" in html  # DOWN p75: -0.5 - 0.5
    assert "+1.00%" in html  # UP p75: +0.5 + 0.5
    assert 'data-buffered-pct="-1.0000"' in html


def test_format_multi_window_as_html_without_buffer():
    html = format_multi_window_as_html(_make_multi_window_result())
    assert "buffer-addendum" not in html


def _make_hourly_data() -> dict:
    return {
        "ticker": "NDX",
        "previous_close": 20000.00,
        "percentiles": [75, 90],
        "slots": {
            "10:00": {
                "label_et": "10:00 AM ET",
                "when_down_day_count": 25,
                "when_up_day_count": 25,
                "when_down": {
                    "day_count": 25,
                    "pct": {"p75": -0.50, "p90": -1.00},
                    "price": {"p75": 19900.00, "p90": 19800.00},
                },
                "when_up": {
                    "day_count": 25,
                    "pct": {"p75": 0.50, "p90": 1.00},
                    "price": {"p75": 20100.00, "p90": 20200.00},
                },
            }
        },
        "slots_primary": {},
        "slots_15min": {},
        "slots_10min": {},
        "slots_5min": {},
        "has_fine_data": False,
        "recommended": {},
    }


def test_format_hourly_moves_as_html_with_buffer():
    html = format_hourly_moves_as_html(_make_hourly_data(), buffer=0.5)
    assert "buffer-addendum" in html
    assert "$19,800.00" in html  # DOWN p75 buffered: 20000 * (1 - 0.01)
    assert "$20,200.00" in html  # UP p75 buffered: 20000 * (1 + 0.01)


def test_format_hourly_moves_as_html_without_buffer():
    html = format_hourly_moves_as_html(_make_hourly_data())
    assert "buffer-addendum" not in html


def test_db_server_js_handles_buffered_pct_attribute():
    """The injected live-price JS must reapply the buffer addendum after replacing
    the price cell — otherwise live-price updates would wipe the addendum."""
    src = (_REPO / "db_server.py").read_text()
    assert "el.dataset.bufferedPct" in src, (
        "applySectionPrices must read el.dataset.bufferedPct so live-price "
        "updates preserve the buffer addendum"
    )
    assert "buffer-addendum" in src, (
        "JS must rebuild the buffer-addendum div when updating price cells"
    )


def test_hourly_ref_close_pins_to_prev_close_in_js():
    """The 'Reference Close' header in the hourly section must always show the
    previous day's close, never the live price. Live price is shown in the
    price-basis line below."""
    src = (_REPO / "db_server.py").read_text()
    # Find the applySectionPrices function body
    start = src.find("function applySectionPrices(")
    assert start != -1, "applySectionPrices function not found"
    # Look at a reasonable chunk of the function for our markers
    body = src[start:start + 4500]
    assert "if (section === 'hourly')" in body, (
        "ref-close branch must early-return for hourly so live price never "
        "overwrites the Reference Close header"
    )
    assert "<strong>Reference Close:</strong> ' + fmtPrice(prevClose)" in body, (
        "hourly ref-close must render fmtPrice(prevClose), not fmtPrice(price)"
    )


def test_price_basis_carries_data_prev_close():
    """price-basis spans must carry data-prev-close so the live-price line
    can render the diff vs previous close."""
    from common import range_percentiles_formatter as fmt
    src = Path(fmt.__file__).read_text()
    basis_spans = src.count('class="price-basis"')
    with_attr = sum(
        1 for line in src.splitlines()
        if 'class="price-basis"' in line
        and 'data-prev-close="{_raw_prev_close(prev_close)}"' in line
    )
    assert basis_spans > 0, "no price-basis spans found"
    assert with_attr == basis_spans, (
        f"mismatch: {basis_spans} price-basis spans but only {with_attr} carry "
        "data-prev-close — every price-basis must expose the attribute so the "
        "live-price diff badge can render"
    )


def test_js_renders_diff_badge_on_live_price_basis():
    """The price-basis live line must render a diff vs previous close badge
    (▲/▼ +/-x.xx% vs prev $...) next to the live price."""
    src = (_REPO / "db_server.py").read_text()
    start = src.find("function applySectionPrices(")
    assert start != -1, "applySectionPrices function not found"
    body = src[start:start + 4500]
    # The live-price branch of the price-basis renderer must compute diff
    # against el.dataset.prevClose and emit a "vs prev" badge.
    assert "basisDiffBadge" in body, (
        "applySectionPrices must build a diff badge for the live price-basis line"
    )
    assert "live price: ' + fmt + '</strong>' + liveBadge + basisDiffBadge" in body, (
        "live price-basis innerHTML must append the diff badge after the LIVE badge"
    )


def test_diff_badge_includes_dollar_amount_moved():
    """The diff badge (both ref-close and price-basis) must show the dollar
    amount moved alongside the percentage — e.g. '▼ -$2.67 (-0.04%) vs prev $...'.
    The amount comes from `price - prevClose` and is rendered with thousand
    separators to two decimals."""
    src = (_REPO / "db_server.py").read_text()
    start = src.find("function applySectionPrices(")
    assert start != -1, "applySectionPrices function not found"
    body = src[start:start + 5500]
    # Both diff-badge code paths must compute the absolute dollar amount and
    # render it before the percentage.
    assert body.count("var diffAmt = price - prevClose;") >= 2, (
        "both diff badges (ref-close and price-basis) must compute diffAmt = price - prevClose"
    )
    assert body.count("Math.abs(diffAmt)") >= 2, (
        "both diff badges must render Math.abs(diffAmt) to avoid a double sign"
    )
    # Percentage must appear in parentheses, after the dollar amount.
    assert body.count("'$' + absAmt") >= 2, (
        "dollar amount must be rendered with a $ prefix"
    )
    assert body.count("' (' + pctSign + diffPct.toFixed(2) + '%)'") >= 2, (
        "percentage must be wrapped in parentheses after the dollar amount"
    )
