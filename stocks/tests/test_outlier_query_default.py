"""Tests for the ?outliers=... endpoint convention.

The /range_percentiles endpoint defaults to FILTERED (winsorized) view —
extreme moves get capped at the Tukey fence (Q1 ± 1.5×IQR). To see the
raw, unfiltered tail, pass ?outliers=1 (or true/yes/on).

These tests cover the parsing logic without spinning up the full HTTP server.
"""

import pytest


def _parse_outliers(query_value: str | None) -> bool:
    """Mirror the endpoint's parsing logic (db_server.py).

    Returns True if outliers should be EXCLUDED (winsorized), False if RAW.
    """
    raw = (query_value if query_value is not None else "").strip().lower()
    return raw not in {"1", "true", "yes", "on"}


class TestDefault:
    """No param → exclude outliers (winsorized view)."""

    def test_no_param_filtered(self):
        assert _parse_outliers(None) is True

    def test_empty_string_filtered(self):
        assert _parse_outliers("") is True


class TestOptInRaw:
    """User opts INTO raw view with ?outliers=1/true/yes/on."""

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "Yes"])
    def test_on_values_show_raw(self, val):
        assert _parse_outliers(val) is False


class TestOtherValuesAreFiltered:
    """Anything not recognized as 'on' falls through to filtered (default)."""

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", "foo", "2", " "])
    def test_off_or_unknown_filtered(self, val):
        assert _parse_outliers(val) is True
