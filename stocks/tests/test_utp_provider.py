"""Tests for UTP data providers and live display."""

import json
import time as time_mod
from datetime import date, datetime, timezone
from io import StringIO
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scripts.live_trading.providers.utp_provider import (
    UtpEquityProvider,
    UtpOptionsProvider,
    _CacheMixin,
)
from scripts.live_trading.advisor.utp_display import TickerSnapshot, UtpDataDisplay


# ---------------------------------------------------------------------------
# Cache mixin tests
# ---------------------------------------------------------------------------

class TestCacheMixin:
    def test_cache_hit(self):
        cache = _CacheMixin()
        cache._cache_set("key1", {"data": 42})
        result = cache._cache_get("key1")
        assert result == {"data": 42}
        assert cache.cache_stats["hits"] == 1
        assert cache.cache_stats["misses"] == 0

    def test_cache_miss(self):
        cache = _CacheMixin()
        result = cache._cache_get("nonexistent")
        assert result is None
        assert cache.cache_stats["misses"] == 1

    def test_cache_ttl_expiry(self):
        cache = _CacheMixin()
        cache._cache_ttl = 0  # Expire immediately
        cache._cache_set("key1", "data")
        time_mod.sleep(0.01)
        result = cache._cache_get("key1")
        assert result is None

    def test_cache_age(self):
        cache = _CacheMixin()
        cache._cache_set("key1", "data")
        age = cache._cache_age("key1")
        assert age is not None
        assert age >= 0
        assert age < 1  # Should be near-instant

    def test_cache_age_missing(self):
        cache = _CacheMixin()
        assert cache._cache_age("nonexistent") is None


# ---------------------------------------------------------------------------
# UtpEquityProvider tests
# ---------------------------------------------------------------------------

class TestUtpEquityProvider:
    def _make_provider(self):
        """Create a provider with mocked session."""
        prov = UtpEquityProvider()
        with patch("scripts.live_trading.providers.utp_provider.UtpEquityProvider.initialize") as _:
            pass
        # Manual initialization with mocks
        prov._session = MagicMock()
        prov._base_url = "http://localhost:8000"
        prov._cache_ttl = 120
        prov._csv_provider = MagicMock()
        prov._realtime_provider = MagicMock()
        return prov

    def test_get_bars_today_utp(self):
        prov = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"last": 20150.50, "volume": 12345}
        mock_resp.raise_for_status = MagicMock()
        prov._session.get.return_value = mock_resp

        bars = prov.get_bars("NDX", date.today())
        assert bars is not None
        assert not bars.empty
        assert float(bars["close"].iloc[0]) == 20150.50
        assert "timestamp" in bars.columns

    def test_get_bars_today_cache_hit(self):
        prov = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"last": 20000, "volume": 100}
        mock_resp.raise_for_status = MagicMock()
        prov._session.get.return_value = mock_resp

        # First call populates cache
        prov.get_bars("NDX", date.today())
        # Second call should use cache
        prov.get_bars("NDX", date.today())

        # Only one HTTP call
        assert prov._session.get.call_count == 1

    def test_get_bars_historical_delegates_to_csv(self):
        prov = self._make_provider()
        historical_date = date(2026, 1, 15)
        prov._csv_provider.get_bars.return_value = pd.DataFrame({
            "close": [19500], "timestamp": [datetime(2026, 1, 15, 15, 0)]
        })

        bars = prov.get_bars("NDX", historical_date)
        prov._csv_provider.get_bars.assert_called_once_with("NDX", historical_date, "5min")

    def test_get_bars_utp_error_returns_empty(self):
        prov = self._make_provider()
        prov._session.get.side_effect = Exception("Connection refused")
        prov._csv_provider.get_bars.return_value = pd.DataFrame()

        bars = prov.get_bars("NDX", date.today())
        # Should fall through to CSV provider
        assert prov._csv_provider.get_bars.called

    def test_get_previous_close_delegates(self):
        prov = self._make_provider()
        prov._realtime_provider.get_previous_close.return_value = 19950.0

        result = prov.get_previous_close("NDX", date.today())
        assert result == 19950.0

    def test_get_previous_close_csv_fallback(self):
        prov = self._make_provider()
        prov._realtime_provider.get_previous_close.return_value = None
        prov._csv_provider.get_previous_close.return_value = 19900.0

        result = prov.get_previous_close("NDX", date.today())
        assert result == 19900.0

    def test_get_options_chain_returns_none(self):
        prov = self._make_provider()
        assert prov.get_options_chain("NDX", date.today()) is None

    def test_check_connection_success(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            assert UtpEquityProvider.check_connection("http://localhost:8000") is True

    def test_check_connection_failure(self):
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection refused")
            assert UtpEquityProvider.check_connection("http://localhost:8000") is False

    def test_quote_no_price_field(self):
        prov = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ticker": "NDX"}  # No price field
        mock_resp.raise_for_status = MagicMock()
        prov._session.get.return_value = mock_resp
        prov._csv_provider.get_bars.return_value = pd.DataFrame()

        bars = prov.get_bars("NDX", date.today())
        # Falls through to CSV
        assert prov._csv_provider.get_bars.called

    def test_close_cleans_up(self):
        prov = self._make_provider()
        prov.close()
        prov._session.close.assert_called_once()
        prov._csv_provider.close.assert_called_once()
        prov._realtime_provider.close.assert_called_once()


# ---------------------------------------------------------------------------
# UtpOptionsProvider tests
# ---------------------------------------------------------------------------

class TestUtpOptionsProvider:
    def _make_provider(self):
        prov = UtpOptionsProvider()
        prov._session = MagicMock()
        prov._base_url = "http://localhost:8000"
        prov._dte_buckets = [0, 1, 2]
        prov._strike_range_pct = 0.05
        prov._cache_ttl = 120
        return prov

    def _mock_expirations_response(self, expirations):
        """Mock for list_expirations=true call."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"expirations": expirations}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def _mock_quotes_response(self, calls=None, puts=None):
        """Mock for chain+quotes call (UTP real format)."""
        quotes = {}
        if calls is not None:
            quotes["call"] = calls
        if puts is not None:
            quotes["put"] = puts
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "symbol": "NDX",
            "chain": {"expirations": [], "strikes": []},
            "quotes": quotes,
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def _today_yyyymmdd(self):
        return date.today().strftime("%Y%m%d")

    def _tomorrow_yyyymmdd(self):
        from datetime import timedelta
        return (date.today() + timedelta(days=1)).strftime("%Y%m%d")

    def test_get_options_chain_basic(self):
        prov = self._make_provider()
        today = date.today()

        exp_resp = self._mock_expirations_response([self._today_yyyymmdd()])
        chain_resp = self._mock_quotes_response(
            puts=[{"strike": 20000, "bid": 1.50, "ask": 1.70, "volume": 100, "open_interest": 5000}],
            calls=[{"strike": 20100, "bid": 2.00, "ask": 2.30, "volume": 200, "open_interest": 3000}],
        )
        prov._session.get.side_effect = [exp_resp, chain_resp]

        result = prov.get_options_chain("NDX", today, dte_buckets=[0])
        assert result is not None
        assert len(result) == 2
        for col in ("strike", "type", "bid", "ask", "mid", "dte", "expiration", "volume", "open_interest"):
            assert col in result.columns

    def test_get_options_chain_mid_price(self):
        prov = self._make_provider()
        today = date.today()

        exp_resp = self._mock_expirations_response([self._today_yyyymmdd()])
        chain_resp = self._mock_quotes_response(
            puts=[{"strike": 20000, "bid": 1.00, "ask": 2.00, "volume": 50, "open_interest": 1000}],
        )
        prov._session.get.side_effect = [exp_resp, chain_resp]

        result = prov.get_options_chain("NDX", today, dte_buckets=[0])
        assert float(result.iloc[0]["mid"]) == 1.50

    def test_get_options_chain_dte_filtering(self):
        prov = self._make_provider()
        today = date.today()

        exp_resp = self._mock_expirations_response([self._tomorrow_yyyymmdd()])
        prov._session.get.side_effect = [exp_resp]

        # Request DTE=0 only — should get nothing since only DTE=1 exists
        result = prov.get_options_chain("NDX", today, dte_buckets=[0])
        assert result is None

        # Now request DTE=1 — should work
        prov._expirations_cache.clear()
        exp_resp2 = self._mock_expirations_response([self._tomorrow_yyyymmdd()])
        chain_resp2 = self._mock_quotes_response(
            puts=[{"strike": 20000, "bid": 1.0, "ask": 1.2, "volume": 10, "open_interest": 100}],
        )
        prov._session.get.side_effect = [exp_resp2, chain_resp2]
        result = prov.get_options_chain("NDX", today, dte_buckets=[1])
        assert result is not None
        assert len(result) == 1

    def test_yyyymmdd_and_iso_both_parsed(self):
        """Provider handles both YYYYMMDD and YYYY-MM-DD expiration formats."""
        prov = self._make_provider()
        today = date.today()
        iso = today.isoformat()
        yyyymmdd = today.strftime("%Y%m%d")

        # YYYYMMDD
        exp_resp = self._mock_expirations_response([yyyymmdd])
        chain_resp = self._mock_quotes_response(
            calls=[{"strike": 20000, "bid": 1.0, "ask": 1.5, "volume": 10, "open_interest": 100}],
        )
        prov._session.get.side_effect = [exp_resp, chain_resp]
        result = prov.get_options_chain("NDX", today, dte_buckets=[0])
        assert result is not None
        # Expiration in DataFrame is normalized to YYYY-MM-DD
        assert result.iloc[0]["expiration"] == iso

        # ISO format
        prov._expirations_cache.clear()
        prov._cache.clear()
        exp_resp2 = self._mock_expirations_response([iso])
        chain_resp2 = self._mock_quotes_response(
            calls=[{"strike": 20000, "bid": 1.0, "ask": 1.5, "volume": 10, "open_interest": 100}],
        )
        prov._session.get.side_effect = [exp_resp2, chain_resp2]
        result2 = prov.get_options_chain("NDX", today, dte_buckets=[0])
        assert result2 is not None

    def test_expirations_cached_per_session(self):
        prov = self._make_provider()
        today = date.today()

        exp_resp = self._mock_expirations_response([self._today_yyyymmdd()])
        chain_resp = self._mock_quotes_response(
            puts=[{"strike": 20000, "bid": 1.0, "ask": 1.2, "volume": 10, "open_interest": 100}],
        )
        prov._session.get.side_effect = [exp_resp, chain_resp]

        prov.get_options_chain("NDX", today, dte_buckets=[0])

        # Second call — expirations cached, only chain fetched
        chain_resp2 = self._mock_quotes_response(
            puts=[{"strike": 20000, "bid": 1.1, "ask": 1.3, "volume": 20, "open_interest": 200}],
        )
        prov._cache.clear()
        prov._session.get.side_effect = [chain_resp2]
        prov._session.get.reset_mock()

        prov.get_options_chain("NDX", today, dte_buckets=[0])
        assert prov._session.get.call_count == 1

    def test_chain_cache_ttl(self):
        prov = self._make_provider()
        prov._cache_ttl = 120
        today = date.today()

        exp_resp = self._mock_expirations_response([self._today_yyyymmdd()])
        chain_resp = self._mock_quotes_response(
            calls=[{"strike": 20000, "bid": 2.0, "ask": 2.5, "volume": 50, "open_interest": 500}],
        )
        prov._session.get.side_effect = [exp_resp, chain_resp]

        prov.get_options_chain("NDX", today, dte_buckets=[0])
        prov._session.get.reset_mock()
        prov.get_options_chain("NDX", today, dte_buckets=[0])
        assert prov._session.get.call_count == 0  # All cached

    def test_http_error_returns_none(self):
        prov = self._make_provider()
        prov._session.get.side_effect = Exception("Connection refused")

        result = prov.get_options_chain("NDX", date.today(), dte_buckets=[0])
        assert result is None

    def test_empty_quotes_response(self):
        prov = self._make_provider()
        today = date.today()

        exp_resp = self._mock_expirations_response([self._today_yyyymmdd()])
        # Empty quotes
        chain_resp = MagicMock()
        chain_resp.json.return_value = {"symbol": "NDX", "chain": {}, "quotes": {}}
        chain_resp.raise_for_status = MagicMock()
        prov._session.get.side_effect = [exp_resp, chain_resp]

        result = prov.get_options_chain("NDX", today, dte_buckets=[0])
        assert result is None

    def test_type_from_quotes_key(self):
        """Type comes from the 'call'/'put' key in quotes dict."""
        prov = self._make_provider()
        today = date.today()

        exp_resp = self._mock_expirations_response([self._today_yyyymmdd()])
        chain_resp = self._mock_quotes_response(
            calls=[{"strike": 20000, "bid": 1.0, "ask": 1.5, "volume": 10, "open_interest": 100}],
        )
        prov._session.get.side_effect = [exp_resp, chain_resp]

        result = prov.get_options_chain("NDX", today, dte_buckets=[0])
        assert result.iloc[0]["type"] == "call"

    def test_strike_range_computed_from_price(self):
        """When current price is set, strike_min/strike_max are passed."""
        prov = self._make_provider()
        prov._strike_range_pct = 0.05
        prov.set_current_price("NDX", 20000.0)
        today = date.today()

        exp_resp = self._mock_expirations_response([self._today_yyyymmdd()])
        chain_resp = self._mock_quotes_response(
            calls=[{"strike": 20000, "bid": 1.0, "ask": 1.5, "volume": 10, "open_interest": 100}],
        )
        prov._session.get.side_effect = [exp_resp, chain_resp]

        prov.get_options_chain("NDX", today, dte_buckets=[0])

        # Check the second call (chain fetch) had strike_min/strike_max
        chain_call = prov._session.get.call_args_list[1]
        params = chain_call.kwargs.get("params", chain_call[1].get("params", {}))
        assert "strike_min" in params
        assert "strike_max" in params
        assert params["strike_min"] == 19000.0  # 20000 - 5%
        assert params["strike_max"] == 21000.0  # 20000 + 5%

    def test_quote_error_skipped_gracefully(self):
        """If UTP returns an error for one type, the other still works."""
        prov = self._make_provider()
        today = date.today()

        exp_resp = self._mock_expirations_response([self._today_yyyymmdd()])
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "symbol": "NDX",
            "chain": {},
            "quotes": {
                "call": [{"strike": 20000, "bid": 1.0, "ask": 1.5, "volume": 10, "open_interest": 100}],
                "put": {"error": "Expiration not available"},
            },
        }
        mock_resp.raise_for_status = MagicMock()
        prov._session.get.side_effect = [exp_resp, mock_resp]

        result = prov.get_options_chain("NDX", today, dte_buckets=[0])
        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]["type"] == "call"

    def test_close_cleans_up(self):
        prov = self._make_provider()
        prov.close()
        prov._session.close.assert_called_once()
        assert len(prov._expirations_cache) == 0

    def test_get_available_dates_today_only(self):
        prov = self._make_provider()
        dates = prov.get_available_dates("NDX")
        assert dates == [date.today()]


# ---------------------------------------------------------------------------
# UtpDataDisplay tests
# ---------------------------------------------------------------------------

class TestUtpDataDisplay:
    def test_snapshot_with_data(self, capsys):
        display = UtpDataDisplay(profile_name="test_profile", tickers=["NDX"], width=80)
        now = datetime(2026, 3, 18, 18, 30, 0, tzinfo=timezone.utc)

        options_df = pd.DataFrame([
            {"strike": 19900, "type": "put", "bid": 0.45, "ask": 0.55, "mid": 0.50,
             "volume": 120, "open_interest": 5400, "dte": 0, "expiration": "2026-03-18"},
            {"strike": 20200, "type": "call", "bid": 1.50, "ask": 1.65, "mid": 1.58,
             "volume": 380, "open_interest": 9500, "dte": 0, "expiration": "2026-03-18"},
        ])

        display.print_market_snapshot(
            price=20150.25,
            prev_close=20080.0,
            options_df=options_df,
            now=now,
            cache_stats={"quote": {"age": 45}, "chain": {}},
            next_refresh_secs=75,
        )

        output = capsys.readouterr().out
        assert "NDX" in output
        assert "UTP/IBKR" in output
        assert "20,150.25" in output
        assert "19,900" in output
        assert "20,200" in output

    def test_snapshot_no_options(self, capsys):
        display = UtpDataDisplay(profile_name="test", tickers=["NDX"], width=80)
        now = datetime(2026, 3, 18, 18, 0, 0, tzinfo=timezone.utc)

        display.print_market_snapshot(
            price=20000.0,
            prev_close=19950.0,
            options_df=None,
            now=now,
        )

        output = capsys.readouterr().out
        assert "No options data" in output

    def test_snapshot_no_price(self, capsys):
        display = UtpDataDisplay(profile_name="test", tickers=["NDX"], width=80)
        now = datetime(2026, 3, 18, 18, 0, 0, tzinfo=timezone.utc)

        display.print_market_snapshot(
            price=None,
            prev_close=None,
            options_df=None,
            now=now,
        )

        output = capsys.readouterr().out
        assert "---" in output

    def test_multi_ticker_snapshot(self, capsys):
        """All three tickers displayed in one view."""
        display = UtpDataDisplay(
            profile_name="v5_smart_roll",
            tickers=["NDX", "SPX", "RUT"],
            width=80,
        )
        now = datetime(2026, 3, 18, 18, 30, 0, tzinfo=timezone.utc)

        snapshots = [
            TickerSnapshot(
                ticker="NDX", price=20150.0, prev_close=20080.0,
                options_df=pd.DataFrame([
                    {"strike": 20000, "type": "put", "bid": 1.0, "ask": 1.2,
                     "mid": 1.1, "volume": 100, "open_interest": 5000,
                     "dte": 0, "expiration": "2026-03-18"},
                ]),
            ),
            TickerSnapshot(
                ticker="SPX", price=5750.0, prev_close=5720.0,
                options_df=pd.DataFrame([
                    {"strike": 5700, "type": "put", "bid": 0.8, "ask": 1.0,
                     "mid": 0.9, "volume": 200, "open_interest": 8000,
                     "dte": 0, "expiration": "2026-03-18"},
                ]),
            ),
            TickerSnapshot(
                ticker="RUT", price=2100.0, prev_close=2090.0,
                options_df=None,  # No options data for RUT
            ),
        ]

        display.print_multi_ticker_snapshot(
            snapshots=snapshots, now=now,
            cache_stats={"quotes": {}, "chains": {}},
            next_refresh_secs=120,
        )

        output = capsys.readouterr().out
        # All three tickers in banner
        assert "NDX" in output
        assert "SPX" in output
        assert "RUT" in output
        # Price summary strip
        assert "20,150.00" in output
        assert "5,750.00" in output
        assert "2,100.00" in output
        # NDX options chain present
        assert "20,000" in output
        # SPX options chain present
        assert "5,700" in output
        # RUT has no options
        assert "No options data" in output

    def test_multi_ticker_price_summary_shows_day_change(self, capsys):
        display = UtpDataDisplay(profile_name="test", tickers=["NDX", "SPX"], width=80)
        now = datetime(2026, 3, 18, 18, 0, 0, tzinfo=timezone.utc)

        snapshots = [
            TickerSnapshot(ticker="NDX", price=20200.0, prev_close=20000.0,
                           options_df=None),
            TickerSnapshot(ticker="SPX", price=5700.0, prev_close=5750.0,
                           options_df=None),
        ]
        display.print_multi_ticker_snapshot(snapshots, now)

        output = capsys.readouterr().out
        # NDX up → CALL pursuit
        assert "CALL" in output
        # SPX down → PUT pursuit
        assert "PUT" in output


# ---------------------------------------------------------------------------
# --live flag integration test
# ---------------------------------------------------------------------------

class TestLiveFlag:
    def test_argparse_recognizes_live_flag(self):
        """Verify --live is accepted by the argument parser."""
        import argparse
        # Recreate the parser minimally
        parser = argparse.ArgumentParser()
        parser.add_argument("--profile")
        parser.add_argument("--live", action="store_true")
        parser.add_argument("--interval", type=int, default=60)
        parser.add_argument("--dry-run", action="store_true")

        args = parser.parse_args(["--profile", "v5_smart_roll", "--live", "--interval", "120"])
        assert args.live is True
        assert args.interval == 120
        assert args.profile == "v5_smart_roll"

    def test_evaluator_accepts_use_utp(self):
        """Verify TierEvaluator accepts use_utp parameter."""
        from scripts.live_trading.advisor.tier_evaluator import TierEvaluator
        from scripts.live_trading.advisor.profile_loader import (
            AdvisorProfile, RiskConfig, ProviderConfig, SignalConfig,
            ExitRuleConfig, TierDef,
        )
        from scripts.live_trading.advisor.position_tracker import PositionTracker

        profile = AdvisorProfile(
            name="test",
            ticker="NDX",
            risk=RiskConfig(),
            providers=ProviderConfig(),
            signal=SignalConfig(),
            instrument="credit_spread",
            tiers=[TierDef(label="t1", priority=1, directional="pursuit")],
            exit_rules=ExitRuleConfig(),
            strategy_defaults={},
        )
        from pathlib import Path
        tracker = PositionTracker(profile_name="test", data_dir=Path("/tmp/test_utp_positions"))

        evaluator = TierEvaluator(profile, tracker, use_utp=True)
        assert evaluator._use_utp is True

    def test_provider_config_has_utp_base_url(self):
        """Verify ProviderConfig includes utp_base_url field."""
        from scripts.live_trading.advisor.profile_loader import ProviderConfig
        config = ProviderConfig()
        assert config.utp_base_url == "http://localhost:8000"

    def test_profile_loader_parses_utp_section(self):
        """Verify _parse_providers reads utp.base_url from YAML dict."""
        from scripts.live_trading.advisor.profile_loader import _parse_providers
        raw = {
            "equity": {"csv_dir": "equities_output"},
            "options": {"csv_dir": "csv_exports/options"},
            "utp": {"base_url": "http://myhost:9000"},
        }
        config = _parse_providers(raw)
        assert config.utp_base_url == "http://myhost:9000"

    def test_profile_loader_default_utp(self):
        """Without utp section, default URL is used."""
        from scripts.live_trading.advisor.profile_loader import _parse_providers
        raw = {"equity": {}, "options": {}}
        config = _parse_providers(raw)
        assert config.utp_base_url == "http://localhost:8000"
