import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def fake_polygon_module(monkeypatch):
    """Provide a minimal fake polygon module so tests don't require polygon-api-client."""
    if 'polygon' in sys.modules:
        yield
        return

    fake_mod = types.ModuleType('polygon')

    class FakeRESTClient:
        def __init__(self, api_key):
            self.api_key = api_key

        # Will be monkeypatched per-test
        def get_aggs(self, *args, **kwargs):
            return []

        def list_options_contracts(self, *args, **kwargs):
            return iter([])

        def get_snapshot_option(self, *args, **kwargs):
            return None

    fake_mod.RESTClient = FakeRESTClient
    sys.modules['polygon'] = fake_mod
    try:
        yield
    finally:
        # Keep it installed for duration of test session to avoid import churn
        pass


def _import_script_module():
    # Import after fake module registered
    from scripts.historical_stock_options import HistoricalDataFetcher
    return HistoricalDataFetcher


def test_compute_market_transition_times_basic():
    HistoricalDataFetcher = _import_script_module()
    # Use a Monday 10:00 ET (market open)
    # 10:00 ET => 14:00 UTC during DST; construct via timezone-aware UTC
    sample_utc = datetime(2024, 6, 3, 14, 0, 0, tzinfo=timezone.utc)
    seconds_to_open, seconds_to_close = HistoricalDataFetcher._compute_market_transition_times(sample_utc, "America/New_York")
    assert seconds_to_open is not None and seconds_to_open >= 0
    assert seconds_to_close is not None and seconds_to_close > 0


def test_is_market_open_true_false():
    HistoricalDataFetcher = _import_script_module()
    # Monday 10:00 local (assumes function uses local time; we pass an aware ET time)
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    open_dt = datetime(2024, 6, 3, 10, 0, 0, tzinfo=et)
    closed_dt = datetime(2024, 6, 3, 8, 0, 0, tzinfo=et)
    assert HistoricalDataFetcher._is_market_open(open_dt) is True
    assert HistoricalDataFetcher._is_market_open(closed_dt) is False


def test_csv_cache_save_and_load(tmp_path, monkeypatch):
    HistoricalDataFetcher = _import_script_module()
    # Stub client out (we won't call it)
    fetcher = HistoricalDataFetcher(api_key="dummy", data_dir=str(tmp_path), quiet=True)

    symbol = "AAPL"
    options_data = {
        "contracts": [
            {
                "ticker": "O:AAPL240621C00180000",
                "type": "call",
                "strike": 180.0,
                "expiration": "2024-06-21",
                "bid": 1.23,
                "ask": 1.30,
                "day_close": 1.25,
                "fmv": 1.27,
                "delta": 0.45,
                "gamma": 0.02,
                "theta": -0.01,
                "vega": 0.10,
            },
            {
                "ticker": "O:AAPL240621P00180000",
                "type": "put",
                "strike": 180.0,
                "expiration": "2024-06-21",
                "bid": 1.10,
                "ask": 1.20,
                "day_close": 1.15,
                "fmv": 1.18,
                "delta": -0.55,
                "gamma": 0.02,
                "theta": -0.01,
                "vega": 0.11,
            },
        ]
    }

    # Save then load latest for same expiration
    fetcher._save_options_to_csv(symbol, options_data)
    loaded = fetcher._load_options_from_csv(symbol, "2024-06-21")
    assert isinstance(loaded, list) and len(loaded) == 2
    # Ensure basic fields persisted
    call = next(c for c in loaded if c["type"] == "call")
    put = next(c for c in loaded if c["type"] == "put")
    assert call["ticker"].startswith("O:AAPL")
    assert pytest.approx(call["bid"], rel=1e-6) == 1.23
    assert pytest.approx(put["ask"], rel=1e-6) == 1.20


def test_format_output_selection_logic():
    HistoricalDataFetcher = _import_script_module()
    fetcher = HistoricalDataFetcher(api_key="dummy", data_dir="data", quiet=True)

    symbol = "AAPL"
    target_date = "2024-06-21"

    stock_result = {
        "success": True,
        "data": {
            "target_date": target_date,
            "trading_date": target_date,
            "open": 190.0,
            "high": 195.0,
            "low": 189.0,
            "close": 192.0,
            "volume": 100_000_000,
        },
    }

    # Build contracts around the close price 192
    def c(ticker, typ, strike):
        return {
            "ticker": ticker,
            "type": typ,
            "strike": float(strike),
            "expiration": target_date,
            "bid": 1.0,
            "ask": 1.1,
            "day_close": 1.05,
            "fmv": 1.06,
            "delta": 0.3 if typ == 'call' else -0.3,
            "gamma": 0.02,
            "theta": -0.01,
            "vega": 0.1,
        }

    contracts = [
        c("OC1", "call", 190),
        c("OC2", "call", 192),
        c("OC3", "call", 195),
        c("OP1", "put", 190),
        c("OP2", "put", 192),
        c("OP3", "put", 195),
    ]

    options_result = {"success": True, "data": {"contracts": contracts}}

    rendered = fetcher.format_output(
        symbol=symbol,
        target_date=target_date,
        stock_result=stock_result,
        options_result=options_result,
        option_type="all",
        strike_range_percent=None,
        options_per_expiry=1,
        max_days_to_expiry=None,
    )

    # With options_per_expiry=1 and ATM 192, we expect 1 call just below/at and 1 above,
    # and similarly for puts. Presence check in rendered text is sufficient here.
    assert "OC2" in rendered or "OC1" in rendered
    assert "OC3" in rendered
    assert "OP2" in rendered or "OP1" in rendered
    assert "OP3" in rendered


def test_get_stock_price_for_date_uses_prev_trading_day(monkeypatch):
    HistoricalDataFetcher = _import_script_module()

    class FakeBar:
        def __init__(self, ts_ms, o, h, l, c, v):
            self.timestamp = ts_ms
            self.open = o
            self.high = h
            self.low = l
            self.close = c
            self.volume = v

    def fake_get_aggs(self, ticker, multiplier, timespan, from_, to, adjusted, sort, limit):
        # Return one bar for 2024-06-05 regardless of request window
        dt = datetime(2024, 6, 5, 0, 0)
        return [FakeBar(int(dt.timestamp() * 1000), 1, 2, 0.5, 1.5, 123)]

    # Instantiate and monkeypatch client method
    fetcher = HistoricalDataFetcher(api_key="dummy", data_dir="data", quiet=True)
    monkeypatch.setattr(fetcher.client, "get_aggs", fake_get_aggs.__get__(fetcher.client))

    out = pytest.run(async_fn=fetcher.get_stock_price_for_date("AAPL", "2024-06-07")) if hasattr(pytest, 'run') else None
    # Fallback simple runner for py3.11 without pytest.run helper
    if out is None:
        import asyncio
        out = asyncio.get_event_loop().run_until_complete(fetcher.get_stock_price_for_date("AAPL", "2024-06-07"))

    assert out["success"] is True
    assert out["data"]["trading_date"] == "2024-06-05"


def test_get_active_options_for_date_filters_and_snapshot(monkeypatch):
    HistoricalDataFetcher = _import_script_module()

    class Contract:
        def __init__(self, ticker, contract_type, strike_price, expiration_date):
            self.ticker = ticker
            self.contract_type = contract_type
            self.strike_price = strike_price
            self.expiration_date = expiration_date

    class LastQuote:
        def __init__(self, bid, ask):
            self.bid = bid
            self.ask = ask

    class Day:
        def __init__(self, close):
            self.close = close

    class Greeks:
        def __init__(self, delta, gamma, theta, vega):
            self.delta = delta
            self.gamma = gamma
            self.theta = theta
            self.vega = vega

    class Snapshot:
        def __init__(self, bid, ask, close, delta):
            self.last_quote = LastQuote(bid, ask)
            self.last_trade = types.SimpleNamespace(price=close)
            self.day = Day(close)
            self.fair_market_value = types.SimpleNamespace(value=(bid + ask) / 2.0)
            self.greeks = Greeks(delta, 0.02, -0.01, 0.1)

    # Fake contract stream (both calls and puts)
    contracts = [
        Contract("OC", "call", 100.0, "2024-06-21"),
        Contract("OP", "put", 100.0, "2024-06-21"),
        Contract("OC_far", "call", 150.0, "2024-06-21"),
    ]

    def fake_list_options_contracts(self, **kwargs):
        return iter(contracts)

    def fake_get_snapshot_option(self, underlying, option_ticker):
        if option_ticker == "OC_far":
            raise RuntimeError("simulate snapshot error for fallback path")
        return Snapshot(1.0, 1.2, 1.1, 0.4)

    def fake_get_aggs(self, ticker, multiplier, timespan, from_, to, adjusted, sort, limit):
        # Used only for fallback of OC_far
        class Bar:
            def __init__(self, close):
                self.close = close
        return [Bar(0.9)]

    fetcher = HistoricalDataFetcher(api_key="dummy", data_dir="data", quiet=True)
    monkeypatch.setattr(fetcher.client, "list_options_contracts", fake_list_options_contracts.__get__(fetcher.client))
    monkeypatch.setattr(fetcher.client, "get_snapshot_option", fake_get_snapshot_option.__get__(fetcher.client))
    monkeypatch.setattr(fetcher.client, "get_aggs", fake_get_aggs.__get__(fetcher.client))

    async def run():
        res = await fetcher.get_active_options_for_date(
            symbol="AAPL",
            target_date_str="2024-06-21",
            option_type="all",
            stock_close_price=100.0,
            strike_range_percent=20,  # Keep OC_far out by strike filter (150 > 120)
            max_days_to_expiry=None,
            include_expired=False,
            use_cache=False,
        )
        return res

    import asyncio
    result = asyncio.get_event_loop().run_until_complete(run())
    assert result["success"] is True
    contracts_out = result["data"]["contracts"]
    tickers = {c.get("ticker") for c in contracts_out}
    assert "OC" in tickers and "OP" in tickers
    assert "OC_far" not in tickers  # filtered by strike range
    oc = next(c for c in contracts_out if c["ticker"] == "OC")
    assert pytest.approx(oc["bid"], rel=1e-6) == 1.0
    assert pytest.approx(oc["ask"], rel=1e-6) == 1.2



def test_db_first_reads_without_hitting_api(monkeypatch):
    HistoricalDataFetcher = _import_script_module()

    # Build a fake DataFrame that mimics DB latest options schema
    import pandas as pd
    df = pd.DataFrame([
        {
            'option_ticker': 'O:TEST240621C00180000',
            'option_type': 'call',
            'strike_price': 180.0,
            'expiration_date': '2024-06-21',
            'bid': 1.23,
            'ask': 1.30,
            'day_close': 1.25,
            'fmv': 1.27,
            'delta': 0.45,
            'gamma': 0.02,
            'theta': -0.01,
            'vega': 0.10,
            'rho': None,
            'implied_volatility': None,
            'volume': None,
            'open_interest': None,
            'last_quote_timestamp': None,
        },
        {
            'option_ticker': 'O:TEST240621P00180000',
            'option_type': 'put',
            'strike_price': 180.0,
            'expiration_date': '2024-06-21',
            'bid': 1.10,
            'ask': 1.20,
            'day_close': 1.15,
            'fmv': 1.18,
            'delta': -0.55,
            'gamma': 0.02,
            'theta': -0.01,
            'vega': 0.11,
            'rho': None,
            'implied_volatility': None,
            'volume': None,
            'open_interest': None,
            'last_quote_timestamp': None,
        },
    ])

    # Fake DB object
    class FakeDB:
        async def get_latest_options_data(self, ticker: str):
            return df

    # Monkeypatch the get_stock_db used inside the script module
    import scripts.historical_stock_options as hso
    monkeypatch.setattr(hso, 'get_stock_db', lambda *args, **kwargs: FakeDB())

    # Instantiate fetcher and ensure API methods would raise if called
    fetcher = HistoricalDataFetcher(api_key="dummy", data_dir="data", quiet=True)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("API should not be called when DB has rows")

    monkeypatch.setattr(fetcher.client, 'list_options_contracts', fail_if_called)
    monkeypatch.setattr(fetcher.client, 'get_snapshot_option', fail_if_called)

    import asyncio
    async def run():
        res = await fetcher.get_active_options_for_date(
            symbol="TEST",
            target_date_str="2024-06-21",
            option_type="all",
            stock_close_price=180.0,
            strike_range_percent=None,
            max_days_to_expiry=None,
            include_expired=False,
            use_cache=False,
            save_to_csv=False,
            use_db=True,
            db_conn="questdb://dummy"
        )
        return res

    result = asyncio.get_event_loop().run_until_complete(run())
    assert result["success"] is True
    out = result["data"]["contracts"]
    assert isinstance(out, list) and len(out) == 2
    tickers = {c.get("ticker") for c in out}
    assert "O:TEST240621C00180000" in tickers
    assert "O:TEST240621P00180000" in tickers

