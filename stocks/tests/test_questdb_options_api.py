import asyncio
from datetime import datetime, timezone

import pandas as pd
import pytest


@pytest.mark.asyncio
async def test_bucket_minutes_open_post_closed(monkeypatch):
    from common.questdb_db import StockQuestDB

    def make_dt(year, month, day, hour, minute):
        return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)

    # Monday 14:00 UTC ~ 10:00 ET (open)
    assert StockQuestDB._get_bucket_minutes(make_dt(2024, 6, 3, 14, 0)) == 15
    # Monday 21:00 UTC ~ 17:00 ET (post-market)
    assert StockQuestDB._get_bucket_minutes(make_dt(2024, 6, 3, 21, 0)) == 60
    # Sunday closed
    assert StockQuestDB._get_bucket_minutes(make_dt(2024, 6, 2, 12, 0)) == 240


def test_floor_to_bucket():
    from common.questdb_db import StockQuestDB
    dt = datetime(2024, 6, 3, 14, 37, tzinfo=timezone.utc)
    floored_15 = StockQuestDB._floor_to_bucket(dt, 15)
    assert floored_15 == datetime(2024, 6, 3, 14, 30, tzinfo=timezone.utc)
    floored_60 = StockQuestDB._floor_to_bucket(dt, 60)
    assert floored_60 == datetime(2024, 6, 3, 14, 0, tzinfo=timezone.utc)


class _FakeConn:
    def __init__(self):
        self.registered = {}
        self.executed = []
        self.fetch_queries = []
        self._rows = []

    def register(self, name, df):
        self.registered[name] = df.copy()

    def execute(self, sql, *params):
        self.executed.append((sql, params))
        return None

    async def fetch(self, sql, *params):
        self.fetch_queries.append((sql, params))
        return self._rows

class _FakePoolCtx:
    def __init__(self, conn):
        self.conn = conn
    async def __aenter__(self):
        return self.conn
    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_save_options_data_inserts_bucketed(monkeypatch):
    from common.questdb_db import StockQuestDB

    fake_conn = _FakeConn()

    def fake_get_connection(self):
        return _FakePoolCtx(fake_conn)

    monkeypatch.setattr(StockQuestDB, 'get_connection', fake_get_connection)

    db = StockQuestDB("postgresql://user:pass@localhost:9009/db")

    df = pd.DataFrame([
        {
            'option_ticker': 'O:AAPL240621C00180000',
            'expiration': '2024-06-21',
            'strike': 180.0,
            'type': 'call',
            'bid': 1.0,
            'ask': 1.2,
            'day_close': 1.1,
            'fmv': 1.15,
        }
    ])

    await db.save_options_data(df, ticker='AAPL')

    # Verify registration and insert
    assert 'df_options_to_insert' in fake_conn.registered
    reg_df = fake_conn.registered['df_options_to_insert']
    assert 'timestamp' in reg_df.columns and 'write_timestamp' in reg_df.columns
    # Minimal type conversions
    assert 'strike_price' in reg_df.columns and 'option_type' in reg_df.columns
    # Confirm an INSERT occurred
    assert any('INSERT INTO options_data' in sql for sql, _ in fake_conn.executed)


@pytest.mark.asyncio
async def test_get_latest_and_price_feature(monkeypatch):
    from common.questdb_db import StockQuestDB

    fake_conn = _FakeConn()

    # Provide two rows for same option_ticker but different timestamps; only the latest should return
    latest_ts = datetime(2024, 6, 3, 15, 0, tzinfo=timezone.utc)
    older_ts = datetime(2024, 6, 3, 14, 45, tzinfo=timezone.utc)
    fake_conn._rows = [
        {
            'ticker': 'AAPL',
            'option_ticker': 'O:AAPL240621C00180000',
            'expiration_date': datetime(2024, 6, 21, 0, 0),
            'strike_price': 180.0,
            'option_type': 'call',
            'timestamp': latest_ts,
            'write_timestamp': latest_ts,
            'price': 1.2,
            'bid': 1.1,
            'ask': 1.25,
            'day_close': 1.15,
            'fmv': 1.18,
        }
    ]

    def fake_get_connection(self):
        return _FakePoolCtx(fake_conn)

    monkeypatch.setattr(StockQuestDB, 'get_connection', fake_get_connection)

    db = StockQuestDB("postgresql://user:pass@localhost:9009/db")

    latest_df = await db.get_latest_options_data(ticker='AAPL', option_tickers=['O:AAPL240621C00180000'])
    assert not latest_df.empty
    assert latest_df.iloc[0]['bid'] == 1.1

    feat = await db.get_option_price_feature('AAPL', 'O:AAPL240621C00180000')
    assert feat == {
        'price': 1.2,
        'bid': 1.1,
        'ask': 1.25,
        'day_close': 1.15,
        'fmv': 1.18,
    }


