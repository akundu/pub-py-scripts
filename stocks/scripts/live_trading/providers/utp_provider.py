"""UTP data providers — fetch equity quotes and option chains from the UTP daemon.

The Universal Trade Platform daemon (live_trading/universal-trade-platform/)
maintains a persistent IBKR connection and exposes HTTP endpoints at
http://localhost:8000.  These providers query UTP for real-time quotes and
option chains, with a 2-minute TTL cache to respect IBKR rate limits.

Historical data (previous close, historical bars) delegates to the existing
CSV/QuestDB providers — UTP only serves current market data.
"""

import logging
import time as time_mod
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from scripts.backtesting.providers.base import DataProvider

logger = logging.getLogger(__name__)

# Default cache TTL: 2 minutes (advisor runs on 60s cycle)
DEFAULT_CACHE_TTL = 120


class _CacheMixin:
    """Simple time-based cache for HTTP responses."""

    def __init__(self):
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_ttl: int = DEFAULT_CACHE_TTL
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def _cache_get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            ts, data = self._cache[key]
            if time_mod.time() - ts < self._cache_ttl:
                self._cache_hits += 1
                return data
        self._cache_misses += 1
        return None

    def _cache_set(self, key: str, data: Any) -> None:
        self._cache[key] = (time_mod.time(), data)

    def _cache_age(self, key: str) -> Optional[float]:
        """Return age in seconds of a cached entry, or None if not cached."""
        if key in self._cache:
            ts, _ = self._cache[key]
            return time_mod.time() - ts
        return None

    @property
    def cache_stats(self) -> Dict[str, int]:
        return {"hits": self._cache_hits, "misses": self._cache_misses}


class UtpEquityProvider(DataProvider, _CacheMixin):
    """Equity provider that fetches current quotes from UTP/IBKR.

    For today's price: calls GET /market/quote/{ticker} on the UTP daemon.
    For previous close / historical bars: delegates to CSV + QuestDB providers.

    Config params:
        utp_base_url: UTP daemon base URL (default: http://localhost:8000)
        csv_dir: Path to equities_output (for historical, default: "equities_output")
        cache_ttl: Cache TTL in seconds (default: 120)
    """

    def __init__(self):
        _CacheMixin.__init__(self)
        self._session = None
        self._base_url: str = "http://localhost:8000"
        self._csv_provider = None
        self._realtime_provider = None

    def initialize(self, config: Dict[str, Any]) -> None:
        import requests
        self._session = requests.Session()
        self._session.headers["Accept"] = "application/json"
        self._base_url = config.get("utp_base_url", "http://localhost:8000").rstrip("/")
        self._cache_ttl = config.get("cache_ttl", DEFAULT_CACHE_TTL)

        # CSV provider for historical data
        from scripts.backtesting.providers.csv_equity_provider import CSVEquityProvider
        self._csv_provider = CSVEquityProvider()
        self._csv_provider.initialize({
            "csv_dir": config.get("csv_dir", "equities_output"),
        })

        # Try to initialize realtime provider for QuestDB fallback
        try:
            from scripts.live_trading.providers.realtime_equity import RealtimeEquityProvider
            self._realtime_provider = RealtimeEquityProvider()
            self._realtime_provider.initialize({
                "csv_dir": config.get("csv_dir", "equities_output"),
            })
        except Exception as e:
            logger.debug(f"Could not initialize realtime provider (QuestDB): {e}")

    def get_available_dates(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[date]:
        dates = self._csv_provider.get_available_dates(ticker, start_date, end_date)
        today = date.today()
        if (start_date is None or today >= start_date) and \
           (end_date is None or today <= end_date):
            if today not in dates:
                dates.append(today)
                dates.sort()
        return dates

    def get_bars(
        self,
        ticker: str,
        trading_date: date,
        interval: str = "5min",
    ) -> pd.DataFrame:
        """Get OHLCV bars. Uses UTP quote for today, CSV for historical."""
        if trading_date == date.today():
            bars = self._get_utp_quote(ticker)
            if bars is not None and not bars.empty:
                return bars

        return self._csv_provider.get_bars(ticker, trading_date, interval)

    def get_previous_close(
        self,
        ticker: str,
        trading_date: date,
    ) -> Optional[float]:
        """Get previous close — delegates to CSV/QuestDB (UTP doesn't serve historical)."""
        if self._realtime_provider:
            prev = self._realtime_provider.get_previous_close(ticker, trading_date)
            if prev is not None:
                return prev
        return self._csv_provider.get_previous_close(ticker, trading_date)

    def get_options_chain(
        self,
        ticker: str,
        trading_date: date,
        dte_buckets: Optional[List[int]] = None,
    ) -> Optional[pd.DataFrame]:
        """Equity provider doesn't serve options."""
        return None

    def close(self) -> None:
        if self._session:
            self._session.close()
        if self._csv_provider:
            self._csv_provider.close()
        if self._realtime_provider:
            self._realtime_provider.close()

    def _get_utp_quote(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch current quote from UTP, return as single-row DataFrame."""
        cache_key = f"quote_{ticker}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            resp = self._session.get(
                f"{self._base_url}/market/quote/{ticker}",
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"UTP quote fetch failed for {ticker}: {e}")
            return None

        # Extract price from UTP response
        last = data.get("last") or data.get("price") or data.get("close")
        if last is None:
            logger.warning(f"UTP quote for {ticker} has no price field: {list(data.keys())}")
            return None

        last = float(last)
        now = datetime.now(timezone.utc)

        df = pd.DataFrame([{
            "timestamp": now,
            "open": last,
            "high": last,
            "low": last,
            "close": last,
            "volume": int(data.get("volume", 0)),
        }])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        self._cache_set(cache_key, df)
        return df

    @staticmethod
    def check_connection(base_url: str = "http://localhost:8000") -> bool:
        """Check if the UTP daemon is reachable by fetching a quote."""
        import requests
        try:
            resp = requests.get(
                f"{base_url.rstrip('/')}/market/quote/SPY",
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False


class UtpOptionsProvider(DataProvider, _CacheMixin):
    """Options provider that fetches chains from UTP/IBKR.

    For each DTE bucket, fetches expirations from UTP, then retrieves the
    option chain filtered by strike range around current price.

    UTP API contract:
    - GET /market/options/{sym}?list_expirations=true → {"expirations": ["YYYYMMDD", ...]}
    - GET /market/options/{sym}?expiration=YYYYMMDD&strike_min=X&strike_max=Y
        → {"quotes": {"call": [{strike, bid, ask, ...}], "put": [...]}}

    Config params:
        utp_base_url: UTP daemon base URL (default: http://localhost:8000)
        dte_buckets: List of DTE values to fetch (default: [0..11])
        strike_range_pct: Percentage of price for strike range (default: 0.05)
        cache_ttl: Cache TTL in seconds (default: 120)
    """

    def __init__(self):
        _CacheMixin.__init__(self)
        self._session = None
        self._base_url: str = "http://localhost:8000"
        self._dte_buckets: List[int] = list(range(0, 12))
        self._strike_range_pct: float = 0.05
        # Expirations are session-cached (don't change intraday)
        self._expirations_cache: Dict[str, List[str]] = {}
        # Current price per ticker (for strike range calc)
        self._current_prices: Dict[str, float] = {}

    def initialize(self, config: Dict[str, Any]) -> None:
        import requests
        self._session = requests.Session()
        self._session.headers["Accept"] = "application/json"
        self._base_url = config.get("utp_base_url", "http://localhost:8000").rstrip("/")
        self._dte_buckets = config.get("dte_buckets", list(range(0, 12)))
        self._strike_range_pct = config.get("strike_range_pct", 0.05)
        self._cache_ttl = config.get("cache_ttl", DEFAULT_CACHE_TTL)

    def set_current_price(self, ticker: str, price: float) -> None:
        """Set the current price for a ticker (used for strike range filtering)."""
        self._current_prices[ticker] = price

    def get_available_dates(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[date]:
        """UTP only serves today's data."""
        today = date.today()
        if (start_date is None or today >= start_date) and \
           (end_date is None or today <= end_date):
            return [today]
        return []

    def get_bars(
        self,
        ticker: str,
        trading_date: date,
        interval: str = "5min",
    ) -> pd.DataFrame:
        """Options provider doesn't serve equity bars."""
        return pd.DataFrame()

    def get_options_chain(
        self,
        ticker: str,
        trading_date: date,
        dte_buckets: Optional[List[int]] = None,
    ) -> Optional[pd.DataFrame]:
        """Fetch option chains from UTP for the given DTE buckets."""
        buckets = dte_buckets or self._dte_buckets
        today = trading_date or date.today()

        # Get available expirations
        expirations = self._get_expirations(ticker)
        if not expirations:
            return None

        # Filter expirations to those matching our DTE buckets.
        # UTP returns YYYYMMDD format; also handle YYYY-MM-DD for flexibility.
        target_expirations = []
        for exp_str in expirations:
            exp_date = self._parse_expiration(exp_str)
            if exp_date is None:
                continue
            dte = (exp_date - today).days
            if dte in buckets:
                # Keep the raw string for passing back to UTP
                target_expirations.append((exp_str, exp_date, dte))

        if not target_expirations:
            logger.debug(f"No expirations match DTE buckets {buckets} for {ticker}")
            return None

        # Fetch chain for each matching expiration
        all_frames = []
        for exp_str, exp_date, dte in target_expirations:
            chain_df = self._get_chain_for_expiration(ticker, exp_str, exp_date, dte)
            if chain_df is not None and not chain_df.empty:
                all_frames.append(chain_df)

        if not all_frames:
            return None

        result = pd.concat(all_frames, ignore_index=True)
        return result

    def close(self) -> None:
        if self._session:
            self._session.close()
        self._expirations_cache.clear()
        self._cache.clear()

    @staticmethod
    def _parse_expiration(exp_str: str) -> Optional[date]:
        """Parse expiration string in YYYYMMDD or YYYY-MM-DD format."""
        for fmt in ("%Y%m%d", "%Y-%m-%d"):
            try:
                return datetime.strptime(exp_str, fmt).date()
            except (ValueError, TypeError):
                continue
        return None

    def _get_expirations(self, ticker: str) -> List[str]:
        """Fetch available expirations for a ticker (session-cached)."""
        if ticker in self._expirations_cache:
            return self._expirations_cache[ticker]

        try:
            resp = self._session.get(
                f"{self._base_url}/market/options/{ticker}",
                params={"list_expirations": "true"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"UTP expirations fetch failed for {ticker}: {e}")
            return []

        if isinstance(data, list):
            expirations = data
        elif isinstance(data, dict):
            expirations = data.get("expirations", [])
        else:
            expirations = []

        self._expirations_cache[ticker] = expirations
        return expirations

    def _get_chain_for_expiration(
        self, ticker: str, expiration_raw: str, exp_date: date, dte: int,
    ) -> Optional[pd.DataFrame]:
        """Fetch option chain for a single expiration, with caching.

        Uses strike_min/strike_max computed from current price to avoid
        fetching the entire chain (which can timeout on IBKR).
        """
        cache_key = f"chain_{ticker}_{expiration_raw}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # Build params — pass expiration in the raw format UTP gave us
        params: Dict[str, Any] = {"expiration": expiration_raw}

        # Compute strike range from current price if available
        price = self._current_prices.get(ticker)
        if price and price > 0:
            margin = price * self._strike_range_pct
            params["strike_min"] = price - margin
            params["strike_max"] = price + margin

        try:
            resp = self._session.get(
                f"{self._base_url}/market/options/{ticker}",
                params=params,
                timeout=15,  # 15s timeout; skip this expiration if IBKR is slow
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"UTP chain fetch failed for {ticker} exp={expiration_raw}: {e}")
            return None

        exp_iso = exp_date.isoformat()  # Normalize to YYYY-MM-DD for our DataFrame
        df = self._parse_chain_response(data, exp_iso, dte)
        if df is not None:
            self._cache_set(cache_key, df)
        return df

    def _parse_chain_response(
        self, data: Any, expiration: str, dte: int,
    ) -> Optional[pd.DataFrame]:
        """Parse UTP option chain JSON into a DataFrame.

        UTP response format (with expiration param):
        {
          "symbol": "NDX",
          "chain": {"expirations": [...], "strikes": [...]},
          "quotes": {
            "call": [{"strike": 20000, "bid": 1.50, "ask": 1.70, ...}, ...],
            "put":  [{"strike": 20000, "bid": 0.45, "ask": 0.55, ...}, ...]
          }
        }
        """
        if not isinstance(data, dict):
            return None

        quotes = data.get("quotes", {})
        if not quotes:
            # Fallback: maybe the data is in "options" or "chain" key
            options_list = data.get("options", [])
            if options_list:
                return self._parse_flat_options(options_list, expiration, dte)
            return None

        rows = []
        for opt_type_key, quote_list in quotes.items():
            # opt_type_key is "call" or "put"
            if isinstance(quote_list, dict) and "error" in quote_list:
                logger.warning(f"UTP quote error for {opt_type_key}: {quote_list['error']}")
                continue
            if not isinstance(quote_list, list):
                continue
            for opt in quote_list:
                if not isinstance(opt, dict):
                    continue
                bid = float(opt.get("bid", 0))
                ask = float(opt.get("ask", 0))
                rows.append({
                    "strike": float(opt.get("strike", 0)),
                    "type": opt_type_key.lower(),
                    "dte": dte,
                    "expiration": expiration,
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2 if bid and ask else 0,
                    "volume": int(opt.get("volume", 0)),
                    "open_interest": int(opt.get("open_interest", opt.get("oi", 0))),
                })

        if not rows:
            return None

        return pd.DataFrame(rows)

    def _parse_flat_options(
        self, options_list: list, expiration: str, dte: int,
    ) -> Optional[pd.DataFrame]:
        """Parse a flat list of option dicts (fallback format)."""
        rows = []
        for opt in options_list:
            if not isinstance(opt, dict):
                continue
            bid = float(opt.get("bid", 0))
            ask = float(opt.get("ask", 0))
            rows.append({
                "strike": float(opt.get("strike", 0)),
                "type": str(opt.get("type", opt.get("option_type", ""))).lower(),
                "dte": dte,
                "expiration": expiration,
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2 if bid and ask else 0,
                "volume": int(opt.get("volume", 0)),
                "open_interest": int(opt.get("open_interest", opt.get("oi", 0))),
            })
        return pd.DataFrame(rows) if rows else None
