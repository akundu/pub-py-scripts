import multiprocessing
from multiprocessing import Queue
from typing import Any, Callable, Dict, Tuple, Optional, List
import math
import sys

# Try to import scipy for Black-Scholes
try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    norm = None

# Try to import redis for refresh deduplication
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# Try to import pandas for timestamp functions
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


def _iteration_wrapper(
    queue: Queue,
    target: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> None:
    """
    Internal wrapper run inside the forked process.

    It executes the target callable and communicates the outcome back to the
    parent process through a multiprocessing.Queue.  The result is always a
    dictionary with at least a ``status`` field of either ``"ok"`` or
    ``"error"``.
    """
    try:
        result = target(*args, **kwargs)
        queue.put({"status": "ok", "result": result})
    except Exception as exc:  # pragma: no cover - defensive, should rarely happen
        import traceback

        queue.put(
            {
                "status": "error",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )


def run_iteration_in_subprocess(
    target: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Execute ``target`` in a forked subprocess and wait for it to finish.

    Args:
        target: Callable to execute inside the subprocess.
        *args: Positional arguments for ``target``.
        **kwargs: Keyword arguments for ``target``.

    Returns:
        Dictionary containing:
            - ``status``: ``"ok"`` when the call succeeded, otherwise ``"error"``.
            - ``result``: Value returned by ``target`` (when successful).
            - ``error`` / ``traceback``: Present when an exception occurred.
            - ``exitcode``: Exit code of the subprocess.
    """
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_iteration_wrapper,
        args=(queue, target, args, kwargs),
    )
    process.start()
    process.join()

    payload: Dict[str, Any]
    if not queue.empty():
        payload = queue.get()
    else:
        # No payload was returned – treat non-zero exit codes as errors.
        payload = {"status": "ok", "result": None}

    payload["exitcode"] = process.exitcode
    if payload["status"] == "ok" and process.exitcode not in (0, None):
        payload = {
            "status": "error",
            "error": f"Child process exited with code {process.exitcode}",
            "result": payload.get("result"),
            "exitcode": process.exitcode,
        }
    return payload


# ============================================================================
# Black-Scholes Option Pricing
# ============================================================================

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate Black-Scholes call option price.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate (annual)
        sigma: Implied volatility (annual)
    
    Returns:
        Call option price
    """
    if T <= 0:
        # Option expired, intrinsic value only
        return max(S - K, 0)
    
    if sigma <= 0:
        # No volatility, return intrinsic value
        return max(S - K, 0)
    
    # Calculate d1 and d2
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    # Calculate cumulative normal distribution
    if SCIPY_AVAILABLE and norm is not None:
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
    else:
        # Fallback to basic normal CDF approximation
        def norm_cdf(x):
            """Approximate cumulative normal distribution."""
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        N_d1 = norm_cdf(d1)
        N_d2 = norm_cdf(d2)
    
    # Black-Scholes formula for call option
    call_price = S * N_d1 - K * math.exp(-r * T) * N_d2
    
    return max(call_price, 0)  # Ensure non-negative


# ============================================================================
# Redis Helper Functions for Refresh Deduplication
# ============================================================================

def get_redis_client_for_refresh(redis_url: Optional[str]) -> Optional[Any]:
    """Get a synchronous Redis client for refresh deduplication."""
    if not REDIS_AVAILABLE or not redis_url:
        return None
    try:
        return redis.from_url(redis_url, decode_responses=True)
    except Exception:
        return None


def check_redis_refresh_pending(redis_client: Optional[Any], ticker: str) -> bool:
    """Check if a refresh is already pending for this ticker in Redis."""
    if redis_client is None:
        return False
    try:
        key = f"refresh_pending:{ticker}"
        exists = redis_client.exists(key)
        return exists > 0
    except Exception:
        return False


def set_redis_refresh_pending(redis_client: Optional[Any], ticker: str, ttl_seconds: int = 900) -> bool:
    """Set a flag in Redis indicating a refresh is pending for this ticker."""
    if redis_client is None:
        return False
    try:
        key = f"refresh_pending:{ticker}"
        redis_client.setex(key, ttl_seconds, "1")
        return True
    except Exception:
        return False


def clear_redis_refresh_pending(redis_client: Optional[Any], ticker: str) -> bool:
    """Clear the refresh pending flag for this ticker in Redis."""
    if redis_client is None:
        return False
    try:
        key = f"refresh_pending:{ticker}"
        redis_client.delete(key)
        return True
    except Exception:
        return False


def get_redis_last_write_timestamp(redis_client: Optional[Any], ticker: str) -> Optional[float]:
    """
    Get the last write timestamp for a ticker from Redis cache.
    Returns the age in seconds (float), or None if not found.
    """
    if not REDIS_AVAILABLE or redis_client is None:
        return None
    try:
        key = f"options:last_write_timestamp:{ticker}"
        value = redis_client.get(key)
        if value is None:
            return None
        # Value is stored as ISO format string, convert to age in seconds
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        
        # Use helper function if pandas is available, otherwise use fallback
        if PANDAS_AVAILABLE:
            return calculate_age_seconds(value)
        else:
            # Fallback for non-pandas case
            from datetime import datetime, timezone
            ts_dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            now_utc = datetime.now(timezone.utc)
            age_seconds = (now_utc - ts_dt).total_seconds()
            return age_seconds
    except Exception:
        return None


def set_redis_last_write_timestamp(redis_client: Optional[Any], ticker: str, timestamp: Any, ttl_seconds: int = 86400) -> bool:
    """
    Set the last write timestamp for a ticker in Redis cache.
    
    Args:
        redis_client: Redis client instance
        ticker: Ticker symbol
        timestamp: Datetime object (should be UTC-aware)
        ttl_seconds: Time to live in seconds (default: 24 hours)
    
    Returns:
        True if successful, False otherwise
    """
    if not REDIS_AVAILABLE or redis_client is None:
        return False
    try:
        key = f"options:last_write_timestamp:{ticker}"
        # Normalize timestamp to UTC and convert to ISO format string
        normalized_ts = normalize_timestamp_to_utc(timestamp)
        if normalized_ts is not None:
            iso_str = normalized_ts.isoformat()
        else:
            # Fallback to string representation if normalization fails
            iso_str = str(timestamp)
        redis_client.setex(key, ttl_seconds, iso_str)
        return True
    except Exception:
        return False


# ============================================================================
# Options Timestamp Checking Functions
# ============================================================================

def normalize_timestamp_to_utc(timestamp: Any) -> Optional[Any]:
    """
    Normalize a timestamp to UTC timezone.
    Handles pd.Timestamp, datetime objects, and string timestamps.
    
    Args:
        timestamp: Timestamp to normalize (pd.Timestamp, datetime, or string)
        
    Returns:
        Normalized timestamp (pd.Timestamp or datetime with UTC timezone), or None if invalid
    """
    if not PANDAS_AVAILABLE:
        return None
    
    if timestamp is None or pd.isna(timestamp):
        return None
    
    try:
        if isinstance(timestamp, pd.Timestamp):
            if timestamp.tz is None:
                timestamp = timestamp.tz_localize('UTC')
            else:
                timestamp = timestamp.tz_convert('UTC')
            return timestamp
        else:
            # Try to parse as datetime
            from datetime import datetime, timezone
            if isinstance(timestamp, datetime):
                if timestamp.tzinfo is None:
                    return timestamp.replace(tzinfo=timezone.utc)
                else:
                    return timestamp.astimezone(timezone.utc)
            else:
                # Try to parse as string
                ts = pd.to_datetime(timestamp, errors='coerce', utc=True)
                if pd.notna(ts):
                    if ts.tz is None:
                        ts = ts.tz_localize('UTC')
                    return ts
                return None
    except Exception:
        return None


def timestamp_to_datetime_utc(timestamp: Any) -> Optional[Any]:
    """
    Convert a timestamp to a UTC-aware datetime object.
    
    Args:
        timestamp: Timestamp to convert (pd.Timestamp, datetime, or string)
        
    Returns:
        UTC-aware datetime object, or None if invalid
    """
    from datetime import datetime, timezone
    
    normalized = normalize_timestamp_to_utc(timestamp)
    if normalized is None:
        return None
    
    try:
        if isinstance(normalized, pd.Timestamp):
            return normalized.to_pydatetime()
        elif isinstance(normalized, datetime):
            return normalized
        else:
            return None
    except Exception:
        return None


def calculate_age_seconds(timestamp: Any) -> Optional[float]:
    """
    Calculate the age in seconds from a timestamp to now.
    
    Args:
        timestamp: Timestamp to calculate age from
        
    Returns:
        Age in seconds (float), or None if timestamp is invalid
    """
    from datetime import datetime, timezone
    
    ts_dt = timestamp_to_datetime_utc(timestamp)
    if ts_dt is None:
        return None
    
    now_utc = datetime.now(timezone.utc)
    age_seconds = (now_utc - ts_dt).total_seconds()
    return age_seconds


def extract_timestamp_from_dataframe(df: Any, debug: bool = False, ticker: str = "") -> Optional[Any]:
    """
    Extract timestamp value from a DataFrame result.
    Tries multiple column name patterns to find the timestamp.
    
    Args:
        df: DataFrame containing timestamp data
        debug: Whether to print debug messages
        ticker: Ticker symbol (for debug messages)
        
    Returns:
        Timestamp value, or None if not found
    """
    if df is None or df.empty:
        return None
    
    # Try to find the timestamp column by name first
    if 'max_write_timestamp' in df.columns:
        timestamp = df.iloc[0]['max_write_timestamp']
    # If column names are lost (e.g., '0', '1'), try to find any column with timestamp data
    elif any('timestamp' in str(col).lower() for col in df.columns):
        for col in df.columns:
            if 'timestamp' in str(col).lower():
                timestamp = df.iloc[0][col]
                break
        else:
            timestamp = None
    elif len(df.columns) > 0:
        # Fallback: get first column value
        timestamp = df.iloc[0].iloc[0]
    else:
        timestamp = None
    
    if debug and timestamp is not None:
        print(f"DEBUG [extract_timestamp_from_dataframe] {ticker}: Raw timestamp from DB: {timestamp} (type: {type(timestamp)})", file=sys.stderr)
    
    return timestamp


async def fetch_latest_write_timestamp_from_db(
    db: Any,
    ticker: str,
    debug: bool = False
) -> Optional[Any]:
    """
    Fetch the latest write_timestamp from the database for a ticker.
    
    Args:
        db: Database connection object (must have execute_select_sql method)
        ticker: Ticker symbol to fetch timestamp for
        debug: Whether to print debug messages
        
    Returns:
        Timestamp value (pd.Timestamp or datetime), or None if not found
    """
    try:
        query = f"""
        SELECT write_timestamp as max_write_timestamp
        FROM options_data
        WHERE ticker = '{ticker}'
        LATEST ON timestamp PARTITION BY ticker;
        """
        
        timestamp_df = await db.execute_select_sql(query)
        
        if debug:
            print(f"DEBUG [fetch_latest_write_timestamp_from_db] {ticker}: Query returned {len(timestamp_df)} rows", file=sys.stderr)
            if not timestamp_df.empty:
                print(f"DEBUG [fetch_latest_write_timestamp_from_db] {ticker}: DataFrame columns: {list(timestamp_df.columns)}", file=sys.stderr)
                print(f"DEBUG [fetch_latest_write_timestamp_from_db] {ticker}: First row: {timestamp_df.iloc[0].to_dict()}", file=sys.stderr)
        
        raw_timestamp = extract_timestamp_from_dataframe(timestamp_df, debug=debug, ticker=ticker)
        
        if raw_timestamp is None:
            return None
        
        # Normalize timestamp to UTC
        normalized_ts = normalize_timestamp_to_utc(raw_timestamp)
        
        if debug and normalized_ts is not None:
            print(f"DEBUG [fetch_latest_write_timestamp_from_db] {ticker}: Normalized timestamp: {normalized_ts}", file=sys.stderr)
        
        return normalized_ts
    except Exception as e:
        if debug:
            print(f"DEBUG [fetch_latest_write_timestamp_from_db] {ticker}: Error fetching timestamp: {e}", file=sys.stderr)
        return None


async def fetch_latest_option_timestamp_standalone(
    db: Any,
    ticker: str,
    cache: Optional[Dict[str, Any]] = None,
    redis_client: Optional[Any] = None,
    debug: bool = False
) -> Optional[float]:
    """
    Standalone function to fetch latest option write timestamp for a single ticker.
    Returns the age in seconds (difference between now and the timestamp).
    Can be used in multiprocessing workers or regular code paths.
    
    Args:
        db: Database connection object (must have execute_select_sql method)
        ticker: Ticker symbol to fetch timestamp for
        cache: Optional dictionary to use/update as cache
        redis_client: Optional Redis client to check for cached timestamp
        debug: Whether to print debug messages
        
    Returns:
        Age in seconds since the latest write timestamp (float), or None if no timestamp found
    """
    if not PANDAS_AVAILABLE:
        if debug:
            print(f"DEBUG: pandas not available for timestamp checking", file=sys.stderr)
        return None
    
    # Use provided cache or create empty dict
    if cache is None:
        cache = {}
    
    # Check Redis cache first - this is the fastest and most reliable
    if redis_client is not None:
        redis_age = get_redis_last_write_timestamp(redis_client, ticker)
        if redis_age is not None:
            if debug:
                print(f"DEBUG [fetch_latest_option_timestamp_standalone] (Redis): {ticker} - age_seconds={redis_age}", file=sys.stderr)
            return redis_age
    
    # Check in-memory cache - if timestamp is cached, recalculate age from it
    if ticker in cache:
        cached_ts = cache[ticker]
        if cached_ts is None or pd.isna(cached_ts):
            return None
        
        # Recalculate age from cached timestamp using helper function
        age_seconds = calculate_age_seconds(cached_ts)
        
        if debug and age_seconds is not None:
            from datetime import datetime, timezone
            now_utc = datetime.now(timezone.utc)
            ts_dt = timestamp_to_datetime_utc(cached_ts)
            print(f"DEBUG [fetch_latest_option_timestamp_standalone] (cached): {ticker} - cached_ts={cached_ts}, cached_dt={ts_dt}, now_utc={now_utc}, age_seconds={age_seconds}", file=sys.stderr)
        
        return age_seconds
    
    # Fetch from database using helper function
    latest_opt_ts = await fetch_latest_write_timestamp_from_db(db, ticker, debug=debug)
    
    # Store timestamp in cache (for future age recalculation)
    cache[ticker] = latest_opt_ts
    
    # Calculate and return age in seconds
    if latest_opt_ts is None or pd.isna(latest_opt_ts):
        return None
    
    age_seconds = calculate_age_seconds(latest_opt_ts)
    
    # Cache the timestamp in Redis for future lookups
    if redis_client is not None and age_seconds is not None:
        ts_dt = timestamp_to_datetime_utc(latest_opt_ts)
        if ts_dt is not None:
            set_redis_last_write_timestamp(redis_client, ticker, ts_dt, ttl_seconds=86400)  # 24 hour TTL
    
    if debug and age_seconds is not None:
        from datetime import datetime, timezone
        now_utc = datetime.now(timezone.utc)
        ts_dt = timestamp_to_datetime_utc(latest_opt_ts)
        print(f"DEBUG [fetch_latest_option_timestamp_standalone]: {ticker} - latest_opt_ts={latest_opt_ts}, ts_dt={ts_dt}, now_utc={now_utc}, age_seconds={age_seconds}", file=sys.stderr)
    
    return age_seconds


async def check_tickers_for_refresh(
    db: Any,
    tickers: List[str],
    refresh_threshold_seconds: int,
    fetch_timestamp_func: Callable[[List[str], Optional[Dict]], Any],
    redis_client: Optional[Any] = None,
    timestamp_cache: Optional[Dict[str, Any]] = None,
    min_write_timestamp: Optional[str] = None,
    debug: bool = False
) -> List[str]:
    """
    Check which tickers need refresh based on their latest write_timestamp.
    Also includes tickers that don't meet the min_write_timestamp criteria.
    
    Args:
        db: Database connection object
        tickers: List of ticker symbols to check
        refresh_threshold_seconds: Age threshold in seconds for refresh
        fetch_timestamp_func: Function to fetch latest option timestamps (tickers, cache) -> Dict[str, Optional[float]]
        redis_client: Optional Redis client for deduplication
        timestamp_cache: Optional cache dictionary to reuse previously fetched timestamps
        min_write_timestamp: Optional minimum write timestamp (EST format) - tickers with data older than this will be refreshed
        debug: Whether to print debug messages
        
    Returns:
        List of ticker symbols that need refresh
    """
    if not PANDAS_AVAILABLE:
        if debug:
            print("DEBUG: pandas not available for refresh checking", file=sys.stderr)
        return tickers  # Return all tickers if we can't check
    
    from datetime import datetime, timezone
    
    tickers_to_refresh = []
    now_utc = datetime.now(timezone.utc)
    
    # Print to stderr so it's always visible regardless of log level
    print(f"\n=== Checking options data freshness for {len(tickers)} ticker(s) ===", file=sys.stderr)
    print(f"Refresh threshold: {refresh_threshold_seconds} seconds", file=sys.stderr)
    if min_write_timestamp:
        print(f"Min write timestamp filter: {min_write_timestamp} EST", file=sys.stderr)
    print("", file=sys.stderr)
    
    # Use cache if provided, otherwise create empty dict
    if timestamp_cache is None:
        timestamp_cache = {}
    
    # Fetch ages using the provided function
    latest_ages = await fetch_timestamp_func(tickers, timestamp_cache)
    
    # Parse min_write_timestamp if provided
    min_ts_utc = None
    if min_write_timestamp:
        try:
            import pytz
            est = pytz.timezone('America/New_York')
            min_ts = pd.to_datetime(min_write_timestamp)
            if min_ts.tz is None:
                min_ts = est.localize(min_ts)
            min_ts_utc = min_ts.astimezone(pytz.UTC)
            print(f"Min write timestamp (UTC): {min_ts_utc}", file=sys.stderr)
            print(f"Current time (UTC): {now_utc}", file=sys.stderr)
            print("", file=sys.stderr)
        except Exception as e:
            print(f"WARNING: Could not parse min_write_timestamp {min_write_timestamp}: {e}", file=sys.stderr)
    
    # Fetch actual timestamps (not just ages) for min_write_timestamp check
    latest_timestamps = {}
    if min_ts_utc:
        for ticker in tickers:
            if ticker in timestamp_cache:
                latest_timestamps[ticker] = timestamp_cache[ticker]
            else:
                # Fetch timestamp if not in cache using helper function
                ts = await fetch_latest_write_timestamp_from_db(db, ticker, debug=debug)
                if ts is not None:
                    latest_timestamps[ticker] = ts
                    timestamp_cache[ticker] = ts
    
    for ticker in tickers:
        try:
            # Check Redis first - if refresh is already pending, skip
            if redis_client and check_redis_refresh_pending(redis_client, ticker):
                print(f"  {ticker}: Refresh already pending (found in Redis cache) - skipping", file=sys.stderr)
                continue
            
            age_seconds = latest_ages.get(ticker)
            
            if age_seconds is None or pd.isna(age_seconds):
                # No data found, needs refresh
                tickers_to_refresh.append(ticker)
                print(f"  {ticker}: No options data found - will refresh", file=sys.stderr)
                continue
            
            age_minutes = age_seconds / 60
            
            # Get actual timestamp for detailed logging
            latest_ts = latest_timestamps.get(ticker)
            if latest_ts is None and ticker in timestamp_cache:
                latest_ts = timestamp_cache[ticker]
            
            needs_refresh = False
            refresh_reason = None
            
            # Check age threshold
            if age_seconds > refresh_threshold_seconds:
                needs_refresh = True
                refresh_reason = f"Data is {age_minutes:.1f} minutes old (>{refresh_threshold_seconds}s threshold)"
            
            # Check min_write_timestamp criteria
            if min_ts_utc:
                if ticker in latest_timestamps:
                    latest_ts = latest_timestamps[ticker]
                elif ticker in timestamp_cache:
                    latest_ts = timestamp_cache[ticker]
                else:
                    latest_ts = None
                
                if latest_ts is None or pd.isna(latest_ts):
                    needs_refresh = True
                    if refresh_reason:
                        refresh_reason += f" and no timestamp found (needs min {min_write_timestamp} EST)"
                    else:
                        refresh_reason = f"No timestamp found (needs min {min_write_timestamp} EST)"
                elif latest_ts < min_ts_utc:
                    needs_refresh = True
                    time_diff = (min_ts_utc - latest_ts).total_seconds() / 60
                    if refresh_reason:
                        refresh_reason += f" and timestamp {latest_ts} is {time_diff:.1f} minutes before min {min_write_timestamp} EST"
                    else:
                        refresh_reason = f"Timestamp {latest_ts} is {time_diff:.1f} minutes before min {min_write_timestamp} EST"
            
            # Print detailed info for each ticker
            if latest_ts is not None and not pd.isna(latest_ts):
                print(f"  {ticker}: Latest write_timestamp: {latest_ts} UTC (age: {age_minutes:.1f} min, {age_seconds:.0f}s)", file=sys.stderr)
            else:
                print(f"  {ticker}: No timestamp found (age: {age_minutes:.1f} min, {age_seconds:.0f}s)", file=sys.stderr)
            
            if needs_refresh:
                tickers_to_refresh.append(ticker)
                print(f"    -> {refresh_reason} - WILL REFRESH", file=sys.stderr)
            else:
                print(f"    -> Data is fresh (age {age_minutes:.1f} min <= {refresh_threshold_seconds}s threshold) - SKIPPING", file=sys.stderr)
        except Exception as e:
            # On error, include ticker to be safe
            tickers_to_refresh.append(ticker)
            print(f"  {ticker}: ERROR checking timestamp ({e}) - will refresh", file=sys.stderr)
    
    print(f"\n=== Summary: {len(tickers_to_refresh)}/{len(tickers)} tickers need refresh ===\n", file=sys.stderr)
    
    return tickers_to_refresh


