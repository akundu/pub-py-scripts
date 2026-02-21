"""
Cache warmup utilities for stock info API endpoints.

This module provides functions to warm up the cache by pre-fetching stock info
for tickers in the background. It includes TTL-based tracking to prevent
unnecessary duplicate requests.
"""

import logging
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Module-level tracking for tickers that are being/have been fetched
# Structure: {ticker: timestamp} - timestamp when ticker was added
_warmup_tickers_lock = threading.Lock()
_warmup_tickers_timestamps: dict[str, float] = {}


def _warmup_cache_for_ticker(ticker: str, host: str = "mm.kundu.dev", port: int = 9100) -> None:
    """Fire-and-forget cache warmup for a single ticker.
    
    This function explicitly sets allow_source_fetch=true to ensure cache warmup
    actually fetches fresh data from the source, not just serves from cache.
    """
    url = f"http://{host}:{port}/api/stock_info/{ticker}?show_iv=true&show_news=true&allow_source_fetch=true"
    
    logger.debug(f"[CACHE WARMUP] Sending HTTP request for {ticker} to {url}")
    
    try:
        # Make request with timeout - fire and forget, errors are expected
        req = urllib.request.Request(url)
        logger.debug(f"[CACHE WARMUP] Making HTTP GET request to: {url}")
        urllib.request.urlopen(req, timeout=5.0)
        success_msg = f"[CACHE WARMUP] Successfully sent request for {ticker} to {host}:{port}"
        logger.debug(success_msg)
    except TimeoutError:
        # Timeout errors are expected and ignored (fire-and-forget)
        logger.debug(f"[CACHE WARMUP] Timeout for {ticker} (expected, ignoring)")
        pass
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        # Other errors are also ignored (fire-and-forget), but log at debug level
        logger.debug(f"[CACHE WARMUP] Error sending request for {ticker} to {url}: {type(e).__name__}: {e}")
        pass


def _cleanup_expired_tickers(ttl_seconds: float) -> None:
    """Remove tickers from tracking that have exceeded TTL."""
    current_time = time.time()
    expired_tickers = [
        ticker for ticker, timestamp in _warmup_tickers_timestamps.items()
        if current_time - timestamp > ttl_seconds
    ]
    
    if expired_tickers:
        for ticker in expired_tickers:
            del _warmup_tickers_timestamps[ticker]
        logger.debug(f"[CACHE WARMUP] Cleaned up {len(expired_tickers)} expired ticker(s) from tracking")


def warmup_stock_info_cache(
    df: pd.DataFrame,
    host: str = "mm.kundu.dev",
    port: int = 9100,
    ttl_seconds: float = 1800.0,  # Default 30 minutes (1/2 of 1 hour default sleep)
    wait_timeout: Optional[float] = None,  # None = fire-and-forget, otherwise wait up to timeout
) -> None:
    """Background cache warmup: Extract unique tickers and fetch stock_info in background.
    
    This function ensures each ticker is only fetched once within the TTL window.
    Uses a module-level dict with thread-safe locking to prevent duplicate fetches.
    
    Args:
        df: DataFrame containing ticker symbols (must have 'ticker', 'TICKER', or 'Ticker' column)
        host: Database server hostname (default: "mm.kundu.dev")
        port: Database server port (default: 9100)
        ttl_seconds: Time-to-live for ticker tracking in seconds. Tickers older than this
                    will be removed from tracking and can be warmed up again.
                    Default: 1800 seconds (30 minutes, which is 1/2 of 1 hour default sleep).
        wait_timeout: Maximum time to wait for warmup threads to complete (in seconds).
                     If None, fire-and-forget (default). If specified, waits up to that time.
    """
    if df.empty:
        return
    
    # Clean up expired tickers first
    with _warmup_tickers_lock:
        _cleanup_expired_tickers(ttl_seconds)
    
    # Extract unique ticker symbols from DataFrame
    ticker_col = None
    for col in ['ticker', 'TICKER', 'Ticker']:
        if col in df.columns:
            ticker_col = col
            break
    
    if not ticker_col:
        return
    
    # Get unique tickers (excluding NaN/None/empty values)
    tickers = df[ticker_col].dropna().astype(str).str.strip()
    tickers = tickers[tickers != '']
    tickers = tickers[tickers != 'N/A']
    unique_tickers = tickers.unique()
    
    if len(unique_tickers) == 0:
        return
    
    # Thread-safe check: Only fetch tickers that haven't been fetched yet (within TTL)
    current_time = time.time()
    with _warmup_tickers_lock:
        # Filter out tickers that are still within TTL
        new_tickers = [
            t for t in unique_tickers
            if t not in _warmup_tickers_timestamps
            or (current_time - _warmup_tickers_timestamps[t]) > ttl_seconds
        ]
        
        if len(new_tickers) == 0:
            # All tickers are still within TTL
            logger.debug(f"[CACHE WARMUP] All {len(unique_tickers)} tickers still within TTL, skipping warmup")
            return
        
        # Mark new tickers as in-progress with current timestamp
        for ticker in new_tickers:
            _warmup_tickers_timestamps[ticker] = current_time
    
    # Fire-and-forget: Start background threads only for new tickers
    def warmup_worker(ticker: str) -> None:
        try:
            _warmup_cache_for_ticker(ticker, host, port)
        except Exception as e:
            # Log but don't fail - this is fire-and-forget
            logger.debug(f"[CACHE WARMUP] Worker error for {ticker}: {e}")
    
    threads = []
    for ticker in new_tickers:
        # Start a non-daemon thread for each new ticker so we can wait for HTTP requests to be sent
        thread = threading.Thread(target=warmup_worker, args=(ticker,), daemon=False)
        thread.start()
        threads.append(thread)
    
    if len(new_tickers) > 0:
        skipped_count = len(unique_tickers) - len(new_tickers)
        print(
            f"Background cache warmup: Fetching stock info for {len(new_tickers)} new tickers "
            f"(skipped {skipped_count} still within TTL)",
            file=sys.stderr
        )
        logger.debug(f"[CACHE WARMUP] Starting warmup for {len(new_tickers)} tickers via {host}:{port}")
        logger.debug(f"[CACHE WARMUP] Tickers: {', '.join(new_tickers[:10])}{'...' if len(new_tickers) > 10 else ''}")
        print(f"[CACHE WARMUP] Starting warmup for {len(new_tickers)} tickers", file=sys.stderr)
        print(f"[CACHE WARMUP] Database server: {host}:{port}", file=sys.stderr)
        print(f"[CACHE WARMUP] HTTP endpoint base: http://{host}:{port}/api/stock_info/", file=sys.stderr)
        print(
            f"[CACHE WARMUP] Tickers to warmup: {', '.join(new_tickers[:20])}{'...' if len(new_tickers) > 20 else ''}",
            file=sys.stderr
        )
        
        # If wait_timeout is specified, wait for threads (with timeout)
        if wait_timeout is not None:
            min_wait = 5.0  # Always wait at least 5 seconds to ensure requests are sent
            max_wait = max(min_wait, min(wait_timeout, 2.0 * len(new_tickers)))
            start_time = time.time()
            
            for thread in threads:
                remaining_time = max_wait - (time.time() - start_time)
                if remaining_time <= 0:
                    break
                thread.join(timeout=remaining_time)
            
            # Ensure we've waited at least min_wait seconds
            elapsed = time.time() - start_time
            if elapsed < min_wait:
                time.sleep(min_wait - elapsed)
                elapsed = time.time() - start_time
            
            logger.debug(f"[CACHE WARMUP] Waited {elapsed:.2f}s for HTTP requests to be sent")
            print(f"[CACHE WARMUP] Waited {elapsed:.2f}s for HTTP requests to be sent", file=sys.stderr)
        else:
            # Fire-and-forget: don't wait, just log
            logger.debug(f"[CACHE WARMUP] Started {len(threads)} background threads (fire-and-forget)")
            print(f"[CACHE WARMUP] Started {len(threads)} background threads (fire-and-forget)", file=sys.stderr)



