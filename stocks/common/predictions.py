"""
Prediction-specific utility functions and classes for close price predictions.

This module provides caching, history tracking, serialization, and data fetching
for the prediction web interface.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from zoneinfo import ZoneInfo

# Import market hours check
from common.market_hours import is_market_hours

# Try to import prediction modules
try:
    from scripts.predict_close_now import predict_close, predict_future_close
    from scripts.close_predictor.models import UnifiedPrediction, UnifiedBand, ET_TZ
    PREDICTIONS_AVAILABLE = True
except ImportError as e:
    PREDICTIONS_AVAILABLE = False
    predict_close = None
    predict_future_close = None
    UnifiedPrediction = None
    UnifiedBand = None
    # Define ET_TZ fallback
    try:
        ET_TZ = ZoneInfo("America/New_York")
    except Exception:
        from datetime import timezone
        ET_TZ = timezone.utc

# Try to import numpy for type conversion
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logger = logging.getLogger(__name__)


# ============================================================================
# Cache Backend Classes
# ============================================================================

from abc import ABC, abstractmethod

class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        pass

    @abstractmethod
    async def get_with_timestamp(self, key: str) -> Optional[Tuple[Any, float]]:
        """Get cached value with timestamp."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any):
        """Set cached value with current timestamp."""
        pass

    @abstractmethod
    async def clear(self):
        """Clear all cached values."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get backend name for logging."""
        pass


class InMemoryCache(CacheBackend):
    """In-memory cache backend using Python dict."""

    def __init__(self):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self.lock:
            if key in self.cache:
                data, _ = self.cache[key]
                return data
            return None

    async def get_with_timestamp(self, key: str) -> Optional[Tuple[Any, float]]:
        async with self.lock:
            return self.cache.get(key)

    async def set(self, key: str, value: Any):
        async with self.lock:
            self.cache[key] = (value, time.time())

    async def clear(self):
        async with self.lock:
            self.cache.clear()

    def get_name(self) -> str:
        return "memory"


class DiskCache(CacheBackend):
    """Disk-based cache backend using JSON files."""

    def __init__(self, cache_dir: str = ".prediction_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.lock = asyncio.Lock()
        logger.info(f"DiskCache initialized: {self.cache_dir.absolute()}")

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Sanitize key for filesystem
        safe_key = key.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_key}.json"

    async def get(self, key: str) -> Optional[Any]:
        try:
            cache_path = self._get_cache_path(key)
            if not cache_path.exists():
                return None

            async with self.lock:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)
                    return cached.get('data')
        except Exception as e:
            logger.warning(f"DiskCache read error for {key}: {e}")
            return None

    async def get_with_timestamp(self, key: str) -> Optional[Tuple[Any, float]]:
        try:
            cache_path = self._get_cache_path(key)
            if not cache_path.exists():
                return None

            async with self.lock:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)
                    return (cached.get('data'), cached.get('timestamp', time.time()))
        except Exception as e:
            logger.warning(f"DiskCache read error for {key}: {e}")
            return None

    async def set(self, key: str, value: Any):
        try:
            cache_path = self._get_cache_path(key)
            tmp_path = cache_path.with_suffix('.tmp')
            cached = {
                'data': value,
                'timestamp': time.time(),
                'key': key
            }

            async with self.lock:
                # Write to a temp file first, then atomically rename to avoid
                # partial writes leaving a corrupt cache file on disk.
                with open(tmp_path, 'w') as f:
                    json.dump(cached, f, indent=2)
                tmp_path.replace(cache_path)
        except Exception as e:
            logger.error(f"DiskCache write error for {key}: {e}")
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    async def clear(self):
        try:
            async with self.lock:
                for cache_file in self.cache_dir.glob("*.json"):
                    cache_file.unlink()
                logger.info("DiskCache cleared")
        except Exception as e:
            logger.error(f"DiskCache clear error: {e}")

    def get_name(self) -> str:
        return "disk"


class RedisCache(CacheBackend):
    """Redis-based cache backend."""

    def __init__(self, redis_client=None, key_prefix: str = "prediction:"):
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.available = redis_client is not None

    def _make_key(self, key: str) -> str:
        """Prefix key for Redis."""
        return f"{self.key_prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        if not self.available:
            return None

        try:
            redis_key = self._make_key(key)
            cached_json = await self.redis.get(redis_key)
            if cached_json:
                cached = json.loads(cached_json)
                return cached.get('data')
            return None
        except Exception as e:
            logger.warning(f"RedisCache read error for {key}: {e}")
            return None

    async def get_with_timestamp(self, key: str) -> Optional[Tuple[Any, float]]:
        if not self.available:
            return None

        try:
            redis_key = self._make_key(key)
            cached_json = await self.redis.get(redis_key)
            if cached_json:
                cached = json.loads(cached_json)
                return (cached.get('data'), cached.get('timestamp', time.time()))
            return None
        except Exception as e:
            logger.warning(f"RedisCache read error for {key}: {e}")
            return None

    async def set(self, key: str, value: Any):
        if not self.available:
            return

        try:
            redis_key = self._make_key(key)
            cached = {
                'data': value,
                'timestamp': time.time(),
                'key': key
            }
            await self.redis.set(redis_key, json.dumps(cached))
        except Exception as e:
            logger.error(f"RedisCache write error for {key}: {e}")

    async def clear(self):
        if not self.available:
            return

        try:
            # Delete all keys with our prefix
            keys = []
            async for key in self.redis.scan_iter(match=f"{self.key_prefix}*"):
                keys.append(key)
            if keys:
                await self.redis.delete(*keys)
                logger.info(f"RedisCache cleared {len(keys)} keys")
        except Exception as e:
            logger.error(f"RedisCache clear error: {e}")

    def get_name(self) -> str:
        return "redis" if self.available else "redis(unavailable)"


# ============================================================================
# Multi-Backend Cache Manager
# ============================================================================

class PredictionCache:
    """Multi-backend cache manager for predictions.

    Supports in-memory, disk, and Redis backends. Reads from backends in priority order,
    writes to all configured backends.

    Default: disk-based cache for persistence across restarts and worker processes.

    Strategy:
    - Always serves from cache if available (no auto-expiration)
    - Only refreshes when explicitly requested via force_refresh parameter
    - Designed for cron-based cache warming: cron calls with ?cache=false to populate
    """

    def __init__(self, backends: Optional[List[str]] = None, redis_client=None, cache_dir: str = ".prediction_cache"):
        """Initialize cache with specified backends.

        Args:
            backends: List of backend names ('memory', 'disk', 'redis'). Default: ['disk']
            redis_client: Redis client instance (required if 'redis' in backends)
            cache_dir: Directory for disk cache
        """
        if backends is None:
            backends = ['disk']  # Default to disk cache

        self.backend_instances: List[CacheBackend] = []

        # Initialize requested backends
        for backend_name in backends:
            if backend_name == 'memory':
                self.backend_instances.append(InMemoryCache())
            elif backend_name == 'disk':
                self.backend_instances.append(DiskCache(cache_dir=cache_dir))
            elif backend_name == 'redis':
                self.backend_instances.append(RedisCache(redis_client=redis_client))
            else:
                logger.warning(f"Unknown cache backend: {backend_name}")

        if not self.backend_instances:
            # Fallback to disk if no valid backends
            logger.warning("No valid cache backends, using disk as fallback")
            self.backend_instances.append(DiskCache(cache_dir=cache_dir))

        backend_names = [b.get_name() for b in self.backend_instances]
        logger.info(f"PredictionCache initialized with backends: {backend_names}")

    async def get(self, key: str) -> Optional[Any]:
        """Get cached value, trying backends in priority order.

        Returns value from first backend that has it.
        """
        for backend in self.backend_instances:
            value = await backend.get(key)
            if value is not None:
                logger.debug(f"Cache HIT on {backend.get_name()} for {key}")
                return value

        logger.debug(f"Cache MISS for {key}")
        return None

    async def get_with_timestamp(self, key: str) -> Optional[Tuple[Any, float]]:
        """Get cached value with timestamp, trying backends in priority order."""
        for backend in self.backend_instances:
            result = await backend.get_with_timestamp(key)
            if result is not None:
                logger.debug(f"Cache HIT on {backend.get_name()} for {key}")
                return result

        logger.debug(f"Cache MISS for {key}")
        return None

    async def set(self, key: str, value: Any):
        """Store value in ALL configured backends."""
        tasks = [backend.set(key, value) for backend in self.backend_instances]
        await asyncio.gather(*tasks, return_exceptions=True)
        backend_names = [b.get_name() for b in self.backend_instances]
        logger.debug(f"Cached {key} to: {backend_names}")

    async def clear(self):
        """Clear all caches in all backends."""
        tasks = [backend.clear() for backend in self.backend_instances]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("All cache backends cleared")

    async def cleanup_expired(self):
        """No-op: cache entries never expire automatically."""
        pass


class PredictionHistory:
    """Store prediction snapshots throughout the trading day for band convergence visualization.

    Uses file-based persistence to survive server restarts.
    """

    def __init__(self, storage_dir: str = ".prediction_history"):
        """Initialize prediction history with file-based storage.

        Args:
            storage_dir: Directory to store history files (default: .prediction_history)
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.lock = asyncio.Lock()

        # In-memory cache for fast access
        self.history: Dict[str, List[Dict[str, Any]]] = {}

        # Load existing snapshots from disk on startup
        self._load_from_disk()

        logger.info(f"PredictionHistory initialized with storage: {self.storage_dir.absolute()}")

    def _load_from_disk(self):
        """Load existing snapshot files from disk into memory."""
        try:
            # Clean up old files first (older than 2 days)
            cutoff_date = date.today() - timedelta(days=2)

            for file_path in self.storage_dir.glob("*.json"):
                try:
                    # Parse filename: {ticker}_{YYYY-MM-DD}.json
                    parts = file_path.stem.split('_')
                    if len(parts) < 4:  # Should be like NDX_2026_02_13
                        continue

                    ticker = parts[0]
                    date_str = f"{parts[1]}-{parts[2]}-{parts[3]}"
                    snapshot_date = datetime.strptime(date_str, '%Y-%m-%d').date()

                    # Delete old files
                    if snapshot_date < cutoff_date:
                        file_path.unlink()
                        logger.info(f"Deleted old prediction history file: {file_path.name}")
                        continue

                    # Load recent files into memory
                    with open(file_path, 'r') as f:
                        snapshots = json.load(f)
                        key = f"{ticker}:{date_str}"
                        self.history[key] = snapshots
                        logger.info(f"Loaded {len(snapshots)} snapshots for {key}")

                except Exception as e:
                    logger.warning(f"Error loading prediction history file {file_path}: {e}")

        except Exception as e:
            logger.error(f"Error loading prediction history from disk: {e}")

    def _get_file_path(self, ticker: str, date_str: str) -> Path:
        """Get the file path for a ticker/date combination.

        Args:
            ticker: Ticker symbol (e.g., 'NDX')
            date_str: Date string in YYYY-MM-DD format

        Returns:
            Path to the JSON file
        """
        # Convert date to filename: NDX_2026_02_13.json
        date_parts = date_str.split('-')
        filename = f"{ticker}_{'_'.join(date_parts)}.json"
        return self.storage_dir / filename

    async def add_snapshot(self, ticker: str, date_str: str, prediction: dict):
        """Add a prediction snapshot with timestamp and save to disk.

        Args:
            ticker: Ticker symbol (e.g., 'NDX')
            date_str: Date string in YYYY-MM-DD format
            prediction: Prediction data dictionary
        """
        async with self.lock:
            key = f"{ticker}:{date_str}"

            # Initialize if needed
            if key not in self.history:
                self.history[key] = []

            # Add timestamp to prediction
            snapshot = {
                'timestamp': datetime.now(ET_TZ).isoformat(),
                'time_label': prediction.get('time_label'),
                'current_price': prediction.get('current_price'),
                'hours_to_close': prediction.get('hours_to_close'),
                'combined_bands': prediction.get('combined_bands'),
                'percentile_bands': prediction.get('percentile_bands'),
                'statistical_bands': prediction.get('statistical_bands'),
            }

            # Add to in-memory history
            self.history[key].append(snapshot)

            # Keep only last 100 snapshots per day
            if len(self.history[key]) > 100:
                self.history[key] = self.history[key][-100:]

            # Save to disk (async file write)
            try:
                file_path = self._get_file_path(ticker, date_str)

                # Use asyncio to write file without blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self._write_snapshots_to_file,
                    file_path,
                    self.history[key]
                )

                logger.debug(f"Saved snapshot for {key} to {file_path.name} ({len(self.history[key])} total)")

            except Exception as e:
                logger.error(f"Error saving prediction snapshot to disk: {e}")
                # Continue even if disk write fails (data still in memory)

    def _write_snapshots_to_file(self, file_path: Path, snapshots: List[Dict[str, Any]]):
        """Write snapshots to file (blocking operation, run in executor).

        Args:
            file_path: Path to write to
            snapshots: List of snapshot dictionaries
        """
        with open(file_path, 'w') as f:
            json.dump(snapshots, f, indent=2)

    async def get_snapshots(self, ticker: str, date_str: str) -> List[Dict[str, Any]]:
        """Get all snapshots for a ticker and date.

        Args:
            ticker: Ticker symbol
            date_str: Date string in YYYY-MM-DD format

        Returns:
            List of snapshot dictionaries
        """
        async with self.lock:
            key = f"{ticker}:{date_str}"

            # Return from memory if available
            if key in self.history:
                return self.history[key]

            # Try loading from disk if not in memory
            file_path = self._get_file_path(ticker, date_str)
            if file_path.exists():
                try:
                    loop = asyncio.get_event_loop()
                    snapshots = await loop.run_in_executor(
                        None,
                        self._read_snapshots_from_file,
                        file_path
                    )
                    self.history[key] = snapshots
                    logger.info(f"Loaded {len(snapshots)} snapshots for {key} from disk")
                    return snapshots
                except Exception as e:
                    logger.error(f"Error reading snapshots from {file_path}: {e}")

            return []

    def _read_snapshots_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Read snapshots from file (blocking operation, run in executor).

        Args:
            file_path: Path to read from

        Returns:
            List of snapshot dictionaries
        """
        with open(file_path, 'r') as f:
            return json.load(f)

    async def clear_old_days(self, keep_days: int = 2):
        """Clear history older than N days from memory and disk.

        Args:
            keep_days: Number of days to keep (default: 2)
        """
        async with self.lock:
            cutoff_date = date.today() - timedelta(days=keep_days)
            keys_to_delete = []

            # Clear from memory
            for key in self.history.keys():
                try:
                    _, date_str = key.split(':')
                    snapshot_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    if snapshot_date < cutoff_date:
                        keys_to_delete.append(key)
                except Exception:
                    continue

            for key in keys_to_delete:
                del self.history[key]
                logger.info(f"Cleared old snapshots from memory: {key}")

            # Clear from disk (run in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._cleanup_old_files,
                cutoff_date
            )

    def _cleanup_old_files(self, cutoff_date: date):
        """Clean up old snapshot files from disk (blocking operation).

        Args:
            cutoff_date: Date before which files should be deleted
        """
        try:
            for file_path in self.storage_dir.glob("*.json"):
                try:
                    # Parse filename
                    parts = file_path.stem.split('_')
                    if len(parts) < 4:
                        continue

                    date_str = f"{parts[1]}-{parts[2]}-{parts[3]}"
                    snapshot_date = datetime.strptime(date_str, '%Y-%m-%d').date()

                    if snapshot_date < cutoff_date:
                        file_path.unlink()
                        logger.info(f"Deleted old prediction history file: {file_path.name}")

                except Exception as e:
                    logger.warning(f"Error processing file {file_path}: {e}")

        except Exception as e:
            logger.error(f"Error cleaning up old files: {e}")


# ============================================================================
# Serialization Functions
# ============================================================================

def _serialize_unified_prediction(pred: Any) -> dict:
    """Convert UnifiedPrediction dataclass to JSON-serializable dict.

    Args:
        pred: UnifiedPrediction object

    Returns:
        dict with all prediction data in JSON-serializable format
    """
    if pred is None:
        return None

    def safe_float(val) -> Optional[float]:
        """Convert to float, returning None for NaN/Inf (not valid JSON)."""
        import math
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f

    def serialize_band(band) -> dict:
        """Convert UnifiedBand to dict."""
        if band is None:
            return None
        return {
            'name': band.name,
            'lo_price': safe_float(band.lo_price),
            'hi_price': safe_float(band.hi_price),
            'lo_pct': safe_float(band.lo_pct),
            'hi_pct': safe_float(band.hi_pct),
            'width_pts': safe_float(band.width_pts),
            'width_pct': safe_float(band.width_pct),
            'source': band.source,
        }

    def serialize_band_dict(band_dict: dict) -> dict:
        """Convert dict of UnifiedBand objects to dict of dicts."""
        if not band_dict:
            return {}
        return {name: serialize_band(band) for name, band in band_dict.items()}

    # Convert numpy types to native Python types
    def convert_value(val):
        """Convert numpy types to native Python types."""
        if val is None:
            return None
        if NUMPY_AVAILABLE and isinstance(val, (np.integer, np.floating)):
            return val.item()
        if isinstance(val, (np.ndarray,)):
            return val.tolist()
        return val

    return {
        'ticker': pred.ticker,
        'current_price': convert_value(pred.current_price),
        'prev_close': convert_value(pred.prev_close),
        'hours_to_close': convert_value(pred.hours_to_close),
        'time_label': pred.time_label,
        'above_prev': bool(pred.above_prev),
        'percentile_bands': serialize_band_dict(pred.percentile_bands),
        'statistical_bands': serialize_band_dict(pred.statistical_bands),
        'combined_bands': serialize_band_dict(pred.combined_bands),
        'confidence': pred.confidence,
        'risk_level': convert_value(pred.risk_level),
        'vix1d': convert_value(pred.vix1d),
        'realized_vol': convert_value(pred.realized_vol),
        'stat_sample_size': convert_value(pred.stat_sample_size),
        'reversal_blend': convert_value(pred.reversal_blend),
        'intraday_vol_factor': convert_value(pred.intraday_vol_factor),
        'data_source': pred.data_source,
        'training_approach': pred.training_approach,
        'similar_days': pred.similar_days,  # Already in dict format
    }


# ============================================================================
# Data Fetching Functions
# ============================================================================

async def fetch_today_prediction(ticker: str, cache: PredictionCache, force_refresh: bool = False, history: Optional[PredictionHistory] = None, lookback: int = 250):
    """Fetch today's prediction, using cache if available and not stale.

    Cache TTL:
    - During market hours (9:30 AM - 4:00 PM ET): 15 minutes, so predictions
      update as time_label and market price change throughout the day.
    - Outside market hours: 1 hour (max cache age enforced for freshness).
    """
    if not PREDICTIONS_AVAILABLE:
        return {'error': 'Predictions module not available'}

    cache_key = f"today_{ticker}_{lookback}"

    # Determine TTL based on market hours
    now_et = datetime.now(ET_TZ)
    if is_market_hours(now_et):
        cache_ttl_seconds = 15 * 60   # 15 minutes during market hours
    else:
        cache_ttl_seconds = 1 * 3600  # 1 hour outside market hours

    if not force_refresh:
        cached_with_ts = await cache.get_with_timestamp(cache_key)
        if cached_with_ts is not None:
            cached_data, cache_timestamp = cached_with_ts
            age_seconds = time.time() - cache_timestamp
            # Also reject a cached entry where current_price is 0 or missing —
            # this can happen if the price fetch failed at cache-write time.
            cached_price = cached_data.get('current_price') if isinstance(cached_data, dict) else None
            if cached_price and cached_price > 0 and age_seconds <= cache_ttl_seconds:
                # Cache is fresh and valid — return it
                return {**cached_data, 'cache_timestamp': cache_timestamp}
            elif not (cached_price and cached_price > 0):
                logger.warning(
                    f"Stale/invalid cache for {cache_key} (current_price={cached_price}). Regenerating."
                )
            else:
                logger.info(
                    f"Cache expired for {cache_key} (age {age_seconds:.0f}s > TTL {cache_ttl_seconds}s). Regenerating."
                )

    try:
        # Call predict_close async function
        pred = await predict_close(ticker=ticker, lookback=lookback, force_retrain=False)

        # Serialize the prediction
        serialized = _serialize_unified_prediction(pred)

        # Only cache if we got a valid price — don't persist zeros to disk.
        if not (serialized.get('current_price') and serialized['current_price'] > 0):
            logger.warning(f"Prediction for {ticker} returned current_price={serialized.get('current_price')}; skipping cache write.")
            serialized['cache_timestamp'] = time.time()
            return serialized

        # Add current timestamp as cache_timestamp for fresh data
        serialized['cache_timestamp'] = time.time()

        # Cache it
        await cache.set(cache_key, serialized)

        # Store snapshot in history for band convergence visualization
        if history is not None:
            date_str = datetime.now(ET_TZ).strftime('%Y-%m-%d')
            await history.add_snapshot(ticker, date_str, serialized)

        return serialized
    except Exception as e:
        logger.error(f"Error fetching today's prediction for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


async def fetch_future_prediction(ticker: str, days_ahead: int, cache: PredictionCache, force_refresh: bool = False, lookback: int = 250):
    """Fetch future prediction for N days ahead, using cache if available."""
    if not PREDICTIONS_AVAILABLE:
        return {'error': 'Predictions module not available'}

    cache_key = f"future_{ticker}_{days_ahead}_{lookback}"

    # Future predictions are capped at 1 hour to ensure freshness
    future_cache_ttl_seconds = 1 * 3600

    if not force_refresh:
        cached_with_ts = await cache.get_with_timestamp(cache_key)
        if cached_with_ts is not None:
            cached_data, cache_timestamp = cached_with_ts
            age_seconds = time.time() - cache_timestamp
            cached_price = cached_data.get('current_price') if isinstance(cached_data, dict) else None
            if cached_price and cached_price > 0 and age_seconds <= future_cache_ttl_seconds:
                # Return a copy with cache timestamp added (don't mutate cached object)
                return {**cached_data, 'cache_timestamp': cache_timestamp}
            elif not (cached_price and cached_price > 0):
                logger.warning(
                    f"Stale/invalid cache for {cache_key} (current_price={cached_price}). Regenerating."
                )
            else:
                logger.info(
                    f"Cache expired for {cache_key} (age {age_seconds:.0f}s > TTL {future_cache_ttl_seconds}s). Regenerating."
                )

    try:
        # Get current price first (needed for future prediction)
        today_pred = await fetch_today_prediction(ticker, cache, force_refresh=False, lookback=lookback)
        if 'error' in today_pred:
            return today_pred

        current_price = today_pred.get('current_price')
        if not (current_price and current_price > 0):
            # Fresh price unavailable (e.g. market closed) — use last valid cached price
            today_cache_key = f"today_{ticker}_{lookback}"
            cached_today = await cache.get(today_cache_key)
            if cached_today and isinstance(cached_today, dict):
                current_price = cached_today.get('current_price', 0)
            if not (current_price and current_price > 0):
                return {'error': f'Invalid current_price ({current_price}) for {ticker}; cannot compute future prediction.'}

        # Call predict_close with days_ahead parameter (routes to _predict_future_close_unified)
        # This includes all 4 ensemble methods: Baseline, Conditional, Ensemble, Ensemble Combined
        result = await predict_close(ticker=ticker, days_ahead=days_ahead, lookback=lookback)

        # Debug: Log what attributes are on the result object
        if hasattr(result, '__dict__'):
            logger.info(f"UnifiedPrediction attributes for {ticker}+{days_ahead}d: {list(result.__dict__.keys())}")

        # Convert UnifiedPrediction to dict
        if hasattr(result, '__dict__'):
            result_dict = {}
            for key, val in result.__dict__.items():
                # Convert UnifiedBand objects to dicts
                if isinstance(val, dict):
                    result_dict[key] = {}
                    for k, v in val.items():
                        if hasattr(v, '__dict__'):
                            result_dict[key][k] = v.__dict__
                        else:
                            result_dict[key][k] = v
                else:
                    result_dict[key] = val
            result = result_dict

        # The result is now a dict, convert all types to JSON-serializable
        def convert_value(val):
            if val is None:
                return None
            # Handle numpy types
            if NUMPY_AVAILABLE and isinstance(val, (np.integer, np.floating)):
                return val.item()
            if NUMPY_AVAILABLE and isinstance(val, (np.ndarray,)):
                return val.tolist()
            if NUMPY_AVAILABLE and isinstance(val, np.bool_):
                return bool(val)
            # Handle Python bool explicitly (convert numpy.bool_ or other bool-likes to native bool)
            if isinstance(val, bool):
                return bool(val)
            # Handle other primitive types
            if isinstance(val, (int, float, str)):
                return val
            # Handle datetime/date objects
            if hasattr(val, 'isoformat'):
                return val.isoformat()
            # For other types, try to convert to string as fallback
            if not isinstance(val, (dict, list, tuple)):
                try:
                    # Try to serialize to check if it's JSON-compatible
                    import json
                    json.dumps(val)
                    return val
                except (TypeError, ValueError):
                    return str(val)
            return val

        def convert_dict(d):
            if d is None:
                return None
            if isinstance(d, dict):
                return {k: convert_dict(v) if isinstance(v, (dict, list)) else convert_value(v) for k, v in d.items()}
            if isinstance(d, (list, tuple)):
                return [convert_dict(item) if isinstance(item, (dict, list)) else convert_value(item) for item in d]
            return convert_value(d)

        serialized = convert_dict(result)

        # Debug: Log what fields are in serialized data
        logger.info(f"Serialized future prediction fields for {ticker}+{days_ahead}d: {list(serialized.keys())}")
        logger.info(f"  target_date_str={serialized.get('target_date_str')}, expected_price={serialized.get('expected_price')}, mean_return={serialized.get('mean_return')}, std_return={serialized.get('std_return')}")

        # Only cache if we have a valid price.
        if not (serialized.get('current_price') and serialized['current_price'] > 0):
            logger.warning(f"Future prediction for {ticker}+{days_ahead}d returned current_price={serialized.get('current_price')}; skipping cache write.")
            serialized['cache_timestamp'] = time.time()
            return serialized

        # Add current timestamp as cache_timestamp for fresh data
        serialized['cache_timestamp'] = time.time()

        # Cache it
        await cache.set(cache_key, serialized)

        return serialized
    except Exception as e:
        logger.error(f"Error fetching future prediction for {ticker} (+{days_ahead}d): {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


async def fetch_all_predictions(ticker: str, cache: PredictionCache):
    """Fetch predictions for all timeframes in parallel."""
    if not PREDICTIONS_AVAILABLE:
        return {'error': 'Predictions module not available'}

    try:
        # Fetch all predictions in parallel
        future_days = [1, 2, 3, 5, 10]
        results = await asyncio.gather(
            fetch_today_prediction(ticker, cache),
            *[fetch_future_prediction(ticker, d, cache) for d in future_days],
            return_exceptions=True
        )

        today = results[0]
        future_results = {d: results[i + 1] for i, d in enumerate(future_days)}

        return {
            'today': today if not isinstance(today, Exception) else {'error': str(today)},
            **{f'future_{d}d': r if not isinstance(r, Exception) else {'error': str(r)}
               for d, r in future_results.items()},
        }
    except Exception as e:
        logger.error(f"Error fetching all predictions for {ticker}: {e}")
        return {'error': str(e)}
