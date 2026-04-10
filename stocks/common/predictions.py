"""
Prediction-specific utility functions and classes for close price predictions.

This module provides caching, history tracking, serialization, and data fetching
for the prediction web interface.
"""

import asyncio
import copy
import json
import logging
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

# Import market hours check
from common.market_hours import is_market_hours

# Try to import prediction modules
try:
    from scripts.predict_close import predict_close, predict_future_close
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
        """Get cached value.

        When multiple backends are configured, returns the entry with the newest
        stored timestamp so disk (shared across workers) can beat stale per-worker
        memory after a prewarm or ?cache=false refresh on another process.
        """
        result = await self.get_with_timestamp(key)
        if result is not None:
            logger.debug(f"Cache HIT (merged) for {key}")
            return result[0]
        logger.debug(f"Cache MISS for {key}")
        return None

    async def get_with_timestamp(self, key: str) -> Optional[Tuple[Any, float]]:
        """Get cached value with timestamp from the backend that has the newest timestamp.

        Queries all backends and picks (data, ts) with maximum ts. Missing or invalid
        timestamps are treated as 0.0 for comparison only.
        """
        best: Optional[Tuple[Any, float]] = None
        best_ts = float("-inf")
        for backend in self.backend_instances:
            result = await backend.get_with_timestamp(key)
            if result is None:
                continue
            data, ts = result
            try:
                ts_f = float(ts)
            except (TypeError, ValueError):
                ts_f = 0.0
            if ts_f > best_ts:
                best_ts = ts_f
                best = (data, ts_f)

        if best is not None:
            logger.debug(
                f"Cache HIT (merged, ts={best_ts}) for {key} "
                f"across {[b.get_name() for b in self.backend_instances]}"
            )
            return best

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

    def resolved_disk_cache_dir(self) -> Optional[str]:
        """Absolute path to on-disk prediction cache if a DiskCache backend exists."""
        for backend in self.backend_instances:
            if isinstance(backend, DiskCache):
                return str(backend.cache_dir.resolve())
        return None


# Max concurrent prewarm worker processes (one ticker per job; pool queues extras).
PREWARM_POOL_MAX_WORKERS = 5

_prewarm_executor: Optional[ProcessPoolExecutor] = None
_prewarm_executor_lock = threading.Lock()


def get_prewarm_process_executor(max_workers: int = PREWARM_POOL_MAX_WORKERS) -> ProcessPoolExecutor:
    """Lazily create a shared process pool for /predictions/api/prewarm."""
    global _prewarm_executor
    with _prewarm_executor_lock:
        if _prewarm_executor is None:
            _prewarm_executor = ProcessPoolExecutor(max_workers=max_workers)
        return _prewarm_executor


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
                'empirical_continuous_bands': prediction.get('empirical_continuous_bands'),
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

    def get_latest_date(self, ticker: str) -> Optional[str]:
        """Get the most recent date with snapshots for a ticker.

        Checks both in-memory cache and disk files.

        Args:
            ticker: Ticker symbol (e.g., 'NDX')

        Returns:
            Date string in YYYY-MM-DD format, or None if no snapshots exist
        """
        dates = set()

        # Check in-memory keys
        prefix = f"{ticker}:"
        for key in self.history:
            if key.startswith(prefix):
                dates.add(key[len(prefix):])

        # Check disk files
        for file_path in self.storage_dir.glob(f"{ticker}_*.json"):
            try:
                parts = file_path.stem.split('_')
                if len(parts) >= 4:
                    date_str = f"{parts[1]}-{parts[2]}-{parts[3]}"
                    dates.add(date_str)
            except Exception:
                continue

        if not dates:
            return None

        return sorted(dates)[-1]

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

    result = {
        'ticker': pred.ticker,
        'current_price': convert_value(pred.current_price),
        'prev_close': convert_value(pred.prev_close),
        'hours_to_close': convert_value(pred.hours_to_close),
        'time_label': pred.time_label,
        'above_prev': bool(pred.above_prev),
        'percentile_bands': serialize_band_dict(pred.percentile_bands),
        'statistical_bands': serialize_band_dict(pred.statistical_bands),
        'combined_bands': serialize_band_dict(pred.combined_bands),
        'empirical_continuous_bands': serialize_band_dict(getattr(pred, 'empirical_continuous_bands', {})),
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

    # Add ensemble_methods if present (for multi-day predictions and 0DTE comparison)
    if hasattr(pred, 'ensemble_methods') and pred.ensemble_methods:
        result['ensemble_methods'] = pred.ensemble_methods
        # Add recommended_method field for easy access
        recommended = next((m['method'] for m in pred.ensemble_methods if m.get('recommended')), None)
        result['recommended_method'] = recommended

    # Add directional analysis if present
    if hasattr(pred, 'directional_analysis') and pred.directional_analysis:
        da = pred.directional_analysis
        result['directional_analysis'] = {
            'momentum_state': {
                'trend_label': da.momentum_state.trend_label,
                'consecutive_days': da.momentum_state.consecutive_days,
                'return_5d': da.momentum_state.return_5d,
                'is_extended_streak': da.momentum_state.is_extended_streak,
            },
            'direction_probability': {
                'p_up': da.direction_probability.p_up,
                'p_down': da.direction_probability.p_down,
                'up_count': da.direction_probability.up_count,
                'down_count': da.direction_probability.down_count,
                'total_samples': da.direction_probability.total_samples,
                'confidence': da.direction_probability.confidence,
                'mean_reversion_prob': da.direction_probability.mean_reversion_prob,
            },
            'asymmetric_bands': serialize_band_dict(da.asymmetric_bands),
        }

    return result


def _rescale_band_dict_for_new_spot(band_dict: Any, old_px: float, new_px: float) -> None:
    """Recompute absolute band levels when spot moves; prefers lo_pct/hi_pct from model."""
    if not isinstance(band_dict, dict) or old_px <= 0 or new_px <= 0:
        return
    for _name, band in band_dict.items():
        if not isinstance(band, dict):
            continue
        lo_pct = band.get('lo_pct')
        hi_pct = band.get('hi_pct')
        try:
            if lo_pct is not None and hi_pct is not None:
                band['lo_price'] = float(new_px) * (1.0 + float(lo_pct) / 100.0)
                band['hi_price'] = float(new_px) * (1.0 + float(hi_pct) / 100.0)
                band['width_pts'] = band['hi_price'] - band['lo_price']
                band['width_pct'] = (band['width_pts'] / float(new_px)) * 100.0
            else:
                r = new_px / old_px
                for k in ('lo_price', 'hi_price', 'width_pts'):
                    if band.get(k) is not None:
                        band[k] = float(band[k]) * r
                if band.get('lo_price') is not None and band.get('hi_price') is not None:
                    band['width_pct'] = ((band['hi_price'] - band['lo_price']) / float(new_px)) * 100.0
        except (TypeError, ValueError):
            continue


def _rescale_prediction_payload_bands(payload: dict, old_px: float, new_px: float) -> None:
    for key in (
        'percentile_bands',
        'statistical_bands',
        'combined_bands',
        'empirical_continuous_bands',
    ):
        bd = payload.get(key)
        if isinstance(bd, dict):
            _rescale_band_dict_for_new_spot(bd, old_px, new_px)
    da = payload.get('directional_analysis')
    if isinstance(da, dict):
        asym = da.get('asymmetric_bands')
        if isinstance(asym, dict):
            _rescale_band_dict_for_new_spot(asym, old_px, new_px)


async def overlay_questdb_spot_on_prediction_payload(
    ticker: str,
    payload: dict,
    stock_db: Any,
) -> None:
    """Align ``current_price`` / ``prev_close`` with QuestDB, bypassing Redis price cache.

    Cached prediction JSON can embed stale spot from PriceService Redis; this refreshes
    display fields from DB before the response is sent.
    """
    if not isinstance(payload, dict) or payload.get('error'):
        return
    getter = getattr(stock_db, 'get_latest_price_with_data', None)
    if not callable(getter):
        return
    db_ticker = ticker.replace('I:', '') if ticker.startswith('I:') else ticker
    try:
        live = await getter(db_ticker, True, bypass_cache=True)
    except TypeError:
        live = await getter(db_ticker, True)
    except Exception as e:
        logger.warning('overlay_questdb_spot: live price fetch failed for %s: %s', ticker, e)
        return
    if not live or live.get('price') is None:
        return
    try:
        new_px = float(live['price'])
    except (TypeError, ValueError):
        return
    if new_px <= 0:
        return
    old_px = float(payload.get('current_price') or 0.0)
    payload['live_overlay_applied'] = True
    payload['live_spot_source'] = live.get('source')
    ts = live.get('timestamp')
    if ts is not None:
        payload['live_spot_timestamp'] = (
            ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
        )
    payload['current_price'] = new_px

    try:
        ts_utc = live.get('timestamp')
        if ts_utc is not None:
            if hasattr(ts_utc, 'to_pydatetime'):
                ts_utc = ts_utc.to_pydatetime()
            if getattr(ts_utc, 'tzinfo', None) is None:
                ts_utc = ts_utc.replace(tzinfo=timezone.utc)
            current_date = ts_utc.astimezone(ET_TZ).date()
        else:
            current_date = datetime.now(ET_TZ).date()
        connmgr = getattr(stock_db, 'connection', None)
        if connmgr is not None and hasattr(connmgr, 'get_connection'):
            async with connmgr.get_connection() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT date, close
                    FROM daily_prices
                    WHERE ticker = $1 AND date < $2
                    ORDER BY date DESC
                    LIMIT 1
                    """,
                    db_ticker,
                    current_date,
                )
            if row is not None and row['close'] is not None:
                payload['prev_close'] = float(row['close'])
                pdt = row['date']
                if hasattr(pdt, 'date'):
                    pdt = pdt.date()
                payload['prev_close_date'] = pdt.isoformat() if hasattr(pdt, 'isoformat') else str(pdt)
    except Exception as e:
        logger.debug('overlay_questdb_spot: prev_close refresh skipped for %s: %s', ticker, e)

    pc = payload.get('prev_close')
    if pc is not None:
        try:
            payload['above_prev'] = new_px >= float(pc)
        except (TypeError, ValueError):
            pass

    if old_px > 0 and abs(new_px - old_px) / old_px > 1e-9:
        _rescale_prediction_payload_bands(payload, old_px, new_px)


# ============================================================================
# Data Fetching Functions
# ============================================================================

async def fetch_today_prediction(
    ticker: str,
    cache: PredictionCache,
    force_refresh: bool = False,
    history: Optional[PredictionHistory] = None,
    lookback: int = 150,
    stock_db: Optional[Any] = None,
):
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
                # Deep copy so overlay + history never mutate shared cache objects
                out = copy.deepcopy(cached_data)
                out['cache_timestamp'] = cache_timestamp
                await overlay_questdb_spot_on_prediction_payload(ticker, out, stock_db)
                if history is not None and is_market_hours(now_et):
                    date_str = now_et.strftime('%Y-%m-%d')
                    await history.add_snapshot(ticker, date_str, out)
                return out
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

        if serialized is None:
            logger.warning(f"predict_close returned None for {ticker} (no data or QuestDB unavailable)")
            return {'error': f'No prediction available for {ticker}', 'ticker': ticker}

        # Only cache if we got a valid price — don't persist zeros to disk.
        if not (serialized.get('current_price') and serialized['current_price'] > 0):
            # Outside market hours, refresh the existing cache entry's timestamp
            # so "Last Updated" stays current even when live price is unavailable.
            if not is_market_hours():
                cached_with_ts = await cache.get_with_timestamp(cache_key)
                if cached_with_ts is not None:
                    cached_data, _ = cached_with_ts
                    if isinstance(cached_data, dict) and cached_data.get('current_price', 0) > 0:
                        # Merge fresh band data into cached entry (handles new fields
                        # like empirical_continuous_bands added after cache was written)
                        merged = copy.deepcopy(cached_data)
                        for band_key in ('percentile_bands', 'statistical_bands',
                                         'combined_bands', 'empirical_continuous_bands',
                                         'ensemble_methods', 'directional_analysis'):
                            fresh_val = serialized.get(band_key)
                            if fresh_val and band_key not in merged:
                                merged[band_key] = fresh_val
                        merged['cache_timestamp'] = time.time()
                        await cache.set(cache_key, merged)
                        logger.info(f"Refreshed cache timestamp for {cache_key} (market closed, reusing last valid price)")
                        await overlay_questdb_spot_on_prediction_payload(ticker, merged, stock_db)
                        return merged
            logger.warning(f"Prediction for {ticker} returned current_price={serialized.get('current_price')}; skipping cache write.")
            serialized['cache_timestamp'] = time.time()
            await overlay_questdb_spot_on_prediction_payload(ticker, serialized, stock_db)
            return serialized

        # Add current timestamp as cache_timestamp for fresh data
        serialized['cache_timestamp'] = time.time()

        # Persist raw model output; overlay mutates the in-memory dict for the response only
        await cache.set(cache_key, copy.deepcopy(serialized))
        await overlay_questdb_spot_on_prediction_payload(ticker, serialized, stock_db)

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


async def fetch_future_prediction(ticker: str, days_ahead: int, cache: PredictionCache, force_refresh: bool = False, lookback: int = 150):
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

        if serialized is None:
            logger.warning(f"predict_close returned None for {ticker}+{days_ahead}d")
            return {'error': f'No prediction available for {ticker} +{days_ahead}d', 'ticker': ticker}

        # Override current_price with the live price from today's prediction.
        # _predict_future_close_unified may fall back to yesterday's CSV close
        # when QuestDB returns None (market closed), but today's cached prediction
        # has the correct live/last-traded price.
        if current_price and current_price > 0:
            old_price = serialized.get('current_price', 0)
            if old_price and old_price > 0 and abs(current_price - old_price) / old_price > 0.001:
                logger.info(
                    f"Overriding current_price for {ticker}+{days_ahead}d: "
                    f"${old_price:,.2f} -> ${current_price:,.2f} (from today's prediction)"
                )
                serialized['current_price'] = current_price
                # Recalculate expected_price using the live price
                mean_return = serialized.get('mean_return')
                if mean_return is not None:
                    serialized['expected_price'] = current_price * (1 + mean_return / 100)
                # Recalculate band prices relative to live price
                for method_key in ('ensemble_methods',):
                    methods = serialized.get(method_key, [])
                    if isinstance(methods, list):
                        for method in methods:
                            bands = method.get('bands', {})
                            if isinstance(bands, dict):
                                for band_name, band in bands.items():
                                    if isinstance(band, dict) and 'lo_pct' in band and 'hi_pct' in band:
                                        band['lo_price'] = current_price * (1 + band['lo_pct'] / 100)
                                        band['hi_price'] = current_price * (1 + band['hi_pct'] / 100)
                                        band['width_pts'] = band['hi_price'] - band['lo_price']
                # Also update primary bands (percentile_bands, combined_bands, statistical_bands, empirical_continuous_bands)
                for band_key in ('percentile_bands', 'combined_bands', 'statistical_bands', 'empirical_continuous_bands'):
                    bands = serialized.get(band_key, {})
                    if isinstance(bands, dict):
                        for band_name, band in bands.items():
                            if isinstance(band, dict) and 'lo_pct' in band and 'hi_pct' in band:
                                band['lo_price'] = current_price * (1 + band['lo_pct'] / 100)
                                band['hi_price'] = current_price * (1 + band['hi_pct'] / 100)
                                band['width_pts'] = band['hi_price'] - band['lo_price']

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


async def prewarm_one_ticker_async(
    ticker: str,
    lookback: int,
    prewarm_days: List[int],
    cache: PredictionCache,
    history: Optional[PredictionHistory],
) -> Dict[str, Any]:
    """Warm today + configured future horizons for one ticker; JSON shape matches prewarm API."""
    try:
        today_result = await fetch_today_prediction(
            ticker, cache, force_refresh=True, history=history, lookback=lookback
        )
        future_results: Dict[str, str] = {}
        for days in prewarm_days:
            fr = await fetch_future_prediction(
                ticker, days, cache, force_refresh=True, lookback=lookback
            )
            future_results[f'{days}d'] = (
                'ok' if isinstance(fr, dict) and 'error' not in fr else 'error'
            )
        today_ok = isinstance(today_result, dict) and 'error' not in today_result
        return {
            'status': 'ok' if today_ok else 'error',
            'today': 'ok' if today_ok else 'error',
            'future': future_results,
            'timestamp': datetime.now(ET_TZ).isoformat(),
        }
    except Exception as e:
        logger.exception("Prewarm failed for %s", ticker)
        return {
            'status': 'error',
            'message': str(e),
            'today': 'error',
            'future': {},
            'timestamp': datetime.now(ET_TZ).isoformat(),
        }


def _prewarm_ticker_worker_job(
    args: Tuple[str, int, Tuple[int, ...], str],
) -> Dict[str, Any]:
    """Run ``prewarm_one_ticker_async`` in a subprocess (picklable entry point).

    Uses a disk-only PredictionCache at ``disk_cache_dir`` so warmed JSON files are
    shared with the web app (merged with Redis/memory on read when configured).

    Args:
        args: (ticker, lookback, prewarm_days, disk_cache_dir)
    """
    ticker, lookback, prewarm_days, disk_cache_dir = args
    if not PREDICTIONS_AVAILABLE:
        return {
            'status': 'error',
            'message': 'Predictions module not available',
            'today': 'error',
            'future': {},
            'timestamp': datetime.now(ET_TZ).isoformat(),
        }

    async def _run() -> Dict[str, Any]:
        cache = PredictionCache(backends=['disk'], cache_dir=disk_cache_dir)
        return await prewarm_one_ticker_async(
            ticker, lookback, list(prewarm_days), cache, None
        )

    try:
        return asyncio.run(_run())
    except Exception as e:
        logger.exception("Prewarm worker failed for %s", ticker)
        return {
            'status': 'error',
            'message': str(e),
            'today': 'error',
            'future': {},
            'timestamp': datetime.now(ET_TZ).isoformat(),
        }


async def fetch_all_predictions(ticker: str, cache: PredictionCache, future_days=None):
    """Fetch predictions for all timeframes in parallel."""
    if not PREDICTIONS_AVAILABLE:
        return {'error': 'Predictions module not available'}

    try:
        # Fetch all predictions in parallel
        if future_days is None:
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


# ============================================================================
# Historical Prediction (past-date backtest for the predictions page)
# ============================================================================

def _compute_historical_predictions_sync(ticker: str, date_str: str, lookback: int) -> dict:
    """Compute 0DTE prediction bands for a past trading day.

    Trains models and generates predictions at each half-hour time slot,
    producing the same data structure the live band convergence chart uses.

    This is CPU-intensive and I/O-heavy (reads many CSVs); call via run_in_executor.

    Returns:
        dict with keys: snapshots, actual_close, actual_prices, latest_prediction
        or dict with key: error
    """
    import pandas as pd
    from zoneinfo import ZoneInfo

    ET = ZoneInfo("America/New_York")
    UTC = ZoneInfo("UTC")

    try:
        from scripts.close_predictor.prediction import (
            _train_statistical,
            make_unified_prediction,
        )
        from scripts.csv_prediction_backtest import (
            load_csv_data,
            get_available_dates,
            get_day_close,
            get_day_open,
            get_previous_close,
            get_first_hour_range,
            get_opening_range,
            get_price_at_time,
            get_vix1d_at_time,
            get_historical_context,
            DayContext,
        )
        from scripts.percentile_range_backtest import (
            collect_all_data,
            get_price_at_slot,
            TIME_SLOTS,
            HOURS_TO_CLOSE,
        )
    except ImportError as e:
        return {'error': f'Required prediction modules not available: {e}'}

    # Load target day's CSV data
    df = load_csv_data(ticker, date_str)
    if df is None or df.empty:
        return {'error': f'No CSV data available for {ticker} on {date_str}'}

    actual_close = get_day_close(df)
    prev_close = get_previous_close(ticker, date_str)
    if prev_close is None:
        return {'error': f'No previous close available for {ticker} before {date_str}'}

    # Get all available dates and find target index
    all_dates = get_available_dates(ticker, 2000)
    try:
        target_idx = all_dates.index(date_str)
    except ValueError:
        return {'error': f'Date {date_str} not found in available trading dates'}

    if target_idx < 10:
        return {'error': f'Not enough historical data before {date_str} for training'}

    # Training window: lookback days before the target date
    train_start = max(0, target_idx - lookback)
    train_date_list = all_dates[train_start:target_idx]
    train_dates_set = set(train_date_list)

    # Train statistical/LightGBM model (uses dates strictly before date_str)
    predictor = _train_statistical(ticker, date_str, lookback)

    # Collect percentile data (include target date for realized_vol computation)
    dates_for_pct = all_dates[train_start:target_idx + 1]
    pct_df = collect_all_data(ticker, dates_for_pct)

    if pct_df is None or pct_df.empty:
        return {'error': f'Could not collect percentile data for {ticker}'}

    # Extract realized vol for target date
    current_vol = None
    target_vol_rows = pct_df[pct_df['date'] == date_str]
    if not target_vol_rows.empty and 'realized_vol' in target_vol_rows.columns:
        vol_val = target_vol_rows.iloc[0]['realized_vol']
        if not pd.isna(vol_val):
            current_vol = float(vol_val)

    # Build DayContext from historical data
    hist_ctx = get_historical_context(ticker, date_str)
    day_open = get_day_open(df)
    vix1d = get_vix1d_at_time(date_str, df.iloc[0]['timestamp'].to_pydatetime())
    fh_high, fh_low = get_first_hour_range(df)
    or_high, or_low = get_opening_range(df)
    price_945 = get_price_at_time(df, 9, 45)

    day_ctx = DayContext(
        prev_close=prev_close,
        day_open=day_open,
        vix1d=vix1d,
        prev_day_close=hist_ctx.get('day_2', {}).get('close'),
        prev_vix1d=hist_ctx.get('day_1', {}).get('vix1d'),
        prev_day_high=hist_ctx.get('day_1', {}).get('high'),
        prev_day_low=hist_ctx.get('day_1', {}).get('low'),
        close_5days_ago=hist_ctx.get('day_5', {}).get('close'),
        first_hour_high=fh_high,
        first_hour_low=fh_low,
        opening_range_high=or_high,
        opening_range_low=or_low,
        price_at_945=price_945,
        ma5=hist_ctx.get('ma5'),
        ma10=hist_ctx.get('ma10'),
        ma20=hist_ctx.get('ma20'),
        ma50=hist_ctx.get('ma50'),
    )

    # Generate predictions for each half-hour time slot
    snapshots = []
    latest_prediction = None

    for hour_et, minute_et in TIME_SLOTS:
        time_label = f"{hour_et}:{minute_et:02d}"

        # Get current price at this slot
        current_price = get_price_at_slot(df, hour_et, minute_et)
        if current_price is None:
            continue

        # Create timezone-aware datetime for this slot
        slot_dt = datetime(
            int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10]),
            hour_et, minute_et, 0, tzinfo=ET
        )
        slot_utc = slot_dt.astimezone(UTC)

        # Compute intraday high/low up to this slot from CSV data
        before = df[df['timestamp'] <= slot_utc]
        market_before = before[before['timestamp'].dt.hour >= 14]
        if market_before.empty:
            market_before = before
        if market_before.empty:
            continue

        day_high = float(market_before['high'].max())
        day_low = float(market_before['low'].min())

        # Make unified prediction (same function used by live predictions)
        pred = make_unified_prediction(
            pct_df=pct_df,
            predictor=predictor,
            ticker=ticker,
            current_price=current_price,
            prev_close=prev_close,
            current_time=slot_dt,
            time_label=time_label,
            day_ctx=day_ctx,
            day_high=day_high,
            day_low=day_low,
            train_dates=train_dates_set,
            current_vol=current_vol,
            vol_scale=True,
            data_source="csv_historical",
        )

        if pred is None:
            continue

        serialized = _serialize_unified_prediction(pred)

        # Create snapshot matching PredictionHistory.add_snapshot shape
        snapshot = {
            'timestamp': slot_dt.isoformat(),
            'time_label': time_label,
            'current_price': float(current_price),
            'hours_to_close': HOURS_TO_CLOSE.get(time_label, 0),
            'combined_bands': serialized.get('combined_bands'),
            'percentile_bands': serialized.get('percentile_bands'),
            'statistical_bands': serialized.get('statistical_bands'),
            'empirical_continuous_bands': serialized.get('empirical_continuous_bands'),
        }
        snapshots.append(snapshot)
        latest_prediction = serialized

    if not snapshots:
        return {'error': f'Could not generate any predictions for {ticker} on {date_str}'}

    # Extract actual intraday prices from CSV for chart overlay
    actual_prices = []
    for _, row in df.iterrows():
        price = float(row['close'])
        if price > 0:
            ts = row['timestamp']
            ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
            if '+' not in ts_str and 'Z' not in ts_str:
                ts_str = ts_str + '+00:00'
            actual_prices.append({'timestamp': ts_str, 'price': price})

    return {
        'snapshots': snapshots,
        'actual_close': float(actual_close),
        'actual_prices': actual_prices,
        'latest_prediction': latest_prediction,
    }


async def fetch_historical_prediction(
    ticker: str, date_str: str, cache: 'PredictionCache', lookback: int = 150
) -> dict:
    """Fetch historical prediction for a past trading day.

    Runs the expensive computation in a thread-pool executor and caches
    the result indefinitely (historical data is immutable).

    Returns a flat dict that merges the latest slot's prediction data
    with chart data (snapshots, actual_prices) so the frontend can use
    it for both summary/table rendering AND the convergence chart.
    """
    cache_key = f"historical_{ticker}_{date_str}_{lookback}"

    # Check cache first — historical data never expires
    cached = await cache.get(cache_key)
    if cached is not None:
        return cached

    # Run CPU/IO-intensive computation in executor
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _compute_historical_predictions_sync(ticker, date_str, lookback),
    )

    if 'error' in result:
        return result

    # Build response: spread latest prediction's fields + historical metadata
    latest = result.get('latest_prediction') or {}
    response = {
        **latest,
        'is_historical': True,
        'date': date_str,
        'actual_close': result['actual_close'],
        'snapshots': result['snapshots'],
        'actual_prices': result['actual_prices'],
        'cache_timestamp': time.time(),
    }

    # Cache the result permanently (historical data is immutable)
    await cache.set(cache_key, response)

    return response
