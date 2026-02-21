"""QuestDB implementation for URL shortener."""

import os
import logging
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import asyncpg

from .base import URLShortenerDBBase
from .models import URLMapping


class URLShortenerQuestDB(URLShortenerDBBase):
    """QuestDB implementation for URL shortener database operations."""
    
    # Table creation SQL
    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS url_mappings (
        short_code SYMBOL INDEX CAPACITY 1024,
        original_url STRING,
        created_at TIMESTAMP,
        access_count LONG,
        last_accessed TIMESTAMP
    ) TIMESTAMP(created_at) PARTITION BY DAY WAL
    DEDUP UPSERT KEYS(created_at, short_code);
    """
    
    def __init__(
        self,
        db_config: str,
        pool_max_size: int = 10,
        connection_timeout_seconds: int = 30,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize QuestDB connection.
        
        Args:
            db_config: Database connection string (questdb://user:pass@host:port/db)
            pool_max_size: Maximum size of connection pool
            connection_timeout_seconds: Connection timeout in seconds
            logger: Optional logger instance
        """
        super().__init__(db_config)
        
        self.logger = logger or logging.getLogger(__name__)
        self.pool_max_size = pool_max_size
        self.connection_timeout_seconds = connection_timeout_seconds
        
        # Parse connection string
        self._parse_connection_string(db_config)
        
        # Connection pool (will be created per event loop)
        self._pools: Dict[int, asyncpg.Pool] = {}
        self._pool_lock = asyncio.Lock()
        
        # Check if we should create tables
        create_tables = os.getenv("QUESTDB_CREATE_TABLES", "0")
        self._should_create_tables = create_tables == "1"
        
        if self._should_create_tables:
            self.logger.info("Table creation enabled via QUESTDB_CREATE_TABLES=1")
            # Initialize tables
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._ensure_tables())
            except RuntimeError:
                # No running loop, will create tables on first operation
                pass
    
    def _parse_connection_string(self, db_config: str) -> None:
        """Parse database connection string."""
        parsed = urlparse(db_config)
        
        self.host = parsed.hostname or "localhost"
        self.port = parsed.port or 8812  # QuestDB PostgreSQL wire protocol port
        self.database = parsed.path.lstrip("/") or "qdb"
        self.user = parsed.username or "admin"
        self.password = parsed.password or "quest"
        
        self.logger.debug(
            f"Parsed connection: host={self.host}, port={self.port}, "
            f"database={self.database}, user={self.user}"
        )
    
    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool for current event loop."""
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        
        if loop_id not in self._pools:
            async with self._pool_lock:
                if loop_id not in self._pools:
                    self.logger.debug(f"Creating connection pool for loop {loop_id}")
                    pool = await asyncpg.create_pool(
                        host=self.host,
                        port=self.port,
                        database=self.database,
                        user=self.user,
                        password=self.password,
                        min_size=1,
                        max_size=self.pool_max_size,
                        timeout=self.connection_timeout_seconds,
                        command_timeout=self.connection_timeout_seconds,
                    )
                    self._pools[loop_id] = pool
                    self.logger.debug(f"Connection pool created for loop {loop_id}")
        
        return self._pools[loop_id]
    
    @asynccontextmanager
    async def _get_connection(self):
        """Get a database connection from the pool."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            yield conn
    
    async def _ensure_tables(self) -> None:
        """Create tables if they don't exist (only if QUESTDB_CREATE_TABLES=1)."""
        if not self._should_create_tables:
            self.logger.debug("Table creation disabled (QUESTDB_CREATE_TABLES != 1)")
            return
        
        try:
            self.logger.info("Creating url_mappings table if not exists...")
            async with self._get_connection() as conn:
                await conn.execute(self.CREATE_TABLE_SQL)
            self.logger.info("Table creation completed successfully")
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise
    
    async def create_short_url(
        self,
        short_code: str,
        original_url: str,
        created_at: Optional[datetime] = None,
    ) -> bool:
        """Create a new short URL mapping.
        
        Args:
            short_code: The short code to use
            original_url: The original long URL
            created_at: Optional creation timestamp (defaults to now UTC)
            
        Returns:
            True if created successfully, False if short_code already exists
        """
        if created_at is None:
            created_at = datetime.now(timezone.utc)
        elif created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        
        # Convert to naive UTC for QuestDB
        created_at_naive = created_at.replace(tzinfo=None)
        
        try:
            # Check if short code already exists
            if await self.short_code_exists(short_code):
                self.logger.warning(f"Short code already exists: {short_code}")
                return False
            
            async with self._get_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO url_mappings (short_code, original_url, created_at, access_count, last_accessed)
                    VALUES ($1, $2, $3, $4, NULL)
                    """,
                    short_code,
                    original_url,
                    created_at_naive,
                    0,
                )
            
            self.logger.info(f"Created short URL: {short_code} -> {original_url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating short URL: {e}")
            raise
    
    async def get_original_url(self, short_code: str) -> Optional[str]:
        """Get the original URL for a short code.
        
        Args:
            short_code: The short code to lookup
            
        Returns:
            The original URL if found, None otherwise
        """
        try:
            async with self._get_connection() as conn:
                row = await conn.fetchrow(
                    "SELECT original_url FROM url_mappings WHERE short_code = $1 LIMIT 1",
                    short_code,
                )
                
                if row:
                    return row["original_url"]
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting original URL: {e}")
            return None
    
    async def get_url_mapping(self, short_code: str) -> Optional[Dict[str, Any]]:
        """Get complete URL mapping information.
        
        Args:
            short_code: The short code to lookup
            
        Returns:
            Dictionary with URL mapping data or None if not found
        """
        try:
            async with self._get_connection() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT short_code, original_url, created_at, access_count, last_accessed
                    FROM url_mappings
                    WHERE short_code = $1
                    LIMIT 1
                    """,
                    short_code,
                )
                
                if row:
                    return dict(row)
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting URL mapping: {e}")
            return None
    
    async def short_code_exists(self, short_code: str) -> bool:
        """Check if a short code already exists.
        
        Args:
            short_code: The short code to check
            
        Returns:
            True if exists, False otherwise
        """
        try:
            async with self._get_connection() as conn:
                row = await conn.fetchrow(
                    "SELECT 1 FROM url_mappings WHERE short_code = $1 LIMIT 1",
                    short_code,
                )
                return row is not None
                
        except Exception as e:
            self.logger.error(f"Error checking short code existence: {e}")
            return False
    
    async def increment_access_count(self, short_code: str) -> None:
        """Increment the access count for a short code.
        
        Note: QuestDB doesn't support UPDATE, so we need to INSERT a new record
        with updated values. The DEDUP UPSERT KEYS will handle the upsert.
        
        Args:
            short_code: The short code to update
        """
        try:
            # Get current mapping
            mapping = await self.get_url_mapping(short_code)
            if not mapping:
                self.logger.warning(f"Cannot increment access count - short code not found: {short_code}")
                return
            
            # Increment access count
            new_count = mapping.get("access_count", 0) + 1
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            
            async with self._get_connection() as conn:
                # Insert new record with updated values (DEDUP UPSERT will handle it)
                await conn.execute(
                    """
                    INSERT INTO url_mappings (short_code, original_url, created_at, access_count, last_accessed)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    short_code,
                    mapping["original_url"],
                    mapping["created_at"],
                    new_count,
                    now,
                )
            
            self.logger.debug(f"Incremented access count for {short_code}: {new_count}")
            
        except Exception as e:
            self.logger.error(f"Error incrementing access count: {e}")
    
    async def delete_short_url(self, short_code: str) -> bool:
        """Delete a short URL mapping.
        
        Note: QuestDB doesn't support DELETE in the traditional sense.
        This is a placeholder for potential future implementation.
        
        Args:
            short_code: The short code to delete
            
        Returns:
            False (not implemented for QuestDB)
        """
        self.logger.warning("DELETE operation not supported in QuestDB")
        return False
    
    async def list_recent_urls(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List recently created short URLs.
        
        Args:
            limit: Maximum number of URLs to return
            
        Returns:
            List of URL mapping dictionaries
        """
        try:
            async with self._get_connection() as conn:
                rows = await conn.fetch(
                    """
                    SELECT short_code, original_url, created_at, access_count, last_accessed
                    FROM url_mappings
                    ORDER BY created_at DESC
                    LIMIT $1
                    """,
                    limit,
                )
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error listing recent URLs: {e}")
            return []
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        try:
            async with self._get_connection() as conn:
                # Get total URL count
                total_row = await conn.fetchrow(
                    "SELECT COUNT(*) as total FROM url_mappings"
                )
                total_urls = total_row["total"] if total_row else 0
                
                # Get total access count
                access_row = await conn.fetchrow(
                    "SELECT SUM(access_count) as total FROM url_mappings"
                )
                total_accesses = access_row["total"] if access_row and access_row["total"] else 0
                
                return {
                    "total_urls": total_urls,
                    "total_accesses": total_accesses,
                    "database": "questdb",
                    "status": "healthy",
                }
                
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {
                "total_urls": 0,
                "total_accesses": 0,
                "database": "questdb",
                "status": "error",
                "error": str(e),
            }
    
    async def health_check(self) -> bool:
        """Check if database is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            async with self._get_connection() as conn:
                await conn.fetchrow("SELECT 1")
            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    async def close(self) -> None:
        """Close all database connections."""
        for loop_id, pool in self._pools.items():
            try:
                await pool.close()
                self.logger.debug(f"Closed connection pool for loop {loop_id}")
            except Exception as e:
                self.logger.error(f"Error closing pool for loop {loop_id}: {e}")
        
        self._pools.clear()






