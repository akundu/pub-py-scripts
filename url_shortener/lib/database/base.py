"""Abstract base class for URL shortener database implementations."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime


class URLShortenerDBBase(ABC):
    """Abstract base class for URL shortener database operations."""
    
    def __init__(self, db_config: str):
        """Initialize database connection.
        
        Args:
            db_config: Database connection string
        """
        self.db_config = db_config
    
    @abstractmethod
    async def create_short_url(
        self, 
        short_code: str, 
        original_url: str,
        created_at: Optional[datetime] = None
    ) -> bool:
        """Create a new short URL mapping.
        
        Args:
            short_code: The short code to use
            original_url: The original long URL
            created_at: Optional creation timestamp (defaults to now)
            
        Returns:
            True if created successfully, False if short_code already exists
        """
        pass
    
    @abstractmethod
    async def get_original_url(self, short_code: str) -> Optional[str]:
        """Get the original URL for a short code.
        
        Args:
            short_code: The short code to lookup
            
        Returns:
            The original URL if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_url_mapping(self, short_code: str) -> Optional[Dict[str, Any]]:
        """Get complete URL mapping information.
        
        Args:
            short_code: The short code to lookup
            
        Returns:
            Dictionary with URL mapping data or None if not found
        """
        pass
    
    @abstractmethod
    async def short_code_exists(self, short_code: str) -> bool:
        """Check if a short code already exists.
        
        Args:
            short_code: The short code to check
            
        Returns:
            True if exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def increment_access_count(self, short_code: str) -> None:
        """Increment the access count for a short code.
        
        Args:
            short_code: The short code to update
        """
        pass
    
    @abstractmethod
    async def delete_short_url(self, short_code: str) -> bool:
        """Delete a short URL mapping.
        
        Args:
            short_code: The short code to delete
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    async def list_recent_urls(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List recently created short URLs.
        
        Args:
            limit: Maximum number of URLs to return
            
        Returns:
            List of URL mapping dictionaries
        """
        pass
    
    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dictionary with statistics (total_urls, total_accesses, etc.)
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close database connections."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if database is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        pass






