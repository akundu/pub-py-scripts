"""Data models for URL shortener."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class URLMapping:
    """Represents a URL mapping in the database."""
    
    short_code: str
    original_url: str
    created_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "short_code": self.short_code,
            "original_url": self.original_url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "URLMapping":
        """Create from dictionary."""
        return cls(
            short_code=data["short_code"],
            original_url=data["original_url"],
            created_at=data["created_at"] if isinstance(data["created_at"], datetime) else datetime.fromisoformat(data["created_at"]),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed"),
        )






