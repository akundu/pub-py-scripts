"""Short code generation utilities."""

import random
import string
import hashlib
import uuid
from typing import Optional


class ShortCodeGenerator:
    """Generate short codes for URLs."""
    
    # Base62 characters (alphanumeric, case-sensitive)
    BASE62_CHARS = string.ascii_letters + string.digits  # a-zA-Z0-9
    
    def __init__(self, default_length: int = 6):
        """Initialize short code generator.
        
        Args:
            default_length: Default length for generated codes
        """
        self.default_length = default_length
    
    def generate_random(self, length: Optional[int] = None) -> str:
        """Generate a random short code.
        
        Args:
            length: Length of the code (uses default if not specified)
            
        Returns:
            Random short code
        """
        length = length or self.default_length
        return ''.join(random.choices(self.BASE62_CHARS, k=length))
    
    def generate_from_uuid(self, length: Optional[int] = None) -> str:
        """Generate short code from UUID.
        
        Args:
            length: Length of the code (uses default if not specified)
            
        Returns:
            Short code based on UUID
        """
        length = length or self.default_length
        
        # Generate UUID and convert to integer
        uuid_int = uuid.uuid4().int
        
        # Convert to base62
        code = self._int_to_base62(uuid_int)
        
        # Return first N characters
        return code[:length]
    
    def generate_from_url(self, url: str, length: Optional[int] = None) -> str:
        """Generate short code from URL hash.
        
        This provides deterministic codes for the same URL, but may have collisions.
        
        Args:
            url: The URL to hash
            length: Length of the code (uses default if not specified)
            
        Returns:
            Short code based on URL hash
        """
        length = length or self.default_length
        
        # Hash the URL
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        
        # Convert hex to int, then to base62
        hash_int = int(url_hash, 16)
        code = self._int_to_base62(hash_int)
        
        # Return first N characters
        return code[:length]
    
    def generate_sequential(self, sequence_number: int, length: Optional[int] = None) -> str:
        """Generate short code from sequence number.
        
        Useful for auto-incrementing IDs.
        
        Args:
            sequence_number: Sequential ID
            length: Minimum length of the code (uses default if not specified)
            
        Returns:
            Short code based on sequence number
        """
        length = length or self.default_length
        
        # Convert sequence to base62
        code = self._int_to_base62(sequence_number)
        
        # Pad with zeros if needed
        if len(code) < length:
            code = code.rjust(length, '0')
        
        return code
    
    def _int_to_base62(self, num: int) -> str:
        """Convert integer to base62 string.
        
        Args:
            num: Integer to convert
            
        Returns:
            Base62 string
        """
        if num == 0:
            return self.BASE62_CHARS[0]
        
        result = []
        base = len(self.BASE62_CHARS)
        
        while num > 0:
            remainder = num % base
            result.append(self.BASE62_CHARS[remainder])
            num = num // base
        
        return ''.join(reversed(result))
    
    def _base62_to_int(self, code: str) -> int:
        """Convert base62 string to integer.
        
        Args:
            code: Base62 string
            
        Returns:
            Integer value
        """
        result = 0
        base = len(self.BASE62_CHARS)
        
        for char in code:
            result = result * base + self.BASE62_CHARS.index(char)
        
        return result
    
    @staticmethod
    def is_valid_format(code: str) -> bool:
        """Check if code has valid format (alphanumeric).
        
        Args:
            code: Code to validate
            
        Returns:
            True if valid format
        """
        return all(c in ShortCodeGenerator.BASE62_CHARS or c in '-_' for c in code)






