"""Tests for short code generation."""

import pytest
from lib.shortcode import ShortCodeGenerator


class TestShortCodeGenerator:
    """Test short code generation."""
    
    def test_generate_random(self):
        """Test random code generation."""
        generator = ShortCodeGenerator(default_length=6)
        
        code = generator.generate_random()
        assert len(code) == 6
        assert generator.is_valid_format(code)
    
    def test_generate_random_custom_length(self):
        """Test random code with custom length."""
        generator = ShortCodeGenerator(default_length=6)
        
        code = generator.generate_random(length=10)
        assert len(code) == 10
        assert generator.is_valid_format(code)
    
    def test_generate_from_uuid(self):
        """Test UUID-based generation."""
        generator = ShortCodeGenerator(default_length=8)
        
        code = generator.generate_from_uuid()
        assert len(code) == 8
        assert generator.is_valid_format(code)
    
    def test_generate_from_url(self):
        """Test URL-based generation (deterministic)."""
        generator = ShortCodeGenerator(default_length=6)
        
        url = "https://example.com/test"
        code1 = generator.generate_from_url(url)
        code2 = generator.generate_from_url(url)
        
        # Should be deterministic
        assert code1 == code2
        assert len(code1) == 6
        assert generator.is_valid_format(code1)
    
    def test_generate_sequential(self):
        """Test sequential generation."""
        generator = ShortCodeGenerator(default_length=6)
        
        code = generator.generate_sequential(123)
        assert len(code) >= 6
        assert generator.is_valid_format(code)
    
    def test_base62_conversion(self):
        """Test base62 encoding/decoding."""
        generator = ShortCodeGenerator()
        
        # Test round-trip conversion
        num = 123456
        code = generator._int_to_base62(num)
        decoded = generator._base62_to_int(code)
        
        assert decoded == num
    
    def test_is_valid_format(self):
        """Test format validation."""
        assert ShortCodeGenerator.is_valid_format("abc123")
        assert ShortCodeGenerator.is_valid_format("ABC_123")
        assert ShortCodeGenerator.is_valid_format("test-code")
        
        # Invalid formats
        assert not ShortCodeGenerator.is_valid_format("abc 123")
        assert not ShortCodeGenerator.is_valid_format("abc@123")
        assert not ShortCodeGenerator.is_valid_format("abc#123")




