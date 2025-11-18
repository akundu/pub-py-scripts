#!/usr/bin/env python3
"""
Test script to verify logging functionality in ticker.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import the setup_logging function from ticker.py
import importlib.util
spec = importlib.util.spec_from_file_location("ticker", "ticker.py")
ticker_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ticker_module)

def test_logging_levels():
    """Test different logging levels."""
    print("=== Testing Logging Levels ===\n")
    
    # Test each log level
    levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    for level in levels:
        print(f"\n--- Testing {level} level ---")
        logger = ticker_module.setup_logging(level)
        
        # Test all log levels
        logger.debug("This is a DEBUG message")
        logger.info("This is an INFO message")
        logger.warning("This is a WARNING message")
        logger.error("This is an ERROR message")
        logger.critical("This is a CRITICAL message")
    
    print("\n=== Logging Test Complete ===")

if __name__ == "__main__":
    test_logging_levels() 