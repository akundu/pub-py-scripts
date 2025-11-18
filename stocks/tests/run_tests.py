#!/usr/bin/env python3
"""
Test runner for QuestDB tests.
Can be run standalone or via pytest.
"""

import sys
import os
import asyncio
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Run all tests."""
    # Run pytest with verbose output
    exit_code = pytest.main([
        'tests/test_questdb_db.py',
        '-v',
        '--tb=short',
        '--color=yes',
        '-x'  # Stop on first failure
    ])
    sys.exit(exit_code)


if __name__ == '__main__':
    main()

