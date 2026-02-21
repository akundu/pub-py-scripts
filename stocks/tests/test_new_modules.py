"""
Quick validation test for newly created modules.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all new modules can be imported."""
    print("Testing imports...")
    
    try:
        from common.web.serializers import dataframe_to_json_records, serialize_mapping_datetime
        print("✓ common.web.serializers")
    except ImportError as e:
        print(f"✗ common.web.serializers: {e}")
        return False
    
    try:
        from common.web.filters import parse_filter_strings, apply_filters
        print("✓ common.web.filters")
    except ImportError as e:
        print(f"✗ common.web.filters: {e}")
        return False
    
    try:
        from common.web.html_generators import format_options_html, generate_stock_info_html
        print("✓ common.web.html_generators")
    except ImportError as e:
        print(f"✗ common.web.html_generators: {e}")
        return False
    
    try:
        from server.logging_config import RequestFormatter, setup_logging
        print("✓ server.logging_config")
    except ImportError as e:
        print(f"✗ server.logging_config: {e}")
        return False
    
    try:
        from server.middleware import logging_middleware
        print("✓ server.middleware")
    except ImportError as e:
        print(f"✗ server.middleware: {e}")
        return False
    
    try:
        from server.websocket_manager import WebSocketManager
        print("✓ server.websocket_manager")
    except ImportError as e:
        print(f"✗ server.websocket_manager: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of new modules."""
    import pandas as pd
    from datetime import datetime
    
    print("\nTesting basic functionality...")
    
    # Test serializers
    from common.web.serializers import dataframe_to_json_records
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    result = dataframe_to_json_records(df)
    assert len(result) == 3
    assert 'a' in result[0]
    assert result[0]['a'] in [1, '1']  # Could be int or string
    print("✓ dataframe_to_json_records works")
    
    # Test filters
    from common.web.filters import parse_filter_strings, apply_filters
    filters = parse_filter_strings("a > 1")
    assert len(filters) == 1
    assert filters[0]['field'] == 'a'
    print("✓ parse_filter_strings works")
    
    filtered = apply_filters(df, filters)
    assert len(filtered) == 2
    print("✓ apply_filters works")
    
    # Test HTML generators
    from common.web.html_generators import format_options_html
    html = format_options_html({})
    assert 'No options data available' in html
    print("✓ format_options_html works")
    
    # Test logging config
    from server.logging_config import RequestFormatter
    formatter = RequestFormatter()
    assert formatter is not None
    print("✓ RequestFormatter works")
    
    # Test WebSocket manager
    from server.websocket_manager import WebSocketManager
    ws_manager = WebSocketManager()
    assert ws_manager.heartbeat_interval == 1.0
    print("✓ WebSocketManager works")
    
    return True

if __name__ == '__main__':
    print("=" * 60)
    print("VALIDATING NEW MODULES")
    print("=" * 60)
    
    if not test_imports():
        print("\n❌ Import test failed!")
        sys.exit(1)
    
    if not test_basic_functionality():
        print("\n❌ Functionality test failed!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    sys.exit(0)

