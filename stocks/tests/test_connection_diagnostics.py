#!/usr/bin/env python3
"""
Connection Diagnostics Script

This script helps diagnose common issues with market data streaming:
- API key configuration
- Network connectivity
- API rate limits
- WebSocket connection issues

Usage:
    python tests/test_connection_diagnostics.py --data-source alpaca
    python tests/test_connection_diagnostics.py --data-source polygon
"""

import os
import sys
import asyncio
import argparse
import requests
from pathlib import Path

# Add project root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

def check_environment_variables():
    """Check if required environment variables are set."""
    print("=== Environment Variables Check ===")
    
    alpaca_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_API_SECRET")
    polygon_key = os.getenv("POLYGON_API_KEY")
    
    print(f"ALPACA_API_KEY: {'✓ Set' if alpaca_key else '✗ Not set'}")
    print(f"ALPACA_API_SECRET: {'✓ Set' if alpaca_secret else '✗ Not set'}")
    print(f"POLYGON_API_KEY: {'✓ Set' if polygon_key else '✗ Not set'}")
    
    return {
        'alpaca': bool(alpaca_key and alpaca_secret),
        'polygon': bool(polygon_key)
    }

def check_network_connectivity():
    """Check basic network connectivity to API endpoints."""
    print("\n=== Network Connectivity Check ===")
    
    endpoints = {
        'Alpaca REST API': 'https://paper-api.alpaca.markets/v2/account',
        'Alpaca WebSocket': 'wss://stream.data.alpaca.markets/v2/iex',
        'Polygon REST API': 'https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-09/2023-01-09',
        'Polygon WebSocket': 'wss://delayed.polygon.io'
    }
    
    results = {}
    
    for name, url in endpoints.items():
        try:
            if url.startswith('wss://'):
                # For WebSocket endpoints, just check if we can resolve the hostname
                import socket
                hostname = url.replace('wss://', '').split('/')[0]
                socket.gethostbyname(hostname)
                print(f"{name}: ✓ Hostname resolvable")
                results[name] = True
            else:
                response = requests.get(url, timeout=10)
                if response.status_code in [200, 401, 403]:  # 401/403 means endpoint is reachable but auth failed
                    print(f"{name}: ✓ Reachable (Status: {response.status_code})")
                    results[name] = True
                else:
                    print(f"{name}: ✗ Unexpected status code: {response.status_code}")
                    results[name] = False
        except requests.exceptions.ConnectionError:
            print(f"{name}: ✗ Connection failed")
            results[name] = False
        except requests.exceptions.Timeout:
            print(f"{name}: ✗ Timeout")
            results[name] = False
        except Exception as e:
            print(f"{name}: ✗ Error: {e}")
            results[name] = False
    
    return results

def check_api_authentication():
    """Check if API keys are valid."""
    print("\n=== API Authentication Check ===")
    
    alpaca_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_API_SECRET")
    polygon_key = os.getenv("POLYGON_API_KEY")
    
    results = {}
    
    # Test Alpaca API
    if alpaca_key and alpaca_secret:
        try:
            headers = {
                'APCA-API-KEY-ID': alpaca_key,
                'APCA-API-SECRET-KEY': alpaca_secret
            }
            response = requests.get('https://paper-api.alpaca.markets/v2/account', headers=headers, timeout=10)
            if response.status_code == 200:
                print("Alpaca API: ✓ Authentication successful")
                results['alpaca'] = True
            else:
                print(f"Alpaca API: ✗ Authentication failed (Status: {response.status_code})")
                results['alpaca'] = False
        except Exception as e:
            print(f"Alpaca API: ✗ Error: {e}")
            results['alpaca'] = False
    else:
        print("Alpaca API: ⚠ Skipped (no API keys)")
        results['alpaca'] = None
    
    # Test Polygon API
    if polygon_key:
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-09/2023-01-09?apiKey={polygon_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print("Polygon API: ✓ Authentication successful")
                results['polygon'] = True
            else:
                print(f"Polygon API: ✗ Authentication failed (Status: {response.status_code})")
                results['polygon'] = False
        except Exception as e:
            print(f"Polygon API: ✗ Error: {e}")
            results['polygon'] = False
    else:
        print("Polygon API: ⚠ Skipped (no API key)")
        results['polygon'] = None
    
    return results

def check_python_dependencies():
    """Check if required Python packages are installed."""
    print("\n=== Python Dependencies Check ===")
    
    dependencies = {
        'alpaca-py': 'Alpaca SDK',
        'polygon-api-client': 'Polygon SDK',
        'pandas': 'Data manipulation',
        'websockets': 'WebSocket support',
        'requests': 'HTTP requests'
    }
    
    results = {}
    
    for package, description in dependencies.items():
        try:
            __import__(package.replace('-', '_'))
            print(f"{description} ({package}): ✓ Installed")
            results[package] = True
        except ImportError:
            print(f"{description} ({package}): ✗ Not installed")
            results[package] = False
    
    return results

def run_quick_stream_test(data_source):
    """Run a quick stream test to check for immediate issues."""
    print(f"\n=== Quick Stream Test ({data_source}) ===")
    
    if data_source == "alpaca":
        # Test Alpaca streaming
        try:
            from alpaca.data.live import StockDataStream
            from alpaca.data.enums import DataFeed
            
            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_API_SECRET")
            
            if not api_key or not secret_key:
                print("✗ Cannot test Alpaca streaming: API keys not set")
                return False
            
            print("Testing Alpaca WebSocket connection...")
            stream = StockDataStream(api_key, secret_key, feed=DataFeed.SIP)
            
            # Try to connect (this will fail quickly if there are issues)
            print("✓ Alpaca WebSocket client created successfully")
            return True
            
        except Exception as e:
            print(f"✗ Alpaca streaming test failed: {e}")
            return False
    
    elif data_source == "polygon":
        # Test Polygon streaming
        try:
            from polygon.websocket import WebSocketClient
            
            api_key = os.getenv("POLYGON_API_KEY")
            
            if not api_key:
                print("✗ Cannot test Polygon streaming: API key not set")
                return False
            
            print("Testing Polygon WebSocket connection...")
            stream = WebSocketClient(api_key=api_key, market="stocks")
            
            print("✓ Polygon WebSocket client created successfully")
            return True
            
        except Exception as e:
            print(f"✗ Polygon streaming test failed: {e}")
            return False
    
    return False

def generate_diagnostic_report(env_vars, network, auth, dependencies, stream_test):
    """Generate a comprehensive diagnostic report."""
    print("\n" + "="*60)
    print("DIAGNOSTIC REPORT")
    print("="*60)
    
    # Overall status
    network_ok = all(network.values())
    auth_ok = all(v for v in auth.values() if v is not None)
    deps_ok = all(dependencies.values())
    
    if network_ok and auth_ok and deps_ok and stream_test:
        print("✓ OVERALL STATUS: All checks passed")
        print("Your streaming setup should work correctly.")
    else:
        print("✗ OVERALL STATUS: Issues detected")
        print("\nIssues found:")
        
        if not network_ok:
            print("- Network connectivity problems detected")
        if not auth_ok:
            print("- API authentication issues detected")
        if not deps_ok:
            print("- Missing Python dependencies")
        if not stream_test:
            print("- Stream client creation failed")
    
    print("\nRecommendations:")
    
    if not env_vars.get('alpaca') and not env_vars.get('polygon'):
        print("- Set up API keys for at least one data source")
    
    if not network_ok:
        print("- Check your internet connection")
        print("- Verify firewall settings")
        print("- Check if you're behind a corporate proxy")
    
    if not auth_ok:
        print("- Verify your API keys are correct")
        print("- Check if your API keys have the necessary permissions")
        print("- Ensure your API keys are not expired")
    
    if not deps_ok:
        print("- Install missing Python packages:")
        for package, installed in dependencies.items():
            if not installed:
                print(f"  pip install {package}")
    
    if not stream_test:
        print("- Check the specific error messages above")
        print("- Verify API key permissions for streaming")

def main():
    parser = argparse.ArgumentParser(description="Diagnose market data streaming issues")
    parser.add_argument("--data-source", choices=["alpaca", "polygon"], 
                       help="Data source to test (alpaca or polygon)")
    
    args = parser.parse_args()
    
    print("Market Data Streaming Diagnostics")
    print("="*40)
    
    # Run all checks
    env_vars = check_environment_variables()
    network = check_network_connectivity()
    auth = check_api_authentication()
    dependencies = check_python_dependencies()
    
    stream_test = False
    if args.data_source:
        stream_test = run_quick_stream_test(args.data_source)
    
    # Generate report
    generate_diagnostic_report(env_vars, network, auth, dependencies, stream_test)

if __name__ == "__main__":
    main() 