#!/usr/bin/env python3
"""
Test script for the new database statistics API endpoints.
This script demonstrates how to use the new /stats/* endpoints.
"""

import requests
import json
import time
import argparse
from typing import Dict, Any


def test_endpoint(base_url: str, endpoint: str, timeout: int = None) -> Dict[str, Any]:
    """Test a stats endpoint and return the response."""
    url = f"{base_url}{endpoint}"
    if timeout:
        url += f"?timeout={timeout}"
    
    print(f"\n🔍 Testing: {url}")
    start_time = time.time()
    
    try:
        response = requests.get(url, timeout=30)  # HTTP client timeout
        elapsed = (time.time() - start_time) * 1000
        
        print(f"   HTTP Status: {response.status_code}")
        print(f"   Client Time: {elapsed:.2f}ms")
        
        if response.status_code == 200:
            data = response.json()
            server_time = data.get('execution_time_ms', 'N/A')
            print(f"   Server Time: {server_time}ms")
            return data
        else:
            error_data = response.json()
            print(f"   Error: {error_data.get('error', 'Unknown error')}")
            return error_data
            
    except requests.exceptions.Timeout:
        print(f"   ❌ HTTP Client timeout after 30 seconds")
        return {"error": "Client timeout"}
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Connection failed - is the server running?")
        return {"error": "Connection failed"}
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Test database statistics API endpoints")
    parser.add_argument('--host', default='localhost', help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=8080, help='Server port (default: 8080)')
    parser.add_argument('--timeout', type=int, help='Custom timeout for endpoints (optional)')
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    
    print("=" * 60)
    print("Database Statistics API Endpoints Test")
    print("=" * 60)
    print(f"Server: {base_url}")
    if args.timeout:
        print(f"Custom timeout: {args.timeout} seconds")
    
    # Test all endpoints
    endpoints = [
        ("/stats/database", "Comprehensive Database Statistics"),
        ("/stats/tables", "Fast Table Counts"),
        ("/stats/performance", "Performance Test Results"),
        ("/stats/pool", "Connection Pool & Cache Status")
    ]
    
    results = {}
    
    for endpoint, description in endpoints:
        print(f"\n📊 {description}")
        print("-" * 40)
        
        # Test with default timeout
        result = test_endpoint(base_url, endpoint, args.timeout)
        results[endpoint] = result
        
        # Show key data from successful responses
        if 'error' not in result:
            if endpoint == "/stats/database":
                stats = result.get('stats', {})
                if 'table_counts' in stats:
                    print("   📋 Table Counts:")
                    for table, count in stats['table_counts'].items():
                        print(f"      {table}: {count:,} rows")
                        
                if 'count_accuracy' in stats:
                    print("   ✅ Count Accuracy:")
                    for table, accurate in stats['count_accuracy'].items():
                        status = "✅ Accurate" if accurate else "❌ Needs refresh"
                        print(f"      {table}: {status}")
                        
            elif endpoint == "/stats/tables":
                table_counts = result.get('table_counts', {})
                total = result.get('total_tables', 0)
                print(f"   📋 Found {total} tables:")
                for table, count in table_counts.items():
                    print(f"      {table}: {count:,} rows")
                    
            elif endpoint == "/stats/performance":
                perf_tests = result.get('performance_tests', {})
                if perf_tests:
                    print("   🚀 Performance Improvements:")
                    for test_name, improvement in perf_tests.items():
                        print(f"      {test_name}: {improvement:.1f}x faster")
                        
            elif endpoint == "/stats/pool":
                pool_status = result.get('pool_status', {})
                cache_status = result.get('cache_status', {})
                
                if pool_status:
                    print("   🏊 Connection Pool:")
                    print(f"      Size: {pool_status.get('pool_current_size', 0)}/{pool_status.get('pool_max_size', 0)}")
                    print(f"      Active: {pool_status.get('active_connections', 0)}")
                    print(f"      Stale: {pool_status.get('stale_connections', 0)}")
                    
                if cache_status:
                    print("   💾 Table Cache:")
                    valid = cache_status.get('cache_valid', False)
                    remaining = cache_status.get('remaining_cache_minutes', 0)
                    print(f"      Status: {'✅ Valid' if valid else '❌ Invalid'}")
                    if valid:
                        print(f"      Remaining: {remaining:.1f} minutes")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Test Summary")
    print("=" * 60)
    
    success_count = sum(1 for result in results.values() if 'error' not in result)
    total_count = len(results)
    
    print(f"Successful endpoints: {success_count}/{total_count}")
    
    if success_count < total_count:
        print("\n❌ Failed endpoints:")
        for endpoint, result in results.items():
            if 'error' in result:
                print(f"   {endpoint}: {result['error']}")
    
    if success_count > 0:
        print(f"\n✅ The new statistics endpoints are working!")
        print("You can now use these endpoints for:")
        print("   • Database monitoring dashboards")
        print("   • Performance analysis")
        print("   • Health checks")
        print("   • Connection pool monitoring")
        
        print(f"\nExample usage:")
        print(f"   curl {base_url}/stats/database")
        print(f"   curl {base_url}/stats/tables")
        print(f"   curl {base_url}/stats/pool")


if __name__ == "__main__":
    main()
