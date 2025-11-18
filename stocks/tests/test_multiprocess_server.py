#!/usr/bin/env python3
"""
Test script to demonstrate multi-process database server functionality.
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any


async def test_health_check(session: aiohttp.ClientSession, port: int) -> Dict[str, Any]:
    """Test the health check endpoint to see process information."""
    async with session.get(f"http://localhost:{port}/health") as response:
        return await response.json()


async def test_database_stats(session: aiohttp.ClientSession, port: int) -> Dict[str, Any]:
    """Test the database stats endpoint."""
    async with session.get(f"http://localhost:{port}/stats/pool") as response:
        return await response.json()


async def test_concurrent_requests(session: aiohttp.ClientSession, port: int, num_requests: int = 10):
    """Test concurrent requests to verify load balancing across workers."""
    print(f"\n🔄 Testing {num_requests} concurrent requests...")
    
    # Create concurrent health check requests
    tasks = []
    for i in range(num_requests):
        task = asyncio.create_task(test_health_check(session, port))
        tasks.append(task)
    
    # Execute all requests concurrently
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.time() - start_time
    
    # Analyze which workers handled requests
    worker_counts = {}
    successful_requests = 0
    
    for result in results:
        if isinstance(result, dict) and 'process' in result:
            successful_requests += 1
            worker_id = result['process'].get('worker_id', 'unknown')
            pid = result['process'].get('pid', 'unknown')
            worker_key = f"Worker-{worker_id} (PID:{pid})"
            worker_counts[worker_key] = worker_counts.get(worker_key, 0) + 1
        elif isinstance(result, Exception):
            print(f"   ❌ Request failed: {result}")
    
    print(f"   ⏱️  {successful_requests}/{num_requests} requests completed in {elapsed:.2f}s")
    print(f"   🏆 Average: {elapsed/num_requests*1000:.1f}ms per request")
    
    if worker_counts:
        print(f"   📊 Load distribution:")
        for worker, count in sorted(worker_counts.items()):
            percentage = (count / successful_requests) * 100
            print(f"      {worker}: {count} requests ({percentage:.1f}%)")
    
    return worker_counts


async def main():
    """Main test function."""
    port = 8080
    
    print("🚀 Multi-Process Database Server Test")
    print("=" * 50)
    print(f"Testing server on http://localhost:{port}")
    print()
    print("⚠️  Make sure the server is running with --workers > 1")
    print("   Example: python db_server.py --db-file test.db --workers 4")
    print()
    
    async with aiohttp.ClientSession() as session:
        try:
            # Test 1: Basic health check
            print("1️⃣ Testing basic connectivity...")
            health = await test_health_check(session, port)
            
            if health.get('status') == 'healthy':
                print("   ✅ Server is healthy")
                
                process_info = health.get('process', {})
                multiprocess_mode = process_info.get('multiprocess_mode', False)
                worker_id = process_info.get('worker_id')
                pid = process_info.get('pid')
                
                print(f"   📋 Process Info:")
                print(f"      Multiprocess mode: {multiprocess_mode}")
                print(f"      Worker ID: {worker_id}")
                print(f"      PID: {pid}")
                
                if not multiprocess_mode:
                    print("   ⚠️  Server is running in single-process mode")
                    print("      Use --workers > 1 to test multi-process features")
                
            else:
                print("   ❌ Server health check failed")
                return
            
            # Test 2: Pool status (if available)
            print("\n2️⃣ Testing pool status...")
            try:
                pool_stats = await test_database_stats(session, port)
                if 'pool_status' in pool_stats:
                    pool = pool_stats['pool_status']
                    print(f"   📊 Connection Pool:")
                    print(f"      Available: {pool.get('available_connections', 'N/A')}")
                    print(f"      Max Size: {pool.get('pool_max_size', 'N/A')}")
                    print(f"      Active: {pool.get('active_connections', 'N/A')}")
                else:
                    print("   ℹ️  Pool status not available")
            except Exception as e:
                print(f"   ⚠️  Pool stats error: {e}")
            
            # Test 3: Concurrent requests
            worker_distribution = await test_concurrent_requests(session, port, 20)
            
            # Test 4: Sustained load
            print("\n3️⃣ Testing sustained load (3 rounds of 15 requests)...")
            total_workers = set()
            
            for round_num in range(1, 4):
                print(f"   Round {round_num}:")
                round_workers = await test_concurrent_requests(session, port, 15)
                total_workers.update(round_workers.keys())
                await asyncio.sleep(1)  # Brief pause between rounds
            
            print(f"\n📈 Summary:")
            print(f"   Total unique workers observed: {len(total_workers)}")
            if len(total_workers) > 1:
                print("   ✅ Load balancing is working across multiple processes!")
            else:
                print("   ℹ️  Only one worker observed (single-process or low load)")
            
            print(f"   Workers: {list(total_workers)}")
            
        except aiohttp.ClientError as e:
            print(f"❌ Connection error: {e}")
            print(f"   Make sure the server is running on port {port}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
