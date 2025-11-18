#!/usr/bin/env python3
"""
Test Redis Cluster with async Python redis client
Validates that the cluster is up and running properly
"""

import asyncio
import time
from redis.asyncio.cluster import RedisCluster, ClusterNode
from redis.exceptions import RedisError, ConnectionError, ClusterDownError


async def test_cluster_connection():
    """Test basic cluster connection"""
    print("\n" + "="*60)
    print("Redis Cluster Connection Test")
    print("="*60 + "\n")
    
    startup_nodes = [
        ClusterNode("192.168.4.163", 6379),
        ClusterNode("192.168.4.163", 6380),
        ClusterNode("192.168.4.163", 6381)
    ]
    
    try:
        print("Connecting to Redis Cluster on 192.168.4.163:6379...")
        cluster = RedisCluster(
            startup_nodes=startup_nodes,
            decode_responses=True,
            socket_connect_timeout=10,
            socket_timeout=10,
            require_full_coverage=False
        )
        # Don't call initialize() - let it auto-initialize on first command
        # Test PING
        print("✓ Testing PING...")
        result = await cluster.ping()
        assert result == True, "PING failed"
        print("  PING successful!")
        
        await cluster.aclose()
        return True
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        return False


async def test_basic_operations(cluster):
    """Test basic SET/GET operations"""
    print("\n✓ Testing basic SET/GET operations...")
    
    try:
        # Test simple string operations
        await cluster.set('test:key1', 'value1')
        value = await cluster.get('test:key1')
        assert value == 'value1', f"Expected 'value1', got '{value}'"
        print("  ✓ SET/GET successful")
        
        # Test multiple keys (will be distributed across nodes)
        keys = ['test:key2', 'test:key3', 'test:key4', 'test:key5']
        for key in keys:
            await cluster.set(key, f'value_{key}')
        
        for key in keys:
            value = await cluster.get(key)
            assert value == f'value_{key}', f"Value mismatch for {key}"
        print(f"  ✓ Distributed SET/GET across {len(keys)} keys successful")
        
        # Cleanup
        await cluster.delete(*keys)
        await cluster.delete('test:key1')
        
        return True
    except Exception as e:
        print(f"  ✗ Basic operations failed: {e}")
        return False


async def test_hash_operations(cluster):
    """Test HASH operations"""
    print("\n✓ Testing HASH operations...")
    
    try:
        # Test HSET/HGETALL
        await cluster.hset('user:123:profile', mapping={
            'name': 'John Doe',
            'email': 'john@example.com',
            'age': '30',
            'city': 'New York'
        })
        
        profile = await cluster.hgetall('user:123:profile')
        assert profile['name'] == 'John Doe', "Hash field mismatch"
        assert profile['email'] == 'john@example.com', "Hash field mismatch"
        print(f"  ✓ HASH operations successful: {profile}")
        
        # Cleanup
        await cluster.delete('user:123:profile')
        
        return True
    except Exception as e:
        print(f"  ✗ Hash operations failed: {e}")
        return False


async def test_list_operations(cluster):
    """Test LIST operations"""
    print("\n✓ Testing LIST operations...")
    
    try:
        # Test RPUSH/LRANGE
        await cluster.delete('test:list')
        await cluster.rpush('test:list', 'item1', 'item2', 'item3')
        
        items = await cluster.lrange('test:list', 0, -1)
        assert len(items) == 3, f"Expected 3 items, got {len(items)}"
        assert items == ['item1', 'item2', 'item3'], "List items mismatch"
        print(f"  ✓ LIST operations successful: {items}")
        
        # Cleanup
        await cluster.delete('test:list')
        
        return True
    except Exception as e:
        print(f"  ✗ List operations failed: {e}")
        return False


async def test_set_operations(cluster):
    """Test SET operations"""
    print("\n✓ Testing SET operations...")
    
    try:
        # Test SADD/SMEMBERS
        await cluster.delete('test:set')
        await cluster.sadd('test:set', 'apple', 'banana', 'cherry')
        
        members = await cluster.smembers('test:set')
        assert len(members) == 3, f"Expected 3 members, got {len(members)}"
        assert 'apple' in members, "Set member missing"
        print(f"  ✓ SET operations successful: {sorted(members)}")
        
        # Cleanup
        await cluster.delete('test:set')
        
        return True
    except Exception as e:
        print(f"  ✗ Set operations failed: {e}")
        return False


async def test_ttl_operations(cluster):
    """Test TTL/Expiration operations"""
    print("\n✓ Testing TTL/Expiration...")
    
    try:
        # Test SETEX
        await cluster.setex('test:ttl', 60, 'temporary_value')
        
        ttl = await cluster.ttl('test:ttl')
        assert ttl > 0 and ttl <= 60, f"TTL should be between 0 and 60, got {ttl}"
        print(f"  ✓ TTL operations successful: {ttl} seconds remaining")
        
        # Cleanup
        await cluster.delete('test:ttl')
        
        return True
    except Exception as e:
        print(f"  ✗ TTL operations failed: {e}")
        return False


async def test_cluster_info(cluster):
    """Test cluster-specific information"""
    print("\n✓ Testing cluster information...")
    
    try:
        # Get cluster info
        cluster_info = await cluster.cluster_info()
        print(f"  Cluster state: {cluster_info.get('cluster_state', 'unknown')}")
        print(f"  Cluster slots assigned: {cluster_info.get('cluster_slots_assigned', 'unknown')}")
        print(f"  Cluster known nodes: {cluster_info.get('cluster_known_nodes', 'unknown')}")
        
        # Get cluster nodes
        nodes = await cluster.cluster_nodes()
        print(f"  Number of nodes: {len(nodes)}")
        for node_id, node_info in list(nodes.items())[:3]:  # Show first 3 nodes
            print(f"    - Node: {node_info.get('host', 'unknown')}:{node_info.get('port', 'unknown')} "
                  f"({node_info.get('node_type', 'unknown')})")
        
        # Get cluster slots
        slots = await cluster.cluster_slots()
        print(f"  Number of slot ranges: {len(slots)}")
        
        return True
    except Exception as e:
        print(f"  ✗ Cluster info failed: {e}")
        return False


async def test_performance(cluster):
    """Test performance with async operations"""
    print("\n✓ Testing performance (1000 async operations)...")
    
    try:
        # Test concurrent SET operations
        start = time.time()
        tasks = [cluster.set(f'perf:async:{i}', f'value_{i}') for i in range(1000)]
        await asyncio.gather(*tasks)
        elapsed = time.time() - start
        print(f"  1000 async SETs: {elapsed:.3f}s ({1000/elapsed:.0f} ops/sec)")
        
        # Test concurrent GET operations
        start = time.time()
        tasks = [cluster.get(f'perf:async:{i}') for i in range(1000)]
        await asyncio.gather(*tasks)
        elapsed = time.time() - start
        print(f"  1000 async GETs: {elapsed:.3f}s ({1000/elapsed:.0f} ops/sec)")
        
        # Cleanup
        tasks = [cluster.delete(f'perf:async:{i}') for i in range(1000)]
        await asyncio.gather(*tasks)
        
        return True
    except Exception as e:
        print(f"  ✗ Performance test failed: {e}")
        return False


async def test_hash_tags(cluster):
    """Test hash tags for ensuring keys are on the same node"""
    print("\n✓ Testing hash tags (keys on same node)...")
    
    try:
        # Use hash tags to ensure keys are on the same node
        await cluster.set('user:{123}:name', 'John Doe')
        await cluster.set('user:{123}:email', 'john@example.com')
        await cluster.set('user:{123}:age', '30')
        
        name = await cluster.get('user:{123}:name')
        email = await cluster.get('user:{123}:email')
        age = await cluster.get('user:{123}:age')
        
        assert name == 'John Doe', "Hash tag key mismatch"
        assert email == 'john@example.com', "Hash tag key mismatch"
        assert age == '30', "Hash tag key mismatch"
        print(f"  ✓ Hash tags successful: name={name}, email={email}, age={age}")
        
        # Cleanup
        await cluster.delete('user:{123}:name', 'user:{123}:email', 'user:{123}:age')
        
        return True
    except Exception as e:
        print(f"  ✗ Hash tags test failed: {e}")
        return False


async def test_memory_info(cluster):
    """Test memory information"""
    print("\n✓ Testing memory information...")
    
    try:
        info = await cluster.info('memory')
        used_memory = info.get('used_memory_human', 'N/A')
        max_memory = info.get('maxmemory_human', 'N/A')
        print(f"  Used memory: {used_memory}")
        print(f"  Max memory: {max_memory}")
        
        # Get total keys across cluster
        # Note: DBSIZE only works on individual nodes in cluster mode
        # We'll get it from one node as an approximation
        try:
            keys_count = await cluster.dbsize()
            print(f"  Approximate keys: {keys_count}")
        except:
            print("  Keys count: N/A (cluster mode)")
        
        return True
    except Exception as e:
        print(f"  ✗ Memory info failed: {e}")
        return False


async def main():
    """Main test function"""
    print("\n" + "="*60)
    print("Redis Cluster Async Test Suite")
    print("="*60)
    
    # Provide all nodes for better cluster discovery
    startup_nodes = [
        ClusterNode("192.168.4.163", 6379),
        ClusterNode("192.168.4.163", 6380),
        ClusterNode("192.168.4.163", 6381)
    ]
    cluster = None
    results = []
    
    try:
        # Connect to cluster - provide all nodes mapped to 192.168.4.163 ports
        print("\nConnecting to Redis Cluster on 192.168.4.163:6379...")
        # Provide all nodes with 192.168.4.163 ports
        # Map internal Docker IPs to 192.168.4.163 ports for connection
        startup_nodes_local = [
            ClusterNode("192.168.4.163", 6379),
            ClusterNode("192.168.4.163", 6380),
            ClusterNode("192.168.4.163", 6381)
        ]
        
        # Create cluster connection
        # Cluster nodes now advertise 127.0.0.1 addresses via cluster-announce settings
        cluster = RedisCluster(
            startup_nodes=startup_nodes_local,
            decode_responses=True,
            socket_connect_timeout=10,
            socket_timeout=10,
            require_full_coverage=False
        )
        
        # Don't call initialize() explicitly - let it auto-initialize on first command
        print("Testing connection (auto-initializing)...")
        result = await cluster.ping()
        if result:
            print("✓ Connected successfully!\n")
        else:
            raise Exception("PING returned False")
        
        # Run all tests
        results.append(("Connection", await test_cluster_connection()))
        results.append(("Basic Operations", await test_basic_operations(cluster)))
        results.append(("Hash Operations", await test_hash_operations(cluster)))
        results.append(("List Operations", await test_list_operations(cluster)))
        results.append(("Set Operations", await test_set_operations(cluster)))
        results.append(("TTL Operations", await test_ttl_operations(cluster)))
        results.append(("Hash Tags", await test_hash_tags(cluster)))
        results.append(("Cluster Info", await test_cluster_info(cluster)))
        results.append(("Performance", await test_performance(cluster)))
        results.append(("Memory Info", await test_memory_info(cluster)))
        
    except ConnectionError as e:
        print(f"\n✗ Connection Error: {e}")
        print("  Make sure Redis Cluster is running:")
        print("    docker-compose up -d")
        print("  Check cluster status:")
        print("    docker-compose ps")
        print("    redis-cli -c -h 192.168.4.163 -p 6379 CLUSTER INFO")
        return False
    except TimeoutError as e:
        print(f"\n✗ Timeout Error: {e}")
        print("  The cluster may be slow to respond. Try:")
        print("    docker-compose restart")
        print("    Wait a few seconds and try again")
        return False
    except ClusterDownError as e:
        print(f"\n✗ Cluster Down Error: {e}")
        print("  The cluster may not be initialized. Check logs:")
        print("    docker-compose logs redis-cluster-init")
        return False
    except RedisError as e:
        print(f"\n✗ Redis Error: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if cluster:
            await cluster.aclose()
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print("\n" + "="*60)
    if passed == total:
        print(f"✅ ALL TESTS PASSED! ({passed}/{total})")
        print("="*60)
        print("\nConnection Info:")
        print("  Host: 192.168.4.163")
        print("  Port: 6379")
        print("  Mode: Redis Cluster (3 nodes)")
        print("  Persistence: None (memory only)")
        print("\n✨ Redis Cluster is working perfectly with async client!\n")
        return True
    else:
        print(f"✗ SOME TESTS FAILED ({passed}/{total} passed)")
        print("="*60 + "\n")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)

