#!/usr/bin/env python3
"""
Test Redis connection and usage with twemproxy.
Tests multiple keys, DataFrame serialization, and connection patterns.
"""

import asyncio
import sys
import os
import json
from pathlib import Path

# Add project root to path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    print("ERROR: redis package not installed. Install with: pip install redis")
    sys.exit(1)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("WARNING: pandas not available, skipping DataFrame tests")
    PANDAS_AVAILABLE = False

async def test_basic_operations(redis_url: str):
    """Test basic Redis operations through twemproxy."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic SET/GET Operations")
    print("=" * 60)
    
    client = None
    try:
        # Create client with twemproxy-compatible settings
        print(f"\n1. Creating Redis client: {redis_url}")
        client = redis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=False,
            socket_keepalive=True,
            retry_on_timeout=True,
            socket_connect_timeout=10,
            socket_timeout=10,
            max_connections=1,
        )
        print("   ✓ Client created")
        
        # Test multiple keys
        test_keys = [
            ("test:key1", b"value1"),
            ("test:key2", b"value2"),
            ("test:key3", b"value3"),
        ]
        
        print(f"\n2. Testing SET operations for {len(test_keys)} keys...")
        for key, value in test_keys:
            try:
                result = await client.set(key, value)
                print(f"   ✓ SET {key}: {result}")
            except Exception as e:
                print(f"   ✗ SET {key} failed: {type(e).__name__}: {e}")
                return False
        
        print(f"\n3. Testing GET operations for {len(test_keys)} keys...")
        for key, expected_value in test_keys:
            try:
                result = await client.get(key)
                if result == expected_value:
                    print(f"   ✓ GET {key}: {result.decode()}")
                else:
                    print(f"   ✗ GET {key}: mismatch (expected {expected_value}, got {result})")
                    return False
            except Exception as e:
                print(f"   ✗ GET {key} failed: {type(e).__name__}: {e}")
                return False
        
        print(f"\n4. Testing DELETE operations...")
        for key, _ in test_keys:
            try:
                result = await client.delete(key)
                print(f"   ✓ DELETE {key}: {result}")
            except Exception as e:
                print(f"   ✗ DELETE {key} failed: {type(e).__name__}: {e}")
                return False
        
        print("\n✓ Basic operations test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Basic operations test FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if client:
            try:
                await client.aclose()
                print("   ✓ Client closed")
            except:
                pass

async def test_dataframe_operations(redis_url: str):
    """Test DataFrame serialization/deserialization through twemproxy."""
    if not PANDAS_AVAILABLE:
        print("\n" + "=" * 60)
        print("TEST 2: DataFrame Operations (SKIPPED - pandas not available)")
        print("=" * 60)
        return True
    
    print("\n" + "=" * 60)
    print("TEST 2: DataFrame Operations")
    print("=" * 60)
    
    client = None
    try:
        print(f"\n1. Creating Redis client: {redis_url}")
        client = redis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=False,
            socket_keepalive=True,
            retry_on_timeout=True,
            socket_connect_timeout=10,
            socket_timeout=10,
            max_connections=1,
        )
        print("   ✓ Client created")
        
        # Create test DataFrame
        print("\n2. Creating test DataFrame...")
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        print(f"   ✓ DataFrame created: shape {df.shape}")
        
        # Serialize DataFrame
        print("\n3. Serializing DataFrame...")
        if df.empty:
            data = json.dumps({'empty': True}).encode('utf-8')
        else:
            json_str = df.to_json(orient='split', date_format='iso')
            data = json.dumps({'empty': False, 'data': json_str}).encode('utf-8')
        print(f"   ✓ Serialized: {len(data)} bytes")
        
        # Store in Redis
        test_key = "test:cache:df"
        print(f"\n4. Storing DataFrame in Redis (key: {test_key})...")
        try:
            result = await client.set(test_key, data)
            print(f"   ✓ SET successful: {result}")
        except Exception as e:
            print(f"   ✗ SET failed: {type(e).__name__}: {e}")
            return False
        
        # Retrieve from Redis
        print(f"\n5. Retrieving DataFrame from Redis...")
        try:
            retrieved_data = await client.get(test_key)
            if retrieved_data is None:
                print("   ✗ GET returned None")
                return False
            print(f"   ✓ GET successful: {len(retrieved_data)} bytes")
        except Exception as e:
            print(f"   ✗ GET failed: {type(e).__name__}: {e}")
            return False
        
        # Deserialize DataFrame
        print("\n6. Deserializing DataFrame...")
        try:
            decoded = json.loads(retrieved_data.decode('utf-8'))
            if decoded.get('empty', False):
                df_retrieved = pd.DataFrame()
            else:
                json_str = decoded['data']
                df_retrieved = pd.read_json(json_str, orient='split')
            print(f"   ✓ Deserialized: shape {df_retrieved.shape}")
        except Exception as e:
            print(f"   ✗ Deserialization failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Verify data
        print("\n7. Verifying DataFrame data...")
        if df.equals(df_retrieved):
            print("   ✓ Data matches perfectly")
        else:
            print("   ✗ Data mismatch!")
            print(f"   Original:\n{df}")
            print(f"   Retrieved:\n{df_retrieved}")
            return False
        
        # Cleanup
        print(f"\n8. Cleaning up (DELETE {test_key})...")
        try:
            await client.delete(test_key)
            print("   ✓ Cleanup successful")
        except Exception as e:
            print(f"   ⚠ Cleanup failed: {type(e).__name__}: {e}")
        
        print("\n✓ DataFrame operations test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ DataFrame operations test FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if client:
            try:
                await client.aclose()
                print("   ✓ Client closed")
            except:
                pass

async def test_multiple_keys_pattern(redis_url: str):
    """Test storing and retrieving multiple keys in a pattern."""
    print("\n" + "=" * 60)
    print("TEST 3: Multiple Keys Pattern (like cache would use)")
    print("=" * 60)
    
    client = None
    try:
        print(f"\n1. Creating Redis client: {redis_url}")
        client = redis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=False,
            socket_keepalive=True,
            retry_on_timeout=True,
            socket_connect_timeout=10,
            socket_timeout=10,
            max_connections=1,
        )
        print("   ✓ Client created")
        
        # Create multiple test keys (simulating cache usage)
        num_keys = 10
        print(f"\n2. Creating {num_keys} test keys...")
        test_data = {}
        for i in range(num_keys):
            key = f"stocks:questdb:test_key:AAPL:{i:04d}"
            value = json.dumps({'ticker': 'AAPL', 'index': i, 'data': f'value_{i}'}).encode('utf-8')
            test_data[key] = value
        
        # Store all keys
        print(f"\n3. Storing {num_keys} keys...")
        stored_count = 0
        for key, value in test_data.items():
            try:
                result = await client.set(key, value)
                if result:
                    stored_count += 1
            except Exception as e:
                print(f"   ✗ SET {key} failed: {type(e).__name__}: {e}")
                return False
        print(f"   ✓ Stored {stored_count}/{num_keys} keys")
        
        # Retrieve all keys
        print(f"\n4. Retrieving {num_keys} keys...")
        retrieved_count = 0
        for key, expected_value in test_data.items():
            try:
                result = await client.get(key)
                if result == expected_value:
                    retrieved_count += 1
                else:
                    print(f"   ✗ GET {key}: data mismatch")
                    return False
            except Exception as e:
                print(f"   ✗ GET {key} failed: {type(e).__name__}: {e}")
                return False
        print(f"   ✓ Retrieved {retrieved_count}/{num_keys} keys")
        
        # Cleanup
        print(f"\n5. Cleaning up {num_keys} keys...")
        deleted_count = 0
        for key in test_data.keys():
            try:
                result = await client.delete(key)
                if result:
                    deleted_count += 1
            except Exception as e:
                print(f"   ⚠ DELETE {key} failed: {type(e).__name__}: {e}")
        print(f"   ✓ Deleted {deleted_count}/{num_keys} keys")
        
        print("\n✓ Multiple keys pattern test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Multiple keys pattern test FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if client:
            try:
                await client.aclose()
                print("   ✓ Client closed")
            except:
                pass

async def test_connection_reuse_vs_ondemand(redis_url: str):
    """Test connection reuse vs on-demand creation."""
    print("\n" + "=" * 60)
    print("TEST 4: Connection Reuse vs On-Demand")
    print("=" * 60)
    
    # Test 1: Reuse same connection
    print("\n1. Testing connection reuse (same client, multiple operations)...")
    client1 = None
    try:
        client1 = redis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=False,
            socket_keepalive=True,
            retry_on_timeout=True,
            socket_connect_timeout=10,
            socket_timeout=10,
            max_connections=1,
        )
        
        operations = 5
        success_count = 0
        for i in range(operations):
            key = f"test:reuse:{i}"
            value = f"value_{i}".encode('utf-8')
            try:
                await client1.set(key, value)
                result = await client1.get(key)
                if result == value:
                    success_count += 1
                await client1.delete(key)
            except Exception as e:
                print(f"   ✗ Operation {i} failed: {type(e).__name__}: {e}")
        
        print(f"   ✓ Connection reuse: {success_count}/{operations} operations successful")
        if success_count == operations:
            print("   ✓ Connection reuse works correctly")
        else:
            print("   ✗ Connection reuse has issues")
            return False
            
    except Exception as e:
        print(f"   ✗ Connection reuse test failed: {type(e).__name__}: {e}")
        return False
    finally:
        if client1:
            try:
                await client1.aclose()
            except:
                pass
    
    # Test 2: On-demand connections
    print("\n2. Testing on-demand connections (new client per operation)...")
    operations = 5
    success_count = 0
    for i in range(operations):
        client2 = None
        try:
            client2 = redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=False,
                socket_keepalive=True,
                retry_on_timeout=True,
                socket_connect_timeout=10,
                socket_timeout=10,
                max_connections=1,
            )
            
            key = f"test:ondemand:{i}"
            value = f"value_{i}".encode('utf-8')
            await client2.set(key, value)
            result = await client2.get(key)
            if result == value:
                success_count += 1
            await client2.delete(key)
            
        except Exception as e:
            print(f"   ✗ Operation {i} failed: {type(e).__name__}: {e}")
        finally:
            if client2:
                try:
                    await client2.aclose()
                except:
                    pass
    
    print(f"   ✓ On-demand connections: {success_count}/{operations} operations successful")
    if success_count == operations:
        print("   ✓ On-demand connections work correctly")
        return True
    else:
        print("   ✗ On-demand connections have issues")
        return False

async def main():
    """Main test function."""
    print("Redis + Twemproxy Connection Test")
    print("=" * 60)
    
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    print(f"Redis URL: {redis_url}")
    
    results = []
    
    # Run all tests
    results.append(("Basic Operations", await test_basic_operations(redis_url)))
    results.append(("DataFrame Operations", await test_dataframe_operations(redis_url)))
    results.append(("Multiple Keys Pattern", await test_multiple_keys_pattern(redis_url)))
    results.append(("Connection Patterns", await test_connection_reuse_vs_ondemand(redis_url)))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n✓ All tests PASSED!")
        sys.exit(0)
    else:
        print("\n✗ Some tests FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

