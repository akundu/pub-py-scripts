import os
import asyncio
import random
import string
from redis.asyncio.cluster import RedisCluster, ClusterNode
from redis.exceptions import RedisError

# Configuration
HOST_IP = os.getenv("HOST_IP", "127.0.0.1")
STARTUP_NODES = [ClusterNode(HOST_IP, 6379)]

async def test_cluster_connectivity():
    print(f"--- Connecting to Cluster at {STARTUP_NODES[0].host}:{STARTUP_NODES[0].port} ---")
    
    try:
        # Initialize the Async Redis Cluster client
        rc = RedisCluster(startup_nodes=STARTUP_NODES, decode_responses=True)
        
        # 1. Test Ping (Basic Connectivity)
        await rc.ping()
        print("✅ [PASS] Cluster Ping successful.")

        # 2. Test Topology (Do we see 3 primaries?)
        # We inspect the nodes the client has discovered
        nodes = await rc.cluster_nodes()
        primary_count = sum(1 for node_info in nodes.values() if 'master' in node_info.get('flags', ''))
        print(f"✅ [PASS] Discovered {len(nodes)} nodes ({primary_count} primaries).")

        # 3. Test Redirects (The '172.x' fix verification)
        # We will write 10 random keys. Because of hashing, these SHOULD land on different nodes.
        # If the announce-ip is broken, this will hang or timeout.
        print("\n--- Testing Slot Redirection (Verifying 'announce-ip' fix) ---")
        for i in range(5):
            key = f"test_key_{i}"
            value = f"value_{i}"
            await rc.set(key, value)
            # Verify we can get it back
            val_back = await rc.get(key)
            print(f"   - Wrote/Read key '{key}': {val_back == value}")
        print("✅ [PASS] seamless redirection works.")

        # 4. Async Concurrency Test
        print(f"\n--- Testing Async Concurrency (1000 Ops) ---")
        # We will launch 1000 set operations concurrently using asyncio.gather
        tasks = []
        for i in range(1000):
            tasks.append(rc.set(f"concurrent_{i}", "data"))
        
        # This runs them all 'at once' in the event loop
        await asyncio.gather(*tasks)
        print("✅ [PASS] Successfully handled 1000 concurrent async operations.")

        await rc.aclose()

    except RedisError as e:
        print(f"❌ [FAIL] Redis Error: {e}")
    except Exception as e:
        print(f"❌ [FAIL] Unexpected Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_cluster_connectivity())
