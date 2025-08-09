#!/usr/bin/env python3
"""
Quick test to verify single-process vs multi-process mode
"""

import asyncio
import aiohttp
import json


async def check_server_mode(port: int = 8080):
    """Check which mode the server is running in."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://localhost:{port}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    process_info = data.get('process', {})
                    
                    multiprocess_mode = process_info.get('multiprocess_mode', False)
                    worker_id = process_info.get('worker_id')
                    pid = process_info.get('pid')
                    
                    print(f"🔍 Server Mode Detection")
                    print(f"{'='*30}")
                    print(f"Multiprocess Mode: {multiprocess_mode}")
                    print(f"Process ID: {pid}")
                    print(f"Worker ID: {worker_id}")
                    
                    if multiprocess_mode:
                        print(f"✅ Running in MULTI-PROCESS mode")
                        print(f"   Worker {worker_id} handled this request")
                    else:
                        print(f"✅ Running in SINGLE-PROCESS mode")
                        print(f"   Original async server (no workers)")
                    
                    return multiprocess_mode
                else:
                    print(f"❌ Server responded with status {response.status}")
                    return None
    except aiohttp.ClientError as e:
        print(f"❌ Cannot connect to server on port {port}: {e}")
        print(f"   Make sure the server is running:")
        print(f"   python db_server.py --db-file data/test.db --port {port}")
        return None


if __name__ == "__main__":
    asyncio.run(check_server_mode())
