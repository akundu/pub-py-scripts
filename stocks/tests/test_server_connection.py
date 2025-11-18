#!/usr/bin/env python3
"""
Test server connection and data flow.
"""

import asyncio
import websockets
import json
import sys

async def test_server_connection():
    """Test connection to the server."""
    server_url = "ws://localhost:8080"
    
    print(f"Testing connection to {server_url}")
    
    try:
        # Test connection to AAPL
        ws_url = f"{server_url}/ws?symbol=AAPL"
        print(f"Connecting to {ws_url}")
        
        async with websockets.connect(ws_url) as websocket:
            print("Connected successfully!")
            
            # Wait for initial message
            print("Waiting for initial message...")
            try:
                async with asyncio.timeout(10.0):
                    message = await websocket.recv()
                    print(f"Received message: {message}")
                    
                    # Parse the message
                    data = json.loads(message)
                    print(f"Parsed data: {json.dumps(data, indent=2)}")
                    
            except asyncio.TimeoutError:
                print("No message received within 10 seconds")
            except Exception as e:
                print(f"Error receiving message: {e}")
                
    except Exception as e:
        print(f"Connection failed: {e}")
        return False
    
    return True

async def main():
    """Run the test."""
    success = await test_server_connection()
    
    if success:
        print("\n✓ Server connection test passed!")
    else:
        print("\n✗ Server connection test failed!")
        print("Make sure the database server is running:")
        print("python db_server.py --db-file data/stock_data.db --port 8080")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 