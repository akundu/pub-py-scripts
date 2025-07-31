#!/usr/bin/env python3

import asyncio
import aiohttp
import json
from datetime import datetime

class WebSocketListener:
    def __init__(self, symbol: str, server_url: str = "ws://localhost:9000", show_quotes: bool = True, show_trades: bool = True, show_heartbeat: bool = False):
        self.symbol = symbol
        self.server_url = f"{server_url}/ws?symbol={symbol}"
        self.ws = None
        self.message_count = 0
        self.show_quotes = show_quotes
        self.show_trades = show_trades
        self.show_heartbeat = show_heartbeat
        
    async def connect(self):
        """Connect to the WebSocket server."""
        try:
            session = aiohttp.ClientSession()
            self.ws = await session.ws_connect(self.server_url)
            print(f"🔗 Connected to WebSocket for {self.symbol}")
            print(f"   URL: {self.server_url}")
            print(f"   Waiting for broadcasts...")
            print()
            return True
        except Exception as e:
            print(f"❌ Failed to connect to WebSocket: {e}")
            return False
    
    async def listen(self):
        """Listen for WebSocket messages."""
        if not self.ws:
            print("❌ Not connected to WebSocket")
            return
        
        try:
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    self.message_count += 1
                    await self.handle_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"❌ WebSocket error: {self.ws.exception()}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    print("🔌 WebSocket connection closed")
                    break
        except Exception as e:
            print(f"❌ Error in WebSocket listener: {e}")
        finally:
            if self.ws:
                await self.ws.close()
    
    async def handle_message(self, data: str):
        """Handle incoming WebSocket messages."""
        try:
            message = json.loads(data)
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            if 'data' in message:
                data_content = message['data']
                data_type = data_content.get('type', 'Unknown')
                
                # Check if we should show this message type
                if data_type == 'quote' and not self.show_quotes:
                    return  # Skip quote messages if not enabled
                elif data_type == 'trade' and not self.show_trades:
                    return  # Skip trade messages if not enabled
                elif data_type == 'heartbeat' and not self.show_heartbeat:
                    return  # Skip heartbeat messages unless explicitly requested
            
            print(f"📡 [{timestamp}] Message #{self.message_count} received:")
            print(f"   Symbol: {message.get('symbol', 'Unknown')}")
            
            if 'data' in message:
                data_content = message['data']
                print(f"   Type: {data_content.get('type', 'Unknown')}")
                print(f"   Event Type: {data_content.get('event_type', 'Unknown')}")
                print(f"   Timestamp: {data_content.get('timestamp', 'Unknown')}")
                
                if data_content.get('type') == 'quote':
                    payload = data_content.get('payload', [])
                    if payload:
                        quote = payload[0]
                        print(f"   Quote Data:")
                        print(f"     Bid Price: ${quote.get('bid_price', 'N/A')}")
                        print(f"     Bid Size: {quote.get('bid_size', 'N/A')}")
                        print(f"     Ask Price: ${quote.get('ask_price', 'N/A')}")
                        print(f"     Ask Size: {quote.get('ask_size', 'N/A')}")
                
                elif data_content.get('type') == 'trade':
                    payload = data_content.get('payload', [])
                    if payload:
                        trade = payload[0]
                        print(f"   Trade Data:")
                        print(f"     Price: ${trade.get('price', 'N/A')}")
                        print(f"     Size: {trade.get('size', 'N/A')}")
                
                elif data_content.get('type') == 'heartbeat':
                    if self.show_heartbeat:
                        print(f"   💓 Heartbeat received")
                
                else:
                    print(f"   Raw Data: {json.dumps(data_content, indent=2)}")
            
            print()  # Empty line for readability
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON message: {e}")
            print(f"   Raw message: {data}")
        except Exception as e:
            print(f"❌ Error handling message: {e}")

async def main():
    """Main function to run the WebSocket listener."""
    import argparse
    
    parser = argparse.ArgumentParser(description="WebSocket listener for stock data broadcasts")
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol to listen for (default: AAPL)")
    parser.add_argument("--server", default="ws://localhost:9000", help="WebSocket server URL (default: ws://localhost:9000)")
    parser.add_argument("--quotes-only", action="store_true", help="Show only quote updates (hide trades)")
    parser.add_argument("--trades-only", action="store_true", help="Show only trade updates (hide quotes)")
    parser.add_argument("--show-heartbeat", action="store_true", help="Show heartbeat messages (default: hidden)")
    
    args = parser.parse_args()
    
    # Determine what to show based on arguments
    show_quotes = not args.trades_only
    show_trades = not args.quotes_only
    
    # If both quotes-only and trades-only are specified, show both (default behavior)
    if args.quotes_only and args.trades_only:
        show_quotes = True
        show_trades = True
    
    symbol = args.symbol
    
    print(f"🎧 Starting WebSocket listener for {symbol}")
    print(f"   Server: {args.server}")
    print(f"   Show quotes: {show_quotes}")
    print(f"   Show trades: {show_trades}")
    print(f"   Show heartbeat: {args.show_heartbeat}")
    print(f"   Run test_broadcast_insert.py in another terminal to send test data")
    print()
    
    listener = WebSocketListener(symbol, args.server, show_quotes, show_trades, args.show_heartbeat)
    
    if await listener.connect():
        try:
            await listener.listen()
        except KeyboardInterrupt:
            print(f"\n⏹️  Listener stopped by user")
        except Exception as e:
            print(f"❌ Listener error: {e}")
    
    print(f"📊 Total messages received: {listener.message_count}")

if __name__ == "__main__":
    asyncio.run(main()) 