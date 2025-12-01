#!/usr/bin/env python3
"""
Minimal Realtime Feed Client (rt.py)
Step 2: Add basic display interface showing all updates
"""

import asyncio
import json
import websockets
import sys
import os
import yaml
from datetime import datetime
import argparse

class SimpleRealtimeClient:
    def __init__(self, server_url: str, symbols: list, max_messages: int = None):
        self.server_url = server_url
        self.symbols = [s.upper() for s in symbols]
        self.max_messages = max_messages
        self.start_time = datetime.now()
        
        # Data storage for display - one per symbol
        self.symbol_data = {}
        for symbol in self.symbols:
            self.symbol_data[symbol] = {
                'current_price': None,
                'last_update_time': None,
                'message_count': 0,
                'message_history': [],
                'max_history': 5  # Keep last 5 messages per symbol
            }
        
        # Global message counter
        self.total_message_count = 0
        
    def log(self, message: str):
        """Simple logging with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] {message}")
    
    def add_to_history(self, symbol: str, message_type: str, data: dict):
        """Add message to history for display."""
        timestamp = datetime.now()
        self.symbol_data[symbol]['last_update_time'] = timestamp
        self.symbol_data[symbol]['message_count'] += 1
        
        history_entry = {
            'timestamp': timestamp,
            'type': message_type,
            'data': data
        }
        
        self.symbol_data[symbol]['message_history'].append(history_entry)
        
        # Keep only the last N messages
        if len(self.symbol_data[symbol]['message_history']) > self.symbol_data[symbol]['max_history']:
            self.symbol_data[symbol]['message_history'] = self.symbol_data[symbol]['message_history'][-self.symbol_data[symbol]['max_history']:]
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def display_interface(self):
        """Display the current interface with all data."""
        self.clear_screen()
        
        # Header
        print("=" * 80)
        print(f"🚀 REALTIME FEED CLIENT - {', '.join(self.symbols)}")
        print(f"Server: {self.server_url} | Total Messages: {self.total_message_count}")
        print(f"Session Duration: {datetime.now() - self.start_time}")
        print("=" * 80)
        
        # Ticker view - show current prices for all symbols
        print("📊 CURRENT PRICES:")
        print("-" * 80)
        for symbol in self.symbols:
            data = self.symbol_data[symbol]
            price = data['current_price']
            last_update = data['last_update_time']
            msg_count = data['message_count']
            
            price_str = f"${price:.2f}" if price else "N/A"
            update_str = last_update.strftime('%H:%M:%S.%f')[:-3] if last_update else "Never"
            
            print(f"{symbol:<6} | {price_str:>10} | Messages: {msg_count:>3} | Last: {update_str}")
        
        print("-" * 80)
        
        # Recent updates section at bottom
        print("📨 RECENT UPDATES (Latest 10 across all symbols):")
        print("-" * 80)
        
        # Collect all recent messages from all symbols
        all_messages = []
        for symbol in self.symbols:
            for msg in self.symbol_data[symbol]['message_history']:
                all_messages.append((symbol, msg))
        
        # Sort by timestamp (newest first) and take latest 10
        all_messages.sort(key=lambda x: x[1]['timestamp'], reverse=True)
        recent_messages = all_messages[:10]
        
        if not recent_messages:
            print("No messages received yet...")
        else:
            for i, (symbol, msg) in enumerate(recent_messages, 1):
                timestamp = msg['timestamp'].strftime('%H:%M:%S.%f')[:-3]
                msg_type = msg['type']
                
                # Extract key info for display
                display_info = f"{symbol}: {msg_type}"
                if msg['data'].get('price'):
                    display_info += f" | ${msg['data']['price']:.2f}"
                if msg['data'].get('event_type'):
                    display_info += f" | {msg['data']['event_type']}"
                
                print(f"{i:2d}. [{timestamp}] {display_info}")
        
        print("-" * 80)
        print("Press Ctrl+C to stop")
        print("=" * 80)
    
    async def connect_and_listen(self):
        """Connect to WebSocket and listen for messages from all symbols."""
        # Ensure proper WebSocket scheme
        if not self.server_url.startswith(('ws://', 'wss://')):
            server_url = f"ws://{self.server_url}"
        else:
            server_url = self.server_url
        
        # Create tasks for all symbols
        tasks = []
        for symbol in self.symbols:
            task = asyncio.create_task(self._listen_to_symbol(server_url, symbol))
            tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _listen_to_symbol(self, server_url: str, symbol: str):
        """Listen to a specific symbol's WebSocket."""
        ws_url = f"{server_url}/ws?symbol={symbol}"
        
        self.log(f"🔗 Connecting to: {ws_url}")
        
        try:
            async with websockets.connect(ws_url) as websocket:
                self.log(f"✅ Connected to {symbol}")
                
                async for message in websocket:
                    self.total_message_count += 1
                    self.process_message(symbol, message)
                    
                    # Always display the interface after each message
                    self.display_interface()
                    
                    # Stop if we've reached max messages
                    if self.max_messages and self.total_message_count >= self.max_messages:
                        elapsed = datetime.now() - self.start_time
                        self.log(f"🛑 Reached max messages ({self.max_messages}). Elapsed time: {elapsed}")
                        return
                    
        except websockets.exceptions.ConnectionClosed:
            self.log(f"❌ Connection closed for {symbol}")
        except Exception as e:
            self.log(f"❌ Error for {symbol}: {e}")
    
    def process_message(self, symbol: str, raw_message: str):
        """Process incoming WebSocket message."""
        elapsed = datetime.now() - self.start_time
        self.log(f"📨 Message #{self.total_message_count} for {symbol} (elapsed: {elapsed})")
        
        try:
            parsed = json.loads(raw_message)
            
            # Extract key data
            received_symbol = parsed.get('symbol', 'UNKNOWN')
            data = parsed.get('data', {})
            msg_type = data.get('type')
            event_type = data.get('event_type')
            timestamp = data.get('timestamp')
            payload = data.get('payload', [])
            
            # Handle heartbeats separately (compact logging)
            if msg_type == 'heartbeat':
                self.log(f"💓 Heartbeat #{self.total_message_count} for {symbol} at {timestamp}")
                # Add heartbeats to history too
                self.add_to_history(symbol, 'heartbeat', {'event_type': 'heartbeat', 'timestamp': timestamp})
                self.log("-" * 40)
                return
            
            # For non-heartbeat messages, show full details
            self.log(f"RAW: {raw_message}")
            self.log(f"SYMBOL: {received_symbol}")
            self.log(f"TYPE: {msg_type}")
            self.log(f"EVENT: {event_type}")
            self.log(f"TIMESTAMP: {timestamp}")
            
            # Extract price if available
            if payload and len(payload) > 0:
                first_item = payload[0]
                if isinstance(first_item, dict):
                    price = first_item.get('price')
                    if price is not None:
                        self.log(f"💰 PRICE: ${price}")
                        self.symbol_data[symbol]['current_price'] = price
                    
                    # Log other fields in the payload
                    for key, value in first_item.items():
                        if key != 'price':
                            self.log(f"   {key}: {value}")
            
            # Add to history for display
            self.add_to_history(symbol, msg_type, {
                'event_type': event_type,
                'price': self.symbol_data[symbol]['current_price'],
                'timestamp': timestamp,
                'payload': payload
            })
            
            self.log(f"FULL PAYLOAD: {payload}")
            
        except json.JSONDecodeError as e:
            self.log(f"❌ JSON decode error for {symbol}: {e}")
        
        self.log("-" * 60)

def load_symbols_from_yaml(yaml_file: str) -> list[str]:
    """Load symbols from a YAML file."""
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict) and 'symbols' in data:
                symbols = data['symbols']
                if isinstance(symbols, list):
                    return symbols
                else:
                    print(f"Error: 'symbols' in {yaml_file} should be a list.", file=sys.stderr)
                    return []
            else:
                print(f"Error: Invalid YAML format in {yaml_file}. Expected 'symbols' key.", file=sys.stderr)
                return []
    except Exception as e:
        print(f"Error loading symbols from {yaml_file}: {e}", file=sys.stderr)
        return []

async def main():
    parser = argparse.ArgumentParser(description="Minimal Realtime Feed Client")
    parser.add_argument("--server", type=str, default="ms1.kundu.dev:9001", 
                        help="WebSocket server (default: ms1.kundu.dev:9001)")
    
    # Create a mutually exclusive group for symbol input methods
    symbol_group = parser.add_mutually_exclusive_group()
    symbol_group.add_argument("--symbols", nargs="+", default=["AAPL"], 
                        help="Stock symbols to track (default: AAPL)")
    symbol_group.add_argument("--symbols-list", type=str,
                        help="Path to a YAML file containing a list of symbols under the 'symbols' key")
    
    parser.add_argument("--max-messages", type=int, default=None,
                        help="Maximum messages to receive before stopping (default: run indefinitely)")
    
    args = parser.parse_args()
    
    # Load symbols based on input method
    if args.symbols_list:
        symbols = load_symbols_from_yaml(args.symbols_list)
        if not symbols:
            print(f"❌ No symbols loaded from {args.symbols_list}. Exiting.")
            sys.exit(1)
        print(f"📋 Loaded {len(symbols)} symbols from {args.symbols_list}")
    else:
        symbols = args.symbols
    
    print("🚀 Minimal Realtime Feed Client")
    print(f"Server: {args.server}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Max Messages: {args.max_messages or 'Unlimited'}")
    print("=" * 60)
    
    client = SimpleRealtimeClient(args.server, symbols, args.max_messages)
    
    try:
        await client.connect_and_listen()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
