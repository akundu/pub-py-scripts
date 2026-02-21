#!/usr/bin/env python3
"""
Redis Pub/Sub Listener Tool

This tool listens to Redis Pub/Sub channels and prints all messages that are published.
Useful for debugging and monitoring what data is flowing through Redis.

Usage:
    # Listen to all realtime channels (quote and trade for all symbols)
    python redis_listener.py --pattern "realtime:*:*"
    
    # Listen to specific channels
    python redis_listener.py --channels realtime:quote:AAPL realtime:trade:AAPL
    
    # Listen to all quote channels
    python redis_listener.py --pattern "realtime:quote:*"
    
    # Listen with pretty JSON formatting
    python redis_listener.py --pattern "realtime:*:*" --pretty
    
    # Save messages to a file
    python redis_listener.py --pattern "realtime:*:*" --output messages.log
"""

import os
import sys
import asyncio
import argparse
import json
from datetime import datetime
from typing import List, Optional
import signal

# Try to import Redis
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    try:
        import redis
        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_AVAILABLE = False
        redis = None
        print("Error: redis library not installed. Install with: pip install redis", file=sys.stderr)
        sys.exit(1)

# Global shutdown flag
shutdown_flag = False

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    global shutdown_flag
    
    def signal_handler(signum, frame):
        global shutdown_flag
        print(f"\n[INFO] Received signal {signum}, shutting down...")
        shutdown_flag = True
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def format_message(channel: str, data: dict, pretty: bool = False) -> str:
    """Format a message for display."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    
    if pretty:
        data_str = json.dumps(data, indent=2)
    else:
        data_str = json.dumps(data)
    
    return f"[{timestamp}] Channel: {channel}\n{data_str}"

async def listen_to_channels(redis_url: str, channels: List[str], pattern: Optional[str] = None,
                            pretty: bool = False, output_file: Optional[str] = None):
    """Listen to Redis Pub/Sub channels and print messages."""
    if not REDIS_AVAILABLE:
        print("Error: Redis library not available", file=sys.stderr)
        return
    
    try:
        # Connect to Redis
        redis_client = redis.from_url(
            redis_url,
            decode_responses=False,  # Keep binary for JSON
            socket_connect_timeout=10,
            socket_timeout=10,
            socket_keepalive=True,
            retry_on_timeout=True
        )
        
        # Test connection
        await redis_client.ping()
        print(f"[INFO] Connected to Redis: {redis_url}")
        
        # Create pubsub client
        pubsub = redis_client.pubsub()
        
        # Subscribe to channels or pattern
        if pattern:
            await pubsub.psubscribe(pattern)
            print(f"[INFO] Subscribed to pattern: {pattern}")
        elif channels:
            for channel in channels:
                await pubsub.subscribe(channel)
                print(f"[INFO] Subscribed to channel: {channel}")
        else:
            print("[ERROR] Must specify either --channels or --pattern", file=sys.stderr)
            await redis_client.aclose()
            return
        
        # Open output file if specified
        output_fp = None
        if output_file:
            output_fp = open(output_file, 'a')
            print(f"[INFO] Writing messages to: {output_file}")
        
        print("[INFO] Listening for messages... (Press Ctrl+C to stop)\n")
        
        message_count = 0
        
        try:
            while not shutdown_flag:
                try:
                    # Get message with timeout
                    message = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0),
                        timeout=1.0
                    )
                    
                    if message and message['type'] == 'message':
                        message_count += 1
                        
                        # Decode channel
                        channel = message['channel']
                        if isinstance(channel, bytes):
                            channel = channel.decode('utf-8')
                        
                        # Decode and parse data
                        data = message['data']
                        if isinstance(data, bytes):
                            data = data.decode('utf-8')
                        
                        try:
                            message_data = json.loads(data)
                        except json.JSONDecodeError:
                            message_data = {"raw": data}
                        
                        # Format and print message
                        formatted = format_message(channel, message_data, pretty=pretty)
                        print(formatted)
                        print("-" * 80)
                        
                        # Write to file if specified
                        if output_fp:
                            output_fp.write(formatted + "\n" + "-" * 80 + "\n")
                            output_fp.flush()
                    
                    elif message and message['type'] == 'pmessage':
                        # Pattern match message
                        message_count += 1
                        
                        # Decode channel and pattern
                        pattern_matched = message['pattern']
                        channel = message['channel']
                        if isinstance(pattern_matched, bytes):
                            pattern_matched = pattern_matched.decode('utf-8')
                        if isinstance(channel, bytes):
                            channel = channel.decode('utf-8')
                        
                        # Decode and parse data
                        data = message['data']
                        if isinstance(data, bytes):
                            data = data.decode('utf-8')
                        
                        try:
                            message_data = json.loads(data)
                        except json.JSONDecodeError:
                            message_data = {"raw": data}
                        
                        # Format and print message
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        if pretty:
                            data_str = json.dumps(message_data, indent=2)
                        else:
                            data_str = json.dumps(message_data)
                        
                        formatted = f"[{timestamp}] Pattern: {pattern_matched} -> Channel: {channel}\n{data_str}"
                        print(formatted)
                        print("-" * 80)
                        
                        # Write to file if specified
                        if output_fp:
                            output_fp.write(formatted + "\n" + "-" * 80 + "\n")
                            output_fp.flush()
                    
                except asyncio.TimeoutError:
                    # Timeout is expected, continue
                    continue
                except Exception as e:
                    print(f"[ERROR] Error processing message: {e}", file=sys.stderr)
                    continue
                    
        except KeyboardInterrupt:
            pass
        finally:
            print(f"\n[INFO] Received {message_count} messages total")
            if output_fp:
                output_fp.close()
                print(f"[INFO] Messages saved to: {output_file}")
            
            # Cleanup
            await pubsub.unsubscribe()
            if pattern:
                await pubsub.punsubscribe()
            await pubsub.aclose()
            await redis_client.aclose()
            print("[INFO] Disconnected from Redis")
            
    except Exception as e:
        print(f"[ERROR] Failed to connect to Redis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Listen to Redis Pub/Sub channels and print messages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--redis-url',
        type=str,
        default=None,
        help='Redis URL (default: from REDIS_URL env var or redis://localhost:6379/0)'
    )
    
    parser.add_argument(
        '--channels',
        nargs='+',
        help='Specific channels to subscribe to (e.g., realtime:quote:AAPL realtime:trade:AAPL)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        help='Pattern to match channels (e.g., "realtime:*:*" for all realtime channels)'
    )
    
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty-print JSON messages with indentation'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file to save messages to (appends to file)'
    )
    
    return parser.parse_args()

async def main():
    """Main function."""
    args = parse_args()
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Get Redis URL
    redis_url = args.redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    # Validate arguments
    if not args.channels and not args.pattern:
        print("[ERROR] Must specify either --channels or --pattern", file=sys.stderr)
        return 1
    
    if args.channels and args.pattern:
        print("[ERROR] Cannot specify both --channels and --pattern", file=sys.stderr)
        return 1
    
    # Start listening
    await listen_to_channels(
        redis_url=redis_url,
        channels=args.channels or [],
        pattern=args.pattern,
        pretty=args.pretty,
        output_file=args.output
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

