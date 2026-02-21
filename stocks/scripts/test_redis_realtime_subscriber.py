#!/usr/bin/env python3
"""
Test Redis realtime channels used by polygon_realtime_streamer.

Subscribe to realtime:quote:* and realtime:trade:* and print each message
so you can verify the streamer is publishing to Redis.

Usage:
  # Subscribe to all realtime channels (quote and trade for any symbol)
  python scripts/test_redis_realtime_subscriber.py

  # Subscribe only to specific symbols
  python scripts/test_redis_realtime_subscriber.py --symbols I:SPX I:NDX

  # Custom Redis URL
  python scripts/test_redis_realtime_subscriber.py --redis-url redis://lin1.kundu.dev:6379/0

Then in another terminal run the streamer (e.g. poll-only):
  python scripts/polygon_realtime_streamer.py --symbols I:SPX I:NDX --feed both \\
    --redis-url redis://lin1.kundu.dev:6379 --poll-only --no-db-write --poll-interval 15

Alternative: redis-cli (subscribe to one channel, see raw JSON):
  redis-cli -u redis://lin1.kundu.dev:6379/0 SUBSCRIBE realtime:quote:I:SPX realtime:trade:I:SPX
"""

import argparse
import json
import os
import sys

try:
    import redis
except ImportError:
    print("redis package required: pip install redis", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Subscribe to Redis realtime channels and print messages")
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        help="Redis URL (default: REDIS_URL or redis://localhost:6379/0)",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Optional: only subscribe to these symbols (e.g. I:SPX I:NDX). If omitted, subscribe to realtime:quote:* and realtime:trade:* via PSUBSCRIBE.",
    )
    args = parser.parse_args()

    r = redis.from_url(
        args.redis_url,
        decode_responses=True,
        socket_connect_timeout=5,
    )

    try:
        r.ping()
    except redis.ConnectionError as e:
        print(f"Redis connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Connected to {args.redis_url}")
    if args.symbols:
        channels = []
        for s in args.symbols:
            channels.append(f"realtime:quote:{s}")
            channels.append(f"realtime:trade:{s}")
        print(f"Subscribing to: {channels}")
        pubsub = r.pubsub()
        pubsub.subscribe(channels)
    else:
        print("Subscribing to pattern: realtime:quote:* and realtime:trade:*")
        pubsub = r.pubsub()
        pubsub.psubscribe("realtime:quote:*", "realtime:trade:*")

    print("Waiting for messages (Ctrl+C to stop)...\n")
    for message in pubsub.listen():
        if message["type"] in ("subscribe", "psubscribe"):
            continue
        if message["type"] not in ("message", "pmessage"):
            continue
        try:
            payload = json.loads(message["data"])
            channel = message.get("channel") or message.get("pattern", "")
            symbol = payload.get("symbol", "")
            data_type = payload.get("data_type", "")
            records = payload.get("records", [])
            ts = payload.get("timestamp", "")
            print(f"[{channel}] {data_type} {symbol} @ {ts}")
            for rec in records:
                price = rec.get("price")
                size = rec.get("size")
                ts_rec = rec.get("timestamp", "")
                print(f"  -> price={price} size={size} timestamp={ts_rec}")
            print()
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  (parse: {e})", message["data"][:200])


if __name__ == "__main__":
    main()
