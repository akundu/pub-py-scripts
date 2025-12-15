#!/usr/bin/env python3
"""
Simple Redis test publisher for the db_server realtime WebSocket pipeline.

This script publishes synthetic quote or trade updates for a single symbol
to the same Redis channels that the production streamer uses:

    realtime:quote:{SYMBOL}
    realtime:trade:{SYMBOL}

The db_server's WebSocketManager will consume these messages, save them to
the DB, and broadcast them to any connected WebSocket clients (including
the stock_display_dashboard).

Example:
    # Publish quotes for TESTSYM every 0.5s using default Redis URL
    python tests/realtime_test_publisher.py --symbol TESTSYM --interval 0.5

    # Publish trades instead of quotes
    python tests/realtime_test_publisher.py --symbol TESTSYM --data-type trade --interval 0.5
"""

import argparse
import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import Optional

from scripts.polygon_realtime_streamer import RedisPublisher


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish synthetic realtime data for a single symbol to Redis "
                    "for testing the db_server WebSocket pipeline."
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="TESTSYM",
        help="Symbol to publish (default: TESTSYM)",
    )

    parser.add_argument(
        "--data-type",
        type=str,
        choices=["quote", "trade"],
        default="quote",
        help="Type of realtime data to publish (default: quote)",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Interval between messages in seconds (default: 0.5)",
    )

    parser.add_argument(
        "--redis-url",
        type=str,
        default=None,
        help="Redis URL (default: use RedisPublisher default / REDIS_URL env var)",
    )

    parser.add_argument(
        "--start-price",
        type=float,
        default=100.0,
        help="Starting price for synthetic data (default: 100.0)",
    )

    parser.add_argument(
        "--max-jump-bps",
        type=float,
        default=50.0,
        help="Maximum random price jump in basis points per tick (default: 50 = 0.5%%)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def _next_price(current: float, max_jump_bps: float) -> float:
    """
    Simple random-walk price generator.

    max_jump_bps: maximum absolute move in basis points (1/100 of a percent)
    """
    if current <= 0:
        current = 100.0
    # Random move between -max_jump_bps and +max_jump_bps bps
    delta_bps = random.uniform(-max_jump_bps, max_jump_bps)
    move = current * (delta_bps / 10_000.0)
    new_price = max(0.01, current + move)
    return round(new_price, 4)


async def _publish_loop(
    symbol: str,
    data_type: str,
    interval: float,
    redis_url: Optional[str],
    start_price: float,
    max_jump_bps: float,
) -> None:
    """Main publishing loop using RedisPublisher."""
    price = start_price

    async with RedisPublisher(redis_url=redis_url) as publisher:
        if not publisher.available or publisher.redis_client is None:
            logger.error("RedisPublisher is not available; check Redis installation / URL")
            return

        logger.info(
            f"Starting test publisher for symbol={symbol}, data_type={data_type}, "
            f"interval={interval}s, redis_url={publisher.redis_url}"
        )

        try:
            while True:
                now = datetime.now(timezone.utc).isoformat()
                price = _next_price(price, max_jump_bps)

                if data_type == "quote":
                    # Minimal quote-like payload; WebSocketManager._handle_redis_message
                    # will convert this into the quote_update structure the dashboard expects.
                    record = {
                        "timestamp": now,
                        "price": price,
                        "size": random.randint(1, 1000),
                    }
                else:  # trade
                    record = {
                        "timestamp": now,
                        "price": price,
                        "size": random.randint(1, 1000),
                    }

                ok = await publisher.publish_realtime_data(
                    symbol=symbol,
                    data_type=data_type,
                    records=[record],
                )

                if ok:
                    logger.debug(f"Published {data_type} for {symbol}: {record}")
                else:
                    logger.warning(f"Failed to publish {data_type} for {symbol}")

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        except KeyboardInterrupt:
            logger.info("Interrupted by user, stopping publisher")
        except Exception as e:
            logger.error(f"Error in publish loop: {e}", exc_info=True)


async def main_async() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%%Y-%%m-%%d %%H:%%M:%%S",
    )

    await _publish_loop(
        symbol=args.symbol,
        data_type=args.data_type,
        interval=args.interval,
        redis_url=args.redis_url,
        start_price=args.start_price,
        max_jump_bps=args.max_jump_bps,
    )
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())


