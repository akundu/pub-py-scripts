#!/usr/bin/env python3
"""
Seed test data into URL shortener database.

Usage:
    python seed_data.py --db-url questdb://admin:quest@localhost:8812/qdb --count 10
"""

import argparse
import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta
import random

# Add parent directories to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lib.database.questdb import URLShortenerQuestDB
from lib.service import URLShortenerService
from lib.shortcode import ShortCodeGenerator
from lib.common.logging_config import setup_logging


# Sample URLs for testing
SAMPLE_URLS = [
    "https://github.com/python/cpython",
    "https://docs.python.org/3/library/asyncio.html",
    "https://fastapi.tiangolo.com/",
    "https://www.questdb.io/docs/",
    "https://redis.io/documentation",
    "https://stackoverflow.com/questions/tagged/python",
    "https://news.ycombinator.com/",
    "https://www.reddit.com/r/programming/",
    "https://medium.com/@username/long-article-title",
    "https://twitter.com/username/status/123456789",
]


async def main():
    parser = argparse.ArgumentParser(description="Seed test data")
    parser.add_argument(
        "--db-url",
        default=os.getenv("QUESTDB_URL", "questdb://admin:quest@localhost:8812/qdb"),
        help="QuestDB connection URL"
    )
    parser.add_argument("--count", type=int, default=10, help="Number of URLs to create")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    logger = setup_logging(level="DEBUG" if args.verbose else "INFO")
    
    try:
        logger.info("Connecting to database...")
        
        db = URLShortenerQuestDB(db_config=args.db_url, logger=logger)
        generator = ShortCodeGenerator(default_length=6)
        service = URLShortenerService(db=db, short_code_generator=generator, logger=logger)
        
        logger.info(f"Creating {args.count} test URLs...")
        
        created = 0
        for i in range(args.count):
            # Pick a random URL
            url = random.choice(SAMPLE_URLS)
            
            # Add some variation
            url = f"{url}?test={i}&seed=true"
            
            try:
                result = await service.create_short_url(url)
                logger.info(f"Created: {result['short_code']} -> {url}")
                created += 1
            except Exception as e:
                logger.warning(f"Failed to create URL {i}: {e}")
        
        logger.info(f"Successfully created {created} URLs")
        
        # Show statistics
        stats = await service.get_statistics()
        logger.info(f"Total URLs in database: {stats['total_urls']}")
        
        await service.close()
        logger.info("Done")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error seeding data: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)






