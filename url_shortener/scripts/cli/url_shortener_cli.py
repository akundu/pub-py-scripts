#!/usr/bin/env python3
"""
Command-line interface for URL shortener service.

Usage:
    python url_shortener_cli.py shorten <url> [--custom-code CODE]
    python url_shortener_cli.py get <short_code>
    python url_shortener_cli.py stats <short_code>
    python url_shortener_cli.py list [--limit N]
    python url_shortener_cli.py health
"""

import argparse
import asyncio
import json
import sys
import os
from typing import Optional
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lib.database.questdb import URLShortenerQuestDB
from lib.database.cache import RedisCache
from lib.service import URLShortenerService
from lib.shortcode import ShortCodeGenerator
from lib.common.logging_config import setup_logging


class URLShortenerCLI:
    """Command-line interface for URL shortener."""
    
    def __init__(self, db_url: str, redis_url: Optional[str] = None, verbose: bool = False):
        """Initialize CLI."""
        self.db_url = db_url
        self.redis_url = redis_url
        self.verbose = verbose
        self.logger = setup_logging(level="DEBUG" if verbose else "INFO")
        self.db = None
        self.cache = None
        self.service = None
    
    async def initialize(self):
        """Initialize database and service."""
        self.logger.info("Initializing URL shortener...")
        
        # Initialize database
        self.db = URLShortenerQuestDB(
            db_config=self.db_url,
            logger=self.logger,
        )
        
        # Initialize cache (optional)
        if self.redis_url:
            self.cache = RedisCache(
                redis_url=self.redis_url,
                logger=self.logger,
            )
            await self.cache.connect()
        
        # Initialize service
        generator = ShortCodeGenerator(default_length=6)
        self.service = URLShortenerService(
            db=self.db,
            cache=self.cache,
            short_code_generator=generator,
            logger=self.logger,
        )
        
        self.logger.info("Initialization complete")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.service:
            await self.service.close()
    
    async def shorten(self, url: str, custom_code: Optional[str] = None):
        """Shorten a URL."""
        try:
            result = await self.service.create_short_url(url, custom_code)
            
            print(json.dumps({
                "success": True,
                "short_code": result["short_code"],
                "original_url": result["original_url"],
                "created_at": result["created_at"].isoformat(),
                "message": f"Successfully shortened URL to: {result['short_code']}"
            }, indent=2))
            
            return 0
            
        except ValueError as e:
            print(json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2), file=sys.stderr)
            return 1
        except Exception as e:
            print(json.dumps({
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }, indent=2), file=sys.stderr)
            return 1
    
    async def get(self, short_code: str):
        """Get original URL for a short code."""
        try:
            original_url = await self.service.get_original_url(
                short_code, 
                increment_count=False  # Don't increment for CLI lookups
            )
            
            if original_url:
                print(json.dumps({
                    "success": True,
                    "short_code": short_code,
                    "original_url": original_url
                }, indent=2))
                return 0
            else:
                print(json.dumps({
                    "success": False,
                    "error": f"Short code '{short_code}' not found"
                }, indent=2), file=sys.stderr)
                return 1
                
        except Exception as e:
            print(json.dumps({
                "success": False,
                "error": f"Error: {str(e)}"
            }, indent=2), file=sys.stderr)
            return 1
    
    async def stats(self, short_code: str):
        """Get statistics for a short code."""
        try:
            info = await self.service.get_url_info(short_code)
            
            if info:
                # Convert datetime objects to strings
                result = {
                    "success": True,
                    "short_code": info["short_code"],
                    "original_url": info["original_url"],
                    "created_at": info["created_at"].isoformat() if isinstance(info["created_at"], datetime) else str(info["created_at"]),
                    "access_count": info["access_count"],
                    "last_accessed": info["last_accessed"].isoformat() if info["last_accessed"] and isinstance(info["last_accessed"], datetime) else None,
                }
                print(json.dumps(result, indent=2))
                return 0
            else:
                print(json.dumps({
                    "success": False,
                    "error": f"Short code '{short_code}' not found"
                }, indent=2), file=sys.stderr)
                return 1
                
        except Exception as e:
            print(json.dumps({
                "success": False,
                "error": f"Error: {str(e)}"
            }, indent=2), file=sys.stderr)
            return 1
    
    async def list_urls(self, limit: int = 100):
        """List recent URLs."""
        try:
            urls = await self.service.list_recent_urls(limit)
            
            # Convert datetime objects to strings
            for url in urls:
                if isinstance(url.get("created_at"), datetime):
                    url["created_at"] = url["created_at"].isoformat()
                if url.get("last_accessed") and isinstance(url["last_accessed"], datetime):
                    url["last_accessed"] = url["last_accessed"].isoformat()
            
            print(json.dumps({
                "success": True,
                "count": len(urls),
                "urls": urls
            }, indent=2))
            
            return 0
            
        except Exception as e:
            print(json.dumps({
                "success": False,
                "error": f"Error: {str(e)}"
            }, indent=2), file=sys.stderr)
            return 1
    
    async def health(self):
        """Check service health."""
        try:
            health_status = await self.service.health_check()
            stats = await self.service.get_statistics()
            
            print(json.dumps({
                "success": True,
                "health": health_status,
                "statistics": stats
            }, indent=2))
            
            return 0 if health_status["overall"] else 1
            
        except Exception as e:
            print(json.dumps({
                "success": False,
                "error": f"Error: {str(e)}"
            }, indent=2), file=sys.stderr)
            return 1


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="URL Shortener CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Shorten a URL
  %(prog)s shorten https://example.com/long/url
  
  # Shorten with custom code
  %(prog)s shorten https://example.com/long/url --custom-code mylink
  
  # Get original URL
  %(prog)s get mylink
  
  # Get statistics
  %(prog)s stats mylink
  
  # List recent URLs
  %(prog)s list --limit 10
  
  # Check health
  %(prog)s health
        """
    )
    
    parser.add_argument(
        "--db-url",
        default=os.getenv("QUESTDB_URL", "questdb://admin:quest@localhost:8812/qdb"),
        help="QuestDB connection URL (default: from QUESTDB_URL env or questdb://admin:quest@localhost:8812/qdb)"
    )
    
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL"),
        help="Redis connection URL (optional, default: from REDIS_URL env)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Shorten command
    shorten_parser = subparsers.add_parser("shorten", help="Shorten a URL")
    shorten_parser.add_argument("url", help="URL to shorten")
    shorten_parser.add_argument("--custom-code", help="Custom short code")
    
    # Get command
    get_parser = subparsers.add_parser("get", help="Get original URL")
    get_parser.add_argument("short_code", help="Short code to lookup")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get URL statistics")
    stats_parser.add_argument("short_code", help="Short code to get stats for")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List recent URLs")
    list_parser.add_argument("--limit", type=int, default=100, help="Maximum number to return")
    
    # Health command
    health_parser = subparsers.add_parser("health", help="Check service health")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize CLI
    cli = URLShortenerCLI(
        db_url=args.db_url,
        redis_url=args.redis_url,
        verbose=args.verbose,
    )
    
    try:
        await cli.initialize()
        
        # Execute command
        if args.command == "shorten":
            return await cli.shorten(args.url, args.custom_code)
        elif args.command == "get":
            return await cli.get(args.short_code)
        elif args.command == "stats":
            return await cli.stats(args.short_code)
        elif args.command == "list":
            return await cli.list_urls(args.limit)
        elif args.command == "health":
            return await cli.health()
        else:
            parser.print_help()
            return 1
            
    finally:
        await cli.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)






