#!/usr/bin/env python3
"""Clear the /range_percentiles Redis response cache.

The cache lives under the ``rp:v1:`` namespace in Redis (see
``common/range_percentiles_cache.py`` for the key format and TTL rule).
By default this script wipes the entire namespace. Pass ``--pattern`` to
narrow the scope to just the HTML responses, just the multi-window API
responses, etc.

Returns 0 silently when Redis is unavailable (no namespace cache to clear,
nothing to do) — same fail-soft behavior as the cache reads/writes.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Allow `python scripts/clear_response_cache.py` from the repo root without
# needing PYTHONPATH gymnastics.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common.range_percentiles_cache import clear_cache


async def _main(pattern: str, redis_url: str | None) -> int:
    n = await clear_cache(pattern, redis_url=redis_url)
    print(f"Deleted {n} key(s) matching {pattern!r}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description='''
Clear the /range_percentiles Redis response cache.

Removes cached HTML/JSON response bodies stored under the rp:v1:
namespace by `db_server.py`'s `_store_rp_cache`. A cleared cache will
be repopulated on the next request to /range_percentiles or its API
variants.
        '''.strip(),
        epilog='''
Examples:
  %(prog)s
      Clear ALL rp:v1:* entries (full namespace wipe — default).

  %(prog)s --pattern 'rp:v1:html:*'
      Clear only the cached HTML responses (leave JSON cache alone).

  %(prog)s --pattern 'rp:v1:multi_window_api:*'
      Clear only the multi-window API responses.

  %(prog)s --redis-url redis://otherhost:6379/0
      Point at a non-default Redis instance (also honored via REDIS_URL
      env var).

  %(prog)s --help
      Show this help message.
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pattern",
        default="rp:v1:*",
        help="Redis key glob pattern to delete (default: rp:v1:*)",
    )
    parser.add_argument(
        "--redis-url",
        default=None,
        help="Override Redis URL (default: $REDIS_URL or redis://localhost:6379/0)",
    )
    args = parser.parse_args()
    return asyncio.run(_main(args.pattern, args.redis_url))


if __name__ == "__main__":
    raise SystemExit(main())
