#!/usr/bin/env python3
"""
Comprehensive timeout optimization and monitoring script for TimescaleDB.
This script helps diagnose and fix timeout issues in your stock data database.
"""

import asyncio
import asyncpg
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import the common modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.timescale_db import StockDBTimescale
from common.logging_utils import get_logger

class TimeoutOptimizer:
    """Class to optimize timeout handling and monitor database performance."""
    
    def __init__(self, db_config: str, logger: Optional[logging.Logger] = None):
        self.db_config = db_config
        self.logger = logger or get_logger("timeout_optimizer", level="INFO")
        self.db: Optional[StockDBTimescale] = None
        
    async def connect(self) -> bool:
        """Connect to the database with optimized timeout settings."""
        try:
            self.logger.info("Connecting to TimescaleDB with optimized timeout settings...")
            
            # Create database instance with aggressive timeout settings
            self.db = StockDBTimescale(
                db_config=self.db_config,
                pool_max_size=20,  # Increased pool size
                pool_connection_timeout_minutes=60,  # Longer connection timeout
                mv_refresh_interval_minutes=10,  # Less frequent refresh
                logger=self.logger,
                log_level="INFO"  # Correct parameter name for StockDBTimescale
            )
            
            # Test connection
            conn = await self.db._get_connection()
            try:
                version = await conn.fetchval("SELECT version()")
                self.logger.info(f"✅ Connected to: {version}")
                return True
            finally:
                await self.db._return_connection(conn)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to connect: {e}")
            return False
    
    async def close(self):
        """Close database connection."""
        if self.db:
            await self.db.close_pool()
    
    async def check_current_timeout_settings(self) -> Dict[str, Any]:
        """Check current timeout settings in the database."""
        if not self.db:
            return {"error": "Not connected"}
        
        settings = {}
        
        try:
            conn = await self.db._get_connection()
            try:
                # Check PostgreSQL timeout settings
                settings['statement_timeout'] = await conn.fetchval("SHOW statement_timeout")
                settings['idle_in_transaction_session_timeout'] = await conn.fetchval("SHOW idle_in_transaction_session_timeout")
                settings['lock_timeout'] = await conn.fetchval("SHOW lock_timeout")
                settings['deadlock_timeout'] = await conn.fetchval("SHOW deadlock_timeout")
                
                # Check connection settings
                settings['max_connections'] = await conn.fetchval("SHOW max_connections")
                settings['shared_buffers'] = await conn.fetchval("SHOW shared_buffers")
                settings['work_mem'] = await conn.fetchval("SHOW work_mem")
                
                # Check TimescaleDB settings
                try:
                    settings['timescaledb_max_background_workers'] = await conn.fetchval("SHOW timescaledb.max_background_workers")
                except:
                    settings['timescaledb_max_background_workers'] = "Not available"
                
                # Check current connection count
                settings['active_connections'] = await conn.fetchval("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
                settings['idle_connections'] = await conn.fetchval("SELECT count(*) FROM pg_stat_activity WHERE state = 'idle'")
                settings['total_connections'] = await conn.fetchval("SELECT count(*) FROM pg_stat_activity")
            finally:
                await self.db._return_connection(conn)
                
        except Exception as e:
            settings['error'] = str(e)
            self.logger.error(f"Failed to check timeout settings: {e}")
        
        return settings
    
    async def optimize_timeout_settings(self) -> Dict[str, Any]:
        """Apply optimized timeout settings to the database."""
        if not self.db:
            return {"error": "Not connected"}
        
        results = {}
        
        try:
            conn = await self.db._get_connection()
            try:
                self.logger.info("Applying optimized timeout settings...")
                
                # Set aggressive timeout settings
                await conn.execute("SET statement_timeout = '1800s'")  # 30 minutes
                await conn.execute("SET idle_in_transaction_session_timeout = '1800s'")  # 30 minutes
                await conn.execute("SET lock_timeout = '300s'")  # 5 minutes
                await conn.execute("SET deadlock_timeout = '1000ms'")
                
                # Set performance settings
                await conn.execute("SET work_mem = '128MB'")
                await conn.execute("SET temp_buffers = '256MB'")
                await conn.execute("SET random_page_cost = 1.1")
                await conn.execute("SET effective_io_concurrency = 400")
                
                # Set connection optimization
                await conn.execute("SET tcp_keepalives_idle = 300")
                await conn.execute("SET tcp_keepalives_interval = 30")
                await conn.execute("SET tcp_keepalives_count = 3")
                
                results['status'] = 'success'
                results['message'] = 'Timeout settings optimized successfully'
            finally:
                await self.db._return_connection(conn)
                
        except Exception as e:
            results['error'] = str(e)
            self.logger.error(f"Failed to optimize timeout settings: {e}")
        
        return results
    
    async def test_query_performance(self, query_type: str = "simple") -> Dict[str, Any]:
        """Test query performance to identify timeout bottlenecks."""
        if not self.db:
            return {"error": "Not connected"}
        
        results = {}
        
        try:
            conn = await self.db._get_connection()
            try:
                self.logger.info(f"Testing {query_type} query performance...")
                
                start_time = time.time()
                
                if query_type == "simple":
                    # Simple count query
                    result = await conn.fetchval("SELECT COUNT(*) FROM stock_data.daily_prices")
                    query_desc = "Simple count query"
                    
                elif query_type == "complex":
                    # Complex aggregation query
                    result = await conn.fetchval("""
                        SELECT ticker, AVG(close), COUNT(*) 
                        FROM stock_data.daily_prices 
                        WHERE date >= CURRENT_DATE - INTERVAL '30 days'
                        GROUP BY ticker 
                        ORDER BY AVG(close) DESC 
                        LIMIT 100
                    """)
                    query_desc = "Complex aggregation query"
                    
                elif query_type == "timeseries":
                    # Timeseries query
                    result = await conn.fetchval("""
                        SELECT time_bucket('1 day', datetime) as day_bucket,
                               AVG(close) as avg_close
                        FROM stock_data.hourly_prices
                        WHERE datetime >= CURRENT_DATE - INTERVAL '7 days'
                        GROUP BY day_bucket
                        ORDER BY day_bucket
                    """)
                    query_desc = "Timeseries aggregation query"
                    
                else:
                    return {"error": f"Unknown query type: {query_type}"}
                
                execution_time = time.time() - start_time
                
                results = {
                    'query_type': query_type,
                    'query_description': query_desc,
                    'execution_time_seconds': round(execution_time, 3),
                    'result': str(result),
                    'status': 'success'
                }
                
                if execution_time > 5.0:
                    results['warning'] = f"Query took {execution_time:.2f}s - consider optimization"
                elif execution_time > 1.0:
                    results['info'] = f"Query performance: {execution_time:.2f}s"
                else:
                    results['info'] = f"Query performance: {execution_time:.2f}s (excellent)"
            finally:
                await self.db._return_connection(conn)
                
        except Exception as e:
            results = {
                'query_type': query_type,
                'error': str(e),
                'status': 'failed'
            }
            self.logger.error(f"Query performance test failed: {e}")
        
        return results
    
    async def analyze_connection_pool(self) -> Dict[str, Any]:
        """Analyze connection pool health and performance."""
        if not self.db:
            return {"error": "Not connected"}
        
        analysis = {}
        
        try:
            conn = await self.db._get_connection()
            try:
                # Get connection pool statistics
                try:
                    pool_stats = await self.db.get_pool_stats()
                    analysis['pool_stats'] = pool_stats
                except:
                    analysis['pool_stats'] = "Pool stats not available"
                
                # Check for connection leaks
                long_running_queries = await conn.fetch("""
                    SELECT pid, query_start, state, query 
                    FROM pg_stat_activity 
                    WHERE state = 'active' 
                    AND query_start < NOW() - INTERVAL '5 minutes'
                    ORDER BY query_start
                """)
                
                analysis['long_running_queries'] = [
                    {
                        'pid': row['pid'],
                        'query_start': str(row['query_start']),
                        'state': row['state'],
                        'query': row['query'][:100] + '...' if len(row['query']) > 100 else row['query']
                    }
                    for row in long_running_queries
                ]
                
                # Check for locks
                locks = await conn.fetch("""
                    SELECT l.pid, l.mode, l.granted, a.query
                    FROM pg_locks l
                    JOIN pg_stat_activity a ON l.pid = a.pid
                    WHERE NOT l.granted
                    ORDER BY l.pid
                """)
                
                analysis['blocked_locks'] = [
                    {
                        'pid': row['pid'],
                        'mode': row['mode'],
                        'granted': row['granted'],
                        'query': row['query'][:100] + '...' if len(row['query']) > 100 else row['query']
                    }
                    for row in locks
                ]
            finally:
                await self.db._return_connection(conn)
                
        except Exception as e:
            analysis['error'] = str(e)
            self.logger.error(f"Failed to analyze connection pool: {e}")
        
        return analysis
    
    async def generate_optimization_report(self) -> str:
        """Generate a comprehensive optimization report."""
        report = []
        report.append("=" * 80)
        report.append("TIMEOUT OPTIMIZATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check current settings
        report.append("\n📊 CURRENT TIMEOUT SETTINGS:")
        settings = await self.check_current_timeout_settings()
        if 'error' not in settings:
            report.append(f"   Statement Timeout: {settings.get('statement_timeout', 'unknown')}")
            report.append(f"   Idle Transaction Timeout: {settings.get('idle_in_transaction_session_timeout', 'unknown')}")
            report.append(f"   Lock Timeout: {settings.get('lock_timeout', 'unknown')}")
            report.append(f"   Active Connections: {settings.get('active_connections', 'unknown')}")
            report.append(f"   Total Connections: {settings.get('total_connections', 'unknown')}")
        else:
            report.append(f"   Error: {settings['error']}")
        
        # Test query performance
        report.append("\n🚀 QUERY PERFORMANCE TESTS:")
        for query_type in ["simple", "complex", "timeseries"]:
            perf_result = await self.test_query_performance(query_type)
            if 'error' not in perf_result:
                report.append(f"   {perf_result['query_description']}: {perf_result['execution_time_seconds']}s")
                if 'warning' in perf_result:
                    report.append(f"      ⚠️  {perf_result['warning']}")
            else:
                report.append(f"   {query_type}: Failed - {perf_result['error']}")
        
        # Connection pool analysis
        report.append("\n🔌 CONNECTION POOL ANALYSIS:")
        pool_analysis = await self.analyze_connection_pool()
        if 'error' not in pool_analysis:
            if pool_analysis.get('long_running_queries'):
                report.append(f"   ⚠️  Long-running queries: {len(pool_analysis['long_running_queries'])}")
            else:
                report.append("   ✅ No long-running queries detected")
                
            if pool_analysis.get('blocked_locks'):
                report.append(f"   🚨 Blocked locks: {len(pool_analysis['blocked_locks'])}")
            else:
                report.append("   ✅ No blocked locks detected")
        else:
            report.append(f"   Error: {pool_analysis['error']}")
        
        # Optimization recommendations
        report.append("\n💡 OPTIMIZATION RECOMMENDATIONS:")
        report.append("   1. Ensure PostgreSQL configuration is optimized (check postgresql.conf)")
        report.append("   2. Verify PgBouncer settings are appropriate for your workload")
        report.append("   3. Monitor connection pool usage and adjust pool sizes")
        report.append("   4. Consider implementing query timeouts in application code")
        report.append("   5. Use batch operations for large data insertions")
        report.append("   6. Implement proper retry logic with exponential backoff")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Optimize TimescaleDB timeout handling")
    parser.add_argument("--db-config", required=True, help="Database connection string")
    parser.add_argument("--optimize", action="store_true", help="Apply timeout optimizations")
    parser.add_argument("--test-performance", action="store_true", help="Test query performance")
    parser.add_argument("--analyze-pool", action="store_true", help="Analyze connection pool")
    parser.add_argument("--output", help="Output file for report")
    
    args = parser.parse_args()
    
    optimizer = TimeoutOptimizer(args.db_config)
    
    try:
        # Connect to database
        if not await optimizer.connect():
            print("❌ Cannot connect to database. Check connection parameters.")
            sys.exit(1)
        
        # Apply optimizations if requested
        if args.optimize:
            print("🔧 Applying timeout optimizations...")
            result = await optimizer.optimize_timeout_settings()
            if 'error' not in result:
                print(f"✅ {result['message']}")
            else:
                print(f"❌ Optimization failed: {result['error']}")
        
        # Test performance if requested
        if args.test_performance:
            print("🚀 Testing query performance...")
            for query_type in ["simple", "complex", "timeseries"]:
                result = await optimizer.test_query_performance(query_type)
                if 'error' not in result:
                    print(f"   {query_type}: {result['execution_time_seconds']}s")
                else:
                    print(f"   {query_type}: Failed - {result['error']}")
        
        # Analyze connection pool if requested
        if args.analyze_pool:
            print("🔌 Analyzing connection pool...")
            analysis = await optimizer.analyze_connection_pool()
            if 'error' not in analysis:
                print(f"   Long-running queries: {len(analysis.get('long_running_queries', []))}")
                print(f"   Blocked locks: {len(analysis.get('blocked_locks', []))}")
            else:
                print(f"   Analysis failed: {analysis['error']}")
        
        # Generate comprehensive report
        print("📊 Generating optimization report...")
        report = await optimizer.generate_optimization_report()
        print(report)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"\n📄 Report saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Optimization interrupted by user")
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        logging.error(f"Optimization failed: {e}", exc_info=True)
    finally:
        await optimizer.close()

if __name__ == "__main__":
    asyncio.run(main())
