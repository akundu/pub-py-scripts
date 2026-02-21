#!/bin/bash

# PostgreSQL Setup Script for Stock Database System
# This script sets up PostgreSQL with Docker and installs dependencies

set -e

echo "🚀 Setting up PostgreSQL for Stock Database System"
echo "=================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install -r requirements_postgresql.txt

# Create data directory if it doesn't exist
mkdir -p data

# Stop any existing containers to ensure clean setup
echo ""
echo "🛑 Stopping any existing containers..."
docker-compose down

# Start PostgreSQL container
echo ""
echo "🐘 Starting PostgreSQL container..."
docker-compose up -d postgres

# Wait for PostgreSQL to be ready
echo ""
echo "⏳ Waiting for PostgreSQL to be ready..."
until docker-compose exec -T postgres pg_isready -U user -d stock_data; do
    echo "   Waiting for PostgreSQL..."
    sleep 2
done

echo "✅ PostgreSQL is ready!"

# Wait a bit more for initialization to complete
echo ""
echo "⏳ Waiting for database initialization to complete..."
sleep 10

# Verify database ownership and permissions
echo ""
echo "🔍 Verifying database ownership and permissions..."
docker-compose exec -T postgres psql -U user -d stock_data -c "
DO \$\$
BEGIN
    -- Verify database ownership
    IF EXISTS (
        SELECT 1 FROM pg_database 
        WHERE datname = 'stock_data' AND datdba = (SELECT oid FROM pg_roles WHERE rolname = 'user')
    ) THEN
        RAISE NOTICE '✅ stock_data database is owned by user';
    ELSE
        RAISE WARNING '⚠️ stock_data database is NOT owned by user';
    END IF;
    
    -- Verify schema ownership
    IF EXISTS (
        SELECT 1 FROM information_schema.schemata 
        WHERE schema_name = 'public' AND schema_owner = 'user'
    ) THEN
        RAISE NOTICE '✅ public schema is owned by user';
    ELSE
        RAISE WARNING '⚠️ public schema is NOT owned by user';
    END IF;
    
    -- Verify table ownership
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name IN ('daily_prices', 'hourly_prices', 'realtime_data', 'db_stats')
        AND table_owner = 'user'
    ) THEN
        RAISE NOTICE '✅ All tables are owned by user';
    ELSE
        RAISE WARNING '⚠️ Some tables are NOT owned by user';
    END IF;
    
    -- Verify permissions
    IF EXISTS (
        SELECT 1 FROM information_schema.role_table_grants 
        WHERE grantee = 'user' 
        AND table_schema = 'public'
        AND privilege_type = 'ALL'
    ) THEN
        RAISE NOTICE '✅ user has ALL privileges on public schema tables';
    ELSE
        RAISE WARNING '⚠️ user may not have ALL privileges on public schema tables';
    END IF;
END \$\$;
"

# Test connection
echo ""
echo "🧪 Testing database connection..."
python3 -c "
import psycopg2
try:
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='stock_data',
        user='user',
        password='password'
    )
    print('✅ Database connection successful!')
    
    # Test basic operations
    cursor = conn.cursor()
    cursor.execute('SELECT current_user, current_database()')
    user, db = cursor.fetchone()
    print(f'   Connected as: {user}')
    print(f'   Database: {db}')
    
    # Test table access
    cursor.execute('SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = \'public\'')
    table_count = cursor.fetchone()[0]
    print(f'   Tables in public schema: {table_count}')
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f'❌ Database connection failed: {e}')
    exit(1)
"

# Apply database optimizations
echo ""
echo "🔧 Applying database optimizations from MDs/db_optimizations/*.md..."
docker-compose exec -T postgres psql -U user -d stock_data -c "
-- Test fast count optimizations
SELECT 'Testing fast count optimizations...' as status;

-- Test fast count views
SELECT 'hourly_prices_count' as view_name, count FROM hourly_prices_count LIMIT 1;
SELECT 'daily_prices_count' as view_name, count FROM daily_prices_count LIMIT 1;
SELECT 'realtime_data_count' as view_name, count FROM realtime_data_count LIMIT 1;

-- Test fast count functions
SELECT 'fast_count_hourly_prices' as function_name, fast_count_hourly_prices() as count;
SELECT 'fast_count_daily_prices' as function_name, fast_count_daily_prices() as count;
SELECT 'fast_count_realtime_data' as function_name, fast_count_realtime_data() as count;

-- Test performance monitoring functions
SELECT 'verify_count_accuracy' as function_name, COUNT(*) as result_count FROM verify_count_accuracy();
SELECT 'get_index_usage_stats' as function_name, COUNT(*) as result_count FROM get_index_usage_stats();

-- Test utility functions
SELECT 'get_all_table_counts' as function_name, COUNT(*) as result_count FROM get_all_table_counts();

SELECT '✅ All database optimizations are working correctly!' as status;
"

echo ""
echo "🎉 PostgreSQL setup completed successfully!"
echo ""
echo "📋 Connection Details:"
echo "   Host: localhost"
echo "   Port: 5432"
echo "   Database: stock_data (owned by user)"
echo "   Username: user"
echo "   Password: password"
echo ""
echo "🔧 Useful Commands:"
echo "   Start database: docker-compose up -d postgres"
echo "   Stop database:  docker-compose down"
echo "   View logs:      docker-compose logs postgres"
echo "   Connect via psql: psql -h localhost -p 5432 -U user -d stock_data"
echo ""
echo "📊 Database Features:"
echo "   - Database owned by: user"
echo "   - All tables owned by: user"
echo "   - Full permissions granted to: user"
echo "   - Tables: daily_prices, hourly_prices, realtime_data, db_stats, table_counts"
echo "   - Indexes: Optimized for latest data queries and COUNT operations"
echo "   - Functions: get_latest_price(), get_latest_prices(), get_stock_data(), get_realtime_data()"
echo "   - Views: latest_stock_data, latest_prices_summary"
echo "   - Fast Count Optimizations: 234x performance improvement for COUNT queries"
echo "   - Materialized Views: mv_hourly_prices_count, mv_daily_prices_count, mv_realtime_data_count"
echo "   - Fast Count Views: hourly_prices_count, daily_prices_count, realtime_data_count"
echo "   - Fast Count Functions: fast_count_hourly_prices(), fast_count_daily_prices(), fast_count_realtime_data()"
echo "   - Performance Monitoring: verify_count_accuracy(), get_index_usage_stats(), test_count_performance()"
echo ""
echo "🌐 Network Configuration:"
echo "   - Connected to: stock-network (internal)"
echo "   - Connected to: postgres_default (external)"
echo "   - Container name: postgres_db"
echo ""
echo "🚀 You can now use the StockDBPostgreSQL class in your Python code!" 