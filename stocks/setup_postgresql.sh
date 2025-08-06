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
until docker-compose exec -T postgres pg_isready -U stock_user -d stock_data; do
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
docker-compose exec -T postgres psql -U stock_user -d stock_data -c "
DO \$\$
BEGIN
    -- Verify database ownership
    IF EXISTS (
        SELECT 1 FROM pg_database 
        WHERE datname = 'stock_data' AND datdba = (SELECT oid FROM pg_roles WHERE rolname = 'stock_user')
    ) THEN
        RAISE NOTICE '✅ stock_data database is owned by stock_user';
    ELSE
        RAISE WARNING '⚠️ stock_data database is NOT owned by stock_user';
    END IF;
    
    -- Verify schema ownership
    IF EXISTS (
        SELECT 1 FROM information_schema.schemata 
        WHERE schema_name = 'public' AND schema_owner = 'stock_user'
    ) THEN
        RAISE NOTICE '✅ public schema is owned by stock_user';
    ELSE
        RAISE WARNING '⚠️ public schema is NOT owned by stock_user';
    END IF;
    
    -- Verify table ownership
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name IN ('daily_prices', 'hourly_prices', 'realtime_data', 'db_stats')
        AND table_owner = 'stock_user'
    ) THEN
        RAISE NOTICE '✅ All tables are owned by stock_user';
    ELSE
        RAISE WARNING '⚠️ Some tables are NOT owned by stock_user';
    END IF;
    
    -- Verify permissions
    IF EXISTS (
        SELECT 1 FROM information_schema.role_table_grants 
        WHERE grantee = 'stock_user' 
        AND table_schema = 'public'
        AND privilege_type = 'ALL'
    ) THEN
        RAISE NOTICE '✅ stock_user has ALL privileges on public schema tables';
    ELSE
        RAISE WARNING '⚠️ stock_user may not have ALL privileges on public schema tables';
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
        user='stock_user',
        password='stock_password'
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

echo ""
echo "🎉 PostgreSQL setup completed successfully!"
echo ""
echo "📋 Connection Details:"
echo "   Host: localhost"
echo "   Port: 5432"
echo "   Database: stock_data (owned by stock_user)"
echo "   Username: stock_user"
echo "   Password: stock_password"
echo ""
echo "🔧 Useful Commands:"
echo "   Start database: docker-compose up -d postgres"
echo "   Stop database:  docker-compose down"
echo "   View logs:      docker-compose logs postgres"
echo "   Connect via psql: psql -h localhost -p 5432 -U stock_user -d stock_data"
echo ""
echo "📊 Database Features:"
echo "   - Database owned by: stock_user"
echo "   - All tables owned by: stock_user"
echo "   - Full permissions granted to: stock_user"
echo "   - Tables: daily_prices, hourly_prices, realtime_data, db_stats"
echo "   - Indexes: Optimized for latest data queries"
echo "   - Functions: get_latest_price(), get_latest_prices(), get_stock_data(), get_realtime_data()"
echo "   - Views: latest_stock_data, latest_prices_summary"
echo ""
echo "🌐 Network Configuration:"
echo "   - Connected to: stock-network (internal)"
echo "   - Connected to: postgres_default (external)"
echo "   - Container name: postgres_db"
echo ""
echo "🚀 You can now use the StockDBPostgreSQL class in your Python code!" 