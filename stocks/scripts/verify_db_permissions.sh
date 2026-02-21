#!/bin/bash

# Database Permissions Verification Script
# This script verifies that user owns the database and has all necessary permissions

set -e

echo "🔍 Verifying Database Ownership and Permissions"
echo "================================================"

# Check if Docker Compose is running
if ! docker-compose ps | grep -q postgres; then
    echo "❌ PostgreSQL container is not running."
    echo "   Please start it with: docker-compose up -d postgres"
    exit 1
fi

echo "✅ PostgreSQL container is running"

# Wait for PostgreSQL to be ready
echo ""
echo "⏳ Waiting for PostgreSQL to be ready..."
until docker-compose exec -T postgres pg_isready -U user -d stock_data; do
    echo "   Waiting for PostgreSQL..."
    sleep 2
done

echo "✅ PostgreSQL is ready!"

# Run the verification script
echo ""
echo "🔍 Running database ownership and permissions verification..."
docker-compose exec -T postgres psql -U user -d stock_data -f /docker-entrypoint-initdb.d/verify_db_permissions.sql

echo ""
echo "✅ Verification complete!"
echo ""
echo "📋 To run verification manually:"
echo "   docker-compose exec postgres psql -U user -d stock_data -f /docker-entrypoint-initdb.d/verify_db_permissions.sql"
echo ""
echo "📋 To connect to database manually:"
echo "   docker-compose exec postgres psql -U user -d stock_data" 
