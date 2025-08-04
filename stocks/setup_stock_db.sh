#!/bin/bash

# Stock Database Setup Script
# This script connects to an existing PostgreSQL instance and sets up the stock database

# Default values - can be overridden by environment variables
POSTGRES_ROOT_USER=${POSTGRES_ROOT_USER}
POSTGRES_ROOT_PASSWORD=${POSTGRES_ROOT_PASSWORD}
POSTGRES_HOST=${POSTGRES_HOST:-localhost}
POSTGRES_PORT=${POSTGRES_PORT:-5432}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Stock Database Setup Script${NC}"
echo "================================"
echo "PostgreSQL Host: $POSTGRES_HOST"
echo "PostgreSQL Port: $POSTGRES_PORT"
echo "PostgreSQL User: $POSTGRES_ROOT_USER"
echo ""

# Function to check if PostgreSQL is ready
check_postgres_ready() {
    echo -e "${YELLOW}Waiting for PostgreSQL to be ready...${NC}"
    until pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_ROOT_USER"; do
        echo "PostgreSQL is not ready yet. Waiting..."
        sleep 2
    done
    echo -e "${GREEN}PostgreSQL is ready!${NC}"
}

# Function to execute SQL command
execute_sql() {
    local sql="$1"
    local db="${2:-postgres}"
    echo "Executing: $sql"
    PGPASSWORD="$POSTGRES_ROOT_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_ROOT_USER" -d "$db" -c "$sql"
}

# Function to execute SQL file
execute_sql_file() {
    local file="$1"
    local db="$2"
    echo "Executing SQL file: $file"
    PGPASSWORD="$POSTGRES_ROOT_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_ROOT_USER" -d "$db" -f "$file"
}

# Main setup process
main() {
    # Check if PostgreSQL is ready
    check_postgres_ready
    
    echo -e "${YELLOW}Setting up stock database...${NC}"
    
    # Create database
    echo "Creating stock_data database..."
    execute_sql "CREATE DATABASE stock_data;"
    
    # Create user
    echo "Creating stock_user..."
    execute_sql "CREATE USER stock_user WITH PASSWORD 'stock_password';"
    
    # Grant privileges on database
    echo "Granting privileges on stock_data database..."
    execute_sql "GRANT ALL PRIVILEGES ON DATABASE stock_data TO stock_user;" "stock_data"
    
    # Grant privileges on schema
    echo "Granting privileges on public schema..."
    execute_sql "GRANT ALL PRIVILEGES ON SCHEMA public TO stock_user;" "stock_data"
    
    # Execute initialization script
    echo "Running database initialization script..."
    if [ -f "init_db.sql" ]; then
        execute_sql_file "init_db.sql" "stock_data"
    else
        echo -e "${RED}Error: init_db.sql file not found!${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Database setup completed successfully!${NC}"
    echo ""
    echo "Connection details for your stock system:"
    echo "  Host: $POSTGRES_HOST"
    echo "  Port: $POSTGRES_PORT"
    echo "  Database: stock_data"
    echo "  Username: stock_user"
    echo "  Password: stock_password"
}

# Check if required tools are available
check_requirements() {
    if ! command -v psql &> /dev/null; then
        echo -e "${RED}Error: psql command not found. Please install PostgreSQL client tools.${NC}"
        exit 1
    fi
    
    if ! command -v pg_isready &> /dev/null; then
        echo -e "${RED}Error: pg_isready command not found. Please install PostgreSQL client tools.${NC}"
        exit 1
    fi
}

# Run setup
check_requirements
main 
