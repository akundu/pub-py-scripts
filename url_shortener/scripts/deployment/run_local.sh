#!/bin/bash
# Script to run URL shortener locally (without Docker)

set -e

echo "=========================================="
echo "URL Shortener - Local Development Setup"
echo "=========================================="
echo ""

# Navigate to project root
cd "$(dirname "$0")/../.." || exit 1

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✅ Python $PYTHON_VERSION found"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install/update dependencies
echo ""
echo "📥 Installing/verifying dependencies..."
pip install -q --upgrade pip

# Check if fastapi is installed (key dependency)
if python -c "import fastapi" 2>/dev/null; then
    echo "✅ Dependencies already installed"
else
    echo "📦 Installing dependencies (this may take a moment)..."
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
fi

# Check for .env file
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo ""
        echo "📝 Creating .env from .env.example..."
        cp .env.example .env
        echo "✅ .env file created"
    else
        echo ""
        echo "⚠️  No .env file found. Using defaults."
    fi
fi

# Check QuestDB - check HTTP console (port 9000) or minimal HTTP (port 9003)
echo ""
echo "🔍 Checking QuestDB..."
QUESTDB_RUNNING=false
QUESTDB_METHOD=""

# Try HTTP console first (port 9000) with 2 second timeout
if curl -sf --connect-timeout 2 --max-time 3 http://localhost:9000/ > /dev/null 2>&1; then
    echo "✅ QuestDB HTTP console is running on port 9000"
    QUESTDB_RUNNING=true
    QUESTDB_METHOD="http_console"
# Try minimal HTTP server (port 9003) with fast timeout
elif timeout 1 bash -c '</dev/tcp/localhost/9003' 2>/dev/null; then
    echo "✅ QuestDB minimal HTTP is running on port 9003"
    QUESTDB_RUNNING=true
    QUESTDB_METHOD="minimal_http"
fi

if [ "$QUESTDB_RUNNING" = false ]; then
    echo "❌ QuestDB is not running!"
    echo ""
    echo "Please start QuestDB first:"
    echo "  Option 1 (Docker): docker run -d -p 9000:9000 -p 9003:9003 -p 9009:9009 questdb/questdb:7.3.10"
    echo "  Option 2 (Binary):  questdb start"
    echo "  Option 3 (Homebrew): brew install questdb && questdb start"
    echo "  Option 4 (Full Stack): cd questdb_docker && docker-compose up -d"
    echo ""
    echo "QuestDB Console will be at: http://localhost:9000"
    exit 1
fi

# Check PgBouncer (if running full stack with connection pooling)
echo ""
echo "🔍 Checking PgBouncer (connection pooler)..."
if timeout 1 bash -c '</dev/tcp/localhost/8812' 2>/dev/null; then
    echo "✅ PgBouncer is running on port 8812 (connection pooling enabled)"
    PGBOUNCER_RUNNING=true
    # Use PgBouncer port in connection string
    export QUESTDB_URL="questdb://admin:quest@localhost:8812/qdb"
else
    echo "⚠️  PgBouncer not detected - using direct QuestDB connection"
    PGBOUNCER_RUNNING=false
    # Use direct connection to QuestDB's PostgreSQL wire port (default 8812)
    # Note: If you started QuestDB without -p 8812:8812, connection will fail
    export QUESTDB_URL="questdb://admin:quest@localhost:8812/qdb"
fi

# Check Redis (optional but recommended for production)
echo ""
echo "🔍 Checking Redis cache..."
REDIS_RUNNING=false

# Try redis-cli first with timeout
if command -v redis-cli &> /dev/null && timeout 1 redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is running on port 6379 (verified with redis-cli)"
    REDIS_RUNNING=true
    REDIS_STATUS="enabled"
    export REDIS_URL="redis://localhost:6379/0"
# Try TCP connection as fallback with fast timeout
elif timeout 1 bash -c '</dev/tcp/localhost/6379' 2>/dev/null; then
    echo "✅ Redis is running on port 6379 (TCP connection verified)"
    REDIS_RUNNING=true
    REDIS_STATUS="enabled"
    export REDIS_URL="redis://localhost:6379/0"
else
    echo "⚠️  Redis is not running (optional - caching disabled)"
    echo "   To enable caching:"
    echo "     Docker:   docker run -d -p 6379:6379 redis:7-alpine"
    echo "     Homebrew: brew services start redis"
    echo "     Full Stack: cd questdb_docker && docker-compose up -d redis"
    REDIS_RUNNING=false
    REDIS_STATUS="disabled"
    # Unset REDIS_URL so the app doesn't try to connect
    unset REDIS_URL
fi

# Set environment for table creation
export QUESTDB_CREATE_TABLES=1

# Initialize tables if needed
echo ""
echo "🗄️  Checking database tables..."
python scripts/database/init_questdb_database.py 2>&1 | grep -q "successfully" && \
    echo "✅ Database tables verified" || \
    echo "⚠️  Table initialization check completed"

echo ""
echo "=========================================="
echo "🚀 Starting URL Shortener Service"
echo "=========================================="
echo ""
echo "Configuration:"
if [ "$PGBOUNCER_RUNNING" = true ]; then
    echo "  - Database:  QuestDB via PgBouncer (localhost:8812)"
else
    echo "  - Database:  QuestDB direct connection (localhost:8812)"
fi
echo "  - Cache:     Redis $REDIS_STATUS"
echo "  - Server:    http://localhost:9200"
echo ""
echo "Direct Access (No Proxy):"
echo "  - Web Interface:  http://localhost:9200/"
echo "  - API Docs:       http://localhost:9200/api/docs"
echo "  - Health Check:   http://localhost:9200/health"
echo "  - Redirect Test:  http://localhost:9200/{short_code}"
echo ""
echo "Via Envoy Proxy (if running):"
echo "  - Web Interface:  http://localhost:9200/s/"
echo "  - API:            http://localhost:9200/s/api/*"
echo "  - Health Check:   http://localhost:9200/s/health"
echo "  - Redirect Test:  http://localhost:9200/s/{short_code}"
echo "  - Envoy Admin:    http://localhost:9901"
echo ""
echo "Management:"
echo "  - QuestDB Console: http://localhost:9000"
if [ "$REDIS_RUNNING" = true ]; then
    echo "  - Redis:           localhost:6379 (use redis-cli)"
fi
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Start the server
python app.py


