#!/bin/bash
# Quick start script for URL shortener

set -e

echo "=========================================="
echo "URL Shortener - Quick Start"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"
echo ""

# Navigate to the project root (url_shortener directory)
cd "$(dirname "$0")/../.." || exit 1

# Check if .env exists, if not create from example (if example exists)
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "📝 Creating .env from .env.example..."
        cp .env.example .env
        echo "✅ .env file created. Please review and update if needed."
    else
        echo "ℹ️  No .env.example found. Using default configuration from docker-compose.yml"
    fi
else
    echo "✅ .env file already exists"
fi

echo ""
echo "🚀 Starting services with Docker Compose..."
echo ""

# Start services
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check health
echo ""
echo "🔍 Checking service health..."

max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -sf http://localhost:9200/health > /dev/null 2>&1; then
        echo "✅ Service is healthy!"
        break
    fi
    
    attempt=$((attempt + 1))
    echo "   Waiting... (attempt $attempt/$max_attempts)"
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ Service did not become healthy in time"
    echo "   Check logs with: docker-compose logs url-shortener"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ URL Shortener is ready!"
echo "=========================================="
echo ""
echo "🌐 Access points:"
echo "   - Web Interface:  http://localhost:9200/s/"
echo "   - API Docs:       http://localhost:9200/api/docs"
echo "   - Health Check:   http://localhost:9200/s/health"
echo ""
echo "📊 Service status:"
echo "   - QuestDB:        http://localhost:8812"
echo "   - Redis:          localhost:6379"
echo "   - Envoy Admin:    http://localhost:9901"
echo ""
echo "💡 Quick test:"
echo "   curl -X POST http://localhost:9200/api/shorten \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"url\": \"https://github.com\"}'"
echo ""
echo "🛑 To stop services:"
echo "   docker-compose down"
echo ""
echo "📝 View logs:"
echo "   docker-compose logs -f url-shortener"
echo ""


