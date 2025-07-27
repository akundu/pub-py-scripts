#!/bin/bash

# Stock Data Services Docker Startup Script
# This script starts all the required services using Docker Compose

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Stock Data Services...${NC}"

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found.${NC}"
    echo -e "${YELLOW}Please create a .env file with your API keys:${NC}"
    echo -e "${YELLOW}cp env.example .env${NC}"
    echo -e "${YELLOW}Then edit .env with your actual API keys.${NC}"
    echo ""
    echo -e "${RED}Continuing without API keys may cause services to fail.${NC}"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running.${NC}"
    echo "Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: docker-compose is not installed.${NC}"
    echo "Please install docker-compose and try again."
    exit 1
fi

# Create necessary directories
echo -e "${GREEN}Creating necessary directories...${NC}"
mkdir -p data/daily data/hourly data/streaming/raw data/lists

# Build the Docker image
echo -e "${GREEN}Building Docker image...${NC}"
docker-compose build

# Start all services
echo -e "${GREEN}Starting all services...${NC}"
docker-compose up -d

# Wait a moment for services to start
sleep 5

# Check service status
echo -e "${GREEN}Checking service status...${NC}"
docker-compose ps

echo ""
echo -e "${GREEN}Services started successfully!${NC}"
echo ""
echo -e "${YELLOW}Service URLs:${NC}"
echo -e "  SQLite DB Server: http://localhost:9001"
echo -e "  DuckDB DB Server: http://localhost:9002"
echo -e "  Streaming DB Server: http://localhost:9000"
echo ""
echo -e "${YELLOW}To view logs:${NC}"
echo -e "  docker-compose logs -f"
echo ""
echo -e "${YELLOW}To stop services:${NC}"
echo -e "  docker-compose down"
echo ""
echo -e "${YELLOW}To restart services:${NC}"
echo -e "  docker-compose restart" 