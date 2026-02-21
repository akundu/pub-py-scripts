#!/bin/bash
# Health check script for URL shortener service

set -e

# Configuration
HOST="${HEALTH_CHECK_HOST:-localhost}"
PORT="${HEALTH_CHECK_PORT:-9200}"
ENDPOINT="${HEALTH_CHECK_ENDPOINT:-/health}"
TIMEOUT="${HEALTH_CHECK_TIMEOUT:-5}"

# Make health check request
response=$(curl -s -f -m "$TIMEOUT" "http://${HOST}:${PORT}${ENDPOINT}" || echo "FAILED")

if [ "$response" = "FAILED" ]; then
    echo "Health check failed: Unable to connect to service"
    exit 1
fi

# Check if response contains "healthy" status
if echo "$response" | grep -q "healthy"; then
    echo "Health check passed"
    exit 0
else
    echo "Health check failed: Service unhealthy"
    echo "Response: $response"
    exit 1
fi





