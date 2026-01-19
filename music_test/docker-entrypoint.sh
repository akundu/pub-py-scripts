#!/bin/bash
set -e

# Install system dependencies if not already installed
if [ ! -f /app/.deps-installed ]; then
    echo "Installing system dependencies..."
    apt-get update && apt-get install -y \
        libsndfile1 \
        portaudio19-dev \
        && rm -rf /var/lib/apt/lists/*
    
    echo "Installing Python dependencies..."
    pip install --no-cache-dir -r /app/requirements.txt
    
    touch /app/.deps-installed
fi

# Run the web server (use environment variable for log level, default to INFO)
exec python web_server.py --port 9103 --workers 2 --log-level ${CHORD_DETECTOR_LOG_LEVEL:-INFO}
