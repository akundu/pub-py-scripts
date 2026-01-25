#!/bin/bash
set -e

echo "Starting entrypoint script..."

# Install system dependencies if not already installed
if [ ! -f /root/.deps-installed ]; then
    echo "Installing system dependencies..."
    apt-get update && apt-get install -y \
        libsndfile1 \
        portaudio19-dev \
        && rm -rf /var/lib/apt/lists/*
    
    echo "Installing Python dependencies..."
    if [ -f /app/requirements.txt ]; then
        pip install --no-cache-dir -r /app/requirements.txt
        echo "Python dependencies installed successfully"
    else
        echo "ERROR: requirements.txt not found at /app/requirements.txt"
        exit 1
    fi
    
    touch /root/.deps-installed
    echo "Dependencies installation complete"
else
    echo "Dependencies already installed, skipping..."
fi

# Verify numpy is installed
python -c "import numpy" 2>/dev/null || {
    echo "ERROR: numpy not installed, installing now..."
    pip install --no-cache-dir -r /app/requirements.txt
}

# Run the web server (use environment variable for log level, default to INFO)
echo "Starting web server..."
exec python web_server.py --port 9103 --workers 2 --log-level ${CHORD_DETECTOR_LOG_LEVEL:-INFO}
