FROM python:3.11-alpine

# Install minimal system dependencies for compilation
RUN apk add --no-cache build-base gcc g++ musl-dev linux-headers

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (install critical packages first without DuckDB)
RUN pip install --no-cache-dir --only-binary=all --prefer-binary \
    psycopg2-binary asyncpg pandas numpy

# Try to install DuckDB as binary only, skip if not available
RUN pip install --no-cache-dir --only-binary=duckdb duckdb || \
    echo "Warning: DuckDB binary wheel not available for this platform, skipping..."

# Install remaining requirements
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt || \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p data/daily data/hourly data/streaming/raw data/lists && \
    chmod -R 755 data/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create a non-root user
RUN adduser -D -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Default command (can be overridden)
CMD ["python", "db_server.py"] 