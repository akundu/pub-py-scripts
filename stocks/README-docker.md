# Stock Data Services - Docker Setup

This Docker setup provides a complete environment for running all stock data services with proper isolation and configuration management.

## Quick Start

1. **Set up environment variables:**
   ```bash
   cp env.example .env
   # Edit .env with your actual API keys
   ```

2. **Start all services:**
   ```bash
   ./scripts/start-docker-services.sh
   ```

## Services Overview

The Docker Compose setup includes the following services:

### Database Servers
- **SQLite DB Server** (Port 9001): `data/stock_data.db`
- **DuckDB Server** (Port 9002): `data/stock_data.duckdb`  
- **Streaming DB Server** (Port 9000): `data/streaming/streaming.duckdb`

### Market Data Service
- **Stream Market Data**: Real-time market data streaming from Polygon.io

## Environment Variables

Create a `.env` file with your API keys:

```bash
# Required API Keys
POLYGON_API_KEY=your_polygon_api_key_here
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_API_SECRET=your_alpaca_api_secret_here
```

## Manual Docker Commands

### Start all services
```bash
docker-compose up -d
```

### View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f db-server-sqlite
```

### Stop services
```bash
docker-compose down
```

### Restart services
```bash
docker-compose restart
```

### Rebuild and restart
```bash
docker-compose up -d --build
```

## Service Details

### Database Servers
Each database server runs with:
- `ulimit -n 65536` for high file descriptor limits
- Heartbeat monitoring every 60 seconds
- Appropriate log levels (INFO for main DBs, ERROR for streaming)

### Market Data Streaming
The streaming service:
- Connects to Polygon.io for real-time data
- Saves data to the streaming database (port 9000)
- Uses the stocks_to_track.yaml symbol list
- Implements retry logic with max 30 retries
- Limits to 5 symbols per connection

## Data Persistence

All data is persisted through Docker volumes:
- `./data:/app/data` - Main data directory
- `./data/lists:/app/data/lists` - Symbol lists

## Network Configuration

All services communicate through a dedicated `stock-network` bridge network.

## Troubleshooting

### Check service status
```bash
docker-compose ps
```

### View service logs
```bash
docker-compose logs [service-name]
```

### Access service directly
```bash
# Connect to a specific service
docker-compose exec db-server-sqlite bash
```

### Reset everything
```bash
docker-compose down -v
docker-compose up -d --build
```

## Security Notes

- API keys are passed via environment variables
- The `.env` file is excluded from Docker builds via `.dockerignore`
- Never commit your actual `.env` file to version control
- Use the provided `env.example` as a template

## Performance Considerations

- Each database server has increased file descriptor limits
- Services are configured with appropriate restart policies
- Data volumes ensure persistence across container restarts
- Network isolation prevents conflicts with other services

## Development

To modify the setup:
1. Edit `docker-compose.yml` for service configuration
2. Edit `Dockerfile` for base image changes
3. Update `requirements.txt` for Python dependencies
4. Rebuild with `docker-compose up -d --build` 