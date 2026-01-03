# URL Shortener Service

A production-ready URL shortening service built with FastAPI, QuestDB, and Redis.

## Features

- ✅ **REST API** - Full-featured API for programmatic access
- 🌐 **Web Interface** - Beautiful, responsive web UI
- 🚀 **High Performance** - QuestDB for time-series data storage
- 💾 **Redis Caching** - Optional caching layer for improved performance
- 🎯 **Custom Short Codes** - Allow users to choose their own codes
- 📊 **Analytics** - Track access counts and timestamps
- 🔒 **Production Ready** - Docker, Envoy proxy, health checks
- 🧪 **Tested** - Comprehensive test suite
- 🖥️ **CLI Tool** - Command-line interface for all operations

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f url-shortener

# Stop services
docker-compose down
```

The service will be available at:
- Web Interface: http://localhost:8888/s/
- API Docs: http://localhost:8080/api/docs
- Health Check: http://localhost:8888/s/health

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize database tables
export QUESTDB_CREATE_TABLES=1

# Run the service
python app.py
```

## Architecture

```
┌─────────────┐
│   Envoy     │  Port 8888 (External)
│   Proxy     │
└──────┬──────┘
       │
       ├─> /s/*  → URL Shortener (Port 8080)
       ├─> /stock_info/* → Stock Service
       └─> /static/* → Static Files
       
┌──────────────────┐
│  URL Shortener   │
│   (FastAPI)      │
└────┬────────┬────┘
     │        │
     │        └─> Redis (Cache)
     └─> QuestDB (Database)
```

### Directory Structure

```
url_shortener/
├── lib/                      # Backend library code
│   ├── database/            # Database layer (QuestDB)
│   │   ├── base.py         # Abstract base class
│   │   ├── questdb.py      # QuestDB implementation
│   │   ├── cache.py        # Redis cache
│   │   └── models.py       # Data models
│   ├── common/             # Common utilities
│   │   ├── validators.py   # URL/code validation
│   │   ├── headers.py      # Header parsing
│   │   ├── url_builder.py  # URL construction
│   │   └── logging_config.py
│   ├── service.py          # Business logic
│   └── shortcode.py        # Code generation
│
├── web_app/                # Web application (FastAPI)
│   ├── api/               # REST API
│   │   ├── routes.py      # API endpoints
│   │   └── schemas.py     # Pydantic models
│   ├── web/               # Web interface
│   │   └── routes.py      # Web endpoints
│   └── middleware/        # Middleware
│
├── ux/web/                # Frontend (CDN-deployable)
│   ├── index.html        # Homepage
│   ├── result.html       # Results page
│   ├── error.html        # Error page
│   ├── css/styles.css    # Styles
│   └── js/app.js         # Client-side JS
│
├── scripts/              # Utility scripts
│   ├── cli/             # Command-line interface
│   ├── database/        # DB utilities
│   └── deployment/      # Deployment scripts
│
├── tests/               # Test suite
├── config.py           # Configuration
├── app.py              # Main entry point
└── docker-compose.yml  # Docker orchestration
```

## API Documentation

### REST API Endpoints

#### Create Short URL
```bash
POST /api/shorten
Content-Type: application/json

{
  "url": "https://example.com/very/long/url",
  "custom_code": "mylink"  // optional
}

Response:
{
  "short_code": "mylink",
  "short_url": "http://localhost:8888/s/mylink",
  "original_url": "https://example.com/very/long/url",
  "created_at": "2024-01-01T12:00:00Z"
}
```

#### Get URL Information
```bash
GET /api/urls/{short_code}

Response:
{
  "short_code": "mylink",
  "original_url": "https://example.com/very/long/url",
  "created_at": "2024-01-01T12:00:00Z",
  "access_count": 42,
  "last_accessed": "2024-01-02T10:30:00Z"
}
```

#### Redirect to Original URL
```bash
GET /{short_code}

Response: 302 Redirect to original URL
```

#### Health Check
```bash
GET /health

Response:
{
  "status": "healthy",
  "database": "healthy",
  "cache": "healthy",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Get Statistics
```bash
GET /api/stats

Response:
{
  "total_urls": 1000,
  "total_accesses": 50000,
  "database": "questdb",
  "cache_enabled": true,
  "custom_codes_enabled": true
}
```

## CLI Usage

```bash
# Shorten a URL
python scripts/cli/url_shortener_cli.py shorten https://example.com/long/url

# Shorten with custom code
python scripts/cli/url_shortener_cli.py shorten https://example.com/long/url --custom-code mylink

# Get original URL
python scripts/cli/url_shortener_cli.py get mylink

# Get statistics for a short code
python scripts/cli/url_shortener_cli.py stats mylink

# List recent URLs
python scripts/cli/url_shortener_cli.py list --limit 10

# Check service health
python scripts/cli/url_shortener_cli.py health

# Use custom database URL
python scripts/cli/url_shortener_cli.py --db-url questdb://admin:quest@localhost:8812/qdb health
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QUESTDB_URL` | `questdb://admin:quest@localhost:8812/qdb` | QuestDB connection URL |
| `QUESTDB_CREATE_TABLES` | `0` | Set to `1` to auto-create tables |
| `REDIS_URL` | None | Redis URL (optional) |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8080` | Server port |
| `BASE_URL` | `http://localhost:8080` | Base URL for short links |
| `PATH_PREFIX` | `` | Path prefix (e.g., `/s`) |
| `SHORT_CODE_LENGTH` | `6` | Default code length |
| `ENABLE_CUSTOM_CODES` | `true` | Allow custom codes |
| `LOG_LEVEL` | `INFO` | Logging level |

### Database Setup

The service automatically creates tables when `QUESTDB_CREATE_TABLES=1`.

To manually initialize tables:
```bash
python scripts/database/init_questdb_database.py
```

To seed test data:
```bash
python scripts/database/seed_data.py --count 100
```

## Testing

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=lib --cov=web_app

# Run specific test file
pytest tests/test_service.py

# Run with verbose output
pytest -v
```

## Deployment

### Docker

```bash
# Build image
docker build -t url-shortener:latest .

# Run container
docker run -d \
  -p 8080:8080 \
  -e QUESTDB_URL=questdb://admin:quest@questdb:8812/qdb \
  -e QUESTDB_CREATE_TABLES=1 \
  url-shortener:latest
```

### Envoy Proxy

The service is configured to work behind Envoy proxy. Routes are defined in `http-proxy/config/envoy.yaml`:

- `/s/*` → URL Shortener service
- `/s/api/*` → REST API
- `/s/health` → Health check

Envoy handles:
- Header forwarding (`X-Forwarded-*`)
- Health checks
- Load balancing
- Timeout management

## Security Considerations

1. **Input Validation**: All URLs and short codes are validated
2. **Reserved Words**: Common paths (api, admin, etc.) are blocked
3. **Rate Limiting**: Should be configured in Envoy/API Gateway
4. **Non-root User**: Docker runs as non-root user
5. **HTTPS**: Use HTTPS in production (configure in Envoy)

## Performance

- **QuestDB**: Optimized for time-series data
- **Redis Caching**: Reduces database load
- **Connection Pooling**: Efficient database connections
- **Async/Await**: Non-blocking I/O throughout
- **Base62 Encoding**: Compact short codes

## Monitoring

### Health Checks

```bash
# Application health
curl http://localhost:8080/health

# Envoy admin
curl http://localhost:9901/stats

# QuestDB metrics
curl http://localhost:8812/metrics
```

### Logs

```bash
# Docker logs
docker-compose logs -f url-shortener

# Application logs (if LOG_FILE is set)
tail -f /path/to/logfile.log
```

## Troubleshooting

### Database Connection Issues

```bash
# Check QuestDB is running
docker-compose ps questdb

# Check QuestDB logs
docker-compose logs questdb

# Verify connection
psql -h localhost -p 9000 -U admin -d qdb
```

### Table Creation

If tables aren't being created:
```bash
# Set environment variable
export QUESTDB_CREATE_TABLES=1

# Or manually create tables
python scripts/database/init_questdb_database.py
```

### Redis Connection

```bash
# Check Redis is running
docker-compose ps redis

# Test Redis connection
redis-cli -h localhost -p 6379 ping
```

## Development

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy lib/ web_app/
```

### Adding New Features

1. Update database schema in `init_db.sql`
2. Add repository methods in `lib/database/questdb.py`
3. Add service methods in `lib/service.py`
4. Add API endpoints in `web_app/api/routes.py`
5. Add tests in `tests/`
6. Update documentation

## License

MIT License - see LICENSE file for details

## Support

For issues and questions, please open an issue on GitHub.

## Credits

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [QuestDB](https://questdb.io/) - High-performance time-series database
- [Redis](https://redis.io/) - In-memory cache
- [Envoy](https://www.envoyproxy.io/) - Cloud-native proxy



