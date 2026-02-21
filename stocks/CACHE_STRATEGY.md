# Prediction Cache Strategy

## Overview

The prediction cache uses an **explicit refresh** strategy designed for cron-based cache warming. This prevents expensive prediction recalculations during market hours while allowing controlled updates via background jobs.

## How It Works

### 1. Cache Behavior

- **Always serves from cache** if available (no auto-expiration)
- **Never expires** based on time during market hours
- Only refreshes when explicitly requested via `?cache=false` parameter

### 2. Client-Side Auto-Refresh (Optional)

The web UI has an optional auto-refresh checkbox (default 30 seconds):
- When enabled, calls the API every 30 seconds
- **Gets cached data** from the server (fast, no recalculation)
- Does NOT trigger fresh predictions
- Useful for seeing updated price data if cache was recently refreshed

### 3. Cron-Based Cache Warming

To populate the cache with fresh predictions, use a cron job:

```bash
# Cron example: Refresh NDX predictions every 5 minutes during market hours
*/5 9-16 * * 1-5 curl -s "http://localhost:8081/predictions/api/lazy/today/NDX?cache=false" > /dev/null 2>&1

# Refresh both NDX and SPX
*/5 9-16 * * 1-5 curl -s "http://localhost:8081/predictions/api/lazy/today/NDX?cache=false" > /dev/null && \
                 curl -s "http://localhost:8081/predictions/api/lazy/today/SPX?cache=false" > /dev/null
```

**Cron Schedule Breakdown:**
- `*/5` - Every 5 minutes
- `9-16` - Hours 9 AM to 4 PM (server local time)
- `* * 1-5` - Monday through Friday

### 4. Manual Refresh

Users can also force refresh via URL parameter:

```bash
# Force refresh (bypasses cache)
curl "http://localhost:8081/predictions/api/lazy/today/NDX?cache=false"

# Use cache (default)
curl "http://localhost:8081/predictions/api/lazy/today/NDX"
```

## Architecture Benefits

### Before (Time-Based Expiration)
- ❌ Cache expired every 300 seconds during market hours
- ❌ Multiple users could trigger expensive recalculations
- ❌ Unpredictable load spikes
- ❌ No control over when predictions run

### After (Explicit Refresh)
- ✅ Cache serves indefinitely (millisecond response times)
- ✅ Only cron job triggers recalculations (controlled)
- ✅ Predictable load (every 5 minutes on schedule)
- ✅ All users get same cached prediction (consistency)
- ✅ Cron can run during low-activity periods

## Performance Impact

### Cache Hit (Normal Operation)
```
Request → Cache → Serialized Response
~5-20ms total
```

### Cache Miss / Force Refresh (Cron Job)
```
Request → predict_close_now.py → Model Training → Prediction → Cache → Response
~3-6 seconds total
```

### Comparison
- **Cached:** 5-20ms (300x faster)
- **Fresh:** 3-6 seconds
- **With 100 requests/5min:** Cache prevents 99 expensive calculations

## Deployment Strategy

### 1. Start Server
```bash
python db_server.py \
  --db-file "questdb://user:pass@host:port/db" \
  --port 8081 \
  --log-level INFO \
  --workers 2
```

### 2. Add Cron Job
```bash
# Edit crontab
crontab -e

# Add prediction refresh jobs (adjust times for your timezone)
# Refresh every 5 minutes during market hours (9:30 AM - 4:00 PM ET)
*/5 9-16 * * 1-5 curl -s "http://localhost:8081/predictions/api/lazy/today/NDX?cache=false" >/dev/null 2>&1
*/5 9-16 * * 1-5 curl -s "http://localhost:8081/predictions/api/lazy/today/SPX?cache=false" >/dev/null 2>&1
```

### 3. Optional: Pre-warm Cache on Startup

Add a startup script to populate cache immediately:

```bash
#!/bin/bash
# warm_prediction_cache.sh

# Wait for server to be ready
sleep 5

# Warm cache for both tickers
curl -s "http://localhost:8081/predictions/api/lazy/today/NDX?cache=false" >/dev/null
curl -s "http://localhost:8081/predictions/api/lazy/today/SPX?cache=false" >/dev/null

echo "Prediction cache warmed"
```

Run after server starts:
```bash
python db_server.py ... &
./warm_prediction_cache.sh &
```

## API Parameters

### Query Parameters

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `cache` | `true`, `false` | `true` | When `false`, bypasses cache and generates fresh prediction |
| `refresh_interval` | 5-300 | 30 | Client-side auto-refresh interval (seconds) |
| `date` | YYYY-MM-DD | today | Specific date for future predictions |
| `days_ahead` | 3,5,10,20 | 0 | Days ahead for future prediction |

### Examples

```bash
# Normal request (uses cache)
GET /predictions/api/lazy/today/NDX

# Force fresh prediction (for cron)
GET /predictions/api/lazy/today/NDX?cache=false

# Custom auto-refresh interval (client-side only)
GET /predictions/NDX?refresh_interval=60

# Future prediction with cache bypass
GET /predictions/api/lazy/future/NDX/5?cache=false
```

## Monitoring Cache Performance

### Log Analysis

Check cache hit rate in server logs:

```bash
# Cache hits (fast responses)
grep "predictions/api/lazy/today" /tmp/db_server_8081.log | grep -E "[0-9]{1,3}ms"

# Cache misses (slow responses - should only be cron)
grep "predictions/api/lazy/today" /tmp/db_server_8081.log | grep -E "[0-9]{4,}ms"
```

### Expected Patterns

**During Market Hours:**
- User requests: 5-20ms (cache hits)
- Cron requests: 3-6 seconds (cache refresh)
- Cache refresh every 5 minutes
- All other requests served from cache

**Outside Market Hours:**
- All requests: 5-20ms (cache hits from last market close)
- No cron jobs running
- Cache preserved until next market open

## Troubleshooting

### Cache Not Updating

**Symptom:** Predictions show old data even after cron runs

**Check:**
```bash
# Verify cron is running
grep CRON /var/log/syslog | grep predictions

# Test manual refresh
curl -v "http://localhost:8081/predictions/api/lazy/today/NDX?cache=false"
# Should take 3-6 seconds and return fresh data
```

### Cron Jobs Not Running

**Check cron service:**
```bash
# Check if cron is running
systemctl status cron  # Linux
# or
sudo launchctl list | grep cron  # macOS

# Check cron logs
tail -f /var/log/syslog | grep CRON  # Linux
tail -f /var/log/system.log | grep cron  # macOS
```

### High Server Load

**Symptom:** Server CPU spiking

**Possible causes:**
- Cron interval too aggressive (try 5-10 minutes minimum)
- Multiple users manually refreshing with `?cache=false`
- WebSocket connections triggering unnecessary updates

**Solution:**
```bash
# Increase cron interval to 10 minutes
*/10 9-16 * * 1-5 curl ...

# Check for users bypassing cache
grep "cache=false" /tmp/db_server_8081.log | grep -v "127.0.0.1"
```

## Migration from Old Behavior

### Before
- Cache expired every 300 seconds
- Each request could trigger recalculation
- Unpredictable performance

### After
- Cache never expires
- Only cron triggers recalculation
- Predictable performance

**No code changes needed for existing API consumers** - they'll automatically get cached responses. Just add the cron job to keep cache fresh.

---

**Related Files:**
- `common/predictions.py` - PredictionCache implementation
- `db_server.py` - API endpoints and cache usage
- `scripts/predict_close_now.py` - Prediction engine
