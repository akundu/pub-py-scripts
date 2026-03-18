# Authentication & Security

The platform supports two authentication methods. Both can be used interchangeably on any protected endpoint.

## API Key Authentication

The simplest method. Pass the shared secret via the `X-API-Key` header:

```bash
curl -H "X-API-Key: change-me" http://localhost:8000/market/quote/SPY
```

- API key is configured via the `API_KEY_SECRET` environment variable
- A valid API key grants **all scopes** (full access)
- Best for internal services, scripts, and development

**Implementation:** `app/auth.py` -- `verify_api_key()` dependency

## OAuth2 / JWT Authentication

For production use with granular access control.

### Step 1: Obtain a Token

```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "username": "trader",
    "password": "secret",
    "scopes": ["trades:write", "market:read"]
  }'
```

Response:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "scopes": ["trades:write", "market:read"]
}
```

### Step 2: Use the Token

```bash
curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
  http://localhost:8000/account/positions
```

### Token Details

| Property | Value |
|----------|-------|
| Algorithm | HS256 |
| Expiry | 60 minutes (configurable via `JWT_EXPIRE_MINUTES`) |
| Signing key | `JWT_SECRET_KEY` environment variable |
| Payload fields | `sub` (username), `scopes` (list), `exp` (expiry) |

## Available Scopes

| Scope | Grants Access To |
|-------|-----------------|
| `trades:read` | Read order history (reserved for future use) |
| `trades:write` | Submit and cancel orders (`POST /trade/execute`) |
| `market:read` | Read quotes and market data (`GET /market/quote/{symbol}`) |
| `account:read` | Read positions and balances (`GET /account/positions`) |

If no scopes are requested when obtaining a token, all scopes are granted by default.

## Auth Resolution Logic

The `require_auth()` dependency checks in order:

1. **API Key** -- if `X-API-Key` header is present and matches `API_KEY_SECRET`, grant full access
2. **JWT Token** -- if `Authorization: Bearer <token>` header is present, decode and validate:
   - Token signature and expiry are verified
   - Required scopes (per-endpoint) are checked against the token's scope list
3. **Neither** -- return `401 Unauthorized`

```
Request arrives
  │
  ├─ X-API-Key header present?
  │   ├─ Yes + valid ──> Authenticated (all scopes)
  │   └─ No ──> Continue
  │
  ├─ Authorization: Bearer header present?
  │   ├─ Yes + valid JWT ──> Check scopes
  │   │   ├─ Has required scopes ──> Authenticated
  │   │   └─ Missing scope ──> 403 Forbidden
  │   └─ Invalid/expired JWT ──> 401 Unauthorized
  │
  └─ Neither present ──> 401 Unauthorized
```

## LAN Trust Authentication

When `TRUST_LOCAL_NETWORK=true` (default), requests from private IPs bypass all authentication:

```
Request arrives
  │
  ├─ Client IP in private network? (127.*, 10.*, 172.16-31.*, 192.168.*)
  │   └─ Yes ──> Authenticated as "lan-user" (all scopes)
  │
  ├─ X-API-Key header present?
  │   └─ (existing flow)
  │
  └─ Authorization: Bearer header?
      └─ (existing flow)
```

**Private networks checked:**
- `127.0.0.0/8` (localhost)
- `10.0.0.0/8` (Class A private)
- `172.16.0.0/12` (Class B private)
- `192.168.0.0/16` (Class C private)
- `::1/128` (IPv6 localhost)

This means any machine on your LAN can call the API without credentials:

```bash
curl http://192.168.1.50:8000/dashboard/summary   # No auth needed from LAN
```

To disable (e.g., for internet-exposed servers):

```bash
TRUST_LOCAL_NETWORK=false
```

## Security Best Practices

### Credential Management

All sensitive values are loaded from environment variables via `pydantic-settings`:

```bash
# .env file (never commit to source control)
API_KEY_SECRET=a-long-random-secret
JWT_SECRET_KEY=another-long-random-secret
ROBINHOOD_TOTP_SEED=base32-encoded-totp-seed
ETRADE_CONSUMER_SECRET=etrade-secret
```

- The `.env.example` file provides a template with empty values
- The `.env` file should be in `.gitignore`
- No credentials are logged or included in API responses

### Production Recommendations

1. **Rotate secrets regularly** -- change `API_KEY_SECRET` and `JWT_SECRET_KEY` periodically
2. **Use short-lived tokens** -- reduce `JWT_EXPIRE_MINUTES` to 15-30 for production
3. **Restrict scopes** -- issue tokens with only the scopes the client needs
4. **Use HTTPS** -- deploy behind a TLS-terminating reverse proxy (nginx, Caddy)
5. **Rate limit** -- add rate limiting middleware for the `/auth/token` endpoint
6. **Audit logging** -- log authentication events (successes and failures)

### Unprotected Endpoints

The following endpoints require no authentication:

| Endpoint | Reason |
|----------|--------|
| `GET /health` | Infrastructure monitoring / load balancer health checks (always unauthenticated) |
| `POST /auth/token` | Token issuance (requires username/password in body) |
| `WS /ws/orders` | WebSocket connection (consider adding token-based auth for production) |

All other endpoints skip auth for LAN clients when `TRUST_LOCAL_NETWORK=true`.
