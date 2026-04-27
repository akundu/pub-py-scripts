# Gemini Proxy — Per-Client Topic-Gated Access

`db_server.py` exposes two endpoints that proxy Gemini calls on behalf of
shipped client apps. The `GEMINI_API_KEY` never leaves the server. Every
client app gets its own long random key and is locked to a specific topic;
off-topic questions are refused with the canonical text:

> `I can't talk about that.`

LAN callers presenting an admin header bypass both auth and topic gating.

## Endpoints

### `POST /api/gemini/ask`
Headers
```
Authorization: Bearer <client_key>
Content-Type: application/json
```
Body
```json
{ "prompt": "Explain drop-D tuning briefly.",
  "model": "gemini-flash-latest",     // optional; defaults to client config
  "max_output_tokens": 512 }           // optional
```
Response (on-topic, 200)
```json
{ "answer": "Drop-D is a tuning where the low E is dropped a whole step to D...",
  "on_topic": true,
  "finish_reason": "STOP",
  "tokens": { "input": 142, "output": 233 },
  "client": "guitar-app-prod" }
```
Response (off-topic with `strict_mode: true`, 200)
```json
{ "answer": "I can't talk about that.",
  "on_topic": false,
  "finish_reason": "BLOCKED_OFF_TOPIC",
  "tokens": { "input": 0, "output": 0 },
  "client": "guitar-app-prod" }
```
Failure codes
- `400` invalid JSON / empty prompt / bad `max_output_tokens`
- `401` missing or unknown client key
- `403` client disabled (`enabled: false`)
- `413` prompt longer than `max_prompt_chars`
- `429` rate limit / daily token budget exceeded
- `502` Gemini upstream error
- `503` `GEMINI_API_KEY` unset, or `GEMINI_PROXY_DISABLED=1`

### `GET /api/gemini/ping`
Same auth. Returns the current daily token budget remaining without burning Gemini quota.
```json
{ "ok": true, "client": "guitar-app-prod", "topic": "guitar and music",
  "remaining_tokens_today": 152000 }
```

## Client registry

Path: env var `GEMINI_PROXY_REGISTRY` (default `~/.config/stocks/gemini_proxy_clients.json`).

Schema
```json
{ "clients": [
    { "client_key": "<48-byte urlsafe base64>",
      "name": "guitar-app-prod",
      "topic_label": "guitar and music",
      "system_prompt": "You are a friendly guitar tutor. Only answer questions about guitar techniques, gear, music theory, and music history.",
      "model": "gemini-flash-latest",
      "strict_mode": false,
      "rate_limit_per_min": 60,
      "daily_token_budget": 200000,
      "max_prompt_chars": 4000,
      "enabled": true } ] }
```

The file is hot-reloaded on every request via mtime check — adding, disabling, or rotating a client takes effect on the next call. If the file's permissions allow group/world reads, a warning is logged on load. **Always run `chmod 600` on it.**

### Mint a new client key
```bash
python -c 'import secrets; print(secrets.token_urlsafe(48))'
```
Paste the output into the registry under `client_key`, save, done.

### Rotate a client key
1. Mint a new key (above).
2. Add a second entry with the new key (keep both for transition).
3. Push a new app build with the new key.
4. After the old build is deprecated, delete the old entry.

### Disable a client
Set `"enabled": false` and save. Their requests return 403.

## Topic gating

The server takes the client's `system_prompt` and appends a fixed suffix that instructs Gemini:

> "If the user asks about anything other than {topic_label}, respond with exactly: `I can't talk about that.`"

This is the **default mode** — one Gemini call per request, relying on instruction-following.

**Strict mode** (`"strict_mode": true`): the server first runs a cheap classifier (`gemini-2.5-flash-lite`, ~5 tokens output) asking "Is this question about {topic_label}? YES or NO." If NO, the server short-circuits with the refusal — no full call, no token spend on the main model. Use strict mode for high-volume clients where abusive off-topic prompts would chew through quota.

LAN admin callers skip topic gating entirely.

## LAN admin bypass

Set env `GEMINI_PROXY_ADMIN_KEY` to a long random string. Then a request from a private IP **and** the header `X-Admin-Key: <that string>` skips both client-key lookup and topic restriction. Useful for development on the home network. If the env var is unset, the bypass is disabled completely.

A request from a public IP that presents the admin header is still rejected — the LAN check is required.

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Upstream Gemini key. Required. |
| `GEMINI_PROXY_REGISTRY` | `~/.config/stocks/gemini_proxy_clients.json` | Registry path. |
| `GEMINI_PROXY_ADMIN_KEY` | unset | LAN admin bypass key. Unset disables the bypass. |
| `GEMINI_PROXY_DISABLED` | unset | `1` makes both endpoints return 503. Kill switch. |
| `GEMINI_PROXY_BUDGET_PATH` | `~/.config/stocks/gemini_proxy_budget.json` | Daily-budget checkpoint. |
| `GEMINI_PROXY_AUDIT_PATH` | `~/.config/stocks/gemini_proxy_audit.jsonl` | Append-only audit log. |

## Audit log

Every request — success and failure — appends one JSON line to `GEMINI_PROXY_AUDIT_PATH`. Each record contains:

- `ts`             ISO8601 UTC timestamp
- `client`         client name (or `?` if unknown)
- `ip`             best-effort client IP (X-Forwarded-For first hop, else peer)
- `status`         `ok`, `missing_key`, `unknown_key`, `client_disabled`, `prompt_too_long`, `rate_limited`, `budget_exhausted`, `off_topic_blocked`, `upstream_error`
- `on_topic`       `true` / `false` / `null` (auth failures, etc.)
- `prompt_len`     character count
- `prompt_preview` first 80 chars of the prompt
- `latency_ms`     handler latency
- on success also: `input_tokens`, `output_tokens`, `finish_reason`, `model`, `strict_mode`

Tail-and-grep example:
```bash
tail -f ~/.config/stocks/gemini_proxy_audit.jsonl | jq 'select(.status != "ok")'
```

## Curl examples

```bash
# On-topic — expect a real answer
curl -sX POST localhost:8080/api/gemini/ask \
     -H "Authorization: Bearer $CLIENT_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"prompt":"Explain drop-D tuning briefly."}' | jq

# Off-topic — expect "I can't talk about that."
curl -sX POST localhost:8080/api/gemini/ask \
     -H "Authorization: Bearer $CLIENT_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"prompt":"What is the capital of France?"}' | jq

# Ping — auth + budget check, no Gemini call
curl -s localhost:8080/api/gemini/ping \
     -H "Authorization: Bearer $CLIENT_KEY" | jq

# LAN admin (from 192.168.x.x with GEMINI_PROXY_ADMIN_KEY=adminsecret on the server)
curl -sX POST localhost:8080/api/gemini/ask \
     -H 'X-Admin-Key: adminsecret' \
     -H 'Content-Type: application/json' \
     -d '{"prompt":"Capital of France?"}' | jq
```

## Deployment

**TLS is mandatory.** The proxy answers contain whatever Gemini returns — without HTTPS those answers (and any sensitive context the user typed) are sniffable on the wire. Put `db_server` behind nginx/Caddy/Cloudflare Tunnel with a real cert before exposing this endpoint to the public internet.

Rotating the upstream `GEMINI_API_KEY` requires only a server restart with the new env value — clients keep using their per-client keys unchanged.
