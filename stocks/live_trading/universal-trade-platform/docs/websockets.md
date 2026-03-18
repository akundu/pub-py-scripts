# WebSocket Real-Time Updates

## Overview

The `/ws/orders` WebSocket endpoint streams order status changes to connected clients in real-time. When a trade is executed via `POST /trade/execute`, a background task polls the broker for status updates and broadcasts them to all WebSocket clients.

## Connection

```
ws://localhost:8000/ws/orders
```

No authentication is required to connect (see [Production Recommendations](#production-recommendations) below).

## Message Format

### Server -> Client: Order Update

```json
{
  "type": "order_update",
  "order_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "broker": "robinhood",
  "status": "FILLED",
  "message": "Multi-leg order submitted: 2 legs",
  "timestamp": "2026-03-09T15:30:01.000000+00:00"
}
```

### Status Progression

A typical order lifecycle:

```
PENDING → SUBMITTED → FILLED
                    → PARTIAL_FILL → FILLED
                    → CANCELLED
                    → REJECTED
```

Terminal states (polling stops): `FILLED`, `CANCELLED`, `REJECTED`, `FAILED`

## Architecture

```
POST /trade/execute
  │
  ├─ Returns OrderResult immediately (HTTP 200)
  │
  └─ Spawns BackgroundTask: poll_order_status()
       │
       ├─ Loop (10 iterations, 1s interval):
       │    ├─ provider.get_order_status(order_id)
       │    └─ ws_manager.broadcast_order_update(result)
       │
       └─ Stops on terminal status or timeout
```

### ConnectionManager

**File:** `app/websocket.py`

The `ConnectionManager` dataclass handles WebSocket lifecycle:

- `accept(ws)` -- accepts the connection, adds to the active list
- `disconnect(ws)` -- removes from the active list
- `broadcast_order_update(result)` -- serializes the `OrderResult` to JSON and sends to all clients; automatically removes stale connections that fail to send

Thread-safety is ensured via `asyncio.Lock`.

## Client Examples

### Python (websockets library)

```python
import asyncio
import json
import websockets

async def listen():
    async with websockets.connect("ws://localhost:8000/ws/orders") as ws:
        print("Connected to order stream")
        while True:
            message = await ws.recv()
            data = json.loads(message)
            print(f"[{data['status']}] Order {data['order_id']}: {data['message']}")

asyncio.run(listen())
```

### JavaScript (browser)

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/orders");

ws.onopen = () => console.log("Connected to order stream");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`[${data.status}] Order ${data.order_id}: ${data.message}`);

  if (["FILLED", "CANCELLED", "REJECTED", "FAILED"].includes(data.status)) {
    console.log("Order reached terminal state");
  }
};

ws.onclose = () => console.log("Disconnected");
```

### curl (websocat)

```bash
websocat ws://localhost:8000/ws/orders
```

## End-to-End Flow

1. Client connects to `ws://localhost:8000/ws/orders`
2. Client sends `POST /trade/execute` (via REST) with a trade payload
3. Server returns `OrderResult` with `status: SUBMITTED`
4. Background task starts polling broker every 1 second
5. Each status change is broadcast to WebSocket clients:
   - `{"type": "order_update", "status": "SUBMITTED", ...}`
   - `{"type": "order_update", "status": "FILLED", ...}`
6. Polling stops when a terminal status is reached

## Production Recommendations

- **Add authentication** -- validate a JWT token in the WebSocket handshake query string (e.g., `ws://host/ws/orders?token=...`)
- **Heartbeat/ping** -- add periodic pings to detect dropped connections
- **Per-user filtering** -- only send updates for orders belonging to the authenticated user
- **Reconnection** -- clients should implement exponential backoff reconnection
- **Rate limiting** -- cap the number of concurrent WebSocket connections per user
