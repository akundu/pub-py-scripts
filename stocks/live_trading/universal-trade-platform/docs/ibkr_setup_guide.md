# IBKR (Interactive Brokers) Setup Guide

## Prerequisites

- An Interactive Brokers account (paper or live)
- IB Gateway or Trader Workstation (TWS) installed
- Python package: `pip install ib_insync>=0.9.86`

## Step 1: Install IB Gateway or TWS

Download from [Interactive Brokers](https://www.interactivebrokers.com/en/trading/tws.php):

- **IB Gateway** (recommended for API-only use): lightweight, headless
- **TWS** (Trader Workstation): full GUI + API access

Choose **paper trading** for testing, **live** for production.

## Step 2: Configure API Settings in TWS/Gateway

1. Open TWS or IB Gateway
2. Navigate to: **Edit → Global Configuration → API → Settings**
3. Enable these settings:
   - **Enable ActiveX and Socket Clients**: checked
   - **Socket port**: `7497` (paper) or `7496` (live)
   - **Allow connections from localhost only**: checked (or add trusted IPs)
   - **Read-Only API**: uncheck only when ready for live trading
4. Click **Apply** and **OK**

## Step 3: Set Environment Variables

Add to your `.env` file or export in shell:

```bash
# Connection
IBKR_HOST=127.0.0.1
IBKR_PORT=7497              # 7497=paper, 7496=live
IBKR_CLIENT_ID=1

# Account identification (triggers real provider instead of stub)
IBKR_ACCOUNT_ID=DU123456    # Your paper/live account ID

# Safety
IBKR_READONLY=true           # Start read-only, change to false when ready

# Market data
IBKR_MARKET_DATA_TYPE=4      # 4=delayed(free), 3=frozen, 1=live(paid subscription)

# Timeout
IBKR_CONNECT_TIMEOUT=30      # seconds
```

**Important**: Setting `IBKR_ACCOUNT_ID` activates the real `IBKRLiveProvider`. Leaving it empty uses the stub provider.

## Step 4: Start IB Gateway/TWS

1. Launch IB Gateway (or TWS)
2. Log in with your paper/live credentials
3. Wait for "Connected" status

## Step 5: Start the Platform

```bash
python server.py
```

## Step 6: Verify Connection

```bash
# Health check
curl http://localhost:8000/health
# → {"status": "ok"}

# Get a real quote (requires market data subscription for type=1)
curl -H "X-API-Key: change-me" http://localhost:8000/market/quote/AAPL?broker=ibkr
# → {"symbol": "AAPL", "bid": 175.50, "ask": 175.55, ...}

# Get real positions
curl -H "X-API-Key: change-me" http://localhost:8000/account/positions
# → {"positions": [...], "total_market_value": ...}
```

## Step 7: Enable Trading (When Ready)

1. Set `IBKR_READONLY=false` in your `.env`
2. Restart the server
3. Submit a test order:

```bash
curl -X POST http://localhost:8000/trade/execute \
  -H "X-API-Key: change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "equity_order": {
      "broker": "ibkr",
      "symbol": "AAPL",
      "side": "BUY",
      "quantity": 1,
      "order_type": "LIMIT",
      "limit_price": 170.00
    }
  }'
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ConnectionRefusedError` | Ensure TWS/Gateway is running and API is enabled |
| `ClientId already in use` | Change `IBKR_CLIENT_ID` to a different number |
| Quotes return 0 | Check `IBKR_MARKET_DATA_TYPE` — use `4` for free delayed data |
| Orders rejected with "read-only" | Set `IBKR_READONLY=false` and restart |
| `ib_insync not installed` | Run `pip install ib_insync>=0.9.86` |

## Market Data Types

| Type | Description | Cost |
|------|-------------|------|
| 1 | Live streaming | Paid subscription required |
| 2 | Frozen (last available) | Free |
| 3 | Delayed frozen | Free |
| 4 | Delayed (15-20 min) | Free |

## Security Notes

- **Always start with `IBKR_READONLY=true`** until you've verified the setup
- Use **paper trading** (`port 7497`) before switching to live (`port 7496`)
- The platform logs all orders to the transaction ledger for audit
- Consider running IB Gateway in a separate process/container for isolation
