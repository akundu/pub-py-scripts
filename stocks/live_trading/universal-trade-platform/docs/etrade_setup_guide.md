# E*TRADE Setup Guide

## Prerequisites

- An active E\*TRADE brokerage account
- Python 3.10+ with `pyetrade>=2.1.0` installed
- UTP platform installed

## Step 1: Get API Credentials

1. Log into E\*TRADE at [https://us.etrade.com](https://us.etrade.com)
2. Navigate to [https://us.etrade.com/etx/ris/apikey](https://us.etrade.com/etx/ris/apikey)
3. Complete the **User Intent Survey** describing how you plan to use the API
4. Read and sign the **API Developer License Agreement**
5. Request an **Individual API Consumer Key** (issued immediately)
6. You will receive two credentials:
   - `consumer_key` -- your application identifier
   - `consumer_secret` -- your application secret (keep this private)

**Important**: Save both values securely. The consumer secret is shown only once.

## Step 2: Sign Market Data Agreement

Market data access requires a signed agreement:

- This is typically available during the API key request process
- Without it, quote endpoints will return errors or empty data
- The agreement covers real-time and delayed quote redistribution terms

## Step 3: Set Environment Variables

Add to your `.env` file or export in shell:

```bash
# API credentials
export ETRADE_CONSUMER_KEY="your_consumer_key"
export ETRADE_CONSUMER_SECRET="your_consumer_secret"

# Start with sandbox for testing
export ETRADE_SANDBOX=true
```

## Step 4: Authorize via OAuth

E\*TRADE uses OAuth 1.0a with a browser-based authorization flow:

```bash
python utp.py etrade-auth --sandbox
```

Walk through the flow:

1. A browser window opens to the E\*TRADE authorization URL
2. Log in with your E\*TRADE credentials and approve API access
3. E\*TRADE displays a **6-digit verification code**
4. Copy the code and paste it back in the terminal
5. Tokens are saved to `data/utp/etrade_tokens.json`
6. Tokens expire at **midnight ET daily** -- re-run this command if needed

### List Available Accounts

After authorization, list your accounts to find the account ID:

```bash
python utp.py etrade-auth --sandbox --list-accounts
```

This displays `accountIdKey`, account name, and status for each account. Copy the `accountIdKey` for the next step.

## Step 5: Configure Account

```bash
export ETRADE_ACCOUNT_ID="your_account_id_key"
export ETRADE_SANDBOX=true   # Keep sandbox until verified
```

## Step 6: Verify Read-Only Access

```bash
# These should work with ETRADE_READONLY=true (default)
python utp.py quote SPY --broker etrade
python utp.py portfolio --broker etrade
python utp.py options SPY --broker etrade
```

If all three commands return data without errors, your setup is working.

## Step 7: Switch to Production

Once you have verified everything works in sandbox:

```bash
export ETRADE_SANDBOX=false
export ETRADE_CONSUMER_KEY="your_prod_key"
export ETRADE_CONSUMER_SECRET="your_prod_secret"
python utp.py etrade-auth   # Re-authorize for production
```

**Note**: Production keys may require emailing the signed Developer Agreement to `etradeapi@etrade.com`. Start with sandbox, verify everything works, then switch.

## Step 8: Enable Trading (When Ready)

Trading is disabled by default as a safety measure:

```bash
export ETRADE_READONLY=false
```

## Token Management

| Behavior | Details |
|----------|---------|
| **Daily expiration** | Tokens expire at midnight ET every day |
| **Inactivity timeout** | Tokens are inactivated after 2 hours of no API requests |
| **Auto-renewal** | The `EtradeLiveProvider` automatically renews tokens every 90 minutes |
| **Manual renewal** | Run `python utp.py etrade-auth` if tokens expire |
| **Token file** | Configurable via `ETRADE_TOKEN_FILE` (default: `data/utp/etrade_tokens.json`) |

## E\*TRADE API Key Facts

| Aspect | Details |
|--------|---------|
| **Auth protocol** | OAuth 1.0a (3-stage: request token, authorize, access token) |
| **Order flow** | Preview required before Place (`previewId` valid for 3 minutes) |
| **Multi-leg** | Supports `SPREADS` order action type for 2-4 leg combos |
| **Quotes** | Up to 25 symbols per request, no streaming (pull-only) |
| **Rate limits** | Per-second + per-hour throttle (exact limits in API docs) |
| **API fees** | None |
| **Sandbox URL** | `https://apisb.etrade.com` |
| **Production URL** | `https://api.etrade.com` |

## Differences from IBKR

| Aspect | IBKR | E\*TRADE |
|--------|------|----------|
| Connection | Persistent socket (TWS/Gateway) | Stateless HTTP (OAuth per-request) |
| Auth | TWS login + client ID | OAuth 1.0a + daily token refresh |
| Order flow | Single submit | Preview then Place (2-step) |
| Streaming | Real-time ticks via `reqMktData` | No streaming, poll-only |
| Multi-leg | ComboLeg + BAG contract | `SPREADS` order action type |
| Reconnect | Socket reconnect with backoff | Token renewal every 90 min |
| Rate limits | 50 msg/sec | Per-second + per-hour throttle |

## Environment Variables Reference

| Variable | Default | Notes |
|----------|---------|-------|
| `ETRADE_CONSUMER_KEY` | (required) | OAuth consumer key from E\*TRADE |
| `ETRADE_CONSUMER_SECRET` | (required) | OAuth consumer secret |
| `ETRADE_ACCOUNT_ID` | (required) | Account ID key from `--list-accounts` |
| `ETRADE_SANDBOX` | `true` | Use sandbox (`true`) or production (`false`) |
| `ETRADE_READONLY` | `true` | Block order submission until set to `false` |
| `ETRADE_TOKEN_FILE` | `data/utp/etrade_tokens.json` | Path to persisted OAuth tokens |

## Troubleshooting

### "pyetrade not installed"

```bash
pip install pyetrade>=2.1.0
```

### "OAuth tokens not available"

Run `python utp.py etrade-auth` to complete the authorization flow. This opens a browser for you to approve API access.

### "E\*TRADE tokens expired"

Tokens expire at midnight ET daily. Re-run `python utp.py etrade-auth` to get fresh tokens.

### "E\*TRADE is in read-only mode"

Set `ETRADE_READONLY=false` in your environment. This is a safety feature to prevent accidental order submission.

### Order preview fails

E\*TRADE requires a preview before placing any order. If the preview fails, check:

- Account has sufficient funds/margin
- Symbol, expiration, and strike are valid
- Market is open for the security type
- The `previewId` has not expired (3-minute window)

### Sandbox vs Production

| Aspect | Sandbox | Production |
|--------|---------|------------|
| Base URL | `https://apisb.etrade.com` | `https://api.etrade.com` |
| Data | Fake/delayed | Real market data |
| Money | Simulated | Real funds |
| Key issuance | Immediate | May require emailing signed agreement |

Start with sandbox, verify all commands work, then switch to production.
