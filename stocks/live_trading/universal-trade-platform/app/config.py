"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All sensitive values come from env vars or a .env file."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key_secret: str = Field(default="change-me", description="Shared API key for simple auth")
    jwt_secret_key: str = Field(default="change-me-jwt", description="Secret for JWT signing")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # Robinhood
    robinhood_username: str = ""
    robinhood_password: str = ""
    robinhood_totp_seed: str = ""

    # E*TRADE
    etrade_consumer_key: str = ""
    etrade_consumer_secret: str = ""
    etrade_oauth_token: str = ""
    etrade_oauth_secret: str = ""
    etrade_sandbox: bool = True  # true = apisb.etrade.com, false = api.etrade.com
    etrade_account_id: str = ""  # accountIdKey from /v1/accounts
    etrade_readonly: bool = True  # Safety: blocks order submission
    etrade_token_file: str = "data/utp/etrade_tokens.json"  # Persisted OAuth tokens

    # IBKR
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497
    ibkr_client_id: int = 1
    ibkr_account_id: str = ""
    ibkr_market_data_type: int = 4
    ibkr_connect_timeout: int = 30
    ibkr_readonly: bool = True
    ibkr_exchange: str = "SMART"
    ibkr_option_chain_cache_dir: str = "data/utp/cache/option_chains"
    ibkr_api_mode: str = "tws"  # "tws" (ib_insync) or "rest" (Client Portal Gateway)
    ibkr_gateway_url: str = "https://localhost:5000"

    # Enabled brokers
    enabled_brokers: str = "robinhood,etrade,ibkr"

    # Persistence
    data_dir: str = "data/utp"

    # Expiration
    eod_auto_close: bool = False
    expiration_check_interval_seconds: int = 60

    # Position sync
    position_sync_interval_seconds: int = 120
    position_sync_enabled: bool = True

    # Order fill tracking
    # Poll interval is floored at 2.0s — sub-2s polls hit IBKR's pacing
    # limits and don't actually surface fills any sooner (TWS only ticks
    # status updates at ~1Hz). Two seconds is the documented contract.
    order_poll_interval_seconds: float = 2.0
    order_poll_timeout_seconds: float = 60.0

    @field_validator("order_poll_interval_seconds")
    @classmethod
    def _floor_poll_interval(cls, v: float) -> float:
        return max(float(v), 2.0)

    # Trade defaults — applied to ANY caller (CLI `utp trade`, playbook,
    # spread_scanner handlers, future integrations) that doesn't explicitly
    # override. One place to tune, everyone benefits.
    default_order_type: str = "MARKET"          # MARKET or LIMIT
    limit_slippage_pct: float = 0.0             # credit * (1 - N/100) when LIMIT
    limit_quote_max_age_sec: float = 10.0       # force provider refresh if older

    # LAN trust
    trust_local_network: bool = True

    # CSV import
    csv_import_dir: str = "data/utp/imports"

    # Trade notifications
    notify_on_fill: bool = False  # Enable email/SMS on trade fill
    notify_channel: str = "email"  # "sms", "email", or "both"
    notify_recipients: str = ""  # Comma-separated email addresses or phone numbers
    notify_tag: str = "[UTP-ALERT]"  # Email subject prefix for filtering
    notify_on_paper: bool = False  # Also notify on paper/dry-run trades
    notify_url: str = "http://localhost:9102"  # db_server URL for HTTP notify

    def broker_list(self) -> list[str]:
        return [b.strip().lower() for b in self.enabled_brokers.split(",") if b.strip()]


settings = Settings()
