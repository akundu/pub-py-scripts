"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All sensitive values come from env vars or a .env file."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

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
    order_poll_interval_seconds: float = 1.0
    order_poll_timeout_seconds: float = 30.0

    # LAN trust
    trust_local_network: bool = True

    # CSV import
    csv_import_dir: str = "data/utp/imports"

    def broker_list(self) -> list[str]:
        return [b.strip().lower() for b in self.enabled_brokers.split(",") if b.strip()]


settings = Settings()
