"""Pydantic schemas for orders, quotes, and positions."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OptionAction(str, Enum):
    BUY_TO_OPEN = "BUY_TO_OPEN"
    SELL_TO_OPEN = "SELL_TO_OPEN"
    BUY_TO_CLOSE = "BUY_TO_CLOSE"
    SELL_TO_CLOSE = "SELL_TO_CLOSE"


class OptionType(str, Enum):
    CALL = "CALL"
    PUT = "PUT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class Broker(str, Enum):
    ROBINHOOD = "robinhood"
    ETRADE = "etrade"
    IBKR = "ibkr"


class LedgerEventType(str, Enum):
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_STATUS_CHANGE = "ORDER_STATUS_CHANGE"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    POSITION_SYNCED = "POSITION_SYNCED"
    CSV_IMPORTED = "CSV_IMPORTED"
    SNAPSHOT = "SNAPSHOT"
    SYSTEM_EVENT = "SYSTEM_EVENT"
    PLAYBOOK_EXECUTED = "PLAYBOOK_EXECUTED"


class PositionSource(str, Enum):
    LIVE_API = "live_api"
    PAPER = "paper"
    CSV_IMPORT = "csv_import"
    EXTERNAL_SYNC = "external_sync"


# ── Order Schemas ──────────────────────────────────────────────────────────────

class OptionLeg(BaseModel):
    """A single leg of a multi-leg options order."""

    symbol: str = Field(..., description="Underlying symbol, e.g. 'SPY'")
    expiration: str = Field(..., description="Expiration date YYYY-MM-DD")
    strike: float
    option_type: OptionType
    action: OptionAction
    quantity: int = Field(ge=1)


class MultiLegOrder(BaseModel):
    """Atomic multi-leg options order (e.g. vertical spread, iron condor)."""

    broker: Broker
    legs: list[OptionLeg] = Field(..., min_length=1, max_length=4)
    order_type: OrderType = OrderType.LIMIT
    net_price: Optional[float] = Field(
        None, description="Net debit/credit per contract (required for LIMIT)"
    )
    quantity: int = Field(default=1, ge=1, description="Contract multiplier")
    time_in_force: str = Field(default="DAY")


class EquityOrder(BaseModel):
    """Simple equity order."""

    broker: Broker
    symbol: str
    side: OrderSide
    quantity: int = Field(ge=1)
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    time_in_force: str = Field(default="DAY")


class TradeRequest(BaseModel):
    """Union wrapper — exactly one of equity_order or multi_leg_order."""

    equity_order: Optional[EquityOrder] = None
    multi_leg_order: Optional[MultiLegOrder] = None
    closing_position_id: Optional[str] = None
    closing_quantity: Optional[int] = None

    def model_post_init(self, __context: object) -> None:
        if not self.equity_order and not self.multi_leg_order:
            raise ValueError("Must provide equity_order or multi_leg_order")
        if self.equity_order and self.multi_leg_order:
            raise ValueError("Provide only one of equity_order or multi_leg_order")


# ── Response Schemas ───────────────────────────────────────────────────────────

class OrderResult(BaseModel):
    order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    broker: Broker
    status: OrderStatus = OrderStatus.PENDING
    message: str = ""
    dry_run: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    filled_price: Optional[float] = None
    filled_quantity: Optional[int] = None


class Quote(BaseModel):
    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str = ""  # e.g. "ibkr", "streaming_cache", "delayed"


class Position(BaseModel):
    broker: Broker
    symbol: str
    quantity: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    source: PositionSource = PositionSource.LIVE_API
    last_synced_at: Optional[datetime] = None
    account_id: Optional[str] = None
    con_id: Optional[int] = None  # IBKR contract ID — unique per instrument
    sec_type: Optional[str] = None  # STK, OPT, FOP, IND
    expiration: Optional[str] = None  # For options: YYYYMMDD
    strike: Optional[float] = None  # For options
    right: Optional[str] = None  # C or P for options


class AggregatedPositions(BaseModel):
    positions: list[Position]
    total_market_value: float
    total_unrealized_pnl: float


# ── Ledger Schemas ────────────────────────────────────────────────────────────

class LedgerEntry(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: LedgerEventType
    broker: Optional[Broker] = None
    order_id: Optional[str] = None
    position_id: Optional[str] = None
    source: PositionSource = PositionSource.LIVE_API
    dry_run: bool = False
    data: dict = Field(default_factory=dict)
    sequence_number: int = 0


class LedgerQuery(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    broker: Optional[Broker] = None
    event_type: Optional[LedgerEventType] = None
    source: Optional[PositionSource] = None
    order_id: Optional[str] = None
    limit: int = 100
    offset: int = 0


# ── Position Tracking Schemas ─────────────────────────────────────────────────

class TrackedPosition(BaseModel):
    position_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "open"
    source: PositionSource = PositionSource.LIVE_API
    broker: Broker
    order_type: str = "equity"
    symbol: str
    side: Optional[str] = None
    quantity: float = 0
    entry_price: float = 0
    entry_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    legs: Optional[list[dict]] = None
    expiration: Optional[str] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    order_id: Optional[str] = None
    pnl: Optional[float] = None
    current_mark: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    last_synced_at: Optional[datetime] = None
    con_id: Optional[int] = None
    sec_type: Optional[str] = None
    strike: Optional[float] = None
    right: Optional[str] = None


# ── Account Balances ──────────────────────────────────────────────────────────

class AccountBalances(BaseModel):
    """Account-level cash and margin balances from a broker."""
    cash: float = 0
    net_liquidation: float = 0
    buying_power: float = 0
    maint_margin_req: float = 0
    available_funds: float = 0
    broker: Optional[str] = None


# ── Dashboard Schemas ─────────────────────────────────────────────────────────

class DashboardSummary(BaseModel):
    active_positions: list[TrackedPosition]
    cash_available: float = 0
    cash_deployed: float = 0
    net_liquidation: float = 0
    buying_power: float = 0
    maint_margin_req: float = 0
    available_funds: float = 0
    total_pnl: float = 0
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    last_sync_time: Optional[datetime] = None
    positions_by_source: dict[str, int] = Field(default_factory=dict)


class PerformanceMetrics(BaseModel):
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0
    net_pnl: float = 0
    roi: float = 0
    profit_factor: float = 0
    sharpe: float = 0
    max_drawdown: float = 0
    avg_pnl: float = 0


class DailyPnL(BaseModel):
    date: date
    realized_pnl: float = 0
    unrealized_pnl: float = 0
    total_pnl: float = 0
    trades_opened: int = 0
    trades_closed: int = 0


class CSVImportResult(BaseModel):
    file_name: str
    broker: Broker
    records_imported: int = 0
    records_skipped: int = 0
    errors: list[str] = Field(default_factory=list)


class SyncResult(BaseModel):
    new_positions: int = 0
    updated_positions: int = 0
    unchanged_positions: int = 0
    brokers_synced: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── Reconciliation Schemas ───────────────────────────────────────────────────

class ReconciliationEntry(BaseModel):
    symbol: str
    broker: str
    system_quantity: Optional[float] = None
    broker_quantity: Optional[float] = None
    discrepancy_type: str  # "matched", "missing_in_system", "missing_at_broker", "quantity_mismatch"
    details: str = ""


class ReconciliationReport(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    broker: str
    total_system_positions: int = 0
    total_broker_positions: int = 0
    matched: int = 0
    discrepancies: list[ReconciliationEntry] = Field(default_factory=list)


# ── Status Dashboard Schemas ─────────────────────────────────────────────────

class StatusReport(BaseModel):
    active_positions: list[TrackedPosition] = Field(default_factory=list)
    in_transit_orders: list[dict] = Field(default_factory=list)
    recent_closed: list[dict] = Field(default_factory=list)
    discrepancies: list[ReconciliationEntry] = Field(default_factory=list)
    cache_stats: dict = Field(default_factory=dict)
    connection_status: dict = Field(default_factory=dict)


# ── Playbook Schemas ─────────────────────────────────────────────────────────

class PlaybookInstruction(BaseModel):
    id: str
    type: str  # equity, single_option, credit_spread, debit_spread, iron_condor
    params: dict = Field(default_factory=dict)


class PlaybookDefinition(BaseModel):
    name: str
    description: str = ""
    broker: Broker = Broker.IBKR
    instructions: list[PlaybookInstruction] = Field(default_factory=list)


class InstructionResult(BaseModel):
    instruction_id: str
    status: str  # "success", "failed", "skipped", "dry_run"
    order_result: Optional[OrderResult] = None
    error: Optional[str] = None
    position_id: Optional[str] = None


class PlaybookResult(BaseModel):
    playbook_name: str
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    results: list[InstructionResult] = Field(default_factory=list)
