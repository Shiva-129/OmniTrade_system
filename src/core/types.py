from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
from .money import Decimal

# Constants
MICROSECONDS_PER_SECOND = 1_000_000

class Packet(BaseModel):
    """
    Standardized internal packet format.
    Immutable once created.
    """
    model_config = {"frozen": True}

    exchange_ts: int  # Canonical exchange timestamp in microseconds
    local_arrival_ts: int # Local monotonic timestamp in microseconds
    drift_us: int # drift = exchange_ts - local_arrival_ts
    source: str # e.g., "binance_ws", "kite_rest"
    topic: str # e.g., "trade.btcusdt", "orderbook.nifty"
    payload: Dict[str, Any] # The raw payload
    sequence_id: Optional[int] = None # if provided by exchange

class JournalEntry(BaseModel):
    """
    Entry for the immutable append-only journal.
    """
    event_type: Literal["PACKET", "STATUS_CHANGE", "ERROR", "GAP"]
    timestamp: int # local_arrival_ts
    data: Dict[str, Any]

class DriftStats(BaseModel):
    mean_us: float
    slope: float
    sample_count: int

class SystemState(BaseModel):
    status: Literal["CONNECTED", "DEGRADED", "HALT"]
    last_seen_ts: int
    gap_count: int

# --- Phase 2: Gatekeeper Types ---

from enum import Enum

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"

class TimeInForce(str, Enum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"

class OrderIntent(BaseModel):
    """
    Intent to place an order. Guaranteed immutable by policy.
    Quantity/price are Decimal (canonical money policy). Pydantic
    serializes them as fixed-point strings on the wire.
    """
    model_config = {"frozen": True}

    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    timestamp: int # local creation time

class ExecutionReport(BaseModel):
    """
    Truth from the exchange. Logic triggers on this.
    All monetary fields are Decimal (canonical money policy).

    Phase 5: `fee` is the absolute commission charged by the venue for
    THIS fill, denominated in quote currency (e.g. USDT). It is part of
    the event-sourced record so fee accounting survives replay.
    Default 0 keeps pre-fee producers backward compatible.
    """
    model_config = {"frozen": True}

    client_order_id: str
    exchange_order_id: str
    symbol: str
    side: OrderSide
    status: Literal["NEW", "PARTIAL_FILL", "FILLED", "CANCELED", "REJECTED"]
    filled_quantity: Decimal
    last_filled_price: Decimal
    remaining_quantity: Decimal
    timestamp: int # Exchange timestamp
    fee: Decimal = Decimal("0")

# --- Phase 4: Event Contracts ---
# Components exchange information ONLY through these typed events
# (EventBus boundary rule). No cross-component method calls.

class MarketEvent(BaseModel):
    """
    Normalized market observation emitted onto the EventBus.
    Wraps the raw Packet with an explicit event envelope.
    """
    model_config = {"frozen": True}

    packet: Packet

class RiskDecision(BaseModel):
    """
    Outcome of a risk evaluation of one OrderIntent.
    Emitted by the Risk engine (Phase 6); consumed by execution path.
    """
    model_config = {"frozen": True}

    client_order_id: str
    approved: bool
    reason: str = ""

class PortfolioUpdate(BaseModel):
    """
    Reserved event contract for Phase 5. Defined now so the bus
    contract surface is complete; nothing publishes it yet.
    """
    model_config = {"frozen": True}

    symbol: str
    quantity_delta: Decimal
