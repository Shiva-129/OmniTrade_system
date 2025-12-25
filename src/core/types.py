from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field

# Constants
MICROSECONDS_PER_SECOND = 1_000_000

class Packet(BaseModel):
    """
    Standardized internal packet format.
    Immutable once created.
    """
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
    """
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float # Decimal preferred in full impl, float for now if acceptable or switch to Decimal
    price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    timestamp: int # local creation time

class ExecutionReport(BaseModel):
    """
    Truth from the exchange. Logic triggers on this.
    """
    client_order_id: str
    exchange_order_id: str
    symbol: str
    side: OrderSide
    status: Literal["NEW", "PARTIAL_FILL", "FILLED", "CANCELED", "REJECTED"]
    filled_quantity: float
    last_filled_price: float
    remaining_quantity: float
    timestamp: int # Exchange timestamp

