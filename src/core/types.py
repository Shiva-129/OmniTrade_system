from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field

# Constants
MICROSECONDS_PER_SECOND = 1_000_000

class Packet(BaseModel):
    """
    Standardized internal packet format.
    Immutable once created.
    """
    exchange_ts: int  # Canonical exchange timestamp in microseconds (Epoch)
    local_arrival_ts: int # Local timestamp in microseconds (Epoch)
    drift_us: int # drift = exchange_ts - local_arrival_ts
    source: str # e.g., "binance_ws", "kite_rest"
    topic: str # e.g., "trade.btcusdt", "orderbook.nifty"
    payload: Dict[str, Any] # The raw payload
    sequence_id: Optional[str] = None # if provided by exchange (Trade ID can be str or int)

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
