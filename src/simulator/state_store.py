"""
OmniTrade Simulator: Simulated State Store

In-memory state store for simulation (no Redis dependency).
Mimics the StateController interface for code path parity.
"""
from decimal import Decimal
from typing import Dict, Any, Optional
from .state_hasher import StateHasher

class SimulatedStateStore:
    """
    In-memory state store for deterministic replay.
    Mirrors StateController but without Redis.
    """
    def __init__(self):
        self.positions: Dict[str, Decimal] = {}
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.system_status: str = "CONNECTED"
        self.gap_count: int = 0
        self.last_seen_ts: int = 0

    def update_position(self, symbol: str, delta: Decimal):
        """Update position by delta amount."""
        current = self.positions.get(symbol, Decimal("0"))
        self.positions[symbol] = current + delta

    def set_position(self, symbol: str, qty: Decimal):
        """Set absolute position."""
        self.positions[symbol] = qty

    def get_position(self, symbol: str) -> Decimal:
        return self.positions.get(symbol, Decimal("0"))

    def set_order(self, client_order_id: str, order_data: Dict[str, Any]):
        """Store order state."""
        self.orders[client_order_id] = order_data

    def get_order(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        return self.orders.get(client_order_id)

    def set_system_status(self, status: str):
        self.system_status = status

    def increment_gap_count(self):
        self.gap_count += 1

    def get_state_hash(self) -> str:
        """Compute hash of current full state."""
        return StateHasher.hash_full_state(
            positions=self.positions,
            orders=self.orders,
            system_status=self.system_status,
            gap_count=self.gap_count
        )

    def snapshot(self) -> Dict[str, Any]:
        """Return a copy of current state."""
        return {
            "positions": dict(self.positions),
            "orders": dict(self.orders),
            "system_status": self.system_status,
            "gap_count": self.gap_count,
            "last_seen_ts": self.last_seen_ts
        }
