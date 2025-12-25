"""
OmniTrade Simulator: State Hasher

Computes deterministic hashes of system state at event boundaries.
Used to detect replay divergence.
"""
import hashlib
import json
from decimal import Decimal
from typing import Dict, Any

class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)

class StateHasher:
    """
    Computes deterministic SHA-256 hashes of system state.
    """
    
    @staticmethod
    def hash_state(state: Dict[str, Any]) -> str:
        """
        Computes a deterministic hash of the given state dictionary.
        Keys are sorted for determinism.
        """
        # Serialize with sorted keys for determinism
        serialized = json.dumps(state, sort_keys=True, cls=DecimalEncoder)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    @staticmethod
    def hash_positions(positions: Dict[str, Decimal]) -> str:
        """Hash position state."""
        return StateHasher.hash_state({"positions": positions})

    @staticmethod
    def hash_orders(orders: Dict[str, Any]) -> str:
        """Hash order state."""
        return StateHasher.hash_state({"orders": orders})

    @staticmethod
    def hash_full_state(
        positions: Dict[str, Decimal],
        orders: Dict[str, Any],
        system_status: str,
        gap_count: int
    ) -> str:
        """
        Hash the complete system state.
        """
        state = {
            "positions": positions,
            "orders": orders,
            "system_status": system_status,
            "gap_count": gap_count
        }
        return StateHasher.hash_state(state)
