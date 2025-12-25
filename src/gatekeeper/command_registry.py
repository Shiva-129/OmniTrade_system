from typing import Dict, Optional
from ..core.types import OrderIntent

class CommandRegistry:
    """
    Ensures Idempotency.
    Registry of ClientOrderIDs.
    """
    def __init__(self):
        # In-memory for now, could be Redis-backed for persistence across restarts
        self._orders: Dict[str, OrderIntent] = {}

    def register(self, intent: OrderIntent) -> bool:
        """
        Registers an intent. 
        Returns True if new, False if duplicate.
        """
        if intent.client_order_id in self._orders:
            return False
        
        self._orders[intent.client_order_id] = intent
        return True

    def get(self, client_order_id: str) -> Optional[OrderIntent]:
        return self._orders.get(client_order_id)
