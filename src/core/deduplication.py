"""
Execution Deduplication (Phase 12) -- deterministic identity for every fill.

The same execution event may arrive via:
  - WebSocket user-data stream
  - REST reconciliation fetch
  - Journal replay

It must mutate Portfolio exactly once. Identity is:

  exchange execution ID (Binance tradeId / t) when available
  otherwise: client_order_id:exchange_order_id:filled_qty

All three paths produce the same dedup key.
"""
from __future__ import annotations

from typing import Dict, Optional, Set

from .money import to_decimal, dec_to_str


class ExecutionDeduplicator:
    """
    Pure dedup registry. No I/O, no side effects beyond the set.
    """

    def __init__(self):
        self._seen: Set[str] = set()

    def make_key(self, client_order_id: str, exchange_order_id: str,
                 execution_id: Optional[str] = None,
                 filled_qty: str = "0") -> str:
        if execution_id:
            return f"{exchange_order_id}:{execution_id}"
        # Fallback: deterministic from order + fill qty
        return f"{client_order_id}:{exchange_order_id}:{filled_qty}"

    def is_duplicate(self, key: str) -> bool:
        return key in self._seen

    def mark_seen(self, key: str) -> bool:
        """
        Returns True if this is the first time seeing the key (i.e. should
        be processed), False if duplicate (should be dropped).
        """
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def seed(self, keys) -> None:
        self._seen.update(keys)

    def snapshot(self) -> Set[str]:
        return set(self._seen)

    @property
    def count(self) -> int:
        return len(self._seen)
