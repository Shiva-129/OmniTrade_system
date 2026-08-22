from typing import Dict, Optional, Any
from decimal import Decimal
import redis
import redis.exceptions
from ..core.types import ExecutionReport, OrderIntent
from ..core.money import to_decimal, ZERO, dec_to_str
from ..core.logger import get_logger

logger = get_logger("StateController")

class StateController:
    """
    Authority for mutating Position and Order state.
    Strictly follows: Mutation ONLY on ExecutionReport.

    Positions are stored as canonical fixed-point STRINGS and mutated via
    optimistic WATCH/MULTI transactions with exact Decimal arithmetic --
    never incrbyfloat (binary float drift violates the money invariant).
    """
    MAX_TX_RETRIES = 5

    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        # Using a distinct prefix for Gatekeeper owned state
        self.PREFIX_POS = "gk:positions"
        self.PREFIX_ORDER = "gk:orders"

    def process_execution_report(self, report: ExecutionReport):
        """
        The ONLY entry point for state mutation.
        """
        self._update_order_state(report)

        if report.status in ["PARTIAL_FILL", "FILLED"]:
            self._update_position(report)

        logger.info("state_updated",
                    client_order_id=report.client_order_id,
                    status=report.status,
                    filled_qty=str(report.filled_quantity))

    def _update_order_state(self, report: ExecutionReport):
        """
        Updates order state in Redis.
        """
        key = f"{self.PREFIX_ORDER}:{report.client_order_id}"
        self.redis.set(key, report.model_dump_json())

    def _update_position(self, report: ExecutionReport):
        """
        Atomically applies a signed fill delta using exact Decimal math.

        WATCH the position key, read current string value, compute the new
        value client-side under the canonical context, then MULTI/SET/EXEC.
        On concurrent modification (WatchError) retry with fresh state.
        """
        key = f"{self.PREFIX_POS}:{report.symbol}"
        signed_delta = report.filled_quantity if report.side == "BUY" else -report.filled_quantity

        for _attempt in range(self.MAX_TX_RETRIES):
            pipe = self.redis.pipeline()
            try:
                pipe.watch(key)
                current_raw = pipe.get(key)
                current = to_decimal(current_raw) if current_raw is not None else ZERO
                new_value = current + signed_delta

                pipe.multi()
                pipe.set(key, dec_to_str(new_value))
                pipe.execute()
                return
            except redis.exceptions.WatchError:
                # Concurrent writer won; retry with fresh read.
                continue
            finally:
                pipe.reset()

        raise RuntimeError(
            f"Position update lost race {self.MAX_TX_RETRIES}x for {key}; "
            "refusing to write possibly stale state."
        )

    def get_position(self, symbol: str) -> Decimal:
        raw = self.redis.get(f"{self.PREFIX_POS}:{symbol}")
        return to_decimal(raw) if raw is not None else ZERO
