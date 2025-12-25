from typing import Dict, Optional, Any
import redis
from ..core.types import ExecutionReport, OrderIntent
from ..core.logger import get_logger

logger = get_logger("StateController")

class StateController:
    """
    Authority for mutating Position and Order state.
    Strictly follows: Mutation ONLY on ExecutionReport.
    """
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
                    filled_qty=report.filled_quantity)

    def _update_order_state(self, report: ExecutionReport):
        """
        Updates order status in Redis.
        """
        key = f"{self.PREFIX_ORDER}:{report.client_order_id}"
        # We store the latest report or a consolidated state
        # For simple KV, just dumping the json
        self.redis.set(key, report.model_dump_json())

    def _update_position(self, report: ExecutionReport):
        """
        Updates position based on fills.
        Atomic INCRBYFLOAT equivalent behavior needed.
        """
        key = f"{self.PREFIX_POS}:{report.symbol}"
        
        signed_qty = report.filled_quantity if report.side == "BUY" else -report.filled_quantity
        
        # Redis incrbyfloat is suitable here
        self.redis.incrbyfloat(key, signed_qty)

    def get_position(self, symbol: str) -> float:
        val = self.redis.get(f"{self.PREFIX_POS}:{symbol}")
        return float(val) if val else 0.0
