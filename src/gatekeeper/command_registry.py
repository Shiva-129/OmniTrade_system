import redis
from ..core.types import OrderIntent
from ..core.logger import get_logger

logger = get_logger("CommandRegistry")

class CommandRegistry:
    """
    Ensures Idempotency via Redis SET NX.

    Phase 4: previously in-memory (idempotency was LOST on restart,
    allowing duplicate orders after a crash-recovery cycle). Now backed
    by Redis so a restarted process still recognizes prior intents.

    Key space: gk:cmdreg:{client_order_id} -> serialized OrderIntent
    """
    PREFIX = "gk:cmdreg"

    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url, decode_responses=True)

    def register(self, intent: OrderIntent) -> bool:
        """
        Registers an intent.
        Returns True if new, False if duplicate (seen by ANY process,
        including ones before a restart).
        """
        key = f"{self.PREFIX}:{intent.client_order_id}"
        # NX = set only if not exists. Single atomic op; no race window.
        was_new = self.redis.set(key, intent.model_dump_json(), nx=True)
        if not was_new:
            logger.info("duplicate_intent_rejected", cloid=intent.client_order_id)
        return bool(was_new)

    def get(self, client_order_id: str):
        raw = self.redis.get(f"{self.PREFIX}:{client_order_id}")
        return OrderIntent.model_validate_json(raw) if raw else None
