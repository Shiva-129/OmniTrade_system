from typing import List, Optional
import redis
from .rate_limiter import TokenBucket
from ..core.clock import Clock
from ..core.logger import get_logger

logger = get_logger("ExecutionGuard")

class ExecutionGuard:
    """
    Level 0 Guard.
    Enforces: Rate Limits, Connectivity Interlock, Safe Mode.
    """
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.rate_limiter = TokenBucket(rate=10.0, capacity=50.0) # 10 orders/sec
        self.in_safe_mode = False

    def validate_intent(self) -> bool:
        """
        Checks all pre-conditions. Raises HardBlockError on failure.
        """
        # 1. Check Safe Mode
        if self.in_safe_mode:
            raise RuntimeError("HARD_BLOCK: System in SAFE_MODE")

        # 2. Check Phase 1 Connectivity
        status = self.redis.get("observer:status")
        if status != "CONNECTED":
            raise RuntimeError(f"HARD_BLOCK: Observer status is {status}")

        # 3. Check Heartbeat (Last update from Observer)
        last_seen = int(self.redis.get("observer:last_update") or 0)
        now = Clock.now_us()
        if (now - last_seen) > 2_000_000: # 2 seconds tolerance
            raise RuntimeError(f"HARD_BLOCK: Observer heartbeat stale (>2s)")

        # 4. Check Rate Limit
        if not self.rate_limiter.consume(1.0):
            raise RuntimeError("HARD_BLOCK: Rate limit exceeded")

        return True

    def enter_safe_mode(self, reason: str):
        logger.error("safe_mode_activated", reason=reason)
        self.in_safe_mode = True
        # Could persist this to Redis to coordinate distributed guards
