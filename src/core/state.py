import json
import redis
import statistics
from typing import List, Optional
from .types import DriftStats, SystemState
from .clock import Clock

class ObserverState:
    """
    Manages the authoritative state of the Observer system.
    Backed by Redis, with local caching for read-heavy operations.
    """
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.drift_samples: List[int] = []
        self._local_cache: Optional[SystemState] = None
        self.max_drift_samples = 50

    def update_drift(self, drift_us: int) -> DriftStats:
        """
        Updates rolling drift statistics.
        """
        self.drift_samples.append(drift_us)
        if len(self.drift_samples) > self.max_drift_samples:
            self.drift_samples.pop(0)

        mean_val = statistics.mean(self.drift_samples) if self.drift_samples else 0.0
        
        # Simple slope calculation (linear regression on the window)
        slope_val = 0.0
        if len(self.drift_samples) > 1:
            # x is just index, y is drift
            x = list(range(len(self.drift_samples)))
            try:
                # Basic linear regression slope
                slope_val = statistics.linear_regression(x, self.drift_samples).slope
            except AttributeError:
                # Fallback for older python versions without linear_regression
                # Calculate slope manually if needed or ignore
                pass

        stats = DriftStats(mean_us=mean_val, slope=slope_val, sample_count=len(self.drift_samples))
        
        # Persist stats to Redis? Optional, maybe just keep in memory for high freq
        # For now, we update the authoritative state periodically or on significant change
        return stats

    def set_system_status(self, status: str):
        """
        Updates the global system status in Redis.
        """
        self.redis.set("observer:status", status)
        self.redis.set("observer:last_update", Clock.now_us())

    def get_system_status(self) -> str:
        """
        Reads authoritative status from Redis.
        """
        return self.redis.get("observer:status") or "UNKNOWN"

    def record_gap(self):
        """
        Increments the gap counter in Redis.
        """
        self.redis.incr("observer:gap_count")

    def get_gap_count(self) -> int:
        return int(self.redis.get("observer:gap_count") or 0)
