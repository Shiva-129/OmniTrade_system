from ..core.clock import Clock

class TokenBucket:
    """
    Deterministic Rate Limiter.
    """
    def __init__(self, rate: float, capacity: float):
        """
        rate: tokens per second
        capacity: max burst tokens
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update_ts = Clock.now_us()

    def consume(self, tokens: float = 1.0) -> bool:
        """
        Attempts to consume tokens. Returns True if allowed.
        """
        now = Clock.now_us()
        
        # Refill
        # duration in seconds
        delta_seconds = (now - self.last_update_ts) / 1_000_000.0
        new_tokens = delta_seconds * self.rate
        
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_update_ts = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
