import time
from typing import Final

# Enforce int64 microsecond precision
MICROSECONDS: Final[int] = 1_000_000

class Clock:
    """
    Authoritative local clock source.
    Uses time.monotonic_ns() for monotonic guarantees, scaled to microseconds.
    """
    
    @staticmethod
    def now_us() -> int:
        """
        Returns current monotonic time in microseconds (int64).
        """
        # monotonic_ns returns nanoseconds. Divide by 1000 to get microseconds.
        return time.monotonic_ns() // 1000

    @staticmethod
    def wall_time_us() -> int:
        """
        Returns wall clock time in microseconds (int64).
        Used ONLY for human-readable logging, NOT for ordering.
        """
        return int(time.time() * MICROSECONDS)

    @staticmethod
    def calculate_drift(exchange_ts: int, local_ts: int) -> int:
        """
        Drift = Exchange TS - Local TS
        """
        return exchange_ts - local_ts
