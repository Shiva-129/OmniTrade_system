import time
from typing import Final

# Enforce int64 microsecond precision
MICROSECONDS: Final[int] = 1_000_000

class Clock:
    """
    Authoritative local clock source.
    Uses time.time_ns() for epoch-based timestamps to align with external exchange clocks.
    """
    
    @staticmethod
    def now_us() -> int:
        """
        Returns current monotonic time in microseconds (int64).
        WARNING: Do not use for drift calculation against epoch timestamps.
        """
        # monotonic_ns returns nanoseconds. Divide by 1000 to get microseconds.
        return time.monotonic_ns() // 1000

    @staticmethod
    def now_epoch_us() -> int:
        """
        Returns current epoch time in microseconds (int64).
        Use this for drift calculation against exchange timestamps.
        """
        return time.time_ns() // 1000

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
        Drift = Exchange TS - Local TS.
        Both timestamps MUST be in the same time domain (Epoch).
        """
        return exchange_ts - local_ts
