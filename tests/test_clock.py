"""
Clock contract tests: int64 microsecond precision, monotonic ordering, drift math.
Migrated from unittest to pytest (Phase 0). Logic unchanged.
"""
import time

from src.core.clock import Clock


def test_now_us_is_int_and_monotonic():
    t1 = Clock.now_us()
    time.sleep(0.001)
    t2 = Clock.now_us()
    assert isinstance(t1, int)
    assert t2 > t1


def test_now_epoch_us_is_int():
    ts = Clock.now_epoch_us()
    assert isinstance(ts, int)
    assert ts > 0


def test_calculate_drift():
    # Fake exchange drift
    local = 1000
    exchange = 1500
    assert Clock.calculate_drift(exchange, local) == 500
