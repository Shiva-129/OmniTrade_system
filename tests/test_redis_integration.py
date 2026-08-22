"""
GATE 0: Integration tests against the REAL Redis instance.

Verifies the application's existing Redis-backed components
(ObserverState, StateController) work against a live server at
localhost:6379 -- the project's documented configuration (GUIDE.txt).

Rules honored by this module:
- No fake/mock Redis fallback.
- Tests FAIL loudly if Redis is unreachable; they are NEVER skipped.
- Uses DB 15 exclusively and flushes only that DB, leaving production DB 0 untouched.
"""
import pytest
import redis

from src.core.state import ObserverState
from src.gatekeeper.state_controller import StateController
from src.core.types import ExecutionReport, OrderSide

TEST_REDIS_URL = "redis://localhost:6379/15"


@pytest.fixture()
def live_redis():
    client = redis.from_url(TEST_REDIS_URL, decode_responses=True)
    try:
        client.ping()
    except Exception as e:
        pytest.fail(
            f"GATE 0 requires a running Redis at localhost:6379 "
            f"(start with: wsl -d kali-linux -e bash -c 'redis-server --daemonize yes'). "
            f"Underlying error: {e}"
        )
    client.flushdb()  # isolated scratch DB for deterministic assertions
    yield client
    client.flushdb()
    client.close()


class TestObserverStateLiveRedis:
    def test_system_status_round_trip(self, live_redis):
        state = ObserverState(redis_url=TEST_REDIS_URL)
        state.set_system_status("CONNECTED")
        assert state.get_system_status() == "CONNECTED"
        state.set_system_status("HALT")
        assert state.get_system_status() == "HALT"

    def test_last_update_is_written_on_status_change(self, live_redis):
        state = ObserverState(redis_url=TEST_REDIS_URL)
        state.set_system_status("CONNECTED")
        raw = live_redis.get("observer:last_update")
        assert raw is not None and int(raw) > 0

    def test_gap_counter_increments(self, live_redis):
        state = ObserverState(redis_url=TEST_REDIS_URL)
        before = state.get_gap_count()
        state.record_gap()
        assert state.get_gap_count() == before + 1


class TestStateControllerLiveRedis:
    def _report(self, cloid="t-1", side=OrderSide.BUY, qty=0.5):
        return ExecutionReport(
            client_order_id=cloid,
            exchange_order_id=f"x-{cloid}",
            symbol="BTCUSDT",
            side=side,
            status="FILLED",
            filled_quantity=qty,
            last_filled_price=50000.0,
            remaining_quantity=0.0,
            timestamp=1000,
        )

    def test_buy_fill_creates_long_position(self, live_redis):
        controller = StateController(TEST_REDIS_URL)
        controller.process_execution_report(self._report())
        assert controller.get_position("BTCUSDT") == 0.5

    def test_sell_fill_creates_short_position(self, live_redis):
        controller = StateController(TEST_REDIS_URL)
        controller.process_execution_report(
            self._report(side=OrderSide.SELL, qty=0.25)
        )
        assert controller.get_position("BTCUSDT") == -0.25

    def test_partial_fills_accumulate(self, live_redis):
        controller = StateController(TEST_REDIS_URL)
        controller.process_execution_report(self._report(cloid="p1", qty=0.5))
        partial = self._report(cloid="p2", qty=0.5)
        partial.status = "PARTIAL_FILL"
        controller.process_execution_report(partial)
        assert controller.get_position("BTCUSDT") == 1.0

    def test_order_state_persisted_by_cloid(self, live_redis):
        controller = StateController(TEST_REDIS_URL)
        controller.process_execution_report(self._report(cloid="persist-1"))
        raw = live_redis.get("gk:orders:persist-1")
        assert raw is not None
        assert '"client_order_id": "persist-1"' in raw or '"client_order_id":"persist-1"' in raw

    def test_unknown_symbol_position_defaults_zero(self, live_redis):
        controller = StateController(TEST_REDIS_URL)
        assert controller.get_position("ETHUSDT") == 0.0
