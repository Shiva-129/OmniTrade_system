"""
Gatekeeper unit tests: idempotency registry (Redis-backed, Phase 4) +
deterministic token bucket. Registry tests use the REAL Redis via the
live_redis fixture -- restart-safety cannot be faked with mocks.
"""
from src.gatekeeper.command_registry import CommandRegistry
from src.gatekeeper.rate_limiter import TokenBucket
from src.core.types import OrderIntent, OrderSide, OrderType

from conftest import TEST_REDIS_URL


def _make_intent(client_order_id="123"):
    return OrderIntent(
        client_order_id=client_order_id,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity="1.0",
        price="50000.0",
        timestamp=1000,
    )


class TestTokenBucket:
    def test_consume_all_tokens_then_empty(self):
        # 10 tokens, refill 10/sec
        bucket = TokenBucket(rate=10, capacity=10)
        assert bucket.consume(10.0) is True
        assert bucket.consume(1.0) is False

    def test_partial_consumption_allowed(self):
        bucket = TokenBucket(rate=1, capacity=5)
        assert bucket.consume(3.0) is True
        assert bucket.consume(3.0) is False
        assert bucket.consume(2.0) is True


class TestCommandRegistryLive:
    def test_register_new_returns_true(self, live_redis):
        registry = CommandRegistry(TEST_REDIS_URL)
        assert registry.register(_make_intent()) is True

    def test_duplicate_rejected(self, live_redis):
        registry = CommandRegistry(TEST_REDIS_URL)
        intent = _make_intent()
        registry.register(intent)
        assert registry.register(intent) is False

    def test_restart_safety_new_instance_sees_prior_intents(self, live_redis):
        """
        THE Phase 4 regression test: a fresh process (new registry object,
        same Redis) must still recognize intents registered before the
        'crash'. The old in-memory registry failed exactly this.
        """
        first_process = CommandRegistry(TEST_REDIS_URL)
        intent = _make_intent("survive-restart-1")
        assert first_process.register(intent) is True

        restarted_process = CommandRegistry(TEST_REDIS_URL)  # e.g. after crash+restart
        assert restarted_process.register(intent) is False

    def test_get_returns_registered_intent_after_restart(self, live_redis):
        writer = CommandRegistry(TEST_REDIS_URL)
        intent = _make_intent("fetch-me")
        writer.register(intent)

        reader = CommandRegistry(TEST_REDIS_URL)
        loaded = reader.get("fetch-me")
        assert loaded == intent

    def test_get_unknown_returns_none(self, live_redis):
        registry = CommandRegistry(TEST_REDIS_URL)
        assert registry.get("missing") is None
