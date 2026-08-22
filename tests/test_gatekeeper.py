"""
Gatekeeper unit tests: idempotency registry + deterministic token bucket.
Migrated from unittest to pytest (Phase 0). Logic unchanged.
"""
from src.gatekeeper.command_registry import CommandRegistry
from src.gatekeeper.rate_limiter import TokenBucket
from src.core.types import OrderIntent, OrderSide, OrderType


def _make_intent(client_order_id="123"):
    return OrderIntent(
        client_order_id=client_order_id,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1.0,
        price=50000.0,
        timestamp=1000,
    )


class TestCommandRegistry:
    def test_register_new_returns_true(self):
        registry = CommandRegistry()
        assert registry.register(_make_intent()) is True

    def test_duplicate_rejected(self):
        registry = CommandRegistry()
        intent = _make_intent()
        registry.register(intent)
        assert registry.register(intent) is False

    def test_get_returns_registered_intent(self):
        registry = CommandRegistry()
        intent = _make_intent("abc")
        registry.register(intent)
        assert registry.get("abc") is intent

    def test_get_unknown_returns_none(self):
        registry = CommandRegistry()
        assert registry.get("missing") is None


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
