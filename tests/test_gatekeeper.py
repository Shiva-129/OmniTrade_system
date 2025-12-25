import unittest
from unittest.mock import MagicMock
from src.gatekeeper.command_registry import CommandRegistry
from src.gatekeeper.rate_limiter import TokenBucket
from src.core.types import OrderIntent, OrderSide, OrderType

class TestGatekeeper(unittest.TestCase):
    def test_idempotency(self):
        registry = CommandRegistry()
        intent = OrderIntent(
            client_order_id="123",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0,
            timestamp=1000
        )
        self.assertTrue(registry.register(intent))
        self.assertFalse(registry.register(intent)) # Duplicate

    def test_rate_limiter(self):
        # 10 tokens, 10/sec
        bucket = TokenBucket(rate=10, capacity=10)
        
        # Consume all
        self.assertTrue(bucket.consume(10.0))
        # Next should fail immediately
        self.assertFalse(bucket.consume(1.0))

if __name__ == '__main__':
    unittest.main()
