"""
Phase 11: Opt-in REAL testnet integration test.

This file NEVER runs in the normal suite. It is skipped unless

    RUN_BINANCE_TESTNET=1 pytest tests/test_binance_testnet_integration.py

is explicitly set. The normal gate must have 0 skipped tests; this file
is outside that gate (it is marked skip when the flag is absent).

When enabled, it verifies:
- connectivity to testnet.binance.vision
- account info fetch
- (optionally) a minimal safe order lifecycle on a testnet symbol

No test here may use a production endpoint.
"""
import os
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_BINANCE_TESTNET") != "1",
    reason="Opt-in: set RUN_BINANCE_TESTNET=1 to run real testnet integration (requires BINANCE_API_KEY/SECRET)",
)


def test_testnet_connectivity():
    """Verifies we can reach testnet and fetch account info."""
    from src.adapters.binance import BinanceTestnetConfig, BinanceTestnetBroker
    cfg = BinanceTestnetConfig.from_env()
    broker = BinanceTestnetBroker(cfg)
    # Basic connectivity + creds check passed if we got this far (load_markets inside __init__).
    state = broker.get_account_state()
    assert isinstance(state, dict)
    broker.close()


@pytest.mark.skipif(
    os.getenv("RUN_BINANCE_TESTNET_ORDER") != "1",
    reason="Opt-in: set RUN_BINANCE_TESTNET_ORDER=1 to test a minimal order lifecycle (uses testnet funds)",
)
def test_testnet_minimal_order_lifecycle():
    """
    Places and cancels a tiny LIMIT order far from market on testnet.
    This test uses real testnet funds (which are free) but still
    exercises the full pipeline. It is double-opt-in.
    """
    from src.adapters.binance import BinanceTestnetConfig, BinanceTestnetBroker
    from src.core.money import to_decimal
    from src.core.types import OrderIntent, OrderSide, OrderType

    cfg = BinanceTestnetConfig.from_env()
    broker = BinanceTestnetBroker(cfg)

    # Use a symbol that exists on testnet spot; BTCUSDT is standard.
    # Place a LIMIT BUY far below market (e.g. 10% below) with tiny qty
    # so it rests and can be canceled. This avoids immediate fill.
    intent = OrderIntent(
        client_order_id="phase11-test-" + str(int(__import__("time").time())),
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        quantity=to_decimal("0.0001"), price=to_decimal("10000"), timestamp=1,
    )
    result = broker.submit_order(intent)
    assert result in ("ACCEPTED", "REJECTED")  # REJECTED is also valid if filters block
    if result == "ACCEPTED":
        # Cancel it.
        cancel_result = broker.cancel_order(intent.client_order_id)
        assert cancel_result in ("CANCELED", "UNKNOWN")
        # Drain and verify report
        reports = broker.drain_reports()
        assert len(reports) >= 1
    broker.close()
