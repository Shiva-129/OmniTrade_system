"""
Phase 4: core contract tests.
Contracts are frozen (immutable) and serialize Decimals as fixed-point
strings with EXACT round-trips -- the wire format is part of the contract.
"""
import pytest
from pydantic import ValidationError

from src.core.types import (
    Packet,
    MarketEvent,
    OrderIntent,
    OrderSide,
    OrderType,
    ExecutionReport,
)


def _intent(**overrides):
    base = dict(
        client_order_id="o-1",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity="0.01000000",
        price="117234.52000000",
        timestamp=1000,
    )
    base.update(overrides)
    return OrderIntent(**base)


def _report(**overrides):
    base = dict(
        client_order_id="o-1",
        exchange_order_id="x-9",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        status="FILLED",
        filled_quantity="0.01000000",
        last_filled_price="117234.52000000",
        remaining_quantity="0",
        timestamp=2000,
    )
    base.update(overrides)
    return ExecutionReport(**base)


class TestDecimalContracts:
    def test_order_intent_coerces_numeric_strings_to_decimal(self):
        intent = _intent()
        from decimal import Decimal
        assert intent.quantity == Decimal("0.01000000")
        assert intent.price == Decimal("117234.52000000")

    def test_order_intent_json_round_trip_exact(self):
        intent = _intent()
        revived = OrderIntent.model_validate_json(intent.model_dump_json())
        assert revived == intent
        assert '"quantity":"0.01000000"' in intent.model_dump_json()

    def test_execution_report_json_round_trip_exact(self):
        report = _report()
        revived = ExecutionReport.model_validate_json(report.model_dump_json())
        assert revived == report
        assert '"filled_quantity":"0.01000000"' in report.model_dump_json()

    def test_execution_report_partial_fill_fields(self):
        r = _report(status="PARTIAL_FILL", filled_quantity="0.004", remaining_quantity="0.006")
        assert str(r.filled_quantity) == "0.004"
        assert str(r.remaining_quantity) == "0.006"


class TestImmutability:
    def test_order_intent_frozen(self):
        intent = _intent()
        with pytest.raises(ValidationError):
            intent.quantity = "999"

    def test_execution_report_frozen(self):
        report = _report()
        with pytest.raises(ValidationError):
            report.status = "REJECTED"

    def test_packet_frozen(self):
        packet = Packet(
            exchange_ts=1, local_arrival_ts=2, drift_us=-1,
            source="s", topic="t", payload={}, sequence_id=None,
        )
        with pytest.raises(ValidationError):
            packet.drift_us = 5


class TestEventContracts:
    def test_market_event_wraps_packet_and_is_frozen(self):
        packet = Packet(
            exchange_ts=1, local_arrival_ts=2, drift_us=-1,
            source="binance_ccxt", topic="BTC/USDT", payload={"id": 7}, sequence_id=7,
        )
        event = MarketEvent(packet=packet)
        assert event.packet.sequence_id == 7
        with pytest.raises(ValidationError):
            event.packet = packet
