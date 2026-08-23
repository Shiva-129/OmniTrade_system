"""P3 regression: prove single-symbol accounting unchanged before extension."""
from src.core.money import to_decimal
from src.core.portfolio import Portfolio
from src.core.types import ExecutionReport, OrderSide


def rpt(cloid, symbol, side, qty, price, fee="0"):
    return ExecutionReport(
        client_order_id=cloid, exchange_order_id=f"x-{cloid}",
        symbol=symbol, side=side, status="FILLED",
        filled_quantity=to_decimal(qty), last_filled_price=to_decimal(price),
        remaining_quantity=to_decimal("0"), timestamp=1, fee=to_decimal(fee),
    )


def test_single_symbol_unchanged():
    pf = Portfolio(starting_cash="10000")
    pf.apply_report(rpt("a", "BTCUSDT", OrderSide.BUY, "1", "100"))
    snap = pf.snapshot()
    # hash must be deterministic and match known single-symbol behavior
    assert snap["positions"]["BTCUSDT"]["quantity"] == "1"
    assert pf.cash == to_decimal("9900")
    assert pf.fees_paid == to_decimal("0")


def test_interleaved_symbols_isolated():
    pf = Portfolio(starting_cash="10000")
    pf.apply_report(rpt("b1", "BTCUSDT", OrderSide.BUY, "1", "100", fee="0.10"))
    pf.apply_report(rpt("e1", "ETHUSDT", OrderSide.SELL, "2", "200", fee="0.20"))
    # positions isolated
    assert pf.positions["BTCUSDT"].quantity == to_decimal("1")
    assert pf.positions["ETHUSDT"].quantity == to_decimal("-2")
    # fees aggregate, not per-symbol lost
    assert pf.fees_paid == to_decimal("0.30")
    # cash: 10000 -100 -0.10 +400 -0.20 = 10299.70
    assert pf.cash == to_decimal("10299.70")
    # marks isolated
    pf.mark_price("BTCUSDT", to_decimal("110"), ts_us=1)
    pf.mark_price("ETHUSDT", to_decimal("190"), ts_us=1)
    # unrealized BTC: (110-100)*1=10, ETH short: (200-190)*2=20
    assert pf.unrealized_pnl("BTCUSDT", now_us=1) == to_decimal("10")
    assert pf.unrealized_pnl("ETHUSDT", now_us=1) == to_decimal("20")


def test_snapshot_roundtrip_still_deterministic():
    pf = Portfolio(starting_cash="5000")
    pf.apply_report(rpt("x", "BTCUSDT", OrderSide.BUY, "0.5", "1000"))
    snap1 = pf.snapshot()
    pf2 = Portfolio.from_snapshot(snap1)
    assert pf2.snapshot() == snap1
