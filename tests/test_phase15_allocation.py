"""Phase 15 R3: AllocationPolicy tests."""
import pytest

from research.allocation import AllocationError, EqualWeight, VolatilityTargeted
from src.core.money import to_decimal


class TestEqualWeight:
    def test_two_symbols_equal(self):
        pol = EqualWeight()
        out = pol.allocate(
            ["BTCUSDT", "ETHUSDT"], to_decimal("10000"),
            {"BTCUSDT": to_decimal("50000"), "ETHUSDT": to_decimal("3000")})
        assert len(out) == 2
        # Each gets 5000 notional; BTC: 5000/50000=0.1, ETH: 5000/3000≈1.66667
        assert float(out["BTCUSDT"]) == pytest.approx(0.1)
        assert float(out["ETHUSDT"]) == pytest.approx(1.66667, rel=1e-4)

    def test_deterministic(self):
        pol = EqualWeight()
        args = (["A", "B"], to_decimal("10000"), {"A": to_decimal("10"), "B": to_decimal("20")})
        r1 = pol.allocate(*args)
        r2 = pol.allocate(*args)
        assert r1 == r2

    def test_zero_price_rejected(self):
        with pytest.raises(AllocationError, match="price"):
            EqualWeight().allocate(["A"], to_decimal("1000"), {"A": to_decimal("0")})

    def test_empty_symbols_rejected(self):
        with pytest.raises(AllocationError):
            EqualWeight().allocate([], to_decimal("1000"), {})

    def test_too_small_rejected(self):
        pol = EqualWeight(min_trade_size=to_decimal("100"))
        with pytest.raises(AllocationError, match="below min"):
            pol.allocate(["A"], to_decimal("10"), {"A": to_decimal("1000")})


class TestVolatilityTargeted:
    def test_low_vol_gets_larger_allocation(self):
        pol = VolatilityTargeted(lookback=5)
        returns = {
            "STABLE": [0.001, -0.001, 0.002, -0.001, 0.001],
            "VOLATILE": [0.1, -0.1, 0.15, -0.12, 0.08],
        }
        out = pol.allocate(
            ["STABLE", "VOLATILE"], to_decimal("10000"),
            {"STABLE": to_decimal("100"), "VOLATILE": to_decimal("100")},
            returns)

        assert float(out["STABLE"]) > float(out["VOLATILE"])

    def test_deterministic(self):
        pol = VolatilityTargeted(lookback=3)
        rets = {"A": [0.01, -0.02, 0.01], "B": [0.05, -0.03, 0.02]}
        args = (["A", "B"], to_decimal("10000"),
                {"A": to_decimal("100"), "B": to_decimal("50")}, rets)
        r1 = pol.allocate(args[0], args[1], args[2], rets)
        r2 = pol.allocate(args[0], args[1], args[2], rets)
        assert r1 == r2

    def test_no_returns_history_rejected(self):
        with pytest.raises(AllocationError, match="returns_history"):
            VolatilityTargeted().allocate(["A"], to_decimal("100"), {"A": to_decimal("10")})

    def test_all_zero_vol_rejected(self):
        pol = VolatilityTargeted()
        with pytest.raises(AllocationError, match="zero"):
            pol.allocate(["A", "B"], to_decimal("1000"),
                         {"A": to_decimal("10"), "B": to_decimal("20")},
                         {"A": [0.0]*5, "B": [0.0]*5})
