"""
Phase 4: canonical money/quantity policy tests.
The invariant: ONE Decimal context (prec=28, ROUND_HALF_EVEN) for the
whole project; floats forbidden in the trading path; fixed-point strings
on the wire.
"""
import decimal
from decimal import Decimal

import pytest

from src.core.money import (
    CANONICAL_CONTEXT,
    init_money_context,
    to_decimal,
    dec_to_str,
    ZERO,
)


class TestCanonicalContext:
    def test_precision_and_rounding_policy(self):
        assert CANONICAL_CONTEXT.prec == 28
        assert CANONICAL_CONTEXT.rounding == decimal.ROUND_HALF_EVEN

    def test_init_installs_global_context(self):
        init_money_context()
        ctx = decimal.getcontext()
        assert ctx.prec == 28
        assert ctx.rounding == decimal.ROUND_HALF_EVEN

    def test_half_even_rounding_in_arithmetic(self):
        init_money_context()
        # ROUND_HALF_EVEN: ties round to the even neighbour
        assert Decimal("2.5").quantize(Decimal("1")) == Decimal("2")
        assert Decimal("3.5").quantize(Decimal("1")) == Decimal("4")


class TestToDecimal:
    def test_accepts_string(self):
        assert to_decimal("0.1") == Decimal("0.1")

    def test_accepts_int(self):
        assert to_decimal(7) == Decimal("7")

    def test_accepts_decimal_passthrough(self):
        assert to_decimal(Decimal("12.34")) == Decimal("12.34")

    def test_rejects_float_loudly(self):
        # A float in the trading path means precision was already lost.
        with pytest.raises(TypeError):
            to_decimal(0.1)

    def test_zero_constant_is_exact(self):
        assert ZERO == Decimal("0")
        assert dec_to_str(ZERO) == "0"


class TestSerialization:
    def test_fixed_point_no_scientific_notation(self):
        assert dec_to_str(Decimal("0.01000000")) == "0.01000000"
        assert dec_to_str(Decimal("1E+3")) == "1000"
        assert dec_to_str(Decimal("1E-8")) == "0.00000001"

    def test_exact_decimal_addition(self):
        # The classic float failure case must be exact under the policy.
        assert to_decimal("0.1") + to_decimal("0.2") == to_decimal("0.3")
