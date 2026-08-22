"""
Canonical money/quantity arithmetic policy for OmniTrade.

PROJECT INVARIANT (Phase 4, locked):
- All prices, quantities, fees, PnL, cash and exposure use Decimal.
- Serialized as fixed-point strings ("0.01000000").
- NEVER convert Decimal -> float anywhere in the trading path.
- One canonical context for the whole project: prec=28, ROUND_HALF_EVEN.
  The simulator MUST use this same context so live path == replay path.
"""
import decimal
from decimal import Decimal
from typing import Final, Union

DecimalLike = Union[str, int, Decimal]

# Single source of truth. The simulator re-exports this context.
CANONICAL_CONTEXT: Final[decimal.Context] = decimal.Context(
    prec=28,
    rounding=decimal.ROUND_HALF_EVEN,  # banker's rounding, exchange-grade
    Emin=-999999,
    Emax=999999,
    capitals=1,
    clamp=0,
    flags=[],
    traps=[decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow],
)

ZERO: Final[Decimal] = Decimal("0")


def init_money_context() -> None:
    """
    Installs the canonical global decimal context.
    Call ONCE at every process entry point (engine, simulator, tests).
    Idempotent by design.
    """
    decimal.setcontext(CANONICAL_CONTEXT)


def to_decimal(value: DecimalLike) -> Decimal:
    """
    Converts a trusted string/int/Decimal into a Decimal under the
    canonical context. Floats are deliberately NOT accepted:
    passing a float means precision was already lost upstream.
    """
    if isinstance(value, float):
        raise TypeError(
            "Float is forbidden in the trading path; "
            f"pass a string or Decimal instead (got {value!r})"
        )
    return CANONICAL_CONTEXT.create_decimal(value)


def dec_to_str(value: Decimal) -> str:
    """Canonical wire/storage format: plain fixed-point string."""
    return format(value, "f")
