"""
Transaction cost model (Phase 8). All rates are canonical Decimals.

- maker_fee / taker_fee: fraction of notional (e.g. "0.001" = 10 bps).
  Next-open marketable fills are TAKER fills by definition.
- slippage_pct: ADVERSE fraction applied to the reference price:
  buys fill at ref*(1+s), sells at ref*(1-s). Never favorable.
- min_order_qty: intents below this are counted as rejected_small.
"""
from pydantic import BaseModel

from src.core.money import Decimal, to_decimal


class CostModel(BaseModel):
    model_config = {"frozen": True}

    maker_fee: Decimal = Decimal("0")
    taker_fee: Decimal = Decimal("0")
    slippage_pct: Decimal = Decimal("0")
    min_order_qty: Decimal = Decimal("0")

    @classmethod
    def zero(cls) -> "CostModel":
        return cls()

    def fee(self, notional: Decimal, side_is_maker: bool = False) -> Decimal:
        rate = self.maker_fee if side_is_maker else self.taker_fee
        return notional * rate

    def fill_price(self, side: str, ref: Decimal) -> Decimal:
        """Adverse-only slippage. side in {BUY, SELL}."""
        s = self.slippage_pct
        if side == "BUY":
            return ref * (Decimal("1") + s)
        if side == "SELL":
            return ref * (Decimal("1") - s)
        raise ValueError(f"unknown side {side!r}")
