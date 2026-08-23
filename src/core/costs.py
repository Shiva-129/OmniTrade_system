"""
Canonical transaction cost policy (Phase 10).

Single fee/slippage model for paper trading, research backtests and
future live adapters. Rates are canonical Decimals:

- maker_fee / taker_fee: fraction of executed notional (e.g. "0.001" = 10 bps).
  MARKET orders are taker; resting LIMIT fills are maker (documented).
- slippage_pct: ADVERSE-only fraction applied to marketable fills:
  BUY at ref*(1+s), SELL at ref*(1-s). Limit fills take no slippage.
- min_order_qty: intents below this size must be rejected upstream.
"""
from pydantic import BaseModel

from .money import Decimal


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

    def fill_price(self, side: str, ref: Decimal,
                   is_maker: bool = False) -> Decimal:
        """Adverse-only execution price for the given side."""
        if is_maker:
            return ref                       # passive limit: no slippage
        s = self.slippage_pct
        if side == "BUY":
            return ref * (Decimal("1") + s)
        if side == "SELL":
            return ref * (Decimal("1") - s)
        raise ValueError(f"unknown side {side!r}")
