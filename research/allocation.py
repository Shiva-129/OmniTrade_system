"""
Allocation Policies (Phase 15 R3) -- research-layer only.

Determines per-symbol trade_size overrides. Output feeds into
strategy trade_size; NEVER bypasses Strategy -> Risk -> Gatekeeper ->
Broker. All arithmetic is Decimal.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional

from src.core.money import ZERO, to_decimal, dec_to_str


class AllocationError(Exception):
    """Raised when allocation would be unsafe or invalid."""


class AllocationPolicy(ABC):
    @abstractmethod
    def allocate(self, symbols: List[str], equity: Decimal,
                 prices: Dict[str, Decimal],
                 returns_history: Optional[Dict[str, List[float]]] = None,
                 ) -> Dict[str, str]:
        """Returns {symbol: trade_size_string}."""


class EqualWeight(AllocationPolicy):
    """
    Splits equity equally across N symbols.
    trade_size = floor(equity / (N * price), step=1) for each symbol.
    Rejects if any computed size is zero (notional too small).
    """

    def __init__(self, min_trade_size: Decimal = Decimal("0.00001")):
        self.min_trade_size = min_trade_size

    def allocate(self, symbols: List[str], equity: Decimal,
                 prices: Dict[str, Decimal],
                 returns_history=None) -> Dict[str, str]:
        if not symbols:
            raise AllocationError("no symbols to allocate")
        n = len(symbols)
        per_symbol = equity / to_decimal(n)
        out: Dict[str, str] = {}
        for sym in symbols:
            price = prices.get(sym)
            if price is None or price <= ZERO:
                raise AllocationError(f"missing or invalid price for {sym}")
            qty = (per_symbol / price).quantize(Decimal("0.00001"), rounding=ROUND_DOWN)
            if qty < self.min_trade_size:
                raise AllocationError(
                    f"allocated quantity {qty} for {sym} below min {self.min_trade_size}"
                )
            out[sym] = dec_to_str(qty)
        return out


class VolatilityTargeted(AllocationPolicy):
    """
    Allocates inversely proportional to realized volatility.
    weight_i = (1/vol_i) / sum(1/vol_j). Requires returns_history.
    Deterministic: same inputs => same outputs.
    """

    def __init__(self, lookback: int = 20,
                 min_trade_size: Decimal = Decimal("0.00001")):
        self.lookback = lookback
        self.min_trade_size = min_trade_size

    def _volatility(self, returns: List[float]) -> float:
        if len(returns) < 2:
            return 0.0
        m = sum(returns) / len(returns)
        var = sum((r - m) ** 2 for r in returns) / len(returns)
        return var ** 0.5

    def allocate(self, symbols: List[str], equity: Decimal,
                 prices: Dict[str, Decimal],
                 returns_history: Optional[Dict[str, List[float]]] = None) -> Dict[str, str]:
        if not symbols:
            raise AllocationError("no symbols")
        if not returns_history:
            raise AllocationError("VolatilityTargeted requires returns_history")
        inv_vols: Dict[str, float] = {}
        for sym in symbols:
            hist = returns_history.get(sym, [])
            window = hist[-self.lookback:] if len(hist) >= self.lookback else hist
            vol = self._volatility(window)
            inv_vols[sym] = 1.0 / vol if vol > 0 else 0.0
        total_inv = sum(inv_vols.values())
        if total_inv <= 0:
            raise AllocationError("all volatilities are zero; cannot allocate")

        out: Dict[str, str] = {}
        for sym in symbols:
            weight = inv_vols[sym] / total_inv
            price = prices.get(sym)
            if price is None or price <= ZERO:
                raise AllocationError(f"invalid price for {sym}")
            allocated = to_decimal(str(weight)) * equity
            qty = (allocated / price).quantize(Decimal("0.00001"), rounding=ROUND_DOWN)
            if qty < self.min_trade_size:
                raise AllocationError(f"qty {qty} for {sym} below minimum")
            out[sym] = dec_to_str(qty)
        return out
