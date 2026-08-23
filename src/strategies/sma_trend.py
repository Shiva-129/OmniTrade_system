"""SMA Trend following (Phase 14). Cross of fast/slow SMA on prior closes."""
from typing import Any, Dict, Optional
from pydantic import model_validator
from .base import BaseStrategy, StrategyConfig, sma
from ..core.money import ZERO
from ..core.types import MarketEvent, OrderIntent, OrderSide

class SmaTrendConfig(StrategyConfig):
    fast_period: int
    slow_period: int
    allow_short: bool = False
    cooldown_events: int = 0
    @model_validator(mode="after")
    def _v(self):
        if self.fast_period < 1 or self.slow_period < 1:
            raise ValueError("periods must be >=1")
        if self.fast_period >= self.slow_period:
            raise ValueError("fast must be < slow")
        if self.trade_size <= ZERO:
            raise ValueError("trade_size must be positive")
        return self

class SmaTrendStrategy(BaseStrategy):
    FLAT, LONG, SHORT = "FLAT", "LONG", "SHORT"
    @classmethod
    def expected_config(cls): return SmaTrendConfig
    def initial_state(self): return {"history": [], "position": self.FLAT}
    def on_market_event(self, event: MarketEvent) -> Optional[OrderIntent]:
        ctx = self._on_symbol_event(event)
        if ctx is None: return None
        price_f = float(ctx["price"]); ts = ctx["ts"]
        hist = self.state["history"]
        # need slow_period prior closes to compute both SMAs (excluding current)
        if len(hist) < self.config.slow_period:
            hist.append(price_f); return None
        # SMAs over PRIOR window
        fast_vals = hist[-self.config.fast_period:]
        slow_vals = hist[-self.config.slow_period:]
        fast_prev = sum(fast_vals) / len(fast_vals)
        slow_prev = sum(slow_vals) / len(slow_vals)
        # current SMAs including current price for decision (prior-window rule: decision price is current, SMAs are prior)
        # Actually signal on current price vs prior SMAs: if price > slow_prev => up
        # Use fast/slow of prior window vs current price position
        # Simplified: if prior fast <= prior slow and price > slow_prev => BUY
        # For determinism, compute SMAs of prior window and compare to price
        pos = self.state["position"]
        intent = None
        if fast_prev <= slow_prev and price_f > slow_prev and pos != self.LONG:
            intent = self.build_intent(OrderSide.BUY, ts, ref_price=ctx["price"])
            self.state["position"] = self.LONG
        elif fast_prev >= slow_prev and price_f < slow_prev and pos != self.SHORT:
            if pos == self.LONG:
                intent = self.build_intent(OrderSide.SELL, ts, ref_price=ctx["price"])
                self.state["position"] = self.FLAT
            elif self.config.allow_short:
                intent = self.build_intent(OrderSide.SELL, ts, ref_price=ctx["price"])
                self.state["position"] = self.SHORT
        hist.append(price_f)
        if len(hist) > self.config.slow_period * 2:
            hist.pop(0)
        return intent
