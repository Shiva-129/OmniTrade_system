"""Bollinger Mean Reversion (Phase 14). Price outside bands reverts."""
import math
from typing import Any, Dict, Optional
from pydantic import model_validator
from .base import BaseStrategy, StrategyConfig
from ..core.money import ZERO
from ..core.types import MarketEvent, OrderIntent, OrderSide

class BollingerConfig(StrategyConfig):
    period: int = 20
    num_std: float = 2.0
    cooldown_events: int = 0
    @model_validator(mode="after")
    def _v(self):
        if self.period < 2: raise ValueError("period must be >=2")
        if self.num_std <= 0: raise ValueError("num_std must be >0")
        if self.trade_size <= ZERO: raise ValueError("trade_size positive")
        return self

class BollingerReversionStrategy(BaseStrategy):
    FLAT, LONG, SHORT = "FLAT", "LONG", "SHORT"
    @classmethod
    def expected_config(cls): return BollingerConfig
    def initial_state(self): return {"history": [], "position": self.FLAT}
    def on_market_event(self, event: MarketEvent) -> Optional[OrderIntent]:
        ctx = self._on_symbol_event(event)
        if ctx is None: return None
        price_f = float(ctx["price"]); ts = ctx["ts"]
        hist = self.state["history"]
        if len(hist) < self.config.period:
            hist.append(price_f); return None
        # bands over PRIOR window
        window = hist[-self.config.period:]
        m = sum(window) / len(window)
        var = sum((x-m)**2 for x in window) / len(window)
        std = math.sqrt(max(0, var))
        upper = m + self.config.num_std * std
        lower = m - self.config.num_std * std
        pos = self.state["position"]
        intent = None
        if price_f < lower and pos == self.FLAT:
            intent = self.build_intent(OrderSide.BUY, ts, ref_price=ctx["price"])
            self.state["position"] = self.LONG
        elif price_f > upper and pos == self.FLAT:
            intent = self.build_intent(OrderSide.SELL, ts, ref_price=ctx["price"])
            self.state["position"] = self.SHORT
        elif pos == self.LONG and price_f > m:
            intent = self.build_intent(OrderSide.SELL, ts, ref_price=ctx["price"])
            self.state["position"] = self.FLAT
        elif pos == self.SHORT and price_f < m:
            intent = self.build_intent(OrderSide.BUY, ts, ref_price=ctx["price"])
            self.state["position"] = self.FLAT
        hist.append(price_f)
        if len(hist) > self.config.period * 2:
            hist.pop(0)
        return intent
