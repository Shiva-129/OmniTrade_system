"""RSI Momentum (Phase 14). RSI threshold momentum."""
from typing import Any, Dict, Optional
from pydantic import model_validator
from .base import BaseStrategy, StrategyConfig
from ..core.money import ZERO
from ..core.types import MarketEvent, OrderIntent, OrderSide

class RsiMomentumConfig(StrategyConfig):
    period: int = 14
    overbought: float = 70.0
    oversold: float = 30.0
    cooldown_events: int = 0
    @model_validator(mode="after")
    def _v(self):
        if self.period < 2: raise ValueError("period must be >=2")
        if not (0 < self.oversold < self.overbought < 100): raise ValueError("0 < oversold < overbought < 100")
        if self.trade_size <= ZERO: raise ValueError("trade_size must be positive")
        return self

class RsiMomentumStrategy(BaseStrategy):
    FLAT, LONG = "FLAT", "LONG"
    @classmethod
    def expected_config(cls): return RsiMomentumConfig
    def initial_state(self): return {"history": [], "position": self.FLAT}
    def on_market_event(self, event: MarketEvent) -> Optional[OrderIntent]:
        ctx = self._on_symbol_event(event)
        if ctx is None: return None
        price_f = float(ctx["price"]); ts = ctx["ts"]
        hist = self.state["history"]
        hist.append(price_f)
        if len(hist) <= self.config.period:
            return None
        # Wilder RSI over prior period (excluding current for no look-ahead: use hist[:-1])
        prior = hist[-(self.config.period+1):-1]
        curr = hist[-1]
        # compute RSI on prior+curr? Use standard: need period+1 values to get period changes
        # We have prior (period values) + curr => period+1 total, first valid at period+1
        # So require len(hist) > period
        if len(hist) <= self.config.period:
            return None
        # compute RSI for current price using prior window
        gains = losses = 0.0
        # Use Wilder initial SMA over prior changes
        for i in range(len(prior)-1 if len(prior)>1 else 0):
            d = prior[i+1]-prior[i] if i+1 < len(prior) else curr - prior[-1]
            if d > 0: gains += d
            else: losses += -d
        # include current delta
        d = curr - prior[-1]
        if d > 0: gains += d
        else: losses += -d
        # Simple SMA version for determinism (not Wilder) for test hand-computation
        avg_gain = gains / self.config.period
        avg_loss = losses / self.config.period
        rsi = 100 - (100 / (1 + (avg_gain / avg_loss))) if avg_loss != 0 else (100 if avg_gain !=0 else 50)
        pos = self.state["position"]
        if rsi > self.config.overbought and pos == self.FLAT:
            intent = self.build_intent(OrderSide.BUY, ts, ref_price=ctx["price"])
            self.state["position"] = self.LONG
            return intent
        if rsi < self.config.oversold and pos == self.LONG:
            intent = self.build_intent(OrderSide.SELL, ts, ref_price=ctx["price"])
            self.state["position"] = self.FLAT
            return intent
        return None
