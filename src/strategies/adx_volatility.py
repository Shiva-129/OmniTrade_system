"""ADX + ATR volatility/regime filter (Phase 14)."""
from typing import Any, Dict, Optional
from pydantic import model_validator
from .base import BaseStrategy, StrategyConfig
from ..core.money import ZERO
from ..core.types import MarketEvent, OrderIntent, OrderSide

class AdxVolatilityConfig(StrategyConfig):
    adx_period: int = 14
    atr_period: int = 14
    adx_threshold: float = 25.0
    cooldown_events: int = 0
    @model_validator(mode="after")
    def _v(self):
        if self.adx_period < 2 or self.atr_period < 2: raise ValueError("periods >=2")
        if self.trade_size <= ZERO: raise ValueError("trade_size positive")
        return self

class AdxVolatilityStrategy(BaseStrategy):
    FLAT, LONG = "FLAT", "LONG"
    @classmethod
    def expected_config(cls): return AdxVolatilityConfig
    def initial_state(self): return {"bars": [], "position": self.FLAT}
    def on_market_event(self, event: MarketEvent) -> Optional[OrderIntent]:
        ctx = self._on_symbol_event(event)
        if ctx is None: return None
        # need OHLC for ADX/ATR; fallback to price for high/low
        payload = event.packet.payload or {}
        try:
            high = float(payload.get("high", ctx["price"]))
            low = float(payload.get("low", ctx["price"]))
            close = float(ctx["price"])
        except: return None
        ts = ctx["ts"]
        bars = self.state["bars"]
        bars.append({"high": high, "low": low, "close": close})
        if len(bars) > self.config.adx_period * 3:
            bars.pop(0)
        from src.indicators.adx import adx as adx_fn
        from src.indicators.atr import atr as atr_fn
        adx_vals = adx_fn(bars, self.config.adx_period)
        atr_vals = atr_fn(bars, self.config.atr_period)
        cur_adx = adx_vals[-1] if adx_vals else None
        cur_atr = atr_vals[-1] if atr_vals else None
        if cur_adx is None or cur_atr is None:
            return None
        # regime: strong trend when ADX > threshold and ATR not collapsing
        # entry long when trending up (close > prior close) and regime true
        if len(bars) < 2: return None
        prev_close = float(bars[-2]["close"])
        pos = self.state["position"]
        trending_up = close > prev_close and cur_adx > self.config.adx_threshold
        if trending_up and pos == self.FLAT:
            intent = self.build_intent(OrderSide.BUY, ts, ref_price=ctx["price"])
            self.state["position"] = self.LONG
            return intent
        if pos == self.LONG and cur_adx < self.config.adx_threshold:
            intent = self.build_intent(OrderSide.SELL, ts, ref_price=ctx["price"])
            self.state["position"] = self.FLAT
            return intent
        return None
