"""
EMA + RSI Filter — V2 example of deterministic strategy modification.

Baseline: EMA crossover (fast/slow)
Candidate: EMA crossover + RSI filter (rsi_period, rsi_threshold)

Deterministic, no future data, warm-up handled via buffers.
Reuses same execution conventions as ema_crossover.
"""
from typing import Any, Dict, Optional

from pydantic import model_validator

from .base import BaseStrategy, StrategyConfig, ema_update, sma
from ..core.money import Decimal, ZERO
from ..core.types import MarketEvent, OrderIntent, OrderSide

# Reuse RSI calculation from rsi_momentum but causal
def _rsi(prices: list[float], period: int) -> Optional[float]:
    if len(prices) <= period:
        return None
    gains = []
    losses = []
    for i in range(1, len(prices)):
        d = prices[i] - prices[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


class EmaRsiConfig(StrategyConfig):
    fast_period: int
    slow_period: int
    rsi_period: int = 14
    rsi_buy_threshold: float = 70.0  # filter BUY if RSI > threshold (overbought)
    rsi_sell_threshold: float = 30.0  # filter SELL if RSI < threshold (oversold)
    allow_short: bool = False
    cooldown_events: int = 0

    @model_validator(mode="after")
    def _validate(self):
        if self.fast_period < 1 or self.slow_period < 1 or self.rsi_period < 2:
            raise ValueError("periods must be >=1, rsi >=2")
        if self.fast_period >= self.slow_period:
            raise ValueError("fast must be < slow")
        if self.trade_size <= ZERO:
            raise ValueError("trade_size positive")
        if not (0 < self.rsi_buy_threshold <= 100 and 0 <= self.rsi_sell_threshold < 100):
            raise ValueError("rsi thresholds 0-100")
        return self


class EmaRsiFilteredStrategy(BaseStrategy):
    FLAT, LONG, SHORT = "FLAT", "LONG", "SHORT"

    @classmethod
    def expected_config(cls) -> type:
        return EmaRsiConfig

    def initial_state(self) -> Dict[str, Any]:
        return {
            "fast_buf": [], "slow_buf": [], "ema_fast": None, "ema_slow": None,
            "prev_fast": None, "prev_slow": None,
            "rsi_prices": [], "position": self.FLAT,
        }

    def on_market_event(self, event: MarketEvent) -> Optional[OrderIntent]:
        ctx = self._on_symbol_event(event)
        if ctx is None:
            return None
        price = float(ctx["price"])
        ts = ctx["ts"]
        st = self.state
        # warm-up EMAs
        if st["ema_fast"] is None:
            st["fast_buf"].append(price)
            if len(st["fast_buf"]) == self.config.fast_period:
                st["ema_fast"] = sma(st["fast_buf"])
                st["fast_buf"] = []
        if st["ema_slow"] is None:
            st["slow_buf"].append(price)
            if len(st["slow_buf"]) == self.config.slow_period:
                st["ema_slow"] = sma(st["slow_buf"])
                st["slow_buf"] = []
        if st["ema_fast"] is None or st["ema_slow"] is None:
            return None
        # RSI prices buffer (causal, includes current price after EMAs seeded)
        st["rsi_prices"].append(price)
        # keep only needed window +1
        if len(st["rsi_prices"]) > self.config.rsi_period + 5:
            st["rsi_prices"] = st["rsi_prices"][-self.config.rsi_period-5:]
        st["prev_fast"], st["prev_slow"] = st["ema_fast"], st["ema_slow"]
        st["ema_fast"] = ema_update(st["ema_fast"], price, self.config.fast_period)
        st["ema_slow"] = ema_update(st["ema_slow"], price, self.config.slow_period)

        crossed_up = st["prev_fast"] <= st["prev_slow"] and st["ema_fast"] > st["ema_slow"]
        crossed_down = st["prev_fast"] >= st["prev_slow"] and st["ema_fast"] < st["ema_slow"]

        # RSI filter (causal, uses prices up to current)
        rsi = _rsi(st["rsi_prices"], self.config.rsi_period)
        if rsi is not None:
            if crossed_up and rsi > self.config.rsi_buy_threshold:
                return None  # overbought filter
            if crossed_down and rsi < self.config.rsi_sell_threshold:
                return None  # oversold filter

        if crossed_up and st["position"] != self.LONG:
            intent = self.build_intent(OrderSide.BUY, ts, ref_price=ctx["price"])
            st["position"] = self.LONG
            return intent
        elif crossed_down:
            if st["position"] == self.LONG:
                intent = self.build_intent(OrderSide.SELL, ts, ref_price=ctx["price"])
                st["position"] = self.FLAT if not self.config.allow_short else self.SHORT
                return intent
            elif self.config.allow_short and st["position"] != self.SHORT:
                intent = self.build_intent(OrderSide.SELL, ts, ref_price=ctx["price"])
                st["position"] = self.SHORT
                return intent
        return None
