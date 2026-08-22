"""
EMA Trend Crossover strategy (Phase 7).

Convention (documented, never mixed):
- EMAs are seeded with the SMA of the first `period` samples; crossover
  detection starts only once BOTH EMAs have a previous AND current value.
- A signal fires on the event whose close COMPLETES the cross
  ("signal-bar execution"). No future data is ever consulted.
- fast crosses ABOVE slow -> LONG signal
- fast crosses BELOW slow -> SHORT signal if allow_short else EXIT-long

State machine on own position view:
  long signal  + not already long  -> BUY trade_size
  short/exit   + currently long    -> SELL trade_size (close)
  short signal + allow_short + not already short -> SELL trade_size (open)
  everything else                  -> None
"""
from typing import Any, Dict, Optional

from pydantic import model_validator

from .base import BaseStrategy, StrategyConfig, ema_update, sma
from ..core.money import Decimal, ZERO
from ..core.types import MarketEvent, OrderIntent, OrderSide


class EmaCrossoverConfig(StrategyConfig):
    fast_period: int
    slow_period: int
    allow_short: bool = False
    cooldown_events: int = 0        # matching-symbol events to skip after emit

    @model_validator(mode="after")
    def _validate(self):
        if self.fast_period < 1 or self.slow_period < 1:
            raise ValueError("periods must be >= 1")
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be < slow_period")
        if self.trade_size <= ZERO:
            raise ValueError("trade_size must be positive")
        if self.cooldown_events < 0:
            raise ValueError("cooldown_events must be >= 0")
        return self


class EmaCrossoverStrategy(BaseStrategy):
    FLAT, LONG, SHORT = "FLAT", "LONG", "SHORT"

    @classmethod
    def expected_config(cls) -> type:
        return EmaCrossoverConfig

    def initial_state(self) -> Dict[str, Any]:
        return {
            "fast_buf": [],        # warm-up buffer for SMA seed (fast)
            "slow_buf": [],        # warm-up buffer for SMA seed (slow)
            "ema_fast": None,
            "ema_slow": None,
            "prev_fast": None,
            "prev_slow": None,
            "position": self.FLAT,
        }

    def on_market_event(self, event: MarketEvent) -> Optional[OrderIntent]:
        ctx = self._on_symbol_event(event)          # symbol filter/cooldown
        if ctx is None:
            return None
        price = float(ctx["price"])
        ts = ctx["ts"]
        st = self.state

        # --- warm-up: seed each EMA with SMA of ITS OWN first N samples.
        # The seeding event itself is consumed by the seed (no signal),
        # so the seeding price is never double-counted.
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
            return None                              # insufficient history

        # --- incremental update + crossover detection (prev vs current) ---
        st["prev_fast"], st["prev_slow"] = st["ema_fast"], st["ema_slow"]
        st["ema_fast"] = ema_update(st["ema_fast"], price, self.config.fast_period)
        st["ema_slow"] = ema_update(st["ema_slow"], price, self.config.slow_period)

        crossed_up = (
            st["prev_fast"] <= st["prev_slow"] and st["ema_fast"] > st["ema_slow"]
        )
        crossed_down = (
            st["prev_fast"] >= st["prev_slow"] and st["ema_fast"] < st["ema_slow"]
        )

        intent: Optional[OrderIntent] = None
        if crossed_up and st["position"] != self.LONG:
            intent = self.build_intent(OrderSide.BUY, ts, ref_price=ctx["price"])
            st["position"] = self.LONG
        elif crossed_down:
            if st["position"] == self.LONG:
                intent = self.build_intent(OrderSide.SELL, ts, ref_price=ctx["price"])
                st["position"] = self.FLAT if not self.config.allow_short else self.SHORT
            elif self.config.allow_short and st["position"] != self.SHORT:
                intent = self.build_intent(OrderSide.SELL, ts, ref_price=ctx["price"])
                st["position"] = self.SHORT
        return intent
