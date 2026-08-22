"""
Z-Score Mean Reversion strategy (Phase 7).

Convention (documented, never mixed):
- Statistics are computed over the PRIOR `window` closes, EXCLUDING the
  current event; the CURRENT price is then normalized against them:
      z = (price - mean(prior_window)) / std(prior_window)
- SIGNALS (state machine on own position view):
    flat & z <= -entry_z  -> BUY  (oversold)
    flat & z >= +entry_z  -> SELL (overbought; short only if allow_short)
    long & z >= +exit_z   -> SELL (target reached)
    short & z <= -exit_z  -> BUY  (target reached)
    otherwise             -> hold (None)

ZERO STANDARD DEVIATION (explicit contract): if every prior close is
identical, z is mathematically undefined. The strategy emits NO signal
and carries no error -- it simply waits for dispersion. No division
hacks, no epsilon fudging.
"""
from typing import Any, Dict, Optional

from pydantic import model_validator

from .base import BaseStrategy, StrategyConfig, mean_std
from ..core.money import Decimal, ZERO
from ..core.types import MarketEvent, OrderIntent, OrderSide


class ZScoreConfig(StrategyConfig):
    window: int                     # prior-close statistics window
    entry_z: float                  # e.g. 2.0
    exit_z: float                   # e.g. 0.5 ; must satisfy exit < entry
    allow_short: bool = False
    cooldown_events: int = 0

    @model_validator(mode="after")
    def _validate(self):
        if self.window < 2:
            raise ValueError("window must be >= 2")
        if self.entry_z <= 0 or self.exit_z < 0:
            raise ValueError("entry_z must be > 0 and exit_z >= 0")
        if self.exit_z >= self.entry_z:
            raise ValueError("exit_z must be < entry_z")
        if self.trade_size <= ZERO:
            raise ValueError("trade_size must be positive")
        return self


class ZScoreMeanReversionStrategy(BaseStrategy):
    FLAT, LONG, SHORT = "FLAT", "LONG", "SHORT"

    @classmethod
    def expected_config(cls) -> type:
        return ZScoreConfig

    def initial_state(self) -> Dict[str, Any]:
        return {"history": [], "position": self.FLAT}

    def on_market_event(self, event: MarketEvent) -> Optional[OrderIntent]:
        ctx = self._on_symbol_event(event)
        if ctx is None:
            return None
        price_f = float(ctx["price"])
        ts = ctx["ts"]
        st = self.state
        history: list = st["history"]

        # insufficient history -> accumulate only
        if len(history) < self.config.window:
            history.append(price_f)
            return None

        # stats over PRIOR window (current price NOT included: no look-ahead)
        mean, std = mean_std(history[-self.config.window:])
        if std == 0.0:
            # undefined z: wait for dispersion; keep the window current.
            history.append(price_f)
            history.pop(0)
            return None

        z = (price_f - mean) / std
        pos = st["position"]
        intent: Optional[OrderIntent] = None

        if pos == self.FLAT:
            if z <= -self.config.entry_z:
                intent = self.build_intent(OrderSide.BUY, ts, ref_price=ctx["price"])
                st["position"] = self.LONG
            elif z >= self.config.entry_z:
                if self.config.allow_short:
                    intent = self.build_intent(OrderSide.SELL, ts, ref_price=ctx["price"])
                    st["position"] = self.SHORT
        elif pos == self.LONG:
            if z >= self.config.exit_z:
                intent = self.build_intent(OrderSide.SELL, ts, ref_price=ctx["price"])
                st["position"] = self.FLAT
        elif pos == self.SHORT:
            if z <= -self.config.exit_z:
                intent = self.build_intent(OrderSide.BUY, ts, ref_price=ctx["price"])
                st["position"] = self.FLAT

        # roll window AFTER decision (prior-window rule)
        history.append(price_f)
        if len(history) > self.config.window:
            history.pop(0)
        return intent
