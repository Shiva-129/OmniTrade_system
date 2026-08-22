"""
Donchian Breakout + ATR stop strategy (Phase 7).

CONVENTIONS (documented explicitly, never mixed):
1. BREAKOUT RULE: "close breaks the PREVIOUS channel" -- the channel
   high/low are computed over the PRIOR `lookback` closes EXCLUDING the
   current event; the current price is compared against them. Strict
   inequality required (> high / < low).
2. EXECUTION TIMING: signal-bar execution -- the order is emitted on the
   same event whose price breaks the channel. (No next-event delay.)
3. ATR on tick data: true ranges require OHLC bars; on a trade-price
   stream we use the documented tick approximation
       TR_i = |price_i - price_{i-1}|,   ATR = SMA(TR, atr_period)
   The FIRST TR needs two events, so ATR warm-up = atr_period + 1 events.
4. STOPS: at entry, stop = entry_ref -/+ atr_mult * ATR(at that moment).
   Position exits when price crosses its stop. No contract changes are
   invented for stops -- the stop lives in strategy state and emits a
   normal exit OrderIntent.
"""
from typing import Any, Dict, List, Optional

from pydantic import model_validator

from .base import BaseStrategy, StrategyConfig
from ..core.money import Decimal, to_decimal, dec_to_str, ZERO
from ..core.types import MarketEvent, OrderIntent, OrderSide


class DonchianConfig(StrategyConfig):
    lookback: int                   # channel length (prior closes)
    atr_period: int                 # tick-ATR length
    atr_stop_multiplier: float = 2.0  # set 0/negative to disable stops
    allow_short: bool = False
    cooldown_events: int = 0

    @model_validator(mode="after")
    def _validate(self):
        if self.lookback < 1:
            raise ValueError("lookback must be >= 1")
        if self.atr_period < 1:
            raise ValueError("atr_period must be >= 1")
        if self.trade_size <= ZERO:
            raise ValueError("trade_size must be positive")
        return self


class DonchianBreakoutStrategy(BaseStrategy):
    FLAT, LONG, SHORT = "FLAT", "LONG", "SHORT"

    @classmethod
    def expected_config(cls) -> type:
        return DonchianConfig

    def initial_state(self) -> Dict[str, Any]:
        return {
            "closes": [],          # rolling closes for the channel
            "prev_price": None,    # for tick TR
            "trs": [],             # rolling true ranges for ATR
            "position": self.FLAT,
            "stop": None,          # canonical Decimal string when armed
        }

    # --------------------------- indicators ------------------------------

    def _atr(self) -> Optional[float]:
        trs: List[float] = self.state["trs"]
        if len(trs) < self.config.atr_period:
            return None
        return sum(trs) / len(trs)

    def _channel(self, exclude_last: bool = True) -> Optional[tuple]:
        """
        (low, high) over the PRIOR `lookback` closes.
        Callers evaluate BEFORE appending the current close, so the
        channel never contains the decision price (no look-ahead).
        """
        closes: List[float] = self.state["closes"]
        if len(closes) < self.config.lookback:
            return None
        window = closes[-self.config.lookback:]
        return min(window), max(window)

    # ------------------------------ main ----------------------------------

    def on_market_event(self, event: MarketEvent) -> Optional[OrderIntent]:
        ctx = self._on_symbol_event(event)
        if ctx is None:
            return None
        price_f = float(ctx["price"])
        ts = ctx["ts"]
        st = self.state

        # --- decisions FIRST: channel and ATR over PRIOR events only ---
        # (stop exit, breakouts, stop arming all use pre-event indicators;
        #  this event's TR/close join the windows afterwards -- no look-ahead)
        intent: Optional[OrderIntent] = None
        pos = st["position"]
        stop = st["stop"]

        # 1) stop exit has priority once armed
        if stop is not None:
            stop_d = to_decimal(stop)
            if pos == self.LONG and ctx["price"] <= stop_d:
                intent = self.build_intent(OrderSide.SELL, ts, ref_price=ctx["price"])
                st["position"], st["stop"] = self.FLAT, None
            elif pos == self.SHORT and ctx["price"] >= stop_d:
                intent = self.build_intent(OrderSide.BUY, ts, ref_price=ctx["price"])
                st["position"], st["stop"] = self.FLAT, None

        # 2) breakouts against the PRIOR channel
        channel = self._channel()
        if intent is None and channel is not None:
            ch_low, ch_high = channel
            if price_f > ch_high and st["position"] != self.LONG:
                if st["position"] == self.SHORT:            # short exit first
                    intent = self.build_intent(OrderSide.BUY, ts, ref_price=ctx["price"])
                    st["position"], st["stop"] = self.FLAT, None
                else:
                    intent = self._enter_long(ctx)
            elif price_f < ch_low and st["position"] != self.SHORT:
                if st["position"] == self.LONG:             # long exit first
                    intent = self.build_intent(OrderSide.SELL, ts, ref_price=ctx["price"])
                    st["position"], st["stop"] = self.FLAT, None
                else:
                    intent = self._enter_short(ctx)

        # --- roll windows AFTER decisions (prior-window rule) ---
        prev = st["prev_price"]
        if prev is not None:
            trs: List[float] = st["trs"]
            trs.append(abs(price_f - prev))
            if len(trs) > self.config.atr_period:
                trs.pop(0)
        st["prev_price"] = price_f

        st["closes"].append(price_f)
        if len(st["closes"]) > self.config.lookback:
            st["closes"].pop(0)
        return intent

    def _arm_stop(self, side: OrderSide, entry_price: Decimal) -> None:
        mult = float(self.config.atr_stop_multiplier)
        atr = self._atr()
        if mult <= 0 or atr is None:                        # disabled / warming up
            return
        atr_d = to_decimal(str(atr))
        m = to_decimal(str(mult))
        stop = entry_price - m * atr_d if side == OrderSide.BUY else entry_price + m * atr_d
        self.state["stop"] = dec_to_str(stop)

    def _enter_long(self, ctx) -> OrderIntent:
        intent = self.build_intent(OrderSide.BUY, ctx["ts"], ref_price=ctx["price"])
        self.state["position"] = self.LONG
        self._arm_stop(OrderSide.BUY, ctx["price"])
        return intent

    def _enter_short(self, ctx) -> OrderIntent:
        intent = self.build_intent(OrderSide.SELL, ctx["ts"], ref_price=ctx["price"])
        self.state["position"] = self.SHORT
        self._arm_stop(OrderSide.SELL, ctx["price"])
        return intent
