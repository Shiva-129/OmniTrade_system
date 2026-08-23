"""MACD Trend (Phase 14). MACD line cross signal."""
from typing import Any, Dict, Optional
from pydantic import model_validator
from .base import BaseStrategy, StrategyConfig
from ..core.money import ZERO
from ..core.types import MarketEvent, OrderIntent, OrderSide

class MacdTrendConfig(StrategyConfig):
    fast: int = 12
    slow: int = 26
    signal: int = 9
    cooldown_events: int = 0
    @model_validator(mode="after")
    def _v(self):
        if self.fast < 1 or self.slow < 1 or self.signal < 1: raise ValueError("periods >=1")
        if self.fast >= self.slow: raise ValueError("fast must be < slow")
        if self.trade_size <= ZERO: raise ValueError("trade_size positive")
        return self

class MacdTrendStrategy(BaseStrategy):
    FLAT, LONG = "FLAT", "LONG"
    @classmethod
    def expected_config(cls): return MacdTrendConfig
    def initial_state(self): return {"history": [], "position": self.FLAT, "prev_macd": None, "prev_sig": None}
    def on_market_event(self, event: MarketEvent) -> Optional[OrderIntent]:
        ctx = self._on_symbol_event(event)
        if ctx is None: return None
        price_f = float(ctx["price"]); ts = ctx["ts"]
        hist = self.state["history"]
        hist.append(price_f)
        n = len(hist)
        # need slow + signal bars to get first macd/signal
        if n < self.config.slow + self.config.signal:
            return None
        # compute EMAs incrementally would be more efficient, but for determinism recompute
        from src.indicators.ema import ema
        ef = ema(hist, self.config.fast)
        es = ema(hist, self.config.slow)
        # macd line where both defined
        macd = [None]*n
        for i in range(n):
            if ef[i] is not None and es[i] is not None:
                macd[i] = ef[i]-es[i]  # type: ignore
        # signal is EMA of macd
        # collect macd values compact
        first = next((i for i,v in enumerate(macd) if v is not None), None)
        if first is None: return None
        seq = [v for v in macd[first:] if v is not None]
        from src.indicators.ema import ema as ema2
        sig_seq = ema2(seq, self.config.signal)  # type: ignore
        # map back: sig_seq corresponds to macd indices first + j
        # current index n-1 maps to j = n-1 - first
        j = n-1 - first
        prev_j = j-1
        if j < 0 or prev_j < 0 or sig_seq[j] is None or sig_seq[prev_j] is None or macd[n-1] is None or macd[n-2] is None:
            return None
        prev_macd, cur_macd = macd[n-2], macd[n-1]
        prev_sig, cur_sig = sig_seq[prev_j], sig_seq[j]
        pos = self.state["position"]
        if prev_macd <= prev_sig and cur_macd > cur_sig and pos == self.FLAT:  # type: ignore
            intent = self.build_intent(OrderSide.BUY, ts, ref_price=ctx["price"])
            self.state["position"] = self.LONG
            return intent
        if prev_macd >= prev_sig and cur_macd < cur_sig and pos == self.LONG:  # type: ignore
            intent = self.build_intent(OrderSide.SELL, ts, ref_price=ctx["price"])
            self.state["position"] = self.FLAT
            return intent
        return None
