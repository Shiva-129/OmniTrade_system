"""
OmniTrade Strategy Contract (Phase 7).

THE DEAL
--------
A strategy is a PURE SIGNAL FUNCTION over a market-event stream:

    MarketEvent(s) -> strategy state -> OrderIntent | None

It NEVER places orders, NEVER touches Redis/Gatekeeper/broker/execution,
and can NEVER mutate Portfolio state (structurally: it holds no reference
to one). Downstream authority is fixed:

    OrderIntent -> RiskManager -> Gatekeeper -> ExecutionReport -> Portfolio

DETERMINISM
-----------
Identical (event sequence, configuration, initial state) => identical
signals, quantities and client order IDs. No wall-clock, no network, no
randomness. Intent timestamps come from the event's exchange timestamp.

NUMERIC POLICY (single, explicit -- no competing money policy)
--------------------------------------------------------------
Indicator math (EMA/z/ATR) runs on IEEE-754 floats: statistical
estimates where sqrt/division are required; deterministic for identical
op sequences. The canonical Decimal policy (core.money) applies at the
TRADING-CONTRACT BOUNDARY ONLY: configured sizes/thresholds that denote
money are Decimal; every emitted quantity/price is converted via
to_decimal(str(x)) at intent-build time. Floats never enter contracts.

EXECUTION CONVENTIONS (documented per strategy, never mixed)
-------------------------------------------------------------
All three strategies act on the event that completes the pattern
("signal bar execution"); look-back windows EXCLUDE the current event,
so no statistic ever contains the decision price.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel

from ..core.money import Decimal, to_decimal
from ..core.types import MarketEvent, OrderIntent, OrderSide, OrderType


class StrategyConfig(BaseModel):
    """
    Externalized, versioned strategy configuration.
    Frozen + hashable => experiments can pin configs by content.
    Concrete strategies extend this with typed parameters.
    """
    model_config = {"frozen": True}

    strategy_name: str
    strategy_version: str = "1.0.0"
    symbol: str
    timeframe: str = "tick"          # granularity LABEL (strategies are per-event)
    trade_size: Decimal              # canonical money policy


class BaseStrategy(ABC):
    """
    Minimal deterministic strategy skeleton.

    State discipline:
      - ALL mutable state lives in self.state (a plain dict of
        JSON-safe values: str/int/float/bool/list/dict).
      - Counters (seq/events/cooldown) are explicit, not hidden globals.
      - export_state()/load_state() give exact save/replay capability;
        reset() restores the pristine initial state.
    """

    def __init__(self, config: StrategyConfig):
        config_type = type(self).expected_config()
        if not isinstance(config, config_type):
            raise TypeError(
                f"{type(self).__name__} requires {config_type.__name__}, "
                f"got {type(config).__name__}"
            )
        self.config = config
        self._seq = 0             # deterministic client-order-id counter
        self._events_seen = 0     # matching-symbol events processed
        self._cooldown_left = 0   # events to skip after last emission
        self.state: Dict[str, Any] = self.initial_state()

    # ------------------------- subclass API -----------------------------

    @classmethod
    @abstractmethod
    def expected_config(cls) -> type:
        """The concrete StrategyConfig subclass this strategy requires."""

    @abstractmethod
    def initial_state(self) -> Dict[str, Any]:
        """Pristine internal state (JSON-safe values only)."""

    @abstractmethod
    def on_market_event(self, event: MarketEvent) -> Optional[OrderIntent]:
        """
        Process ONE event for the configured symbol.
        Returns an OrderIntent (via build_intent) or None. Must stay pure
        w.r.t. everything except self.state/self counters.
        """

    # ------------------------ lifecycle helpers --------------------------

    def reset(self) -> None:
        self._seq = 0
        self._events_seen = 0
        self._cooldown_left = 0
        self.state = self.initial_state()

    def export_state(self) -> Dict[str, Any]:
        return {
            "seq": self._seq,
            "events_seen": self._events_seen,
            "cooldown_left": self._cooldown_left,
            "state": self.state,
        }

    def load_state(self, snapshot: Dict[str, Any]) -> None:
        self._seq = snapshot["seq"]
        self._events_seen = snapshot["events_seen"]
        self._cooldown_left = snapshot["cooldown_left"]
        self.state = snapshot["state"]

    # ----------------------- plumbing for subclasses ----------------------

    def _on_symbol_event(self, event: MarketEvent) -> Optional[Dict[str, Any]]:
        """
        Common pre-processing: symbol filter, event counting, cooldown
        accounting. Returns the event price as canonical Decimal, or None
        when the event must be ignored / carries no usable price.
        """
        if event.packet.topic != self.config.symbol:
            return None
        price = extract_price(event)
        if price is None:
            return None
        self._events_seen += 1
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return None
        return {"price": price, "ts": event.packet.exchange_ts}

    def build_intent(self, side: OrderSide, ts: int,
                     ref_price: Optional[Decimal] = None) -> OrderIntent:
        """
        THE single signal->contract mapping for all strategies.
        Deterministic ID: name:version:symbol:seq (seq = emission counter).
        LIMIT at last observed price => no invented prices downstream.
        """
        self._seq += 1
        self._cooldown_left = getattr(self.config, "cooldown_events", 0)
        return OrderIntent(
            client_order_id=(
                f"{self.config.strategy_name}:{self.config.strategy_version}:"
                f"{self.config.symbol}:{self._seq}"
            ),
            symbol=self.config.symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=self.config.trade_size,
            price=ref_price,
            timestamp=ts,
        )


def extract_price(event: MarketEvent) -> Optional[Decimal]:
    """Wire-float -> canonical Decimal at the ingestion boundary."""
    payload = event.packet.payload or {}
    raw = payload.get("price")
    if raw is None:
        return None
    return to_decimal(str(raw))


# ---------------------- shared indicator math -------------------------
# float domain by documented numeric policy; Decimal only at boundary.

def ema_update(prev: Optional[float], price: float, period: int) -> float:
    alpha = 2.0 / (period + 1.0)
    if prev is None:
        return price
    return alpha * price + (1.0 - alpha) * prev


def sma(values) -> float:
    return sum(values) / len(values)


def mean_std(values) -> tuple[float, float]:
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return m, var ** 0.5
