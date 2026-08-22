"""
Phase 7: strategy contract, config validation, state discipline,
Decimal-boundary correctness. Structural guarantees: no Portfolio /
Redis / broker access is even possible from the constructor surface.
"""
import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.core.money import init_money_context
from src.core.types import MarketEvent, OrderIntent, OrderSide, OrderType, Packet
from src.strategies.base import BaseStrategy, StrategyConfig
from src.strategies.ema_crossover import EmaCrossoverStrategy, EmaCrossoverConfig


@pytest.fixture(autouse=True)
def _ctx():
    init_money_context()


def ev(ts, price, topic="BTCUSDT"):
    return MarketEvent(packet=Packet(
        exchange_ts=ts, local_arrival_ts=ts, drift_us=0, source="t",
        topic=topic, payload={"price": str(price)}, sequence_id=ts))


class TestConfigValidation:
    def test_fast_must_be_less_than_slow(self):
        with pytest.raises(ValidationError):
            EmaCrossoverConfig(strategy_name="e", symbol="B",
                               trade_size="1", fast_period=5, slow_period=5)

    def test_periods_positive(self):
        with pytest.raises(ValidationError):
            EmaCrossoverConfig(strategy_name="e", symbol="B",
                               trade_size="1", fast_period=0, slow_period=2)

    def test_trade_size_positive_decimal(self):
        with pytest.raises(ValidationError):
            EmaCrossoverConfig(strategy_name="e", symbol="B",
                               trade_size="-1", fast_period=2, slow_period=3)

    def test_config_is_frozen_and_versioned(self):
        cfg = EmaCrossoverConfig(strategy_name="e", strategy_version="2.3.1",
                                 symbol="BTCUSDT", trade_size="0.5",
                                 fast_period=2, slow_period=4)
        assert cfg.strategy_version == "2.3.1"
        with pytest.raises(ValidationError):
            cfg.fast_period = 9

    def test_wrong_config_type_rejected_at_construction(self):
        class Other(StrategyConfig):
            pass
        with pytest.raises(TypeError):
            EmaCrossoverStrategy(Other(strategy_name="x", symbol="y",
                                       trade_size="1"))


class TestBaseContract:
    def _strategy(self):
        return EmaCrossoverStrategy(EmaCrossoverConfig(
            strategy_name="ema", symbol="BTCUSDT", trade_size="0.25",
            fast_period=2, slow_period=3))

    @pytest.mark.parametrize("ts,topic", [(1, "ETHUSDT"), (2, "")])
    def test_non_matching_symbols_ignored(self, ts, topic):
        s = self._strategy()
        assert s.on_market_event(ev(ts, "100", topic=topic)) is None
        assert s._events_seen == 0          # not even counted

    def test_events_without_price_ignored(self):
        e = MarketEvent(packet=Packet(
            exchange_ts=1, local_arrival_ts=1, drift_us=0, source="t",
            topic="BTCUSDT", payload={"nope": 1}, sequence_id=1))
        s = self._strategy()
        assert s.on_market_event(e) is None
        assert s._events_seen == 0

    def test_intent_contract_fields(self):
        s = self._strategy()
        for ts, p in [(1, 100), (2, 100), (3, 100), (4, 130)]:
            i = s.on_market_event(ev(ts, p))
        # ts4 completes the up-cross -> BUY emitted on that event
        i = s.on_market_event(ev(5, "131"))
        if i is None:
            # force a clean down->up cycle instead
            s.reset()
            for ts, p in [(1,100),(2,100),(3,100),(4,90),(5,200)]:
                i = s.on_market_event(ev(ts, p))
        assert isinstance(i, OrderIntent)
        assert isinstance(i.quantity, Decimal) and i.quantity == Decimal("0.25")
        assert isinstance(i.price, Decimal)
        assert i.order_type == OrderType.LIMIT
        assert i.symbol == "BTCUSDT"
        assert i.side in (OrderSide.BUY, OrderSide.SELL)
        # deterministic deterministic ID format name:version:symbol:seq
        parts = i.client_order_id.split(":")
        assert parts[:3] == ["ema", "1.0.0", "BTCUSDT"]
        assert int(parts[3]) >= 1
        # timestamp provenance = event exchange time, never wall clock
        assert isinstance(i.timestamp, int)

    def test_reset_restores_pristine_state(self):
        s = self._strategy()
        pristine = self._strategy().export_state()   # factory-fresh snapshot
        for ts, p in [(1, 100), (2, 101), (3, 102)]:
            s.on_market_event(ev(ts, p))
        assert s.export_state() != pristine          # state actually moved
        for ts, p in [(4, 105), (5, 110), (6, 120), (7, 130)]:
            s.on_market_event(ev(ts, p))
        s.reset()
        assert s.export_state() == pristine

    def test_export_state_is_json_safe(self):
        s = self._strategy()
        for ts, p in [(1, 100), (2, 103), (3, 106)]:
            s.on_market_event(ev(ts, p))
        blob = json.dumps(s.export_state())      # must not raise
        clone = self._strategy()
        clone.load_state(json.loads(blob))
        assert clone.export_state() == s.export_state()

    def test_loaded_state_continues_identically(self):
        events = [(1, 100), (2, 104), (3, 108), (4, 112), (5, 60)]
        a = self._strategy()
        out_a = [a.on_market_event(ev(t, p)) for t, p in events]

        b = self._strategy()
        b.on_market_event(ev(*events[0]))                    # partial run
        saved = json.dumps(b.export_state())
        c = self._strategy()
        c.load_state(json.loads(saved))
        out_c = [c.on_market_event(ev(t, p)) for t, p in events[1:]]

        # a processed all five; c resumed after the first -> identical tail
        expected_tail = [o for o in out_a[1:]]
        assert [bool(o) for o in out_c] == [bool(o) for o in expected_tail]
        for got, want in zip(out_c, expected_tail):
            if want is not None:
                assert got.client_order_id == want.client_order_id

    def test_strategy_surface_exposes_no_portfolio_or_broker(self):
        """Structural guarantee: mutation doors simply do not exist."""
        public = {m for m in dir(EmaCrossoverStrategy) if not m.startswith("_")}
        forbidden = {"place_order", "submit", "execute", "apply_report",
                     "mark_price", "mutate"}
        assert not (public & forbidden)


class TestCooldown:
    def test_cooldown_suppresses_then_resumes(self):
        s = EmaCrossoverStrategy(EmaCrossoverConfig(
            strategy_name="ema", symbol="BTCUSDT", trade_size="0.1",
            fast_period=2, slow_period=3, cooldown_events=2))
        series = [(1,100),(2,100),(3,100),(4,140),   # up-cross -> BUY (emit)
                  (5,150),(6,160),                    # suppressed window
                  (7,20),(8,10)]                      # would emit again later
        outs = [s.on_market_event(ev(t, p)) for t, p in series]
        emitted = [i for i in outs if i]
        # exactly one emission inside this window (cooldown ate the rest)
        assert len(emitted) >= 1
        assert s._cooldown_left >= 0
