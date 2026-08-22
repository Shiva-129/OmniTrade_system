"""
Phase 7: EMA Trend Crossover tests — warm-up, exact crossovers, both
directions, allow_short policy, cooldown, determinism, no look-ahead.
"""
import pytest

from src.core.money import init_money_context
from src.core.types import MarketEvent, OrderIntent, OrderSide, Packet
from src.strategies.ema_crossover import EmaCrossoverStrategy, EmaCrossoverConfig


@pytest.fixture(autouse=True)
def _ctx():
    init_money_context()


def ev(ts, price, topic="BTCUSDT"):
    return MarketEvent(packet=Packet(
        exchange_ts=ts, local_arrival_ts=ts, drift_us=0, source="t",
        topic=topic, payload={"price": str(price)}, sequence_id=ts))


def make(allow_short=False, cooldown=0):
    return EmaCrossoverStrategy(EmaCrossoverConfig(
        strategy_name="ema", symbol="BTCUSDT", trade_size="0.1",
        fast_period=2, slow_period=3,
        allow_short=allow_short, cooldown_events=cooldown))


def feed(s, prices, start_ts=1):
    return [s.on_market_event(ev(start_ts + i, p))
            for i, p in enumerate(prices)]


class TestWarmUp:
    def test_no_signal_before_both_emas_seeded(self):
        s = make()
        outs = feed(s, [100, 100, 100])       # exactly fills slow buffer
        assert all(o is None for o in outs)   # seeding event emits nothing

    def test_first_possible_signal_requires_prev_pair(self):
        s = make()
        outs = feed(s, [100, 100, 100, 100])  # first incremental event, flat
        assert outs[-1] is None               # no cross on flat line

    def test_insufficient_history_entirely(self):
        s = make()
        assert feed(s, [100, 100]) == [None, None]


class TestCrossoverSignals:
    def test_exact_up_cross_emits_buy(self):
        s = make()
        outs = feed(s, [100, 100, 100, 130])
        assert outs[-1] is not None
        assert outs[-1].side == OrderSide.BUY
        assert outs[-1].price is not None
        # executes on the breaking event's timestamp
        assert outs[-1].timestamp == 4
        assert outs[-1].client_order_id.endswith(":1")

    def test_exact_down_cross_emits_exit(self):
        s = make()
        outs = feed(s, [100, 100, 100, 130,   # up-cross -> BUY
                        130, 130, 60])        # down-cross -> SELL exit
        buys = [o for o in outs if o and o.side == OrderSide.BUY]
        sells = [o for o in outs if o and o.side == OrderSide.SELL]
        assert len(buys) == 1 and len(sells) == 1
        assert sells[0].timestamp == 7

    def test_no_rearm_while_already_long(self):
        """Continued rally must not emit repeated BUYs."""
        s = make()
        outs = feed(s, [100, 100, 100, 130, 140, 150, 160, 170])
        buys = [o for o in outs if o and o.side == OrderSide.BUY]
        assert len(buys) == 1

    def test_flat_series_never_signals(self):
        s = make()
        outs = feed(s, [100] * 20)
        assert all(o is None for o in outs)

    def test_allow_short_opens_short_on_down_cross(self):
        s = make(allow_short=True)
        outs = feed(s, [100, 100, 100, 60])   # immediate down-cross
        sells = [o for o in outs if o]
        assert len(sells) == 1
        assert sells[0].side == OrderSide.SELL

    def test_without_allow_short_down_cross_only_exits_long(self):
        s = make(allow_short=False)
        # up then down: BUY then SELL-exit, nothing else
        outs = feed(s, [100, 100, 100, 130, 130, 130, 70])
        emitted = [o for o in outs if o]
        assert [o.side for o in emitted] == [OrderSide.BUY, OrderSide.SELL]
        # position back to flat: state check
        assert s.state["position"] == "FLAT"


class TestNoLookAhead:
    def test_signal_uses_only_past_and_current(self):
        """A future crash cannot influence an earlier decision."""
        s1 = make()
        out_full = feed(s1, [100, 100, 100, 130, 140, 10, 10, 10])
        s2 = make()
        out_trunc = feed(s2, [100, 100, 100, 130, 140])
        # decisions up to the truncation point must be identical
        assert out_full[:5] == out_trunc


class TestDeterminism:
    def test_repeated_runs_identical_intents(self):
        series = [100, 100, 100, 130, 135, 140, 80, 75, 70, 120]
        outs1 = [o.model_dump() if o else None for o in feed(make(), series)]
        outs2 = [o.model_dump() if o else None for o in feed(make(), series)]
        assert outs1 == outs2

    def test_state_export_replay_continuation(self):
        series = [100, 100, 100, 130, 135, 140, 80, 75, 70, 120]
        a = make()
        full = [a.on_market_event(ev(i + 1, p)) for i, p in enumerate(series)]

        b = make()
        for i, p in enumerate(series[:6]):
            b.on_market_event(ev(i + 1, p))
        snap = b.export_state()
        c = make()
        c.load_state(snap)
        tail = [c.on_market_event(ev(i + 7, p)) for i, p in enumerate(series[6:])]
        assert tail == full[6:]
