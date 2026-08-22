"""
Phase 7: Donchian Breakout + ATR tests — channel warm-up, upper/lower
breakout, no-breakout, prior-window (no look-ahead), tick-ATR warm-up,
stop arming + stop exit, determinism.
"""
import pytest

from src.core.money import init_money_context
from src.core.types import MarketEvent, OrderSide, Packet
from src.strategies.donchian_breakout import (
    DonchianBreakoutStrategy, DonchianConfig)


@pytest.fixture(autouse=True)
def _ctx():
    init_money_context()


def ev(ts, price):
    return MarketEvent(packet=Packet(
        exchange_ts=ts, local_arrival_ts=ts, drift_us=0, source="t",
        topic="BTCUSDT", payload={"price": str(price)}, sequence_id=ts))


def make(lookback=4, atr_period=3, mult=2.0, allow_short=False):
    return DonchianBreakoutStrategy(DonchianConfig(
        strategy_name="don", symbol="BTCUSDT", trade_size="0.1",
        lookback=lookback, atr_period=atr_period,
        atr_stop_multiplier=mult, allow_short=allow_short))


def feed(s, prices, start_ts=1):
    return [s.on_market_event(ev(start_ts + i, p))
            for i, p in enumerate(prices)]


class TestWarmUp:
    def test_no_signal_before_channel_ready(self):
        s = make(lookback=4)
        outs = feed(s, [100, 101, 102])       # < lookback closes
        assert all(o is None for o in outs)

    def test_breakout_on_first_eligible_event(self):
        s = make(lookback=4)
        outs = feed(s, [100, 100, 100, 100, 130])
        # 5th event: prior window full; 130 > max(100) -> BUY on that event
        assert outs[-1] is not None
        assert outs[-1].side == OrderSide.BUY
        assert outs[-1].timestamp == 5

    def test_atr_warmup_disables_initial_stop(self):
        """ATR needs atr_period TRs => lookback+1 events; early entry has no stop."""
        s = make(lookback=2, atr_period=10)   # ATR far from ready
        feed(s, [100, 100, 100, 110])         # breakout at event 4
        assert s.state["position"] == "LONG"
        assert s.state["stop"] is None        # armed only when ATR exists


class TestBreakouts:
    def test_upper_breakout_long(self):
        s = make(lookback=3, atr_period=2, mult=0)   # no stops: pure signal test
        outs = feed(s, [50, 50, 50, 60])
        assert outs[-1] is not None and outs[-1].side == OrderSide.BUY

    def test_lower_breakout_exit_then_short(self):
        s = make(lookback=3, atr_period=2, mult=0, allow_short=True)
        outs = feed(s, [50, 50, 50, 40])
        # flat -> lower breakout opens short directly
        assert outs[-1] is not None and outs[-1].side == OrderSide.SELL
        assert s.state["position"] == "SHORT"

    def test_lower_breakdown_without_short_only_exits_existing_long(self):
        s = make(lookback=3, atr_period=2, mult=0, allow_short=False)
        outs = feed(s, [50, 50, 50, 70,       # up-cross -> LONG
                        45])                  # down-break -> SELL exit
        sides = [o.side for o in outs if o]
        assert sides == [OrderSide.BUY, OrderSide.SELL]
        assert s.state["position"] == "FLAT"

    def test_inside_channel_no_signal(self):
        s = make(lookback=3, atr_period=2, mult=0)
        outs = feed(s, [50, 60, 55, 58])      # 58 within [50..60]
        assert all(o is None for o in outs)


class TestNoLookAhead:
    def test_current_price_not_in_its_own_channel(self):
        """A huge single print cannot break its own channel."""
        s = make(lookback=3, atr_period=2, mult=0)
        outs = feed(s, [50, 50, 50, 200])
        # 200 breaks the PRIOR window [50,50,50] -> exactly one BUY here...
        assert outs[-1] is not None
        # ...but a second giant print right after must NOT re-fire (already long)
        outs2 = feed(s, [300], start_ts=9)
        assert all(o is None for o in outs2)

    def test_truncation_equivalence(self):
        series = [100, 99, 101, 100, 120, 121, 119, 90, 89, 95]
        a = make(lookback=3, atr_period=2, mult=0)
        full = feed(a, series)
        b = make(lookback=3, atr_period=2, mult=0)
        trunc = feed(b, series[:6])
        assert full[:6] == trunc


class TestATRStops:
    def test_stop_armed_with_multiplier(self):
        s = make(lookback=2, atr_period=2, mult=1.0)
        feed(s, [100, 100])                   # TRs start at event 2
        feed(s, [100])                        # TR=0 -> ATR ready (0.0)... 
        # note: ATR==0 is a valid value; stop = entry - 1*0 = entry
        outs = feed(s, [140])                 # breakout: close>max(100)
        assert outs[-1] is not None
        assert s.state["stop"] is not None

    def test_stop_exit_fires_on_cross(self):
        s = make(lookback=2, atr_period=2, mult=1.0)
        feed(s, [100, 104, 104])              # TRs: 4, 0 -> ATR=2
        # breakout long at 140: prior channel max=104; stop=140-1*2=138
        outs = feed(s, [140])
        assert s.state["position"] == "LONG"
        assert s.state["stop"] is not None
        exit_outs = feed(s, [137])            # price <= stop -> exit
        emitted = [o for o in exit_outs if o]
        assert len(emitted) == 1
        assert emitted[0].side == OrderSide.SELL
        assert s.state["position"] == "FLAT"
        assert s.state["stop"] is None

    def test_price_above_stop_keeps_position(self):
        s = make(lookback=2, atr_period=2, mult=1.0)
        feed(s, [100, 104, 104])
        feed(s, [140])
        keep = feed(s, [139, 141])            # above stop 138 -> hold
        assert all(o is None for o in keep)
        assert s.state["position"] == "LONG"


class TestDeterminism:
    def test_repeated_runs_identical(self):
        series = [50, 50, 52, 54, 80, 82, 81, 40, 41, 42, 60, 61]
        mk = lambda: make(lookback=3, atr_period=2, mult=1.0, allow_short=True)
        outs1 = [o.model_dump() if o else None for o in feed(mk(), series)]
        outs2 = [o.model_dump() if o else None for o in feed(mk(), series)]
        assert outs1 == outs2
