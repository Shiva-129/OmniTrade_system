"""
Phase 7: Z-Score Mean Reversion tests — window warm-up, entry/exit
thresholds, zero-std contract, long & short sides, determinism.
"""
import pytest

from src.core.money import init_money_context
from src.core.types import MarketEvent, OrderSide, Packet
from src.strategies.zscore_mean_reversion import (
    ZScoreMeanReversionStrategy, ZScoreConfig)


@pytest.fixture(autouse=True)
def _ctx():
    init_money_context()


def ev(ts, price):
    return MarketEvent(packet=Packet(
        exchange_ts=ts, local_arrival_ts=ts, drift_us=0, source="t",
        topic="BTCUSDT", payload={"price": str(price)}, sequence_id=ts))


def make(allow_short=False, entry=2.0, exit_=0.5, window=5):
    return ZScoreMeanReversionStrategy(ZScoreConfig(
        strategy_name="z", symbol="BTCUSDT", trade_size="0.2",
        window=window, entry_z=entry, exit_z=exit_, allow_short=allow_short))


def feed(s, prices, start_ts=1):
    return [s.on_market_event(ev(start_ts + i, p))
            for i, p in enumerate(prices)]


class TestConfig:
    def test_window_minimum(self):
        with pytest.raises(Exception):
            ZScoreConfig(strategy_name="z", symbol="B", trade_size="1",
                         window=1, entry_z=2.0, exit_z=0.5)

    def test_exit_below_entry_required(self):
        with pytest.raises(Exception):
            ZScoreConfig(strategy_name="z", symbol="B", trade_size="1",
                         window=5, entry_z=1.0, exit_z=1.5)


class TestWarmUpAndZeroStd:
    def test_no_signal_before_window_filled(self):
        s = make(window=5)
        outs = feed(s, [100, 100, 100, 100])
        assert all(o is None for o in outs)

    def test_zero_std_emits_nothing_and_stays_alive(self):
        s = make()
        outs = feed(s, [100] * 8)             # flat line: std == 0
        assert all(o is None for o in outs)
        assert s.state["history"][-1] == 100  # window kept current

    def test_dispersion_after_flat_line_signals(self):
        s = make(entry=1.0, exit_=0.2, window=4)
        outs = feed(s, [100, 100, 100, 100, 80])   # z = (80-100)/0 = guard...
        # the 80 event itself hits the zero-std guard (prior window flat)
        assert outs[4] is None
        outs2 = feed(s, [80])                       # now prior window mixed
        # prior window [100,100,100,80]: mean 95, std ~8.29, z=(80-95)/8.29
        assert outs2[0] is not None                 # z <= -1 -> BUY


class TestSignals:
    BASE = [99, 101, 100, 102, 98]   # mean 100, std ~1.414 (dispersed!)

    def test_oversold_entry_long(self):
        s = make(entry=2.0, exit_=0.5, window=5)
        outs = feed(s, self.BASE + [80])     # z = (80-100)/1.414 = -14
        assert outs[-1] is not None
        assert outs[-1].side == OrderSide.BUY
        assert s.state["position"] == "LONG"

    def test_long_exit_on_reversion(self):
        s = make(entry=2.0, exit_=0.5, window=5)
        feed(s, self.BASE + [80])            # entry long
        # the 80 stays in the rolling window for 5 events, inflating std;
        # z crosses +0.5 on the 4th recovery print (z ~ 0.561)
        outs = feed(s, [100, 100, 100, 100], start_ts=7)
        sells = [o for o in outs if o and o.side == OrderSide.SELL]
        assert len(sells) == 1
        assert sells[0].timestamp == 10
        assert s.state["position"] == "FLAT"

    def test_overbought_short_entry_when_allowed(self):
        s = make(allow_short=True, entry=2.0, exit_=0.5, window=5)
        outs = feed(s, self.BASE + [125])    # z strongly positive
        assert outs[-1] is not None
        assert outs[-1].side == OrderSide.SELL
        assert s.state["position"] == "SHORT"

    def test_overbought_ignored_when_short_disallowed(self):
        s = make(allow_short=False, entry=2.0, exit_=0.5, window=5)
        outs = feed(s, self.BASE + [125])
        assert outs[-1] is None
        assert s.state["position"] == "FLAT"

    def test_between_thresholds_holds(self):
        s = make(entry=3.0, exit_=0.5, window=5)
        # BASE: mean 100, std ~1.414 -> z(97) ~ -2.12 ; |z| < 3 -> hold
        outs = feed(s, self.BASE + [97])
        assert all(o is None for o in outs)
        assert s.state["position"] == "FLAT"


class TestDeterminism:
    def test_repeated_runs_identical(self):
        series = [100, 100, 100, 100, 100, 78, 79, 100, 101, 100, 60, 100]
        outs1 = [o.model_dump() if o else None for o in feed(make(allow_short=True), series)]
        outs2 = [o.model_dump() if o else None for o in feed(make(allow_short=True), series)]
        assert outs1 == outs2

    def test_no_look_ahead_truncation(self):
        series = [100, 100, 100, 100, 100, 80, 85, 100]
        a = make()
        full = feed(a, series)
        b = make()
        trunc = feed(b, series[:6])
        assert full[:6] == trunc
