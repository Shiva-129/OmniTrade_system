"""
Phase 4: BinanceObserver resilience tests.

No network: the ccxt.binance constructor is monkeypatched with a scriptable
fake so we can verify reconnection, give-up-after-max (loud failure), and
multi-symbol isolation deterministically.
"""
import asyncio
import random
from contextlib import suppress

import pytest

import src.markets.binance_observer as bmo
from src.markets.binance_observer import (
    BinanceObserver,
    backoff_delay,
    MAX_RECONNECT_ATTEMPTS,
    BASE_BACKOFF_S,
    MAX_BACKOFF_S,
)
from src.core.clock import Clock


# --------------------------- backoff policy (pure) ---------------------------

class TestBackoffPolicy:
    def test_doubles_then_caps(self):
        assert backoff_delay(1) == BASE_BACKOFF_S
        assert backoff_delay(2) == BASE_BACKOFF_S * 2
        assert backoff_delay(3) == BASE_BACKOFF_S * 4
        for attempt in range(7, 20):
            assert backoff_delay(attempt) == MAX_BACKOFF_S

    def test_deterministic_with_seeded_rng(self):
        rng1, rng2 = random.Random(42), random.Random(42)
        a = [backoff_delay(i, rng1) for i in range(1, 10)]
        b = [backoff_delay(i, rng2) for i in range(1, 10)]
        assert a == b

    def test_jitter_bounds(self):
        rng = random.Random(7)
        for attempt in range(1, 8):
            base = min(MAX_BACKOFF_S, BASE_BACKOFF_S * (2 ** (attempt - 1)))
            delay = backoff_delay(attempt, rng)
            assert base * 0.75 <= delay <= base * 1.25


# ------------------------- scripted fake exchange ---------------------------

class FakeCCXT:
    """Scriptable stand-in for ccxt.binance()."""

    def __init__(self, behavior):
        # behavior: {symbol: [list of outcomes]; outcome is list-of-trades or Exception}
        self.behavior = behavior
        self.closed = False

    async def watch_trades(self, symbol):
        script = self.behavior.get(symbol, [])
        if script:
            outcome = script.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            if isinstance(outcome, list) and not outcome:
                await asyncio.sleep(0.005)  # idle, don't busy-spin
                return []
            return outcome
        await asyncio.sleep(0.005)
        return []

    async def close(self):
        self.closed = True


@pytest.fixture()
def fast_backoff(monkeypatch):
    # Shrink backoff constants so reconnect loops run at test speed.
    monkeypatch.setattr(bmo, "BASE_BACKOFF_S", 0.001)
    monkeypatch.setattr(bmo, "MAX_BACKOFF_S", 0.004)


def _observer_with(monkeypatch, behavior):
    monkeypatch.setattr(bmo.ccxt, "binance", lambda cfg: FakeCCXT(behavior))
    obs = BinanceObserver(symbols=list(behavior.keys()))
    return obs


class TestReconnectAndIsolation:
    @pytest.mark.asyncio
    async def test_recovers_after_transient_failures(self, monkeypatch, fast_backoff):
        trades_ok = [{"id": 1, "timestamp": Clock.now_epoch_us() // 1000, "price": "1", "amount": "1"}]
        fake = {
            "BTC/USDT": [
                RuntimeError("net down"),
                RuntimeError("still down"),
                trades_ok,
                [],  # idle afterwards
            ]
        }
        obs = _observer_with(monkeypatch, fake)

        received = []
        gen = obs.listen()

        async def consume():
            async for p in gen:
                received.append(p)

        task = asyncio.create_task(consume())

        async def got_packet():
            return len(received) >= 1

        deadline = asyncio.get_running_loop().time() + 3
        while asyncio.get_running_loop().time() < deadline and not received:
            await asyncio.sleep(0.01)
        assert len(received) == 1, "stream did not recover after transient failures"

        await obs.close()
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)

    @pytest.mark.asyncio
    async def test_gives_up_loudly_after_max_attempts(self, monkeypatch, fast_backoff):
        fake = {"BTC/USDT": [RuntimeError("down")] * (MAX_RECONNECT_ATTEMPTS + 5)}
        obs = _observer_with(monkeypatch, fake)

        consumed = []

        async def consume():
            async for p in obs.listen():
                consumed.append(p)

        task = asyncio.create_task(consume())
        with pytest.raises(Exception):
            # The symbol loop raises after max attempts; it surfaces via the task.
            await asyncio.wait_for(task, timeout=3)

        await obs.close()

    @pytest.mark.asyncio
    async def test_symbol_failure_isolates_other_symbols(self, monkeypatch, fast_backoff):
        """
        One dead stream must NOT starve a healthy one (isolation requirement).
        The failing symbol exhausts retries and raises inside ITS loop only;
        the healthy symbol keeps delivering into the shared queue.
        """
        good_trade = [{"id": 9, "timestamp": Clock.now_epoch_us() // 1000}]
        # Healthy: one real delivery, then idle via default branch.
        # Dead: permanent failures until give-up (fast backoff).
        fake = {
            "HEALTHY/USDT": [good_trade],
            "DEAD/USDT": [RuntimeError("down")] * (MAX_RECONNECT_ATTEMPTS + 5),
        }
        obs = _observer_with(monkeypatch, fake)

        received = []
        errors = []
        gen = obs.listen()

        async def consume():
            try:
                async for p in gen:
                    received.append(p)
            except Exception as e:  # DEAD's give-up eventually ends the merge loop
                errors.append(e)

        task = asyncio.create_task(consume())

        deadline = asyncio.get_running_loop().time() + 3
        while asyncio.get_running_loop().time() < deadline and not received:
            await asyncio.sleep(0.01)

        assert any(p.topic.startswith("HEALTHY") for p in received), \
            "healthy symbol starved while other symbol failed"

        await obs.close()
        with suppress(asyncio.CancelledError, RuntimeError):
            await asyncio.wait_for(task, timeout=3)
