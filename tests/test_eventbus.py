"""
Phase 4: EventBus contract tests.
Guarantees under test: FIFO ordering per subscriber, subscriber isolation,
loud backpressure failure (never silent drops), clean close via sentinel,
subscribe/publish after close rejected.
"""
import asyncio

import pytest

from src.core.events import EventBus, drain
from src.core.types import MarketEvent


def _event(i: int) -> MarketEvent:
    from src.core.types import Packet

    return MarketEvent(
        packet=Packet(
            exchange_ts=i, local_arrival_ts=i, drift_us=0,
            source="test", topic="t", payload={"i": i}, sequence_id=i,
        )
    )


class TestOrderingAndIsolation:
    @pytest.mark.asyncio
    async def test_fifo_delivery_per_subscriber(self):
        bus = EventBus()
        q = bus.subscribe(MarketEvent)
        for i in range(50):
            await bus.publish(MarketEvent, _event(i))

        received = []
        while not q.empty():
            received.append(q.get_nowait().packet.payload["i"])
        assert received == list(range(50))

    @pytest.mark.asyncio
    async def test_subscribers_are_isolated(self):
        """A stalled consumer cannot block or starve another consumer."""
        bus = EventBus()
        fast = bus.subscribe(MarketEvent)
        slow = bus.subscribe(MarketEvent)  # never drained during publish

        for i in range(10):
            await bus.publish(MarketEvent, _event(i))

        got_fast = []
        while not fast.empty():
            got_fast.append(fast.get_nowait().packet.payload["i"])
        assert got_fast == list(range(10))
        assert slow.qsize() == 10  # intact, waiting for its own consumer

    @pytest.mark.asyncio
    async def test_no_cross_type_leakage(self):
        from src.core.types import RiskDecision

        bus = EventBus()
        market_q = bus.subscribe(MarketEvent)
        risk_q = bus.subscribe(RiskDecision)

        await bus.publish(MarketEvent, _event(1))
        assert market_q.qsize() == 1
        assert risk_q.qsize() == 0


class TestLoudFailureSemantics:
    @pytest.mark.asyncio
    async def test_backpressure_raises_never_drops(self):
        bus = EventBus(maxsize=1)
        q = bus.subscribe(MarketEvent)
        await bus.publish(MarketEvent, _event(0))
        with pytest.raises(RuntimeError, match="queue full"):
            await bus.publish(MarketEvent, _event(1))
        assert q.qsize() == 1  # first event untouched

    @pytest.mark.asyncio
    async def test_publish_after_close_rejected(self):
        bus = EventBus()
        bus.subscribe(MarketEvent)
        bus.close()
        with pytest.raises(RuntimeError, match="closed"):
            await bus.publish(MarketEvent, _event(0))

    def test_subscribe_after_close_rejected(self):
        bus = EventBus()
        bus.close()
        with pytest.raises(RuntimeError, match="closed"):
            bus.subscribe(MarketEvent)


class TestCleanShutdown:
    @pytest.mark.asyncio
    async def test_close_wakes_consumer_with_sentinel(self):
        bus = EventBus()
        q = bus.subscribe(MarketEvent)

        seen = []

        async def handler(ev):
            seen.append(ev.packet.payload["i"])

        task = asyncio.create_task(drain(q, handler))
        for i in range(3):
            await bus.publish(MarketEvent, _event(i))

        await asyncio.sleep(0.01)  # let consumer process
        assert seen == [0, 1, 2]

        bus.close()
        handled = await asyncio.wait_for(task, timeout=2)
        assert handled == 3  # loop exited cleanly on sentinel

    @pytest.mark.asyncio
    async def test_handler_exception_propagates_to_caller(self):
        bus = EventBus()
        q = bus.subscribe(MarketEvent)

        def boom(_):
            raise ValueError("handler exploded")

        task = asyncio.create_task(drain(q, boom))
        await bus.publish(MarketEvent, _event(0))

        with pytest.raises(ValueError, match="handler exploded"):
            await asyncio.wait_for(task, timeout=2)
