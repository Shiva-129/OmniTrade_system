"""
Phase 4: TradingEngine orchestrator tests.

Uses a FakeExchange (no network) to verify the GATE 4 guarantees:
- write-ahead journaling: journaled count == published count
- zero silent drops: processed_count == packet count
- deterministic FIFO processing order
- gap detection transitions CONNECTED -> DEGRADED
- drift violation transitions -> HALT
- heartbeat freshness after every packet (Phase 4 bug regression)
- clean shutdown via stop() AND via task cancellation (Windows Ctrl+C path)

Redis-backed assertions use the REAL instance on scratch DB 15.
"""
import asyncio
from contextlib import suppress

import pytest

from src.core.clock import Clock
from src.core.engine import TradingEngine
from src.core.types import MarketEvent, Packet
from src.markets.exchange_interface import ExchangeInterface

from conftest import TEST_REDIS_URL


class FakeExchange(ExchangeInterface):
    """Scripted exchange: yields preloaded packets then idles until closed."""

    def __init__(self, name="fake", topic="BTC/USDT"):
        self.name = name
        self.topic = topic
        self.packets = []
        self.closed = False

    def add(self, seq, drift_us=0):
        self.packets.append(
            Packet(
                exchange_ts=Clock.now_epoch_us() + drift_us,
                local_arrival_ts=Clock.now_epoch_us(),
                drift_us=drift_us,
                source=self.name,
                topic=self.topic,
                payload={"seq": seq},
                sequence_id=seq,
            )
        )
        return self

    async def connect(self):
        pass

    async def listen(self):
        idx = 0
        while True:
            if idx < len(self.packets):
                yield self.packets[idx]
                idx += 1
            else:
                await asyncio.sleep(0.005)  # idle like a live stream

    async def close(self):
        self.closed = True


def _build_engine(tmp_path):
    return TradingEngine(redis_url=TEST_REDIS_URL, journal_path=str(tmp_path / "j.jsonl"))


async def _wait_until(predicate, timeout=3.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


class TestEnginePipeline:
    @pytest.mark.asyncio
    async def test_all_packets_journaled_and_processed_zero_drops(self, tmp_path, live_redis):
        ex = FakeExchange()
        for i in range(1, 26):
            ex.add(i)
        engine = _build_engine(tmp_path)
        engine.add_exchange(ex)

        stage_q = engine.register_stage(MarketEvent)  # extra subscriber sees same stream

        start_task = asyncio.create_task(engine.start())
        assert await _wait_until(lambda: engine.processed_count == 25), "engine never finished processing"

        await engine.stop()
        with suppress(asyncio.InvalidStateError):
            start_task.result()

        # Zero drops: every packet journaled, processed, and delivered.
        assert engine.processed_count == 25
        with open(tmp_path / "j.jsonl", encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if line.strip()]
            packet_events = [line for line in lines if '"PACKET"' in line]
        assert len(packet_events) == 25

        delivered = []
        while not stage_q.empty():
            item = stage_q.get_nowait()
            if item is not None:  # skip close() wake-up sentinel
                delivered.append(item)
        assert len(delivered) == 25
        assert [e.packet.payload["seq"] for e in delivered] == list(range(1, 26))

    @pytest.mark.asyncio
    async def test_fifo_processing_order_single_consumer(self, tmp_path, live_redis):
        ex = FakeExchange()
        for i in range(1, 51):
            ex.add(i)
        engine = _build_engine(tmp_path)
        engine.add_exchange(ex)

        start_task = asyncio.create_task(engine.start())
        assert await _wait_until(lambda: engine.processed_count == 50)

        # Single FIFO consumer => last sequence seen is the highest.
        assert engine.sequence_tracker["fake:BTC/USDT"] == 50
        await engine.stop()

    @pytest.mark.asyncio
    async def test_gap_detection_degrades_status(self, tmp_path, live_redis):
        ex = FakeExchange()
        ex.add(1)
        ex.add(5)  # gap of 3
        engine = _build_engine(tmp_path)
        engine.add_exchange(ex)

        start_task = asyncio.create_task(engine.start())
        assert await _wait_until(lambda: engine.processed_count == 2)

        status = engine.state.get_system_status()
        assert status == "DEGRADED"
        assert int(live_redis.get("observer:gap_count")) >= 1

        journal_text = open(tmp_path / "j.jsonl", encoding="utf-8").read()
        assert '"GAP"' in journal_text
        await engine.stop()

    @pytest.mark.asyncio
    async def test_drift_violation_halts_system(self, tmp_path, live_redis):
        ex = FakeExchange()
        for i in range(1, 61):
            ex.add(i, drift_us=900_000)  # mean stays far above 500ms budget
        engine = _build_engine(tmp_path)
        engine.add_exchange(ex)

        start_task = asyncio.create_task(engine.start())
        assert await _wait_until(lambda: engine.processed_count == 60)

        assert engine.state.get_system_status() == "HALT"
        journal_text = open(tmp_path / "j.jsonl", encoding="utf-8").read()
        assert '"Drift Violation"' in journal_text
        await engine.stop()

    @pytest.mark.asyncio
    async def test_heartbeat_fresh_after_every_packet(self, tmp_path, live_redis):
        """THE Phase 4 bug regression: heartbeat must track packets, not transitions."""
        ex = FakeExchange()
        ex.add(1)
        engine = _build_engine(tmp_path)
        engine.add_exchange(ex)

        start_task = asyncio.create_task(engine.start())
        assert await _wait_until(lambda: engine.processed_count == 1)

        last_update = int(live_redis.get("observer:last_update"))
        now = Clock.now_us()
        assert abs(now - last_update) < 2_000_000  # fresh within guard tolerance
        await engine.stop()


class TestShutdown:
    @pytest.mark.asyncio
    async def test_stop_is_clean_and_idempotent(self, tmp_path, live_redis):
        ex = FakeExchange()
        ex.add(1)
        engine = _build_engine(tmp_path)
        engine.add_exchange(ex)

        start_task = asyncio.create_task(engine.start())
        assert await _wait_until(lambda: engine.processed_count == 1)

        await engine.stop()
        assert ex.closed
        assert engine.journal.is_closed
        assert not engine.running
        assert all(t.done() or t.cancelled() for t in engine._tasks)

        await engine.stop()  # second call is a no-op, no exception

    @pytest.mark.asyncio
    async def test_cancellation_still_finalizes_journal(self, tmp_path, live_redis):
        """Windows Ctrl+C path: cancel start() => cleanup still runs."""
        ex = FakeExchange()
        ex.add(1)
        engine = _build_engine(tmp_path)
        engine.add_exchange(ex)

        start_task = asyncio.create_task(engine.start())
        assert await _wait_until(lambda: engine.processed_count == 1)

        start_task.cancel()
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(start_task, timeout=3)

        assert engine.journal.is_closed

    @pytest.mark.asyncio
    async def test_producer_failure_fails_loudly(self, tmp_path, live_redis):
        class ExplodingExchange(FakeExchange):
            async def listen(self):
                yield self.packets[0]
                raise RuntimeError("stream died")

        ex = ExplodingExchange()
        ex.add(1)
        engine = _build_engine(tmp_path)
        engine.add_exchange(ex)

        start_task = asyncio.create_task(engine.start())
        with suppress(RuntimeError):
            await asyncio.wait_for(start_task, timeout=3)

        assert engine.state.get_system_status() == "HALT"
        journal_text = open(tmp_path / "j.jsonl", encoding="utf-8").read()
        assert "Critical Failure" in journal_text
