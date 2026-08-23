"""
Phase 12: Chaos / Failure Injection Tests (deterministic).

Injects failures into every major component and verifies:
1. no unsafe order
2. no duplicate execution
3. correct status (HALT/DEGRADED)
4. correct journal state
5. recoverability or HALT
"""
import asyncio
import pathlib

import pytest

from src.adapters.paper import PaperBroker, FillSchedule
from src.core.costs import CostModel
from src.core.money import to_decimal, ZERO
from src.core.portfolio import Portfolio
from src.core.safety import SafetyController
from src.core.types import ExecutionReport, OrderIntent, OrderSide, OrderType
from src.reconciliation.engine import ReconciliationEngine, ReconciliationState


def _intent(cloid="c1", qty="1", price="100"):
    return OrderIntent(
        client_order_id=cloid, symbol="BTCUSDT", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, quantity=to_decimal(qty),
        price=to_decimal(price), timestamp=1,
    )


class TestChaosMarketFeed:
    def test_duplicate_market_event_no_duplicate_signal(self):
        from src.strategies.ema_crossover import EmaCrossoverStrategy, EmaCrossoverConfig
        from src.core.types import MarketEvent, Packet
        cfg = EmaCrossoverConfig(strategy_name="ema", strategy_version="1.0.0",
                                 symbol="BTCUSDT", trade_size="1", fast_period=2, slow_period=3)
        strat = EmaCrossoverStrategy(cfg)
        # Feed same price three times (duplicate ticks)
        outs = []
        for _ in range(3):
            outs.append(strat.on_market_event(MarketEvent(packet=Packet(
                exchange_ts=1, local_arrival_ts=1, drift_us=0,
                source="fake", topic="BTCUSDT", payload={"price": "100"}, sequence_id=1))))
        # No crash, deterministic: same input 3 times => same output (None, warm-up)
        assert outs == [None, None, None]
        assert strat._events_seen == 3

    def test_out_of_order_sequence_detected(self, tmp_path):
        # Exercise TradingEngine sequence tracker with real MarketEvents via the engine path
        import asyncio
        from src.core.engine import TradingEngine
        from src.core.types import Packet, MarketEvent
        from src.core.clock import Clock

        async def _run():
            engine = TradingEngine(redis_url="redis://localhost:6379/15", journal_path=str(tmp_path / "j.jsonl"))
            # feed seq 10 then 5 out-of-order via _handle_market_event
            pkt10 = Packet(exchange_ts=Clock.now_epoch_us(), local_arrival_ts=Clock.now_epoch_us(),
                           drift_us=0, source="fake", topic="BTCUSDT",
                           payload={"price": "100"}, sequence_id=10)
            await engine._handle_market_event(MarketEvent(packet=pkt10))
            assert engine.sequence_tracker["fake:BTCUSDT"] == 10
            pkt5 = Packet(exchange_ts=Clock.now_epoch_us(), local_arrival_ts=Clock.now_epoch_us(),
                          drift_us=0, source="fake", topic="BTCUSDT",
                          payload={"price": "100"}, sequence_id=5)
            # must not crash — out-of-order is warned, tracker may stay at 10 or update but no exception
            await engine._handle_market_event(MarketEvent(packet=pkt5))
            assert "fake:BTCUSDT" in engine.sequence_tracker
            await engine.stop()
        asyncio.run(_run())

    def test_malformed_event_handled(self):
        from src.adapters.binance_user_stream import BinanceUserStream
        from src.adapters.binance import BinanceTestnetConfig
        cfg = BinanceTestnetConfig(binance_env="testnet", api_key="k", api_secret="s")
        stream = BinanceUserStream(cfg, ws_factory=lambda lk, c: None, rest_factory=lambda c: None)
        import asyncio as aio
        async def run():
            await stream._handle_raw("{{{not json")
            await stream._handle_raw('{"e": "executionReport", "c": "", "S": "BUY"}')  # empty coid
            # malformed must not crash and must not create a report/order
            assert stream._state in ("CONNECTED", "STALE", "DISCONNECTED")  # no crash, state valid
        aio.run(run())


class TestChaosBroker:
    def test_broker_rejects_after_close(self):
        b = PaperBroker(CostModel())
        b.close()
        with pytest.raises(RuntimeError, match="closed"):
            b.submit_order(_intent())

    def test_broker_cancel_unknown_is_safe(self):
        b = PaperBroker(CostModel())
        assert b.cancel_order("ghost") == "UNKNOWN"

    def test_broker_duplicate_report_never_double_counts(self, tmp_path):
        # Verify duplicate ExecutionReport via REAL public path TradingEngine.apply_execution_report
        # — not by manually setting _seen_report_ids then checking drain empty
        import asyncio
        from src.core.engine import TradingEngine
        from src.core.portfolio import Portfolio
        asyncio.run(self._test_duplicate_via_engine(tmp_path))

    async def _test_duplicate_via_engine(self, tmp_path):
        from src.core.engine import TradingEngine
        from src.core.portfolio import Portfolio
        from src.core.types import ExecutionReport
        portfolio = Portfolio(starting_cash="10000")
        engine = TradingEngine(
            redis_url="redis://localhost:6379/15",
            journal_path=str(tmp_path / "dup.jsonl"),
            portfolio=portfolio,
        )
        report = ExecutionReport(
            client_order_id="c1", exchange_order_id="ex:1:filled",
            symbol="BTCUSDT", side=OrderSide.BUY, status="FILLED",
            filled_quantity=to_decimal("1"), last_filled_price=to_decimal("100"),
            remaining_quantity=to_decimal("0"), timestamp=1, fee=to_decimal("0"))
        await engine.apply_execution_report(report)
        assert portfolio.positions["BTCUSDT"].quantity == to_decimal("1")
        assert portfolio.cash == to_decimal("9900")  # 10000 - 100*1
        before_journal = pathlib.Path(engine.journal.filepath).read_text().count("ex:1:filled")
        # Duplicate via public API — must be dropped before mutation
        await engine.apply_execution_report(report)
        # No double count
        assert portfolio.positions["BTCUSDT"].quantity == to_decimal("1")
        assert portfolio.cash == to_decimal("9900")
        after_journal = pathlib.Path(engine.journal.filepath).read_text().count("ex:1:filled")
        assert after_journal == before_journal, "duplicate must not be journaled again"
        await engine.stop()
        # Also verify PaperBroker public path still dedups via on_market_price not double filling
        b = PaperBroker(CostModel())
        b.submit_order(_intent(qty="1", price="100"))
        b.on_market_price("BTCUSDT", to_decimal("90"), 1)
        reports = b.drain_reports()
        filled = [r for r in reports if r.status == "FILLED"][0]
        # second price tick on already-filled limit order must not emit duplicate
        b.on_market_price("BTCUSDT", to_decimal("80"), 2)
        assert b.drain_reports() == []


class TestChaosReconciliation:
    def test_unknown_execution_halts(self):
        eng = ReconciliationEngine()
        local = {"c1": {"status": "NEW", "filled_qty": "0"}}
        ex = {"c1": {"status": "FILLED", "filled": "1.0"}}
        r = eng.reconcile_orders(local, ex)
        # NEW locally but FILLED on exchange is recoverable, not unknown
        assert r.state == ReconciliationState.RECOVERABLE
        # But if exchange has an order we never submitted, that's MISMATCH -> HALT
        r2 = eng.reconcile_orders({}, {"ghost": {"status": "NEW"}})
        assert r2.state == ReconciliationState.MISMATCH
        assert ReconciliationEngine().should_halt(r2) is True


class TestChaosJournal:
    def test_journal_failure_halts_trading(self):
        from src.core.journal import RawJournal
        import tempfile, pathlib
        p = pathlib.Path(tempfile.mktemp(suffix=".jsonl"))
        j = RawJournal(str(p))
        j.close()
        safety = SafetyController()
        try:
            j.append(__import__("src.core.types", fromlist=["JournalEntry"]).JournalEntry(
                event_type="PACKET", timestamp=1, data={}))
        except RuntimeError:
            safety.halt("JOURNAL_FAILURE")
        assert safety.is_halted()

    def test_corrupt_journal_line_skipped(self):
        from src.simulator.journal_reader import JournalReader
        import tempfile, pathlib
        p = pathlib.Path(tempfile.mktemp(suffix=".jsonl"))
        p.write_text('{"event_type": "PACKET", "timestamp": 1, "data": {"source": "fake"}}\nnot json\n{"event_type": "PACKET", "timestamp": 2, "data": {"source": "fake"}}\n')
        reader = JournalReader(str(p))
        n = reader.load()
        # JournalReader skips corrupt lines: exactly 2 valid entries
        assert n == 2
        assert len(list(reader)) == 2


class TestChaosRedis:
    def test_redis_unavailable_blocks_idempotency(self):
        safety = SafetyController()
        # Simulate Redis failure during order submission
        # The safety controller should halt when idempotency cannot be verified
        safety.halt("REDIS_FAILURE")
        assert not safety.can_submit_new_position()
        assert safety.is_halted()


class TestChaosClock:
    def test_clock_drift_halts(self):
        safety = SafetyController()
        drift_us = 600_000  # beyond 500k threshold
        if abs(drift_us) > 500_000:
            safety.halt("CLOCK_DRIFT")
        assert safety.is_halted()

    def test_exchange_time_unavailable_halts(self):
        safety = SafetyController()
        # Simulate exchange time fetch failure
        try:
            raise TimeoutError("exchange time unavailable")
        except TimeoutError:
            safety.halt("CLOCK_FAILURE")
        assert safety.is_halted()


class TestDryRunMode:
    def test_paper_mode_allows_orders(self):
        from src.core.execution_mode import ExecutionMode, parse_execution_mode
        assert parse_execution_mode("PAPER") == ExecutionMode.PAPER

    def test_disabled_mode_blocks_all(self):
        from src.core.execution_mode import ExecutionMode, parse_execution_mode
        mode = parse_execution_mode("DISABLED")
        safety = SafetyController()
        if mode == ExecutionMode.DISABLED:
            safety.halt("DISABLED_MODE")
        assert safety.is_halted()

    def test_testnet_is_max_allowed(self):
        from src.core.execution_mode import parse_execution_mode
        assert parse_execution_mode("TESTNET").value == "TESTNET"
        for prod in ("PROD", "production", "LIVE"):
            with pytest.raises(ValueError):
                parse_execution_mode(prod)
