"""Phase 15 R4: PortfolioSession orchestrator tests."""
import asyncio

import pytest

from src.adapters.paper import PaperBroker
from src.core.costs import CostModel
from src.core.engine import TradingEngine
from src.core.execution_mode import ExecutionMode
from src.core.money import to_decimal
from src.core.portfolio import Portfolio
from src.core.risk_manager import RiskManager, RiskLimits
from src.core.safety import SafetyController, SafetyState
from src.core.session import PortfolioSession, SessionError
from src.reconciliation.engine import ReconciliationEngine, ReconciliationState
from src.strategies.ema_crossover import EmaCrossoverConfig, EmaCrossoverStrategy


def _limits():
    return RiskLimits(
        max_order_size=to_decimal("10"), max_position_size=to_decimal("20"),
        max_open_positions=3, max_daily_loss=to_decimal("100"),
        max_drawdown_pct=to_decimal("10"), stale_data_us=600_000_000,
        cooldown_us=30)


def _make_session(tmp_path, mode="PAPER", broker=None):
    safety = SafetyController()
    portfolio = Portfolio(starting_cash="10000")
    strategy = EmaCrossoverStrategy(EmaCrossoverConfig(
        strategy_name="ema", symbol="BTCUSDT", trade_size="0.1",
        fast_period=2, slow_period=3))
    engine = TradingEngine(
        redis_url="redis://localhost:6379/15",
        journal_path=str(tmp_path / "j.jsonl"),
        portfolio=portfolio, strategy=strategy,
        risk_manager=RiskManager(portfolio, _limits(), lambda: "CONNECTED"),
        gatekeeper=None,  # no gatekeeper for unit test (no Redis dependency)
        broker=broker or PaperBroker(CostModel()),
        safety=safety, execution_mode=mode,
    )
    session = PortfolioSession(engine)
    return session, engine


class TestSessionLifecycle:
    def test_disabled_mode_starts_halted(self, tmp_path):
        session, _ = _make_session(tmp_path, mode="DISABLED")
        assert session.safety.is_halted()

    def test_production_mode_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            _make_session(tmp_path, mode="PRODUCTION")

    @pytest.mark.asyncio
    async def test_session_start_and_stop(self, tmp_path):
        session, engine = _make_session(tmp_path)
        # Start without exchanges -- should not crash
        start = asyncio.create_task(engine.start())
        await asyncio.sleep(0.05)
        assert session._running or engine.running
        await session.stop()
        assert not session.is_running

    def test_pause_resume(self, tmp_path):
        session, _ = _make_session(tmp_path)
        session.pause()
        assert session.safety.is_degraded()
        assert not session.safety.can_submit_new_position()
        assert session.safety.can_submit_reducing() is True
        session.resume()
        assert session.safety.state == SafetyState.HEALTHY

    def test_cannot_resume_from_halt(self, tmp_path):
        session, _ = _make_session(tmp_path)
        session.engine.safety.halt("test")
        with pytest.raises(SessionError, match="HALT"):
            session.resume()

    @pytest.mark.asyncio
    async def test_double_start_rejected(self, tmp_path):
        import unittest.mock as mock
        session, _ = _make_session(tmp_path)
        # Patch engine.start to avoid blocking the event loop (real engine.start awaits forever on gath
        # The guard under test is PortfolioSession.start()'s `if self._running: raise SessionError`
        orig_start = session.engine.start
        session.engine.start = mock.AsyncMock(return_value=None)
        try:
            # 1. first start succeeds via public API
            await session.start()
            assert session.is_running is True
            # 2. second start actually executes and must raise SessionError
            with pytest.raises(SessionError, match="already"):
                await session.start()
            # 3. session remains running
            assert session.is_running is True
            # 4. cleanup via public API
            await session.stop()
            assert session.is_running is False
            # 5. after stop, start again succeeds (proves cleanup)
            await session.start()
            assert session.is_running is True
        finally:
            session.engine.start = orig_start
            await session.stop()

    def test_non_engine_rejected(self):
        with pytest.raises(TypeError):
            PortfolioSession("not an engine")


class TestSessionReconciliationGate:
    @pytest.mark.asyncio
    async def test_mismatch_halts_before_trading(self, tmp_path):
        session, engine = _make_session(tmp_path)

        class MismatchBroker:
            def startup_reconcile(self):
                return {"ok": False, "mismatches": ["ghost order"],
                        "open_orders": []}
            def get_positions(self): return {}

        engine.broker = MismatchBroker()
        with pytest.raises(SessionError, match="reconciliation"):
            await session.start()
        assert engine.safety.is_halted()
        assert session.is_running is False

    @pytest.mark.asyncio
    async def test_consistent_reconciliation_allows_trading(self, tmp_path):
        session, engine = _make_session(tmp_path)

        class OkBroker:
            def startup_reconcile(self):
                return {"ok": True, "mismatches": [], "open_orders": [], "positions": {}}
            def get_positions(self): return {}
        engine.broker = OkBroker()
        # Just call perform_startup_reconciliation directly
        ok = await engine.perform_startup_reconciliation()
        assert ok is True
        assert not engine.safety.is_halted()


class TestSessionReconciliationTrigger:
    def test_trigger_reconciliation_consistent(self, tmp_path):
        session, engine = _make_session(tmp_path)
        result = session.trigger_reconciliation(
            exchange_orders={},
            exchange_balances={"BTCUSDT": "0"})
        assert result["state"] == "CONSISTENT"
        assert not engine.safety.is_halted()

    def test_trigger_reconciliation_mismatch_halts(self, tmp_path):
        session, engine = _make_session(tmp_path)
        from src.core.types import ExecutionReport, OrderSide
        pf_rep = ExecutionReport(
            client_order_id="s", exchange_order_id="sx", symbol="BTCUSDT",
            side=OrderSide.BUY, status="FILLED", filled_quantity=to_decimal("5"),
            last_filled_price=to_decimal("100"), remaining_quantity=to_decimal("0"),
            timestamp=1, fee=to_decimal("0"))
        engine.portfolio.apply_report(pf_rep)

        result = session.trigger_reconciliation(
            exchange_orders={},
            exchange_balances={"BTCUSDT": "99"})
        assert result["state"] == "MISMATCH"
        assert engine.safety.is_halted()


class TestSessionHealthAndAlerts:
    def test_health_snapshot_available(self, tmp_path):
        session, _ = _make_session(tmp_path)
        snap = session.health_snapshot()
        assert "status" in snap
        assert snap["status"] == "HEALTHY"

    def test_health_shows_degraded_after_pause(self, tmp_path):
        session, _ = _make_session(tmp_path)
        session.pause()
        snap = session.health_snapshot()
        assert snap["safety_state"] == "DEGRADED"

    def test_metrics_track_transitions(self, tmp_path):
        session, _ = _make_session(tmp_path)
        before = session.engine.metrics.get_counter("halt_total")
        # halt via engine path increments metric; use engine's safety to trigger transition
        session.engine.safety.halt("metric test")
        # Direct SafetyController halt does not auto-inc metric — engine does on transition
        # So verify at least safety is halted and metric is not decreased
        assert session.engine.safety.is_halted()
        after = session.engine.metrics.get_counter("halt_total")
        assert after == before  # direct halt does not increment; engine transition would
        # Now test via engine transition path
        session2, _ = _make_session(tmp_path)
        before2 = session2.engine.metrics.get_counter("halt_total")
        session2.engine._transition_status("HALT", "test halt", {})
        after2 = session2.engine.metrics.get_counter("halt_total")
        assert after2 == before2 + 1
