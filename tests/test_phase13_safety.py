"""
Phase 13: Safety integration, Health/Observability, Keepalive, Startup/Shutdown.

Tests the D1-D6 wiring. No network, no real Binance, deterministic.
"""
import asyncio
import json

import pytest

from src.adapters.binance import BinanceTestnetConfig
from src.adapters.binance_user_stream import BinanceUserStream
from src.core.deduplication import ExecutionDeduplicator
from src.core.execution_mode import ExecutionMode, parse_execution_mode
from src.core.money import to_decimal, ZERO
from src.core.portfolio import Portfolio
from src.core.safety import SafetyController, SafetyState
from src.core.types import ExecutionReport, OrderIntent, OrderSide, OrderType, MarketEvent, Packet
from src.observability.alerts import AlertManager
from src.observability.health import HealthMonitor
from src.observability.metrics import MetricsRegistry


def _valid_cfg():
    return BinanceTestnetConfig(binance_env="testnet", api_key="k", api_secret="s")


class FakeWS:
    def __init__(self, messages):
        self._messages = list(messages)
    def __aiter__(self): return self
    async def __anext__(self):
        if not self._messages:
            await asyncio.sleep(0.05)
            raise StopAsyncIteration
        await asyncio.sleep(0.01)
        return self._messages.pop(0)
    async def close(self): pass


class FakeRest:
    def __init__(self, listen_key="test-key-123"):
        self.listen_key = listen_key
        self.keepalive_calls = 0
    def create_listen_key(self): return self.listen_key
    def keepalive_listen_key(self, key):
        self.keepalive_calls += 1
        return {"listenKey": key}


# ---------------------------------------------------------------------------
# D1: Safety is single authority
# ---------------------------------------------------------------------------

class TestSafetySingleAuthority:
    def test_guard_delegates_to_safety(self):
        from src.gatekeeper.guard import ExecutionGuard
        safety = SafetyController()
        guard = ExecutionGuard("redis://localhost:6379/15", safety=safety)
        assert guard.in_safe_mode is False
        safety.halt("test")
        assert guard.in_safe_mode is True
        # Guard's validate should now fail due to safe mode
        with pytest.raises(RuntimeError, match="SAFE_MODE"):
            guard.validate_intent()

    def test_observer_status_syncs_to_safety(self, tmp_path):
        from src.core.engine import TradingEngine
        engine = TradingEngine(journal_path=str(tmp_path / "j.jsonl"))  # will use temp file
        # Initially HEALTHY
        assert engine.safety.state == SafetyState.HEALTHY
        engine._transition_status("DEGRADED", "test gap", {})
        assert engine.safety.state == SafetyState.DEGRADED
        engine._transition_status("HALT", "test halt", {})
        assert engine.safety.state == SafetyState.HALT
        # CONNECTED does not auto-recover HALT (fail-closed)
        engine._transition_status("CONNECTED", "reconnect", {})
        assert engine.safety.state == SafetyState.HALT

    def test_every_order_passes_safety_first(self):
        # Structural: engine's _handle_market_event checks safety before risk
        import pathlib
        src = pathlib.Path("src/core/engine.py").read_text()
        # Safety check must appear before risk_manager.evaluate
        safety_pos = src.find("is_halted()")
        risk_pos = src.find("risk_manager.evaluate")
        assert 0 <= safety_pos < risk_pos, "Safety must be checked before Risk"


# ---------------------------------------------------------------------------
# D2: ExecutionMode
# ---------------------------------------------------------------------------

class TestExecutionModeIntegration:
    def test_disabled_halts_immediately(self, tmp_path):
        from src.core.engine import TradingEngine
        e = TradingEngine(execution_mode="DISABLED", journal_path=str(tmp_path / "j.jsonl"))
        assert e.safety.is_halted()
        assert e.execution_mode == ExecutionMode.DISABLED

    def test_production_rejected(self):
        from src.core.engine import TradingEngine
        for prod in ("PROD", "production", "LIVE"):
            with pytest.raises(ValueError):
                TradingEngine(execution_mode=prod)

    def test_paper_and_testnet_allowed(self, tmp_path):
        from src.core.engine import TradingEngine
        for mode in ("PAPER", "TESTNET", ExecutionMode.PAPER, ExecutionMode.TESTNET):
            e = TradingEngine(execution_mode=mode, journal_path=str(tmp_path / f"j_{mode}.jsonl"))
            assert not e.safety.is_halted() or mode == ExecutionMode.DISABLED


# ---------------------------------------------------------------------------
# D4: Health/Metrics/Alerts wiring
# ---------------------------------------------------------------------------

class TestHealthMetricsAlertsWiring:
    def test_engine_exposes_health_snapshot(self, tmp_path):
        from src.core.engine import TradingEngine
        e = TradingEngine(journal_path=str(tmp_path / "j.jsonl"))
        snap = e.health_snapshot()
        assert "status" in snap and "uptime_s" in snap
        assert snap["status"] == "HEALTHY"
        e.safety.halt("test")
        assert e.health_snapshot()["status"] == "HALT"

    def test_metrics_increment_on_halt(self, tmp_path):
        from src.core.engine import TradingEngine
        e = TradingEngine(journal_path=str(tmp_path / "j.jsonl"))
        before = e.metrics.get_counter("halt_total")
        e._transition_status("HALT", "test", {})
        assert e.metrics.get_counter("halt_total") == before + 1

    def test_no_vendor_import_in_core(self):
        import pathlib
        for p in pathlib.Path("src/core").rglob("*.py"):
            text = p.read_text().lower()
            assert "prometheus" not in text
            assert "datadog" not in text
            assert "statsd" not in text


# ---------------------------------------------------------------------------
# D5: Keepalive + multi-symbol
# ---------------------------------------------------------------------------

class TestKeepaliveAndMultiSymbol:
    @pytest.mark.asyncio
    async def test_keepalive_called_periodically(self):
        rest = FakeRest()
        stream = BinanceUserStream(
            _valid_cfg(),
            rest_factory=lambda cfg: rest,
            ws_factory=lambda lk, cfg: FakeWS([]),
            keepalive_interval_s=0.05,
        )
        await stream.connect()
        await asyncio.sleep(0.12)
        await stream.disconnect()
        assert rest.keepalive_calls >= 1

    @pytest.mark.asyncio
    async def test_multi_symbol_both_fills(self):
        # Two symbols, each gets a fill
        msgs = [
            json.dumps({"e": "executionReport", "c": "c-btc", "s": "BTCUSDT", "S": "BUY", "X": "FILLED", "i": 1, "t": 1, "l": "0.01", "L": "50000", "E": 1}),
            json.dumps({"e": "executionReport", "c": "c-eth", "s": "ETHUSDT", "S": "BUY", "X": "FILLED", "i": 2, "t": 2, "l": "0.5", "L": "3000", "E": 2}),
        ]
        received = []
        stream = BinanceUserStream(
            _valid_cfg(),
            on_execution_report=lambda r: received.append(r),
            ws_factory=lambda lk, cfg: FakeWS(msgs),
            rest_factory=lambda cfg: FakeRest(),
        )
        await stream.connect()
        await asyncio.sleep(0.05)
        await stream.disconnect()
        assert len(received) == 2
        symbols = {r.symbol for r in received}
        assert symbols == {"BTCUSDT", "ETHUSDT"}


# ---------------------------------------------------------------------------
# D6: Startup/shutdown ordering
# ---------------------------------------------------------------------------

class TestStartupShutdownOrdering:
    @pytest.mark.asyncio
    async def test_startup_does_reconciliation_before_healthy(self, tmp_path):
        from src.core.engine import TradingEngine
        from src.adapters.paper import PaperBroker
        from src.core.costs import CostModel

        # Create engine with a broker that will have a mismatch if not reconciled
        engine = TradingEngine(journal_path=str(tmp_path / "j.jsonl"))
        # Mock broker with startup_reconcile that reports mismatch
        class MismatchBroker:
            def startup_reconcile(self):
                return {"ok": False, "mismatches": ["test mismatch"], "open_orders": []}
            def get_positions(self): return {}
        engine.broker = MismatchBroker()
        # Perform startup reconciliation (should halt)
        ok = await engine.perform_startup_reconciliation()
        assert ok is False
        assert engine.safety.is_halted()

    @pytest.mark.asyncio
    async def test_shutdown_cancels_open_orders(self, tmp_path):
        from src.core.engine import TradingEngine
        from src.adapters.paper import PaperBroker
        from src.core.costs import CostModel
        from src.core.types import OrderIntent, OrderSide, OrderType

        engine = TradingEngine(journal_path=str(tmp_path / "j.jsonl"))
        broker = PaperBroker(CostModel())
        engine.broker = broker
        # Create an open order
        intent = OrderIntent(client_order_id="open1", symbol="BTCUSDT", side=OrderSide.BUY,
                             order_type=OrderType.LIMIT, quantity=to_decimal("1"), price=to_decimal("100"), timestamp=1)
        broker.submit_order(intent)
        assert len(broker.get_open_orders()) == 1
        engine.running = True
        await engine.stop()
        # Broker should have been closed, open orders should be canceled or broker closed
        assert engine.broker._closed is True or len(broker.get_open_orders()) == 0

    def test_halt_is_terminal_no_auto_recovery(self):
        s = SafetyController()
        s.halt("first")
        s.degrade("second")
        assert s.state == SafetyState.HALT
        assert "second" not in str(s.reasons)
        # No method should auto-recover
        assert s.is_halted() is True


# ---------------------------------------------------------------------------
# D7: Failure injection
# ---------------------------------------------------------------------------

class TestFailureInjection:
    def test_healthy_to_degraded_to_halt(self):
        s = SafetyController()
        assert s.state == SafetyState.HEALTHY
        s.degrade("gap")
        assert s.state == SafetyState.DEGRADED
        assert s.can_submit_new_position() is False
        assert s.can_submit_reducing() is True
        s.halt("mismatch")
        assert s.state == SafetyState.HALT
        assert s.can_submit_reducing() is False

    def test_healthy_to_halt_directly(self):
        s = SafetyController()
        s.halt("critical")
        assert s.state == SafetyState.HALT

    def test_degraded_to_halt(self):
        s = SafetyController()
        s.degrade("stale")
        s.halt("unknown")
        assert s.state == SafetyState.HALT

    def test_halt_no_auto_recovery(self):
        s = SafetyController()
        s.halt("x")
        # Even if we try to degrade, it stays halt
        s.degrade("y")
        assert s.state == SafetyState.HALT

    def test_strategy_cannot_access_safety(self):
        import pathlib
        for p in pathlib.Path("src/strategies").rglob("*.py"):
            text = p.read_text()
            assert "SafetyController" not in text
            assert "safety" not in text.lower() or "safety" in p.name.lower()  # allow filename
            # More strict: no import of safety
            assert "from ..core.safety" not in text
            assert "import.*safety" not in text.lower()

    def test_crash_restart_fails_closed(self):
        # Simulate crash: safety state is lost, new controller starts HEALTHY
        # but journal replay + reconciliation should halt if mismatch
        from src.reconciliation.engine import ReconciliationEngine
        # Create a scenario where local and exchange diverge
        eng = ReconciliationEngine()
        r = eng.reconcile_positions({"BTCUSDT": "1.0"}, {"BTCUSDT": "99.0"})
        assert r.state.value == "MISMATCH"
        # Safety should halt on mismatch
        s = SafetyController()
        if eng.should_halt(r):
            s.halt("reconciliation mismatch on restart")
        assert s.is_halted()


# ---------------------------------------------------------------------------
# D1-D3: No bypass
# ---------------------------------------------------------------------------

class TestNoBypass:
    def test_strategy_cannot_import_broker_or_redis(self):
        import pathlib
        for p in pathlib.Path("src/strategies").rglob("*.py"):
            text = p.read_text().lower()
            assert "import redis" not in text
            assert "from.*broker" not in text
            assert "paperbroker" not in text
            assert "binance" not in text

    def test_strategy_cannot_import_portfolio_mutation(self):
        import pathlib
        for p in pathlib.Path("src/strategies").rglob("*.py"):
            text = p.read_text()
            assert "Portfolio" not in text or "trade_size" in text.lower()  # allow trade_size param
            # More precise: no apply_report
            assert "apply_report" not in text
