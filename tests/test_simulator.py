"""
Unit tests for the Deterministic Simulator.
Tests: Decimal context, state hashing, replay determinism.
Migrated from unittest to pytest (Phase 0). Logic unchanged; tempfile replaced by tmp_path fixture.
"""
import json
from decimal import Decimal

from src.simulator.context import (
    init_decimal_context,
    DeterministicRNG,
    SimulatorConfig,
    DECIMAL_CONTEXT,
)
from src.simulator.state_hasher import StateHasher
from src.simulator.state_store import SimulatedStateStore
from src.simulator.replay_engine import ReplayEngine
from src.simulator.verdict import VerdictStatus


class TestDecimalContext:
    def test_context_initialization(self):
        init_decimal_context()
        assert DECIMAL_CONTEXT.prec == 28

    def test_decimal_determinism(self):
        init_decimal_context()
        a = Decimal("1.123456789012345678901234567")
        b = Decimal("2.987654321098765432109876543")
        result1 = a + b
        result2 = a + b
        assert result1 == result2


class TestDeterministicRNG:
    def test_reproducibility(self):
        rng1 = DeterministicRNG(seed=12345)
        rng2 = DeterministicRNG(seed=12345)

        for _ in range(100):
            assert rng1.randint(0, 1000) == rng2.randint(0, 1000)

    def test_different_seeds(self):
        rng1 = DeterministicRNG(seed=1)
        rng2 = DeterministicRNG(seed=2)
        # Very unlikely to be equal across 10 iterations
        different = False
        for _ in range(10):
            if rng1.randint(0, 1000000) != rng2.randint(0, 1000000):
                different = True
                break
        assert different


class TestStateHasher:
    def test_hash_determinism(self):
        state1 = {"positions": {"BTCUSDT": "1.5"}, "orders": {}}
        state2 = {"positions": {"BTCUSDT": "1.5"}, "orders": {}}
        assert StateHasher.hash_state(state1) == StateHasher.hash_state(state2)

    def test_hash_sensitivity(self):
        state1 = {"positions": {"BTCUSDT": "1.5"}}
        state2 = {"positions": {"BTCUSDT": "1.6"}}
        assert StateHasher.hash_state(state1) != StateHasher.hash_state(state2)


class TestSimulatedStateStore:
    def test_position_updates(self):
        store = SimulatedStateStore()
        store.update_position("BTCUSDT", Decimal("1.5"))
        store.update_position("BTCUSDT", Decimal("-0.5"))
        assert store.get_position("BTCUSDT") == Decimal("1.0")

    def test_state_hash_changes(self):
        store = SimulatedStateStore()
        hash1 = store.get_state_hash()
        store.update_position("BTCUSDT", Decimal("1.0"))
        assert store.get_state_hash() != hash1


def _write_journal(tmp_path, events):
    journal_path = tmp_path / "journal.jsonl"
    lines = [json.dumps(event) + "\n" for event in events]
    journal_path.write_text("".join(lines), encoding="utf-8")
    return str(journal_path)


def _make_config(journal_path):
    return SimulatorConfig(
        config_hash="test",
        rng_seed=42,
        journal_path=journal_path,
    )


class TestReplayEngine:
    def test_empty_journal(self, tmp_path):
        journal_path = _write_journal(tmp_path, [])

        engine = ReplayEngine(_make_config(journal_path))
        verdict = engine.run()

        assert verdict.status == VerdictStatus.PASS
        assert verdict.events_processed == 0

    def test_replay_produces_hash_log(self, tmp_path):
        event = {
            "event_type": "PACKET",
            "timestamp": 1000000,
            "data": {"source": "binance_ws", "drift_us": 100},
        }
        journal_path = _write_journal(tmp_path, [event])

        engine = ReplayEngine(_make_config(journal_path))
        verdict = engine.run()

        assert verdict.status == VerdictStatus.PASS
        assert len(engine.hash_log) == 1

    def test_status_change_event_processed(self, tmp_path):
        event = {
            "event_type": "STATUS_CHANGE",
            "timestamp": 2000000,
            "data": {"status": "DEGRADED", "reason": "gap", "payload": {}},
        }
        journal_path = _write_journal(tmp_path, [event])

        engine = ReplayEngine(_make_config(journal_path))
        verdict = engine.run()

        assert verdict.status == VerdictStatus.PASS
        assert engine.state.system_status == "DEGRADED"

    def test_execution_report_updates_positions(self, tmp_path):
        packet = {
            "event_type": "PACKET",
            "timestamp": 3000000,
            "data": {
                "source": "execution_report",
                "status": "FILLED",
                "symbol": "BTCUSDT",
                "client_order_id": "o-1",
                "side": "BUY",
                "filled_quantity": 0.5,
                "drift_us": 0,
            },
        }
        journal_path = _write_journal(tmp_path, [packet])

        engine = ReplayEngine(_make_config(journal_path))
        verdict = engine.run()

        assert verdict.status == VerdictStatus.PASS
        assert engine.state.get_position("BTCUSDT") == Decimal("0.5")
