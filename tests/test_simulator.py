"""
Unit tests for the Deterministic Simulator.
Tests: Decimal context, state hashing, replay determinism.
"""
import unittest
import tempfile
import json
import os
from decimal import Decimal
from src.simulator.context import (
    init_decimal_context, 
    DeterministicRNG, 
    SimulatorConfig,
    DECIMAL_CONTEXT
)
from src.simulator.state_hasher import StateHasher
from src.simulator.state_store import SimulatedStateStore
from src.simulator.replay_engine import ReplayEngine
from src.simulator.verdict import VerdictStatus

class TestDecimalContext(unittest.TestCase):
    def test_context_initialization(self):
        init_decimal_context()
        # Verify precision
        self.assertEqual(DECIMAL_CONTEXT.prec, 28)

    def test_decimal_determinism(self):
        init_decimal_context()
        a = Decimal("1.123456789012345678901234567")
        b = Decimal("2.987654321098765432109876543")
        result1 = a + b
        result2 = a + b
        self.assertEqual(result1, result2)

class TestDeterministicRNG(unittest.TestCase):
    def test_reproducibility(self):
        rng1 = DeterministicRNG(seed=12345)
        rng2 = DeterministicRNG(seed=12345)
        
        for _ in range(100):
            self.assertEqual(rng1.randint(0, 1000), rng2.randint(0, 1000))

    def test_different_seeds(self):
        rng1 = DeterministicRNG(seed=1)
        rng2 = DeterministicRNG(seed=2)
        # Very unlikely to be equal across 10 iterations
        different = False
        for _ in range(10):
            if rng1.randint(0, 1000000) != rng2.randint(0, 1000000):
                different = True
                break
        self.assertTrue(different)

class TestStateHasher(unittest.TestCase):
    def test_hash_determinism(self):
        state1 = {"positions": {"BTCUSDT": "1.5"}, "orders": {}}
        state2 = {"positions": {"BTCUSDT": "1.5"}, "orders": {}}
        self.assertEqual(StateHasher.hash_state(state1), StateHasher.hash_state(state2))

    def test_hash_sensitivity(self):
        state1 = {"positions": {"BTCUSDT": "1.5"}}
        state2 = {"positions": {"BTCUSDT": "1.6"}}
        self.assertNotEqual(StateHasher.hash_state(state1), StateHasher.hash_state(state2))

class TestSimulatedStateStore(unittest.TestCase):
    def test_position_updates(self):
        store = SimulatedStateStore()
        store.update_position("BTCUSDT", Decimal("1.5"))
        store.update_position("BTCUSDT", Decimal("-0.5"))
        self.assertEqual(store.get_position("BTCUSDT"), Decimal("1.0"))

    def test_state_hash_changes(self):
        store = SimulatedStateStore()
        hash1 = store.get_state_hash()
        store.update_position("BTCUSDT", Decimal("1.0"))
        hash2 = store.get_state_hash()
        self.assertNotEqual(hash1, hash2)

class TestReplayEngine(unittest.TestCase):
    def test_empty_journal(self):
        # Create empty journal
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write("")
            journal_path = f.name
        
        try:
            config = SimulatorConfig(
                config_hash="test",
                rng_seed=42,
                journal_path=journal_path
            )
            engine = ReplayEngine(config)
            verdict = engine.run()
            self.assertEqual(verdict.status, VerdictStatus.PASS)
            self.assertEqual(verdict.events_processed, 0)
        finally:
            os.unlink(journal_path)

    def test_replay_produces_hash_log(self):
        # Create journal with one event
        event = {
            "event_type": "PACKET",
            "timestamp": 1000000,
            "data": {"source": "binance_ws", "drift_us": 100}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps(event) + "\n")
            journal_path = f.name
        
        try:
            config = SimulatorConfig(
                config_hash="test",
                rng_seed=42,
                journal_path=journal_path
            )
            engine = ReplayEngine(config)
            verdict = engine.run()
            self.assertEqual(verdict.status, VerdictStatus.PASS)
            self.assertEqual(len(engine.hash_log), 1)
        finally:
            os.unlink(journal_path)

if __name__ == '__main__':
    unittest.main()
