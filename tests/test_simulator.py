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


# ==================== Phase 5: portfolio replay parity ====================

from decimal import Decimal as _D

from src.core.money import init_money_context as _init_money
from src.core.portfolio import Portfolio as _Portfolio
from src.core.types import ExecutionReport as _ExecutionReport, OrderSide as _OrderSide
from src.simulator.state_hasher import StateHasher as _StateHasher


def _packet_event(ts, topic, price):
    return {
        "event_type": "PACKET",
        "timestamp": ts,
        "data": {
            "source": "binance_ccxt",
            "topic": topic,
            "exchange_ts": ts,
            "local_arrival_ts": ts,
            "drift_us": 0,
            "payload": {"price": price, "seq": ts},
            "sequence_id": None,
        },
    }


def _fill_event(report):
    # EXACT engine journaling format: PACKET / source="execution_report"
    return {
        "event_type": "PACKET",
        "timestamp": report.timestamp,
        "data": {"source": "execution_report", **report.model_dump(mode="json")},
    }


def _report(cloid, side, qty, price, fee):
    return _ExecutionReport(
        client_order_id=cloid,
        exchange_order_id=f"x-{cloid}",
        symbol="BTCUSDT",
        side=side,
        status="FILLED",
        filled_quantity=qty,
        last_filled_price=price,
        remaining_quantity="0",
        timestamp=0,
        fee=fee,
    )


class TestPortfolioReplay:
    """
    GATE 5 core requirement:
    live-style event processing == journal replay == repeated replay.
    """

    def _events(self):
        r1 = _report("r1", _OrderSide.BUY, "0.5", "100", "0.10")
        r2 = _report("r2", _OrderSide.SELL, "0.2", "110", "0.05")
        return [
            _packet_event(1000, "BTCUSDT", "100"),
            _fill_event(r1),
            _packet_event(2000, "BTCUSDT", "110"),
            _fill_event(r2),
            _packet_event(3000, "BTCUSDT", "120"),
        ], [r1, r2]

    def test_replay_portfolio_matches_live_style_processing(self, tmp_path):
        events, reports = self._events()
        journal_path = _write_journal(tmp_path, events)

        # --- live-style processing (engine-equivalent call order) ---
        _init_money()
        live = _Portfolio(starting_cash=_D("5000"))
        it = iter(reports)
        for ev in events:
            data = ev["data"]
            if data.get("source") == "execution_report":
                live.apply_report(next(it))
            else:
                from src.core.money import to_decimal
                live.mark_price(data["topic"], to_decimal(str(data["payload"]["price"])),
                                ts_us=data["exchange_ts"])
                live.update_equity(now_us=data["exchange_ts"])

        # --- replay ---
        config = SimulatorConfig(config_hash="p5", rng_seed=42,
                                 journal_path=journal_path, initial_cash="5000")
        engine = ReplayEngine(config)
        verdict = engine.run()

        assert verdict.status == VerdictStatus.PASS
        assert engine.portfolio is not None

        # byte-identical financial state through BOTH paths
        assert engine.portfolio.snapshot() == live.snapshot()
        assert (StateHasher.hash_state(engine.portfolio.snapshot())
                == _StateHasher.hash_state(live.snapshot()))

        # spot-check the actual numbers
        pf = engine.portfolio
        assert pf.cash == _D("4971.85")            # 5000-50-.1+22-.05
        assert pf.realized_pnl == _D("2.00")       # (110-100)*0.2
        assert pf.positions["BTCUSDT"].quantity == _D("0.3")
        assert pf.peak_equity == _D("5007.85")     # 4971.85+0.3*120

    def test_repeated_replay_hash_stable_with_portfolio(self, tmp_path):
        events, _ = self._events()
        journal_path = _write_journal(tmp_path, events)

        def run():
            cfg = SimulatorConfig(config_hash="p5", rng_seed=42,
                                  journal_path=journal_path, initial_cash="5000")
            eng = ReplayEngine(cfg)
            v = eng.run()
            return v, eng

        v1, e1 = run()
        v2, e2 = run()
        assert v1.status == VerdictStatus.PASS and v2.status == VerdictStatus.PASS
        assert e1.hash_log == e2.hash_log

    def test_pre_phase5_config_stays_portfolio_free(self, tmp_path):
        """initial_cash=None => no portfolio participation, legacy behavior."""
        event = {"event_type": "PACKET", "timestamp": 1, "data":
                 {"source": "s", "drift_us": 0}}
        path = _write_journal(tmp_path, [event])
        cfg = SimulatorConfig(config_hash="legacy", rng_seed=1, journal_path=path)
        eng = ReplayEngine(cfg)
        verdict = eng.run()
        assert verdict.status == VerdictStatus.PASS
        assert eng.portfolio is None
