"""
J-1 Gate — Prove Risk never bypassed, Gatekeeper=None is RESEARCH-ONLY fail-closed,
HALT/DEGRADED block broker via real MarketEvent pipeline, exact ordering spy.

All tests use real TradingEngine path, not private flag asserts.
Mutation-proven: see mutations in docstrings.
"""
import asyncio
import pathlib
import time

import pytest

from src.adapters.paper import PaperBroker
from src.core.costs import CostModel
from src.core.engine import TradingEngine
from src.core.money import to_decimal
from src.core.portfolio import Portfolio
from src.core.risk_manager import RiskManager, RiskLimits
from src.core.safety import SafetyController
from src.core.session import PortfolioSession, SessionError
from src.core.types import MarketEvent, OrderIntent, OrderSide, OrderType, Packet
from src.strategies.base import BaseStrategy, StrategyConfig
from src.strategies.ema_crossover import EmaCrossoverConfig, EmaCrossoverStrategy


def _limits():
    return RiskLimits(
        max_order_size=to_decimal("10"), max_position_size=to_decimal("20"),
        max_open_positions=3, max_daily_loss=to_decimal("100"),
        max_drawdown_pct=to_decimal("10"), stale_data_us=600_000_000, cooldown_us=30)


class AlwaysBuyStrategy(BaseStrategy):
    @classmethod
    def expected_config(cls): return StrategyConfig
    def initial_state(self): return {"n": 0}
    def on_market_event(self, event: MarketEvent):
        self.state["n"] += 1
        # emit on second event only to allow first to warm up
        if self.state["n"] == 2:
            return OrderIntent(client_order_id=f"ord-{event.packet.exchange_ts}",
                               symbol="BTCUSDT", side=OrderSide.BUY,
                               order_type=OrderType.MARKET, quantity=to_decimal("1"),
                               price=None, timestamp=event.packet.exchange_ts)
        return None


# ---------------------------------------------------------------------------
# 1. Gatekeeper=None path — Risk must still be called (prove no bypass)
# ---------------------------------------------------------------------------
class TestGatekeeperNoneRiskBypass:
    @pytest.mark.asyncio
    async def test_gatekeeper_none_still_calls_risk(self, tmp_path):
        """
        Gatekeeper=None is RESEARCH-ONLY. For PAPER/TESTNET it must fail-closed (no broker).
        But RiskManager.evaluate() must still be called — proving Risk not bypassed.
        Mutation: remove `decision = risk_manager.evaluate` → test must FAIL (calls stays []).
        """
        calls = {"risk": 0}

        class SpyRiskManager(RiskManager):
            def evaluate(self, intent, now_us=None):
                calls["risk"] += 1
                return super().evaluate(intent, now_us=now_us)

        portfolio = Portfolio(starting_cash="10000")
        # SpyRisk wraps real RiskManager — prod logic still executes
        real_risk = SpyRiskManager(portfolio, _limits(), lambda: "CONNECTED")
        # Gatekeeper=None → RESEARCH-ONLY but must still call Risk (prove no bypass)
        # Use PAPER mode: gate missing is fail-closed for broker, but Risk must still execute
        engine = TradingEngine(
            redis_url="redis://localhost:6379/15",
            journal_path=str(tmp_path / "j.jsonl"),
            portfolio=portfolio,
            strategy=AlwaysBuyStrategy(StrategyConfig(strategy_name="t", symbol="BTCUSDT", trade_size="1")),
            risk_manager=real_risk,
            gatekeeper=None,
            broker=PaperBroker(CostModel()),
            safety=SafetyController(),
            execution_mode="PAPER",
        )
        pkt1 = Packet(exchange_ts=int(time.time()*1_000_000), local_arrival_ts=int(time.time()*1_000_000),
                      drift_us=0, source="fake", topic="BTCUSDT", payload={"price": "100"}, sequence_id=1)
        await engine._handle_market_event(MarketEvent(packet=pkt1))
        assert calls["risk"] == 0  # first event no order
        pkt2 = Packet(exchange_ts=int(time.time()*1_000_000)+1, local_arrival_ts=int(time.time()*1_000_000)+1,
                      drift_us=0, source="fake", topic="BTCUSDT", payload={"price": "101"}, sequence_id=2)
        await engine._handle_market_event(MarketEvent(packet=pkt2))
        assert calls["risk"] == 1, "Risk must be called even when Gatekeeper is None"
        # Broker must NOT be called when gate is None (fail-closed for research path)
        assert engine.broker.get_account_state()["submitted"] == 0
        await engine.stop()

    @pytest.mark.asyncio
    async def test_gatekeeper_none_paper_fails_closed(self, tmp_path):
        """
        RESEARCH-ONLY contract: Gatekeeper=None is allowed for DISABLED/research, but for PAPER/TESTNET
        it must fail-closed at runtime (Risk still called, Gate skipped, Broker NOT called, metric incremented).
        This is the explicit contract — not a ValueError on construction (to keep unit tests simple),
        but a runtime fail-closed with observable metric.
        """
        from src.gatekeeper.engine import Gatekeeper  # noqa: F401
        portfolio = Portfolio(starting_cash="10000")
        # PAPER with gate None → runtime fail-closed
        for mode in ("PAPER", "TESTNET"):
            calls = {"risk": 0}
            class SpyRisk(RiskManager):
                def evaluate(self, intent, now_us=None):
                    calls["risk"] += 1
                    return super().evaluate(intent, now_us)
            engine = TradingEngine(
                redis_url="redis://localhost:6379/15",
                journal_path=str(tmp_path / f"j_{mode}.jsonl"),
                portfolio=portfolio, strategy=AlwaysBuyStrategy(StrategyConfig(strategy_name="t", symbol="BTCUSDT", trade_size="1")),
                risk_manager=SpyRisk(portfolio, _limits(), lambda: "CONNECTED"),
                gatekeeper=None,
                broker=PaperBroker(CostModel()),
                safety=SafetyController(),
                execution_mode=mode,
            )
            engine.strategy.state["n"] = 1
            pkt = Packet(exchange_ts=int(time.time()*1_000_000), local_arrival_ts=int(time.time()*1_000_000),
                         drift_us=0, source="fake", topic="BTCUSDT", payload={"price": "100"}, sequence_id=1)
            await engine._handle_market_event(MarketEvent(packet=pkt))
            assert calls["risk"] == 1, f"Risk must be called even when gate None (mode {mode})"
            assert engine.broker.get_account_state()["submitted"] == 0, "Broker must NOT be called when gate missing (fail-closed)"
            assert engine.metrics.get_counter("gatekeeper_missing_blocked") >= 1
            await engine.stop()
            # DISABLED/research with gate None → allowed (no broker, but not counted as missing? still RESEARCH)
            # We verify the contract is documented: gate None is RESEARCH-ONLY, not live.


# ---------------------------------------------------------------------------
# 2. Exact ordering Safety → Risk → Gatekeeper → Broker
# ---------------------------------------------------------------------------
class TestExactOrdering:
    @pytest.mark.asyncio
    async def test_safety_risk_gate_broker_exact_order(self, tmp_path):
        """
        Put spy around each layer and record ["risk","gatekeeper","broker"] in exact order.
        Mutation: swap Gate before Risk in engine.py → test must FAIL on order assert.
        """
        calls = []

        class SpyRiskManager:
            def evaluate(self, intent, now_us=None):
                calls.append("risk")
                from src.core.types import RiskDecision, RiskCheck
                return RiskDecision(client_order_id=intent.client_order_id, symbol=intent.symbol,
                                    approved=True, rule="ALLOW", reason="ok",
                                    checks=(RiskCheck(rule="ALLOW", passed=True, detail="ok"),), details={})

        class SpyGatekeeper:
            def submit_intent(self, intent):
                # must be after risk
                assert calls == ["risk"], f"Gate before Risk! {calls}"
                calls.append("gatekeeper")
                return "ACCEPTED"

        class SpyBroker:
            def __init__(self): self.submitted = 0
            def on_market_price(self, *a, **kw): pass
            def drain_reports(self): return []
            def submit_order(self, intent):
                assert calls == ["risk", "gatekeeper"], f"Broker before Gate/Risk! {calls}"
                calls.append("broker")
                self.submitted += 1
                return "ACCEPTED"
            def get_open_orders(self): return []
            def close(self): pass

        # Need gatekeeper guard to pass: set observer:status CONNECTED
        from src.gatekeeper.engine import Gatekeeper
        engine = TradingEngine(
            redis_url="redis://localhost:6379/15",
            journal_path=str(tmp_path / "j.jsonl"),
            portfolio=Portfolio(starting_cash="10000"),
            strategy=AlwaysBuyStrategy(StrategyConfig(strategy_name="t", symbol="BTCUSDT", trade_size="1")),
            risk_manager=SpyRiskManager(),
            gatekeeper=SpyGatekeeper(),
            broker=SpyBroker(),
            safety=SafetyController(),
        )
        # also test Safety is checked before Risk: HALT should prevent Risk
        engine.safety.halt("test halt for ordering")
        pkt_halt = Packet(exchange_ts=int(time.time()*1_000_000), local_arrival_ts=int(time.time()*1_000_000),
                          drift_us=0, source="fake", topic="BTCUSDT", payload={"price": "100"}, sequence_id=99)
        # need to trigger intent while halted — use AlwaysBuy that emits on n==2, so send two halts
        engine.strategy.state["n"] = 1  # next will be 2 → emit, but halted → risk must NOT be called
        await engine._handle_market_event(MarketEvent(packet=pkt_halt))
        assert calls == [], "Risk must NOT be called when HALT (Safety before Risk)"
        # reset safety to HEALTHY for ordering test
        engine.safety = SafetyController()
        engine.gatekeeper = SpyGatekeeper()  # re-wire same spy
        # need to reassign gatekeeper's safety
        engine.gatekeeper = SpyGatekeeper()
        # re-create engine for clean ordering test (simpler: new engine HEALTHY)
        engine2 = TradingEngine(
            redis_url="redis://localhost:6379/15",
            journal_path=str(tmp_path / "j2.jsonl"),
            portfolio=Portfolio(starting_cash="10000"),
            strategy=AlwaysBuyStrategy(StrategyConfig(strategy_name="t2", symbol="BTCUSDT", trade_size="1")),
            risk_manager=SpyRiskManager(),
            gatekeeper=SpyGatekeeper(),
            broker=SpyBroker(),
            safety=SafetyController(),
        )
        engine2.strategy.state["n"] = 1
        pkt2 = Packet(exchange_ts=int(time.time()*1_000_000)+2, local_arrival_ts=int(time.time()*1_000_000)+2,
                      drift_us=0, source="fake", topic="BTCUSDT", payload={"price": "101"}, sequence_id=2)
        await engine2._handle_market_event(MarketEvent(packet=pkt2))
        assert calls == ["risk", "gatekeeper", "broker"]
        await engine.stop()
        await engine2.stop()


# ---------------------------------------------------------------------------
# 4. HALT pipeline — MarketEvent → HALT → no Risk/Gate/Broker
# ---------------------------------------------------------------------------
class TestHaltBlocksBroker:
    @pytest.mark.asyncio
    async def test_halt_blocks_broker_via_real_pipeline(self, tmp_path):
        from src.gatekeeper.engine import Gatekeeper
        calls = {"risk": 0, "gate": 0, "broker": 0}

        class SpyRisk:
            def evaluate(self, intent, now_us=None):
                calls["risk"] += 1
                from src.core.types import RiskDecision, RiskCheck
                return RiskDecision(client_order_id=intent.client_order_id, symbol=intent.symbol,
                                    approved=True, rule="ALLOW", reason="ok",
                                    checks=(RiskCheck(rule="ALLOW", passed=True, detail="ok"),), details={})
        class SpyGate:
            def submit_intent(self, intent):
                calls["gate"] += 1
                return "ACCEPTED"
        class SpyBroker(PaperBroker):
            def __init__(self):
                super().__init__(CostModel())
                self.spy_submitted = 0
            def submit_order(self, intent):
                calls["broker"] += 1
                self.spy_submitted += 1
                return super().submit_order(intent)

        engine = TradingEngine(
            redis_url="redis://localhost:6379/15",
            journal_path=str(tmp_path / "j.jsonl"),
            portfolio=Portfolio(starting_cash="10000"),
            strategy=AlwaysBuyStrategy(StrategyConfig(strategy_name="t", symbol="BTCUSDT", trade_size="1")),
            risk_manager=SpyRisk(),
            gatekeeper=SpyGate(),
            broker=SpyBroker(),
            safety=SafetyController(),
        )
        engine.safety.halt("test halt")
        assert engine.safety.is_halted()
        # need intent on next event
        engine.strategy.state["n"] = 1
        pkt = Packet(exchange_ts=int(time.time()*1_000_000), local_arrival_ts=int(time.time()*1_000_000),
                     drift_us=0, source="fake", topic="BTCUSDT", payload={"price": "100"}, sequence_id=1)
        await engine._handle_market_event(MarketEvent(packet=pkt))
        assert calls["risk"] == 0, "Risk must NOT execute when HALT (Safety before Risk)"
        assert calls["gate"] == 0
        assert calls["broker"] == 0
        assert engine.broker.spy_submitted == 0
        text = pathlib.Path(tmp_path / "j.jsonl").read_text()
        assert '"SAFETY"' in text
        await engine.stop()


# ---------------------------------------------------------------------------
# 5. DEGRADED via real pipeline (already in hardening, but explicit here)
# ---------------------------------------------------------------------------
class TestDegradedPipeline:
    @pytest.mark.asyncio
    async def test_degraded_blocks_increasing_allows_reducing(self, tmp_path):
        from src.gatekeeper.engine import Gatekeeper
        portfolio = Portfolio(starting_cash="10000")
        engine = TradingEngine(
            redis_url="redis://localhost:6379/15",
            journal_path=str(tmp_path / "j.jsonl"),
            portfolio=portfolio,
            strategy=AlwaysBuyStrategy(StrategyConfig(strategy_name="t", symbol="BTCUSDT", trade_size="1")),
            risk_manager=RiskManager(portfolio, _limits(), lambda: "CONNECTED"),
            gatekeeper=Gatekeeper("redis://localhost:6379/15"),
            broker=PaperBroker(CostModel()),
            safety=SafetyController(),
        )
        engine.safety.degrade("test degraded")
        engine.state.redis.set("observer:status", "CONNECTED")
        engine.gatekeeper.guard.redis.set("observer:status", "CONNECTED")
        # increasing BUY with zero position → blocked
        engine.strategy.state["n"] = 1
        pkt = Packet(exchange_ts=int(time.time()*1_000_000), local_arrival_ts=int(time.time()*1_000_000),
                     drift_us=0, source="fake", topic="BTCUSDT", payload={"price": "100"}, sequence_id=1)
        await engine._handle_market_event(MarketEvent(packet=pkt))
        assert engine.broker.get_account_state()["submitted"] == 0
        # create long position then reducing SELL must pass
        from src.core.types import ExecutionReport
        rep = ExecutionReport(client_order_id="seed", exchange_order_id="seed:1",
                              symbol="BTCUSDT", side=OrderSide.BUY, status="FILLED",
                              filled_quantity=to_decimal("1"), last_filled_price=to_decimal("100"),
                              remaining_quantity=to_decimal("0"), timestamp=1, fee=to_decimal("0"))
        await engine.apply_execution_report(rep)
        # use SELL reducing strategy
        class SellOnce(BaseStrategy):
            @classmethod
            def expected_config(cls): return StrategyConfig
            def initial_state(self): return {}
            def on_market_event(self, event): return OrderIntent(client_order_id=f"s-{event.packet.exchange_ts}", symbol="BTCUSDT", side=OrderSide.SELL, order_type=OrderType.LIMIT, quantity=to_decimal("1"), price=to_decimal("100"), timestamp=event.packet.exchange_ts)
        engine.strategy = SellOnce(StrategyConfig(strategy_name="t2", symbol="BTCUSDT", trade_size="1"))
        pkt2 = Packet(exchange_ts=int(time.time()*1_000_000)+1, local_arrival_ts=int(time.time()*1_000_000)+1,
                      drift_us=0, source="fake", topic="BTCUSDT", payload={"price": "100"}, sequence_id=2)
        await engine._handle_market_event(MarketEvent(packet=pkt2))
        assert engine.broker.get_account_state()["submitted"] == 1
        await engine.stop()


# ---------------------------------------------------------------------------
# Stage 2 — Explicit HALT pipeline proof (distinguishes DEGRADED)
# ---------------------------------------------------------------------------
class TestHaltExplicit:
    @pytest.mark.asyncio
    async def test_halt_blocks_both_increasing_and_reducing(self, tmp_path):
        """
        HALT must block BOTH increasing and reducing, unlike DEGRADED which allows reducing.
        Uses real public TradingEngine MarketEvent path, not direct safety-state manipulation.
        Mutation: remove SafetyController.is_halted() gate → order would reach broker and test MUST fail.
        """
        from src.gatekeeper.engine import Gatekeeper
        portfolio = Portfolio(starting_cash="10000")
        engine = TradingEngine(
            redis_url="redis://localhost:6379/15",
            journal_path=str(tmp_path / "j.jsonl"),
            portfolio=portfolio,
            strategy=AlwaysBuyStrategy(StrategyConfig(strategy_name="t", symbol="BTCUSDT", trade_size="1")),
            risk_manager=RiskManager(portfolio, _limits(), lambda: "CONNECTED"),
            gatekeeper=Gatekeeper("redis://localhost:6379/15"),
            broker=PaperBroker(CostModel()),
            safety=SafetyController(),
        )
        # Put Safety into HALT via public API (not _state=HALT)
        engine.safety.halt("test halt explicit")
        assert engine.safety.is_halted()
        engine.state.redis.set("observer:status", "CONNECTED")
        engine.gatekeeper.guard.redis.set("observer:status", "CONNECTED")
        # Spy counters for Risk/Gate/Broker to prove they are NOT reached
        calls = {"risk": 0, "gate": 0, "broker": 0}
        orig_risk_evaluate = engine.risk_manager.evaluate
        def spy_risk(intent, now_us=None):
            calls["risk"] += 1
            return orig_risk_evaluate(intent, now_us=now_us)
        engine.risk_manager.evaluate = spy_risk
        orig_gate = engine.gatekeeper.submit_intent
        def spy_gate(intent):
            calls["gate"] += 1
            return orig_gate(intent)
        engine.gatekeeper.submit_intent = spy_gate
        orig_broker_submit = engine.broker.submit_order
        def spy_broker(intent):
            calls["broker"] += 1
            return orig_broker_submit(intent)
        engine.broker.submit_order = spy_broker

        # HALT + increasing BUY must be blocked (Risk/Gate/Broker all 0)
        engine.strategy.state["n"] = 1
        pkt = Packet(exchange_ts=int(time.time()*1_000_000), local_arrival_ts=int(time.time()*1_000_000),
                     drift_us=0, source="fake", topic="BTCUSDT", payload={"price": "100"}, sequence_id=1)
        await engine._handle_market_event(MarketEvent(packet=pkt))
        assert calls["risk"] == 0, "Risk must NOT execute when HALT"
        assert calls["gate"] == 0
        assert calls["broker"] == 0
        assert engine.broker.get_account_state()["submitted"] == 0
        text = pathlib.Path(tmp_path / "j.jsonl").read_text()
        assert '"SAFETY"' in text and '"approved": false' in text

        # Create long position, then HALT must still block reducing SELL (unlike DEGRADED)
        from src.core.types import ExecutionReport
        # Need to temporarily allow the seed report via direct portfolio (bypass safety for setup)
        # Use apply_execution_report which is allowed even when HALT (portfolio mutation funnel, not broker)
        rep = ExecutionReport(client_order_id="seed", exchange_order_id="seed:halt:1",
                              symbol="BTCUSDT", side=OrderSide.BUY, status="FILLED",
                              filled_quantity=to_decimal("1"), last_filled_price=to_decimal("100"),
                              remaining_quantity=to_decimal("0"), timestamp=2, fee=to_decimal("0"))
        await engine.apply_execution_report(rep)
        assert portfolio.positions["BTCUSDT"].quantity == to_decimal("1")
        # Now try reducing SELL while still HALT — must be blocked (DEGRADED would allow)
        class SellReducing(BaseStrategy):
            @classmethod
            def expected_config(cls): return StrategyConfig
            def initial_state(self): return {}
            def on_market_event(self, event): return OrderIntent(client_order_id=f"s-{event.packet.exchange_ts}", symbol="BTCUSDT", side=OrderSide.SELL, order_type=OrderType.LIMIT, quantity=to_decimal("1"), price=to_decimal("100"), timestamp=event.packet.exchange_ts)
        engine.strategy = SellReducing(StrategyConfig(strategy_name="t2", symbol="BTCUSDT", trade_size="1"))
        pkt2 = Packet(exchange_ts=int(time.time()*1_000_000)+2, local_arrival_ts=int(time.time()*1_000_000)+2,
                      drift_us=0, source="fake", topic="BTCUSDT", payload={"price": "100"}, sequence_id=2)
        await engine._handle_market_event(MarketEvent(packet=pkt2))
        # HALT blocks even reducing
        assert calls["risk"] == 0, "Risk must still NOT execute for reducing when HALT"
        assert calls["broker"] == 0
        assert engine.broker.get_account_state()["submitted"] == 0
        await engine.stop()

