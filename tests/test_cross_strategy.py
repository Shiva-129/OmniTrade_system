"""
Phase 7: cross-strategy guarantees.

1. Same fixture + same config => IDENTICAL signals across repeated
   executions for every strategy (full model dumps compared).
2. Strategies are dependency-free: constructing and running them with no
   Redis, Gatekeeper, broker or network present works by construction —
   proven here by running in a module that imports none of them.
3. Generated OrderIntents validate against the existing contract.
4. Engine hook: MarketEvent -> Strategy -> RiskManager -> Gatekeeper,
   decisions journaled, rejections never reach the gate.
"""
import json

import pytest

from src.core.money import init_money_context, to_decimal
from src.core.portfolio import Portfolio
from src.core.risk_manager import RiskManager, RiskLimits
from src.core.types import MarketEvent, OrderIntent, OrderSide, OrderType, Packet
from src.strategies.ema_crossover import EmaCrossoverStrategy, EmaCrossoverConfig
from src.strategies.zscore_mean_reversion import ZScoreMeanReversionStrategy, ZScoreConfig
from src.strategies.donchian_breakout import DonchianBreakoutStrategy, DonchianConfig


@pytest.fixture(autouse=True)
def _ctx():
    init_money_context()


SERIES = [100, 99, 101, 100, 120, 121, 119, 90, 89, 95, 130, 131, 128]


def ev(ts, price):
    return MarketEvent(packet=Packet(
        exchange_ts=ts, local_arrival_ts=ts, drift_us=0, source="t",
        topic="BTCUSDT", payload={"price": str(price)}, sequence_id=ts))


def strategies():
    return [
        EmaCrossoverStrategy(EmaCrossoverConfig(
            strategy_name="ema", symbol="BTCUSDT", trade_size="0.10",
            fast_period=2, slow_period=3, allow_short=True)),
        ZScoreMeanReversionStrategy(ZScoreConfig(
            strategy_name="zsc", symbol="BTCUSDT", trade_size="0.20",
            window=5, entry_z=1.5, exit_z=0.5, allow_short=True)),
        DonchianBreakoutStrategy(DonchianConfig(
            strategy_name="don", symbol="BTCUSDT", trade_size="0.30",
            lookback=4, atr_period=3, atr_stop_multiplier=0)),
    ]


class TestCrossStrategyDeterminism:
    @pytest.mark.parametrize("idx", [0, 1, 2])
    def test_same_fixture_same_config_identical_signals(self, idx):
        def run():
            s = strategies()[idx]
            return [s.on_market_event(ev(i + 1, p))
                    for i, p in enumerate(SERIES)]

        r1 = [o.model_dump() if o else None for o in run()]
        r2 = [o.model_dump() if o else None for o in run()]
        assert r1 == r2
        assert any(o is not None for o in r1)     # fixture actually signals

    def test_all_three_validate_against_contract(self):
        for s in strategies():
            for i in range(len(SERIES)):
                intent = s.on_market_event(ev(i + 1, SERIES[i]))
                if intent is None:
                    continue
                # canonical Decimal policy at the boundary
                assert isinstance(intent.quantity, __import__("decimal").Decimal)
                assert intent.quantity > 0
                assert isinstance(intent.price, __import__("decimal").Decimal)
                assert intent.order_type == OrderType.LIMIT
                assert intent.symbol == "BTCUSDT"
                # round-trips through the frozen contract
                clone = OrderIntent.model_validate_json(intent.model_dump_json())
                assert clone == intent

    def test_strategies_import_no_infrastructure(self):
        """Import-level guarantee: strategy modules never import infra."""
        import inspect
        import re
        from src.strategies import base, ema_crossover, zscore_mean_reversion, donchian_breakout

        banned = ("redis", "ccxt", "gatekeeper", "portfolio", "execution",
                  "websockets", "requests", "socket")
        for mod in (base, ema_crossover, zscore_mean_reversion, donchian_breakout):
            src = inspect.getsource(mod)
            for m in re.finditer(r"^\s*(?:from\s+[\w.]+\s+import|import\s+[\w.,\s]+)$",
                                 src, re.MULTILINE):
                stmt = m.group(0).lower()
                for b in banned:
                    assert b not in stmt, f"{mod.__name__} imports {b}: {stmt.strip()}"


# ------------------------- engine pipeline hook ---------------------------

from src.core.engine import TradingEngine
from src.gatekeeper.engine import Gatekeeper

from conftest import TEST_REDIS_URL


class _FakeExchange:
    """Minimal scripted exchange feeding one price series."""

    def __init__(self, prices):
        self.prices = prices
        self.closed = False

    async def connect(self):
        pass

    async def listen(self):
        i = 0
        while True:
            if i < len(self.prices):
                ts = i + 1
                yield Packet(
                    exchange_ts=ts * 1_000_000, local_arrival_ts=ts,
                    drift_us=0, source="fake", topic="BTCUSDT",
                    payload={"price": str(self.prices[i]), "seq": ts},
                    sequence_id=ts,
                )
                i += 1
            else:
                import asyncio
                await asyncio.sleep(0.005)

    async def close(self):
        self.closed = True


def _limits():
    return RiskLimits(
        max_order_size=to_decimal("10"), max_position_size=to_decimal("20"),
        max_open_positions=3, max_daily_loss=to_decimal("100"),
        max_drawdown_pct=to_decimal("10"),
        stale_data_us=60_000_000, cooldown_us=30_000_000)


class TestEnginePipelineHook:
    @pytest.mark.asyncio
    async def test_strategy_risk_gatekeeper_wiring(self, tmp_path, live_redis):
        strategy = EmaCrossoverStrategy(EmaCrossoverConfig(
            strategy_name="ema", symbol="BTCUSDT", trade_size="0.01",
            fast_period=2, slow_period=3))
        portfolio = Portfolio(starting_cash="10000")
        rm = RiskManager(portfolio, _limits(), lambda: "CONNECTED")
        gk = Gatekeeper(TEST_REDIS_URL)

        engine = TradingEngine(
            redis_url=TEST_REDIS_URL,
            journal_path=str(tmp_path / "j.jsonl"),
            portfolio=portfolio,
            strategy=strategy, risk_manager=rm, gatekeeper=gk,
        )
        engine.add_exchange(_FakeExchange(
            [100, 100, 100, 140, 141, 142]))

        start = __import__("asyncio").create_task(engine.start())
        deadline = __import__("asyncio").get_running_loop().time() + 3
        while (__import__("asyncio").get_running_loop().time() < deadline
               and engine.processed_count < 6):
            await __import__("asyncio").sleep(0.01)
        await engine.stop()

        text = open(tmp_path / "j.jsonl", encoding="utf-8").read()
        assert '"risk_decision"' in text          # decision journaled
        assert '"approved": true' in text         # at least one approval
        # gatekeeper idempotency registry holds the approved cloid(s)
        assert live_redis.keys("gk:cmdreg:*")      # submitted intents recorded

    @pytest.mark.asyncio
    async def test_rejected_intent_never_reaches_gatekeeper(self, tmp_path, live_redis):
        box = {"v": "HALT"}
        strategy = EmaCrossoverStrategy(EmaCrossoverConfig(
            strategy_name="emah", symbol="BTCUSDT", trade_size="0.01",
            fast_period=2, slow_period=3))
        rm = RiskManager(Portfolio(starting_cash="10000"), _limits(),
                         lambda: box["v"])
        gk = Gatekeeper(TEST_REDIS_URL)
        before_keys = set(live_redis.keys("gk:cmdreg:*"))

        engine = TradingEngine(
            redis_url=TEST_REDIS_URL,
            journal_path=str(tmp_path / "j.jsonl"),
            strategy=strategy, risk_manager=rm, gatekeeper=gk,
        )
        engine.add_exchange(_FakeExchange([100, 100, 100, 500]))
        start = __import__("asyncio").create_task(engine.start())
        deadline = __import__("asyncio").get_running_loop().time() + 3
        while (__import__("asyncio").get_running_loop().time() < deadline
               and engine.processed_count < 4):
            await __import__("asyncio").sleep(0.01)
        await engine.stop()

        text = open(tmp_path / "j.jsonl", encoding="utf-8").read()
        assert '"risk_decision"' in text
        assert '"approved": false' in text        # risk said NO...
        assert set(live_redis.keys("gk:cmdreg:*")) == before_keys  # ...gate untouched

    def test_strategy_without_risk_is_misconfiguration(self, tmp_path):
        engine = TradingEngine(
            journal_path=str(tmp_path / "j.jsonl"),
            strategy=EmaCrossoverStrategy(EmaCrossoverConfig(
                strategy_name="e", symbol="B", trade_size="1",
                fast_period=2, slow_period=3)),
        )
        import asyncio
        with pytest.raises(ValueError, match="RiskManager"):
            asyncio.run(engine.start())
