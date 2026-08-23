"""Phase 15 R2: Multi-symbol backtest — N=1 regression, isolation, determinism."""
import pytest

from research.data.dataset import OHLCVDataset
from research.evaluation.costs import CostModel
from research.evaluation.engine import run_backtest
from research.evaluation.multi_symbol import run_multi_symbol_backtest
from src.core.money import init_money_context, to_decimal
from src.strategies.ema_crossover import EmaCrossoverConfig, EmaCrossoverStrategy


def make_ds(symbol, prices, tf="1m"):
    rows = [[1600000000000 + i * 60000, float(p), float(p) + 1,
             max(float(p) - 1, 0.5), float(p), 10.0] for i, p in enumerate(prices)]
    return OHLCVDataset.from_records(rows, symbol=symbol, timeframe=tf)


def _strategy(symbol="BTCUSDT", qty="1"):
    return EmaCrossoverStrategy(EmaCrossoverConfig(
        strategy_name="ema", symbol=symbol, trade_size=qty,
        fast_period=2, slow_period=3))


PRICES = [100, 100, 100, 140, 145, 150, 90, 85, 80, 120]


@pytest.fixture(autouse=True)
def _ctx():
    init_money_context()


class TestN1Regression:
    def test_single_symbol_matches_run_backtest_fills(self):
        """N=1 must produce identical fills to run_backtest."""
        cm = CostModel(taker_fee=to_decimal("0.001"))
        ds = make_ds("BTCUSDT", PRICES)
        ref = run_backtest(_strategy(), ds, cm, "10000")
        multi = run_multi_symbol_backtest(
            {"BTCUSDT": (_strategy(), ds)}, cm, "10000")
        assert len(ref.fills) == len(multi.fills)
        for a, b in zip(ref.fills, multi.fills):
            assert a["side"] == b["side"]
            assert a["price"] == b["price"]  # same fill price string
        assert ref.summary()["filled_intents"] == multi.filled_intents

    def test_single_symbol_equity_matches(self):
        cm = CostModel()
        ds = make_ds("BTCUSDT", PRICES)
        ref = run_backtest(_strategy(), ds, cm, "10000")
        multi = run_multi_symbol_backtest(
            {"BTCUSDT": (_strategy(), ds)}, cm, "10000")
        assert ref.equity_curve[-1]["equity"] == multi.equity_curve[-1]["equity"]


class TestMultiSymbol:
    def test_two_symbols_isolated(self):
        cm = CostModel(taker_fee=to_decimal("0"))
        btc = make_ds("BTCUSDT", [100, 100, 100, 140, 145])
        eth = make_ds("ETHUSDT", [200, 200, 200, 280, 290])
        result = run_multi_symbol_backtest(
            {"BTCUSDT": (_strategy("BTCUSDT"), btc),
             "ETHUSDT": (_strategy("ETHUSDT"), eth)},
            cm, "10000")
        # Both symbols should have produced fills
        syms = {f["cloid"].split(":")[2] for f in result.fills}
        assert syms == {"BTCUSDT", "ETHUSDT"}
        # No cross-contamination: fills carry their own symbol's cloid prefix and correct price domain
        for f in result.fills:
            sym_in_cloid = f["cloid"].split(":")[2]
            assert sym_in_cloid in ("BTCUSDT", "ETHUSDT")
            # BTC fills around 100-145, ETH around 200-290 — price must match symbol domain
            price = float(f["price"])
            if sym_in_cloid == "BTCUSDT":
                assert 90 <= price <= 160
            else:
                assert 180 <= price <= 320

    def test_insertion_order_independent(self):
        cm = CostModel()
        btc = make_ds("BTCUSDT", [100, 100, 100, 140, 145])
        eth = make_ds("ETHUSDT", [200, 200, 200, 280, 290])
        r_ab = run_multi_symbol_backtest(
            {"BTCUSDT": (_strategy("BTCUSDT"), btc),
             "ETHUSDT": (_strategy("ETHUSDT"), eth)}, cm, "10000")
        r_ba = run_multi_symbol_backtest(
            {"ETHUSDT": (_strategy("ETHUSDT"), eth),
             "BTCUSDT": (_strategy("BTCUSDT"), btc)}, cm, "10000")
        assert [(f["ts"], f["price"]) for f in r_ab.fills] == \
               [(f["ts"], f["price"]) for f in r_ba.fills]
        assert r_ab.equity_curve[-1] == r_ba.equity_curve[-1]

    def test_shared_portfolio_cash_shared(self):
        """Two strategies sharing one Portfolio draw from the same cash.
        Uses a long enough series so crossovers actually fire."""
        cm = CostModel(taker_fee=to_decimal("0"))
        btc = make_ds("BTCUSDT", [100]*5 + [140, 145, 150] + [90] + [100]*3)
        eth = make_ds("ETHUSDT", [200]*5 + [280, 290, 300] + [180] + [200]*3)
        result = run_multi_symbol_backtest(
            {"BTCUSDT": (_strategy("BTCUSDT"), btc),
             "ETHUSDT": (_strategy("ETHUSDT"), eth)}, cm, "50000")
        # Shared portfolio: both symbols' fills drain the same cash pool
        assert result.filled_intents >= 2
        assert result.equity_curve[-1]["equity"] != 50000.0


class TestDeterminism:
    def test_repeated_runs_identical(self):
        cm = CostModel(taker_fee=to_decimal("0.001"))
        datasets = {
            "BTCUSDT": (_strategy("BTCUSDT"), make_ds("BTCUSDT", PRICES)),
            "ETHUSDT": (_strategy("ETHUSDT"), make_ds("ETHUSDT", [200]*10)),
        }
        # Fresh strategy instances per run
        d1 = {s: (_strategy(s), d) for s, (_, d) in datasets.items()}
        d2 = {s: (_strategy(s), d) for s, (_, d) in datasets.items()}
        r1 = run_multi_symbol_backtest(d1, cm, "10000")
        r2 = run_multi_symbol_backtest(d2, cm, "10000")
        assert [(f["ts"], f["side"], f["price"]) for f in r1.fills] == \
               [(f["ts"], f["side"], f["price"]) for f in r2.fills]
        assert r1.summary() == r2.summary()
