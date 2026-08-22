"""
Phase 8: experiment reproducibility + split-discipline tests.
"""
import json
import math

import pytest

from research.data import OHLCVDataset
from research.evaluation.costs import CostModel
from research.evaluation.experiment import (
    ExperimentConfig, build_config, run_experiment, save_results)
from src.core.money import init_money_context
from src.strategies.ema_crossover import EmaCrossoverConfig, EmaCrossoverStrategy

PRICES10 = [100, 100, 100, 140, 145, 150, 90, 85, 80, 120]
# 18 bars: train 10 | validation 3 | test 5 -- test tail carries a full
# warm-up + crossover so fresh-per-slice strategies can actually trade.
PRICES18 = ([100, 100, 100, 140, 145, 150, 90, 85, 80, 95]
            + [50, 55, 60]
            + [100, 100, 100, 140, 145])


def make_ds(prices=None):
    prices = PRICES10 if prices is None else prices
    rows = [[1600000000000 + i * 60000,
             float(p), float(p) + 1, max(float(p) - 1, 0.5), float(p), 10.0]
            for i, p in enumerate(prices)]
    return OHLCVDataset.from_records(rows, symbol="BTC/USDT",
                                     timeframe="1m")


def make_strategy_config():
    return EmaCrossoverConfig(
        strategy_name="ema", strategy_version="1.0.0",
        symbol="BTC/USDT", timeframe="1m", trade_size="0.5",
        fast_period=2, slow_period=3)


def make_exp(ds, taker="0.001", slip="0.0005"):
    return build_config(strategy_config=make_strategy_config(), dataset=ds,
                        taker_fee=taker, slippage_pct=slip)


@pytest.fixture(autouse=True)
def _ctx():
    init_money_context()


class TestExperimentConfig:
    def test_config_hash_deterministic_and_sensitive(self):
        a = make_exp(make_ds())
        b = make_exp(make_ds())
        assert a.config_hash == b.config_hash

        tweaked = list(PRICES10)
        tweaked[0] = 99.999
        c = make_exp(make_ds(tweaked))
        assert c.config_hash != a.config_hash          # dataset identity matters

    def test_dataset_hash_mismatch_rejected(self):
        exp = make_exp(make_ds())
        other = make_ds([5])
        with pytest.raises(ValueError, match="hash mismatch"):
            run_experiment(exp, other, lambda: EmaCrossoverStrategy(
                make_strategy_config()))

    def test_records_full_provenance(self):
        exp = make_exp(make_ds())
        payload = json.loads(exp.canonical())
        for key in ("strategy_name", "strategy_version", "parameters",
                    "symbol", "timeframe", "dataset_hash", "start_ts",
                    "end_ts", "taker_fee", "slippage_pct", "initial_capital",
                    "software_version"):
            assert key in payload


class TestReproducibility:
    def test_identical_experiments_byte_identical_results(self, tmp_path):
        ds = make_ds()
        exp = make_exp(ds)

        r1 = run_experiment(exp, ds, lambda: EmaCrossoverStrategy(
            make_strategy_config()))
        r2 = run_experiment(exp, ds, lambda: EmaCrossoverStrategy(
            make_strategy_config()))

        p1 = save_results(r1, tmp_path / "a.json")
        p2 = save_results(r2, tmp_path / "b.json")
        assert p1.read_bytes() == p2.read_bytes()

    def test_results_are_machine_readable_with_required_sections(self):
        ds = make_ds()
        res = run_experiment(make_exp(ds), ds, lambda: EmaCrossoverStrategy(
            make_strategy_config()))
        for section in ("config", "config_hash", "dataset_quality",
                        "splits", "train", "validation", "test",
                        "benchmark_test"):
            assert section in res
        for split in ("train", "validation", "test"):
            assert "metrics" in res[split]
            assert "equity_curve" in res[split]
            assert "trades" in res[split]
        m = res["test"]["metrics"]
        for metric in ("total_return", "cagr", "sharpe", "sortino",
                       "max_drawdown", "calmar", "win_rate",
                       "profit_factor", "trade_count", "fees_paid",
                       "slippage_cost", "turnover"):
            assert metric in m


class TestSplitDiscipline:
    def test_test_slice_is_disjoint_tail(self):
        ds = make_ds(PRICES18)                     # 12 bars
        exp = make_exp(ds)
        res = run_experiment(exp, ds, lambda: EmaCrossoverStrategy(
            make_strategy_config()), include_benchmark=False)
        tr, va, te = (res["splits"]["train"],
                      res["splits"]["validation"],
                      res["splits"]["test"])
        assert tr[1] == va[0] and va[1] == te[0]   # contiguous
        assert te[1] == len(PRICES18)              # test is the FINAL tail
        assert te not in ([tr], [va])              # disjoint ranges
        assert te[0] >= tr[1] and te[0] >= va[1]

    def test_selection_cannot_touch_test_slice(self):
        """
        Structural demo: parameter selection consumes ONLY train/val
        metrics; the test slice never enters the selection function.
        """
        ds = make_ds(PRICES18)

        candidates = [("fast2", 2), ("fast3", 3)]   # toy search space

        def select_on_validation_only(params_list):
            best, best_score = None, -math.inf
            for name, fast in params_list:
                cfg = EmaCrossoverConfig(
                    strategy_name="ema", strategy_version="1.0.0",
                    symbol="BTC/USDT", timeframe="1m", trade_size="0.5",
                    fast_period=fast, slow_period=4)
                exp = build_config(strategy_config=cfg, dataset=ds)
                res = run_experiment(exp, ds,
                                     lambda: EmaCrossoverStrategy(cfg),
                                     include_benchmark=False)
                score = res["validation"]["metrics"]["sharpe"]
                if score > best_score:
                    best, best_score = name, score
            return best                              # <- test never consulted

        winner = select_on_validation_only(candidates)
        assert winner in ("fast2", "fast3")

    def test_fees_change_outcome_visibility(self):
        """Cost-sensitivity guard: results must expose fee assumptions."""
        ds = make_ds(PRICES18)
        free = run_experiment(make_exp(ds, taker="0"), ds,
                              lambda: EmaCrossoverStrategy(
                                  make_strategy_config()),
                              include_benchmark=False)
        paid = run_experiment(make_exp(ds, taker="0.01"), ds,
                              lambda: EmaCrossoverStrategy(
                                  make_strategy_config()),
                              include_benchmark=False)
        assert free["test"]["metrics"]["trade_count"] >= 0
        assert paid["test"]["metrics"]["fees_paid"] > \
               free["test"]["metrics"]["fees_paid"]


