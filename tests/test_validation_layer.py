"""
Phase 9: walk-forward evaluation, robustness probes, and research
verdict gates (including the forbidden-claims guard).
"""
import pytest

from research.data import OHLCVDataset
from research.validation.param_space import (
    BaseSpec, ParameterSpace, build_strategy)
from research.validation.robustness import (
    monte_carlo_trades, neighborhood_sensitivity, regime_consistency)
from research.validation.selection import SelectionRule
from research.validation.verdict import (
    DISCLAIMER, FORBIDDEN_CLAIMS, VerdictThresholds, decide)
from research.evaluation.split import train_val_test

from src.core.money import init_money_context
from src.simulator.context import DeterministicRNG

BASE = BaseSpec(strategy_name="ema_crossover", symbol="BTC/USDT",
                timeframe="1m", trade_size="0.5")


def make_ds(prices):
    rows = [[1600000000000 + i * 60000,
             float(p), float(p) + 1, max(float(p) - 1, 0.5), float(p), 10.0]
            for i, p in enumerate(prices)]
    return OHLCVDataset.from_records(rows, symbol="BTC/USDT", timeframe="1m")


def ema_space():
    return ParameterSpace(
        strategy_name="ema_crossover",
        grid={"fast_period": ("2", "3"), "slow_period": ("5",),
              "cooldown_events": ("0",)})


# Oscillating fixture: repeated up/down cycles so EMA crossovers fire
# across every window.
CYCLES = []
for _i in range(6):
    CYCLES += [100, 100, 100, 140, 145, 150, 90, 85, 80, 95]


@pytest.fixture(autouse=True)
def _ctx():
    init_money_context()


class TestWalkForward:
    def test_windows_roll_and_lock_then_evaluate(self):
        ds = make_ds(CYCLES)
        from research.validation.walkforward import run_walk_forward

        report = run_walk_forward(ema_space(), BASE, ds,
                                  train_bars=30, test_bars=10, step_bars=10,
                                  taker_fee="0", slippage_pct="0",
                                  rule=SelectionRule(min_val_trades=0))
        assert report.n_windows == 3          # starts 0,10,20 (span 40 <= 60)
        for w in report.windows:
            assert w.train_range[1] == w.test_range[0]   # test right after train
            if w.oos_return is not None:
                assert w.locked is not None
                assert w.locked.params["slow_period"] == "5"
        evaluated = [w for w in report.windows if w.oos_return is not None]
        assert len(evaluated) >= 2
        assert report.mean_oos_return is not None
        assert report.positive_rate is not None

    def test_walkforward_is_deterministic(self):
        from research.validation.walkforward import run_walk_forward
        ds = make_ds(CYCLES)
        r1 = run_walk_forward(ema_space(), BASE, ds, train_bars=30,
                              test_bars=10, step_bars=10,
                              rule=SelectionRule(min_val_trades=0))
        r2 = run_walk_forward(ema_space(), BASE, ds, train_bars=30,
                              test_bars=10, step_bars=10,
                              rule=SelectionRule(min_val_trades=0))
        assert r1.model_dump() == r2.model_dump()

    def test_oos_slices_disjoint_from_selection_slices(self):
        from research.validation.walkforward import run_walk_forward
        ds = make_ds(CYCLES)
        report = run_walk_forward(ema_space(), BASE, ds, train_bars=30,
                                  test_bars=10, step_bars=10,
                                  rule=SelectionRule(min_val_trades=0))
        for w in report.windows:
            tr = w.train_range
            te = w.test_range
            assert te[0] >= tr[1]                    # never overlaps


class TestRobustness:
    def test_neighborhood_sensitivity_counts_positives(self):
        ds_full = make_ds(CYCLES[:40])
        _, val, _ = train_val_test(40)             # use the tail as validation
        val_ds = ds_full.slice_indices(val.start, val.stop)

        rep = neighborhood_sensitivity(
            ema_space(), BASE,
            locked_params={"fast_period": "2", "slow_period": "5",
                           "cooldown_events": "0"},
            locked_score=0.0,
            perturbations={"fast_period": ["3"]},
            validation_ds=val_ds, metric="sharpe")
        assert rep.n_evaluable >= 1
        assert all(p.axis == "fast_period" for p in rep.probes)

    def test_monte_carlo_seeded_reproducible(self):
        pnls = [10, -5, 20, -8, 15, -3, 12, -7]
        a = monte_carlo_trades(pnls, n_sims=500, rng_seed=42)
        b = monte_carlo_trades(pnls, n_sims=500, rng_seed=42)
        assert a.model_dump() == b.model_dump()

    def test_monte_carlo_percentiles_ordered(self):
        pnls = [DeterministicRNG(7).randint(-50, 50) for _ in range(40)]
        mc = monte_carlo_trades([float(x) for x in pnls], n_sims=800,
                                rng_seed=123)
        assert mc.p05_total_pnl <= mc.p50_total_pnl <= mc.p95_total_pnl

    def test_monte_carlo_empty_trades_rejected(self):
        with pytest.raises(ValueError):
            monte_carlo_trades([], n_sims=10)

    def test_regime_consistency_same_sign(self):
        curve_up = [{"ts": i, "equity": e} for i, e in
                    enumerate([1000, 1100, 1200, 1300, 1400])]
        rep = regime_consistency(curve_up, initial_capital=1000.0)
        assert rep.consistent_sign is True

    def test_regime_flip_detected(self):
        curve = [{"ts": i, "equity": e} for i, e in
                 enumerate([1000, 1200, 1250, 900, 700])]
        rep = regime_consistency(curve, initial_capital=1000.0)
        assert rep.consistent_sign is False


class TestVerdictGates:
    def _wf(self, returns):
        """Builds a minimal WalkForwardReport-shaped stub."""
        from research.validation.walkforward import WindowResult, WalkForwardReport
        windows = [WindowResult(window_index=i, train_range=[0, 1],
                                test_range=[1, 2], locked=None,
                                oos_metrics={}, oos_return=r)
                   for i, r in enumerate(returns)]
        mean_r = sum(returns) / len(returns) if returns else None
        pos = (sum(1 for r in returns if r > 0) / len(returns)) if returns else None
        var = ((sum((r - mean_r) ** 2 for r in returns) / len(returns))
               if returns else None)
        return WalkForwardReport(
            n_windows=len(windows), windows=windows,
            mean_oos_return=mean_r,
            std_oos_return=(var ** 0.5) if var is not None else None,
            positive_rate=pos, windows_with_trades=len(returns))

    def _sen(self, frac):
        from research.validation.robustness import AxisProbe, SensitivityReport
        probes = [AxisProbe(axis="x", value=str(i), score=1.0,
                            delta_vs_locked=0.0, positive=(i < frac * 10))
                  for i in range(int(frac * 10))]
        evaluable = [p.positive for p in probes]
        fraction = (sum(1 for x in evaluable if x) / len(evaluable)) \
            if evaluable else None
        return SensitivityReport(locked_score=1.0, probes=probes,
                                 fraction_positive=fraction,
                                 n_evaluable=len(evaluable))

    def test_all_gates_pass_yields_candidate_not_profitable_claim(self):
        from research.validation.robustness import MonteCarloReport, RegimeReport
        mc = MonteCarloReport(n_sims=100, rng_seed=1, p05_total_pnl=2.0,
                              p50_total_pnl=5.0, p95_total_pnl=9.0,
                              mean_total_pnl=5.0)
        regime = RegimeReport(first_half_return=0.02,
                              second_half_return=0.01,
                              consistent_sign=True)
        v = decide(self._wf([0.01, 0.02, 0.03]),
                   self._sen(0.8),
                   mc_report=mc, regime_report=regime,
                   thresholds=VerdictThresholds())
        assert v.verdict == "PASS_CANDIDATE"
        assert all(g.passed is True for g in v.gates)
        blob = v.model_dump_json().lower()
        for claim in FORBIDDEN_CLAIMS:
            assert claim not in blob or claim in DISCLAIMER.lower()

    def test_failing_oos_gate_rejects(self):
        v = decide(self._wf([-0.05, -0.02, 0.001]),
                   self._sen(0.8),
                   thresholds=VerdictThresholds(require_mc_p05_positive=False))
        assert v.verdict == "REJECT"

    def test_too_few_windows_inconclusive(self):
        v = decide(self._wf([0.01]), self._sen(1.0),
                   thresholds=VerdictThresholds(require_mc_p05_positive=False))
        assert v.verdict == "INCONCLUSIVE"
        names = {g.name: g.passed for g in v.gates}
        assert names["walk_forward_sufficient_windows"] is None

    def test_low_positive_rate_rejects(self):
        # only 1 of 3 windows positive -> below 0.6 threshold
        v = decide(self._wf([0.02, -0.03, -0.01]), self._sen(1.0),
                   thresholds=VerdictThresholds(require_mc_p05_positive=False))
        assert v.verdict == "REJECT"
        assert any(g.name == "walk_forward_positive_rate" and g.passed is False for g in v.gates)

    def test_disclaimer_always_present(self):
        v = decide(self._wf([]), self._sen(0.5))
        assert v.disclaimer == DISCLAIMER
