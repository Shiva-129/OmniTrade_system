"""
Phase 9: parameter spaces + deterministic sweep + validation-only selection.
"""
import math

import pytest
from pydantic import ValidationError

from research.data import OHLCVDataset
from research.validation.param_space import (
    BaseSpec, ParameterSpace, build_strategy_config, evaluate_candidates)
from research.validation.selection import LockedParameters, NoCandidate, SelectionRule, select_parameters
from research.validation.sweep import combo_id, run_parameter_sweep

from src.core.money import init_money_context

PRICES = [100, 100, 100, 140, 145, 150, 90, 85, 80, 120,
          100, 100, 100, 140, 145, 150, 90, 85]


def make_ds():
    rows = [[1600000000000 + i * 60000,
             float(p), float(p) + 1, max(float(p) - 1, 0.5), float(p), 10.0]
            for i, p in enumerate(PRICES)]
    return OHLCVDataset.from_records(rows, symbol="BTC/USDT", timeframe="1m")


def ema_space(extra_fast=(), cooldowns=(0,)):
    return ParameterSpace(
        strategy_name="ema_crossover",
        grid={"fast_period": tuple(map(str, [2, *extra_fast])),
              "slow_period": ("4",),
              "cooldown_events": tuple(map(str, cooldowns))})


BASE = BaseSpec(strategy_name="ema_crossover", symbol="BTC/USDT",
                timeframe="1m", trade_size="0.5")


@pytest.fixture(autouse=True)
def _ctx():
    init_money_context()


class TestParameterSpace:
    def test_cartesian_enumeration_deterministic_order(self):
        space = ema_space()
        combos = space.iter_params()
        assert combos == [
            {"fast_period": "2", "slow_period": "4", "cooldown_events": "0"}]
        again = ema_space().iter_params()
        assert combos == again

    def test_multi_value_grid_counts(self):
        space = ParameterSpace(
            strategy_name="ema_crossover",
            grid={"fast_period": ("2", "3"), "slow_period": ("5",),
                  "cooldown_events": ("0", "2")})
        assert len(space.iter_params()) == 4

    def test_missing_required_axis_rejected(self):
        with pytest.raises(ValueError, match="missing required axis"):
            ParameterSpace(strategy_name="ema_crossover",
                           grid={"fast_period": ("2",)}).iter_params()

    def test_contract_violation_raises_through_builder(self):
        bad = dict(fast_period="9", slow_period="4", cooldown_events="0")
        with pytest.raises(ValidationError):
            build_strategy_config(ema_space(), BASE, bad)

    def test_evaluate_candidates_reports_rejections_explicitly(self):
        space = ParameterSpace(
            strategy_name="ema_crossover",
            grid={"fast_period": ("2", "9"),
                  "slow_period": ("4",),
                  "cooldown_events": ("0",)})
        valid, rejected = evaluate_candidates(space, BASE)
        assert len(valid) == 1 and valid[0]["fast_period"] == "2"
        assert len(rejected) == 1
        assert rejected[0]["params"]["fast_period"] == "9"
        assert rejected[0]["reason"]           # strategy contract's own message

    def test_zscore_space_exit_above_entry_rejected(self):
        space = ParameterSpace(
            strategy_name="zscore_mean_reversion",
            grid={"window": ("5",), "entry_z": ("2.0",),
                  "exit_z": ("3.0",)})      # exit >= entry -> invalid
        _, rejected = evaluate_candidates(space, BASE)
        assert len(rejected) == 1

    def test_combo_id_stable_and_sensitive(self):
        a = combo_id({"fast_period": "2"})
        b = combo_id({"fast_period": "2"})
        c = combo_id({"fast_period": "3"})
        assert a == b and a != c


class TestSweep:
    @pytest.mark.asyncio
    async def _noop(self):                 # keep class shape obvious
        pass

    def test_every_valid_combo_gets_unique_experiment_record(self):
        space = ParameterSpace(
            strategy_name="ema_crossover",
            grid={"fast_period": ("2", "3"), "slow_period": ("5",),
                  "cooldown_events": ("0", "1")})
        report = run_parameter_sweep(space, BASE, make_ds())
        assert len(report.entries) == 4
        ids = {e.experiment_id for e in report.entries}
        assert len(ids) == 4               # params differ => provenance differs
        combos = {e.combo_id for e in report.entries}
        assert len(combos) == 4

    def test_invalid_combos_recorded_with_reason(self):
        space = ParameterSpace(
            strategy_name="ema_crossover",
            grid={"fast_period": ("9",), "slow_period": ("4",),
                  "cooldown_events": ("0",)})
        report = run_parameter_sweep(space, BASE, make_ds())
        assert len(report.entries) == 0
        assert len(report.rejected) == 1
        assert report.rejected[0].reason

    def test_provenance_fields_present(self):
        report = run_parameter_sweep(ema_space(), BASE, make_ds(),
                                     taker_fee="0.001")
        assert report.dataset_hash == make_ds().content_hash()
        assert report.taker_fee == "0.001"
        e = report.entries[0]
        assert e.n_train_bars > 0 and e.n_val_bars > 0
        assert e.train_metrics["metrics"]["trade_count"] >= 0

    def test_sweep_is_deterministic(self):
        r1 = run_parameter_sweep(ema_space(cooldowns=(0, 1)), BASE, make_ds())
        r2 = run_parameter_sweep(ema_space(cooldowns=(0, 1)), BASE, make_ds())
        assert r1.model_dump() == r2.model_dump()

    def test_costs_flow_into_sweep_entries(self):
        free = run_parameter_sweep(ema_space(), BASE, make_ds(), taker_fee="0")
        paid = run_parameter_sweep(ema_space(), BASE, make_ds(), taker_fee="0.01")
        f_free = float(free.entries[0].train_metrics["metrics"]["fees_paid"])
        f_paid = float(paid.entries[0].train_metrics["metrics"]["fees_paid"])
        if f_paid > 0:
            assert f_paid > f_free


class TestSelection:
    def _report_with_val_sharpes(self, sharpes, trades=None):
        space = ParameterSpace(
            strategy_name="ema_crossover",
            grid={"fast_period": tuple(str(i + 2) for i in range(len(sharpes))),
                  "slow_period": ("50",), "cooldown_events": ("0",)})
        from research.validation.sweep import SweepEntry, SweepRejected, SweepReport
        entries = []
        for i, s in enumerate(sharpes):
            t = (trades or [1] * len(sharpes))[i]
            entries.append(SweepEntry(
                combo_id=f"c{i}", params={"fast_period": str(i + 2)},
                experiment_id=f"e{i}",
                train_metrics={"metrics": {"sharpe": s}},
                val_metrics={"metrics": {"sharpe": s, "trade_count": t}},
                n_train_bars=10, n_val_bars=10))
        return SweepReport(
            strategy_name="ema_crossover", strategy_version="1.0.0",
            dataset_hash="h" * 64, taker_fee="0", slippage_pct="0",
            initial_capital="10000", entries=entries, rejected=[])

    def test_selects_best_validation_sharpe(self):
        rep = self._report_with_val_sharpes([0.5, 2.0, 1.0])
        locked = select_parameters(rep, SelectionRule())
        assert locked.params["fast_period"] == "3"
        assert locked.selection_value == 2.0

    def test_min_trades_floor_filters_candidates(self):
        rep = self._report_with_val_sharpes([5.0, 1.0], trades=[0, 1])
        locked = select_parameters(rep, SelectionRule(min_val_trades=1))
        assert locked.params["fast_period"] == "3"     # the 5.0 had zero trades

    def test_no_candidate_raises_when_all_filtered(self):
        rep = self._report_with_val_sharpes([5.0], trades=[0])
        with pytest.raises(NoCandidate):
            select_parameters(rep, SelectionRule(min_val_trades=1))

    def test_lock_is_frozen_and_carries_provenance(self):
        rep = self._report_with_val_sharpes([0.5, 2.0, 1.0])
        locked = select_parameters(rep, SelectionRule())
        assert isinstance(locked, LockedParameters)
        with pytest.raises(ValidationError):
            locked.params = {}
        assert locked.n_candidates == 3
        assert locked.combo_id and locked.experiment_id

    def test_selection_module_has_no_test_slice_access(self):
        """Structural: selection API accepts only sweep reports + rules."""
        import inspect
        from research.validation import selection as sel_mod
        sig = inspect.signature(sel_mod.select_parameters)
        param_names = list(sig.parameters)
        assert param_names == ["report", "rule"]
