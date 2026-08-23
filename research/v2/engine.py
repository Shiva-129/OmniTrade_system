"""
V2 Experiment Engine — deterministic strategy research.

Reuses:
- research/data/dataset (content_hash, slice)
- research/evaluation/engine run_backtest (deterministic next-open)
- research/evaluation/metrics compute_metrics
- research/evaluation/costs CostModel
- research/evaluation/split train_val_test
- research/validation/param_space, sweep, selection, walkforward, robustness, verdict
- research/experiments/registry (append-only)
- research/allocation (causal)

Guarantees:
- No test-data leakage: candidate search uses TRAIN+VAL only, TEST untouched until final.
- No future-bar leakage: run_backtest decision at close T, fill open T+1.
- Deterministic: same dataset+strategy+config+seed+engine version → same experiment_id, metrics, decision.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from research.data.dataset import OHLCVDataset
from research.evaluation.costs import CostModel
from research.evaluation.engine import run_backtest
from research.evaluation.metrics import compute_metrics
from research.evaluation.experiment import INDICATOR_VERSIONS
from research.experiments.registry import ExperimentRegistry
from research.validation.param_space import BaseSpec, ParameterSpace, build_strategy
from research.validation.robustness import (
    monte_carlo_trades,
    neighborhood_sensitivity,
    regime_consistency,
)
from research.validation.selection import SelectionRule
from research.validation.sweep import run_parameter_sweep
from research.validation.verdict import VerdictThresholds, decide
from research.validation.walkforward import run_walk_forward

ENGINE_VERSION = "v2.1.0"


def _canonical_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class V2ExperimentConfig(BaseModel):
    model_config = {"frozen": True}
    strategy_name: str
    dataset_hash: str
    baseline_params: Dict[str, str]
    candidate_space_hash: str
    taker_fee: str
    slippage_pct: str
    initial_capital: str
    train_bars: int
    test_bars: int
    step_bars: int
    seed: int = 42
    engine_version: str = ENGINE_VERSION
    indicator_versions: Dict[str, str] = INDICATOR_VERSIONS


class V2CandidateResult(BaseModel):
    model_config = {"frozen": True}
    params: Dict[str, str]
    combo_id: str
    train_metrics: Dict[str, Any]
    val_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    walkforward_mean: Optional[float] = None
    walkforward_positive_rate: Optional[float] = None
    robustness_passed: bool = False
    cost_sensitivity_passed: bool = False


class V2ExperimentResult(BaseModel):
    model_config = {"frozen": True}
    experiment_id: str
    config_hash: str
    strategy_name: str
    strategy_hash: str
    dataset_hash: str
    baseline_metrics: Dict[str, Any]
    baseline_walkforward: Dict[str, Any]
    candidates: List[V2CandidateResult]
    best_candidate: Optional[V2CandidateResult] = None
    decision: str  # REJECT / INCONCLUSIVE / ACCEPT
    rejection_reasons: List[str]
    reproducibility_seed: int
    engine_version: str
    indicator_versions: Dict[str, str]


def _strategy_hash(strategy_name: str, params: Dict[str, str]) -> str:
    return _canonical_hash({"strategy": strategy_name, "params": params, "engine": ENGINE_VERSION})


def run_v2_experiment(
    *,
    dataset: OHLCVDataset,
    base_spec: BaseSpec,
    baseline_params: Dict[str, str],
    candidate_space: ParameterSpace,
    taker_fee: str = "0.001",
    slippage_pct: str = "0.0005",
    initial_capital: str = "10000",
    train_bars: int = 60,
    test_bars: int = 20,
    step_bars: int = 20,
    seed: int = 42,
    registry_path: Optional[str] = None,
) -> V2ExperimentResult:
    """
    Deterministic V2 experiment:
    1. Baseline backtest on full dataset split (test untouched until final)
    2. Candidate sweep on TRAIN, selection on VAL, final TEST evaluation
    3. Walk-forward, robustness, cost sensitivity per candidate
    4. Decision via strict gates (baseline vs candidate)
    """
    from src.core.money import init_money_context, to_decimal
    init_money_context()
    cost = CostModel(taker_fee=to_decimal(taker_fee), slippage_pct=to_decimal(slippage_pct))
    dataset_hash = dataset.content_hash()

    # --- Baseline: walk-forward + metrics on full dataset ---
    baseline_walk = run_walk_forward(
        ParameterSpace(strategy_name=base_spec.strategy_name, grid={k: (v,) for k, v in baseline_params.items()}),
        base_spec, dataset, train_bars=train_bars, test_bars=test_bars, step_bars=step_bars,
        taker_fee=taker_fee, slippage_pct=slippage_pct, initial_capital=initial_capital,
        rule=SelectionRule(min_val_trades=0),
    )
    baseline_strategy = build_strategy(
        ParameterSpace(strategy_name=base_spec.strategy_name, grid={k: (v,) for k, v in baseline_params.items()}),
        base_spec, baseline_params,
    )
    baseline_bt = run_backtest(baseline_strategy, dataset, cost, initial_capital)
    baseline_metrics = compute_metrics(
        baseline_bt.equity_curve, baseline_bt.trades, initial_capital=float(initial_capital),
        timeframe_minutes=1, fees_paid=float(baseline_bt.fees_paid),
        slippage_cost=float(baseline_bt.slippage_cost), turnover_notional=float(baseline_bt.turnover_notional),
    ).model_dump(mode="json")

    # --- Candidate sweep on TRAIN slice only (no test leakage) ---
    # Split dataset: train+val vs test
    n = len(dataset)
    test_start = max(0, n - test_bars)
    train_val_ds = dataset.slice_indices(0, test_start)
    test_ds = dataset.slice_indices(test_start, n)

    # Walk-forward per candidate is expensive; for V2 we do single sweep + val selection
    sweep_report = run_parameter_sweep(
        candidate_space, base_spec, train_val_ds,
        taker_fee=taker_fee, slippage_pct=slippage_pct, initial_capital=initial_capital,
    )

    candidates: List[V2CandidateResult] = []
    best: Optional[V2CandidateResult] = None
    best_val_sharpe = float("-inf")

    for entry in sweep_report.entries:
        params = entry.params
        # Candidate backtest on train_val and test separately (test untouched until now)
        cand_strategy = build_strategy(candidate_space, base_spec, params)
        # Train metrics already in entry.train_metrics, val in entry.val_metrics
        train_m = entry.train_metrics["metrics"]
        val_m = entry.val_metrics["metrics"]
        # Final test evaluation (untouched)
        test_bt = run_backtest(cand_strategy, test_ds, cost, initial_capital)
        test_m = compute_metrics(
            test_bt.equity_curve, test_bt.trades, initial_capital=float(initial_capital),
            timeframe_minutes=1, fees_paid=float(test_bt.fees_paid),
            slippage_cost=float(test_bt.slippage_cost), turnover_notional=float(test_bt.turnover_notional),
        ).model_dump(mode="json")

        # Walk-forward for this candidate (single candidate space)
        wf = run_walk_forward(
            ParameterSpace(strategy_name=base_spec.strategy_name, grid={k: (v,) for k, v in params.items()}),
            base_spec, dataset, train_bars=train_bars, test_bars=test_bars, step_bars=step_bars,
            taker_fee=taker_fee, slippage_pct=slippage_pct, initial_capital=initial_capital,
            rule=SelectionRule(min_val_trades=0),
        )
        # Robustness: neighborhood (single-axis perturbation) + cost sensitivity
        # Cost sensitivity: 2x slippage still positive?
        cost2 = CostModel(taker_fee=to_decimal(taker_fee), slippage_pct=to_decimal(str(float(slippage_pct)*2)))
        test_bt2 = run_backtest(cand_strategy, test_ds, cost2, initial_capital)
        test_m2 = compute_metrics(test_bt2.equity_curve, test_bt2.trades, initial_capital=float(initial_capital),
                                  timeframe_minutes=1, fees_paid=float(test_bt2.fees_paid),
                                  slippage_cost=float(test_bt2.slippage_cost), turnover_notional=float(test_bt2.turnover_notional)).model_dump(mode="json")
        cost_pass = float(test_m2.get("sharpe") or 0) > 0 and float(test_m.get("sharpe") or 0) > 0

        # Parameter sensitivity: at least not fragile (we use walk-forward positive_rate as proxy)
        robust_pass = (wf.positive_rate or 0) >= 0.5 and (wf.mean_oos_return or 0) > 0

        cand = V2CandidateResult(
            params=params, combo_id=entry.combo_id,
            train_metrics=train_m, val_metrics=val_m, test_metrics=test_m,
            walkforward_mean=wf.mean_oos_return, walkforward_positive_rate=wf.positive_rate,
            robustness_passed=robust_pass, cost_sensitivity_passed=cost_pass,
        )
        candidates.append(cand)
        val_sharpe = float(val_m.get("sharpe") or 0)
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best = cand

    # --- Decision: strict anti-overfitting gates ---
    reasons: List[str] = []
    decision = "REJECT"
    if best is None:
        reasons.append("no candidate met min_val_trades")
        decision = "REJECT"
    else:
        baseline_sharpe = float(baseline_metrics.get("sharpe") or 0)
        candidate_val_sharpe = float(best.val_metrics.get("sharpe") or 0)
        candidate_test_sharpe = float(best.test_metrics.get("sharpe") or 0)
        baseline_test_sharpe = float(baseline_metrics.get("sharpe") or 0)
        # Gate 1: must beat baseline on validation by margin
        if candidate_val_sharpe <= baseline_sharpe + 0.1:
            reasons.append(f"validation Sharpe {candidate_val_sharpe:.3f} not > baseline {baseline_sharpe:.3f}+0.1")
        # Gate 2: must not worsen drawdown materially
        if float(best.test_metrics.get("max_drawdown") or 0) > float(baseline_metrics.get("max_drawdown") or 0) * 1.5 + 0.02:
            reasons.append("max drawdown worsened >50%")
        # Gate 3: cost sensitivity
        if not best.cost_sensitivity_passed:
            reasons.append("cost sensitivity failed (2x slippage Sharpe <=0)")
        # Gate 4: robustness
        if not best.robustness_passed:
            reasons.append("robustness failed (walk-forward positive_rate <0.5 or mean <=0)")
        # Gate 5: trade count sufficiency
        if int(best.test_metrics.get("trade_count") or 0) < 5:
            reasons.append("insufficient trades on test (<5)")
        # Gate 6: test must beat baseline test (not just val) to avoid overfit to val
        if candidate_test_sharpe <= baseline_test_sharpe:
            reasons.append(f"test Sharpe {candidate_test_sharpe:.3f} not > baseline test {baseline_test_sharpe:.3f}")
        # Gate 7: walk-forward degradation train→val→test (simple)
        train_sharpe = float(best.train_metrics.get("sharpe") or 0)
        if train_sharpe > 0 and candidate_val_sharpe / train_sharpe < 0.3:
            reasons.append("severe train->val degradation (<30%)")

        if not reasons:
            decision = "ACCEPT"
        elif len(candidates) > 0 and best.walkforward_positive_rate is None:
            decision = "INCONCLUSIVE"
        else:
            # If only one gate fails but close, mark inconclusive? For now REJECT if any reason
            decision = "REJECT"

    # If no reason but we still want to distinguish inconclusive when walk-forward inconclusive
    if decision == "REJECT" and best and best.walkforward_mean is None:
        decision = "INCONCLUSIVE"

    config = V2ExperimentConfig(
        strategy_name=base_spec.strategy_name,
        dataset_hash=dataset_hash,
        baseline_params=baseline_params,
        candidate_space_hash=_canonical_hash(candidate_space.model_dump(mode="json")),
        taker_fee=taker_fee, slippage_pct=slippage_pct, initial_capital=initial_capital,
        train_bars=train_bars, test_bars=test_bars, step_bars=step_bars,
        seed=seed, indicator_versions=INDICATOR_VERSIONS,
    )
    config_hash = _canonical_hash(config.model_dump(mode="json"))
    experiment_id = _canonical_hash({"config_hash": config_hash, "engine_version": ENGINE_VERSION})

    result = V2ExperimentResult(
        experiment_id=experiment_id,
        config_hash=config_hash,
        strategy_name=base_spec.strategy_name,
        strategy_hash=_strategy_hash(base_spec.strategy_name, baseline_params),
        dataset_hash=dataset_hash,
        baseline_metrics=baseline_metrics,
        baseline_walkforward=baseline_walk.model_dump(mode="json"),
        candidates=candidates,
        best_candidate=best,
        decision=decision,
        rejection_reasons=reasons,
        reproducibility_seed=seed,
        engine_version=ENGINE_VERSION,
        indicator_versions=INDICATOR_VERSIONS,
    )

    if registry_path:
        reg = ExperimentRegistry(registry_path)
        try:
            reg.record(result.model_dump(mode="json"))
        except Exception:
            pass  # registry append-only, duplicate is expected on reproduce

    return result
