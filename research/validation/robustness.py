"""
Robustness analysis (Phase 9).

Three independent probes; all deterministic:

1. NEIGHBORHOOD SENSITIVITY -- single-axis perturbations around the
   locked parameters, scored on the VALIDATION slice. A real edge lives
   on a plateau, not a spike: we report the fraction of neighbors that
   keep a positive selection metric.
2. MONTE CARLO TRADE RESHUFFLE -- seeded bootstrap over realized trade
   PnLs (project RNG policy: simulator DeterministicRNG). Reports the
   percentile band of cumulative PnL under resequencing.
3. REGIME CONSISTENCY -- first vs second half of the out-of-sample
   equity curve must agree in sign for the result to be called stable.
"""
import statistics
from typing import Any, Dict, List

from pydantic import BaseModel

from src.core.money import init_money_context
from src.simulator.context import DeterministicRNG

from .param_space import BaseSpec, ParameterSpace, build_strategy


class AxisProbe(BaseModel):
    model_config = {"frozen": True}

    axis: str
    value: str
    score: float | None               # None when probe produced no trades
    delta_vs_locked: float | None
    positive: bool | None


class SensitivityReport(BaseModel):
    locked_score: float
    probes: List[AxisProbe]
    fraction_positive: float | None   # None when no evaluable neighbors
    n_evaluable: int


def neighborhood_sensitivity(space: ParameterSpace, base: BaseSpec,
                             locked_params: Dict[str, str],
                             locked_score: float,
                             perturbations: Dict[str, List[str]],
                             validation_ds, taker_fee: str = "0",
                             slippage_pct: str = "0",
                             initial_capital: str = "10000",
                             metric: str = "sharpe") -> SensitivityReport:
    from ..evaluation.costs import CostModel
    from ..evaluation.engine import run_backtest
    from ..evaluation.metrics import compute_metrics
    from src.core.money import to_decimal

    init_money_context()
    cost = CostModel(taker_fee=to_decimal(taker_fee),
                     slippage_pct=to_decimal(slippage_pct))

    def _score(params: Dict[str, str]) -> float | None:
        strategy = build_strategy(space, base, params)
        bt = run_backtest(strategy, validation_ds, cost, initial_capital)
        m = compute_metrics(bt.equity_curve, bt.trades,
                            initial_capital=float(initial_capital),
                            timeframe_minutes=1,
                            fees_paid=float(bt.fees_paid),
                            slippage_cost=float(bt.slippage_cost),
                            turnover_notional=float(bt.turnover_notional))
        raw = m.model_dump(mode="json")[metric]
        return None if raw is None else float(raw)

    probes: List[AxisProbe] = []
    for axis, values in perturbations.items():
        for v in values:
            if v == locked_params.get(axis):
                continue                       # skip the locked value itself
            alt = dict(locked_params)
            alt[axis] = v
            try:
                score = _score(alt)
            except Exception:
                continue                        # invalid neighbor: excluded
            probes.append(AxisProbe(
                axis=axis, value=v,
                score=score,
                delta_vs_locked=(score - locked_score) if score is not None else None,
                positive=(score > 0) if score is not None else None))

    evaluable = [p.positive for p in probes if p.positive is not None]
    frac = (sum(1 for x in evaluable if x) / len(evaluable)) if evaluable else None
    return SensitivityReport(locked_score=locked_score, probes=probes,
                             fraction_positive=frac,
                             n_evaluable=len(evaluable))


class MonteCarloReport(BaseModel):
    model_config = {"frozen": True}

    n_sims: int
    rng_seed: int
    p05_total_pnl: float
    p50_total_pnl: float
    p95_total_pnl: float
    mean_total_pnl: float


def monte_carlo_trades(trade_pnls: List[float], n_sims: int = 2000,
                       rng_seed: int = 42) -> MonteCarloReport:
    """
    Seeded bootstrap: resample the realized trade PnL sequence with
    replacement N times; each sim's statistic is its cumulative PnL.
    """
    if not trade_pnls:
        raise ValueError("no trades to bootstrap")
    rng = DeterministicRNG(rng_seed)
    totals = []
    for _ in range(n_sims):
        total = 0.0
        for _ in trade_pnls:
            idx = rng.randint(0, len(trade_pnls) - 1)
            total += float(trade_pnls[idx])
        totals.append(total)
    totals.sort()

    def pct(p: float) -> float:
        k = max(0, min(len(totals) - 1, int(round(p * (len(totals) - 1)))))
        return totals[k]

    return MonteCarloReport(
        n_sims=n_sims, rng_seed=rng_seed,
        p05_total_pnl=pct(0.05), p50_total_pnl=pct(0.50),
        p95_total_pnl=pct(0.95),
        mean_total_pnl=statistics.fmean(totals))


class RegimeReport(BaseModel):
    model_config = {"frozen": True}

    first_half_return: float
    second_half_return: float
    consistent_sign: bool | None      # None when either half is flat/zero


def regime_consistency(equity_curve: List[Dict[str, Any]],
                       initial_capital: float) -> RegimeReport:
    eq = [p["equity"] for p in equity_curve]
    half = max(1, len(eq) // 2)
    start = float(initial_capital)

    r1 = eq[half - 1] / start - 1.0 if start > 0 and eq[half - 1] else 0.0
    mid = eq[half - 1]
    r2 = (eq[-1] / mid - 1.0) if mid else 0.0

    if abs(r1) < 1e-12 or abs(r2) < 1e-12:
        consistent = None
    else:
        consistent = (r1 > 0) == (r2 > 0)
    return RegimeReport(first_half_return=r1, second_half_return=r2,
                        consistent_sign=consistent)
