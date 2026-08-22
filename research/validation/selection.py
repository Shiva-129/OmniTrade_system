"""
Validation-only parameter selection (Phase 9).

STRUCTURAL GUARANTEE: this module's public API accepts SweepReport +
SelectionRule ONLY. It cannot receive -- and therefore can never leak --
a test slice. The locked parameters are the ONLY artifact allowed to
cross into final out-of-sample evaluation (walkforward.py / callers).
"""
from typing import Any, Dict

from pydantic import BaseModel

from .sweep import SweepReport


class SelectionRule(BaseModel):
    model_config = {"frozen": True}

    metric: str = "sharpe"             # key inside val_metrics["metrics"]
    min_val_trades: int = 1            # evidence floor; 0 disables
    higher_is_better: bool = True


class LockedParameters(BaseModel):
    """Immutable selection output + full provenance."""
    model_config = {"frozen": True}

    strategy_name: str
    params: Dict[str, str]
    combo_id: str
    experiment_id: str
    selection_metric: str
    selection_value: float
    n_candidates: int
    n_rejected: int


class NoCandidate(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def select_parameters(report: SweepReport,
                      rule: SelectionRule) -> LockedParameters:
    """
    Picks the best candidate by the rule's metric measured on VALIDATION.
    Ties resolve to the earliest candidate in sweep order (deterministic).
    """
    best = None
    best_value = None
    for entry in report.entries:
        m = entry.val_metrics.get("metrics", {})
        trades = int(m.get("trade_count", 0))
        if trades < rule.min_val_trades:
            continue
        value = float(m.get(rule.metric) or 0.0)
        if best_value is None:
            best, best_value = entry, value
            continue
        if rule.higher_is_better:
            if value > best_value:
                best, best_value = entry, value
        else:
            if value < best_value:
                best, best_value = entry, value

    if best is None:
        raise NoCandidate(
            f"no candidate met min_val_trades={rule.min_val_trades} "
            f"on metric={rule.metric}")

    return LockedParameters(
        strategy_name=report.strategy_name,
        params=best.params,
        combo_id=best.combo_id,
        experiment_id=best.experiment_id,
        selection_metric=rule.metric,
        selection_value=float(best.val_metrics["metrics"][rule.metric] or 0.0),
        n_candidates=len(report.entries),
        n_rejected=len(report.rejected),
    )
