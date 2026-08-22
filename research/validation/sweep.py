"""
Deterministic parameter sweep (Phase 9).

SCOPE DISCIPLINE: the sweep consumes ONE dataset slice the caller hands
it -- research code passes the TRAIN slice, selection scores on
VALIDATION. The final test slice is never an input anywhere in this
module (structurally impossible: no parameter exists for it).

Every candidate gets its own ExperimentConfig -> experiment_id (sha256
over full provenance incl. parameters), so any row of a sweep can be
re-run standalone from Phase 8 machinery.
"""
import hashlib
import json
from typing import Any, Dict, List

from pydantic import BaseModel

from ..data.dataset import OHLCVDataset, timeframe_minutes
from ..evaluation.costs import CostModel
from ..evaluation.engine import run_backtest
from ..evaluation.experiment import ExperimentConfig
from ..evaluation.metrics import compute_metrics
from .param_space import BaseSpec, ParameterSpace, build_strategy


def combo_id(params: Dict[str, str]) -> str:
    payload = json.dumps(params, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class SweepEntry(BaseModel):
    model_config = {"frozen": True}

    combo_id: str
    params: Dict[str, str]
    experiment_id: str                 # full provenance hash (Phase 8 style)
    train_metrics: Dict[str, Any]
    val_metrics: Dict[str, Any]
    n_train_bars: int
    n_val_bars: int


class SweepRejected(BaseModel):
    model_config = {"frozen": True}

    params: Dict[str, str]
    reason: str


class SweepReport(BaseModel):
    strategy_name: str
    strategy_version: str
    dataset_hash: str                  # hash of the RESEARCH slice passed in
    taker_fee: str
    slippage_pct: str
    initial_capital: str
    entries: List[SweepEntry]
    rejected: List[SweepRejected]

    def by_combo(self, cid: str) -> SweepEntry:
        return next(e for e in self.entries if e.combo_id == cid)


def _slice_metrics(result, tf_min: int, capital: float) -> Dict[str, Any]:
    m = compute_metrics(
        result.equity_curve, result.trades,
        initial_capital=capital, timeframe_minutes=tf_min,
        fees_paid=float(result.fees_paid),
        slippage_cost=float(result.slippage_cost),
        turnover_notional=float(result.turnover_notional),
    )
    return {"metrics": m.model_dump(mode="json"),
            "execution": result.summary()}


def _experiment_id(space: ParameterSpace, base: BaseSpec, params,
                   ds_hash: str, taker: str, slip: str, capital: str) -> str:
    exp = ExperimentConfig(
        strategy_name=space.strategy_name,
        strategy_version="1.0.0",
        parameters=params,
        symbol=base.symbol,
        timeframe=base.timeframe,
        dataset_hash=ds_hash,
        start_ts=0, end_ts=0,
        taker_fee=taker, slippage_pct=slip, initial_capital=capital,
    )
    return exp.config_hash


def run_parameter_sweep(space: ParameterSpace, base: BaseSpec,
                        research_slice: OHLCVDataset,
                        taker_fee: str = "0", slippage_pct: str = "0",
                        initial_capital: str = "10000"
                        ) -> SweepReport:
    """
    Sweeps ALL valid combos over the given research slice.
    Caller convention: pass the TRAIN+VALIDATION region; NEVER the test.
    The slice is split internally 50/50 into train/val halves so both
    score sets come from disjoint contiguous bars.
    """
    from src.core.money import init_money_context, to_decimal
    init_money_context()

    ds_hash = research_slice.content_hash()
    n = len(research_slice)
    half = max(1, n // 2)
    train_ds = research_slice.slice_indices(0, half)
    val_ds = research_slice.slice_indices(half, n)
    tf_min = timeframe_minutes(base.timeframe)

    entries: List[SweepEntry] = []
    rejected: List[SweepRejected] = []
    cost = CostModel(taker_fee=to_decimal(taker_fee),
                     slippage_pct=to_decimal(slippage_pct))

    for params in space.iter_params():
        try:
            strategy = build_strategy(space, base, params)
        except Exception as e:                     # contract violation
            reason = getattr(e, "errors", lambda: None)()
            msg = reason[0]["msg"] if reason else str(e)
            rejected.append(SweepRejected(params=params, reason=msg))
            continue

        eid = _experiment_id(space, base, params, ds_hash,
                             taker_fee, slippage_pct, initial_capital)

        tr = run_backtest(strategy, train_ds, cost, initial_capital)
        va = run_backtest(strategy, val_ds, cost, initial_capital)

        entries.append(SweepEntry(
            combo_id=combo_id(params),
            params=params,
            experiment_id=eid,
            train_metrics=_slice_metrics(tr, tf_min, float(initial_capital)),
            val_metrics=_slice_metrics(va, tf_min, float(initial_capital)),
            n_train_bars=len(train_ds),
            n_val_bars=len(val_ds),
        ))

    return SweepReport(
        strategy_name=space.strategy_name,
        strategy_version="1.0.0",
        dataset_hash=ds_hash,
        taker_fee=taker_fee,
        slippage_pct=slippage_pct,
        initial_capital=initial_capital,
        entries=entries,
        rejected=rejected,
    )
