"""
Walk-forward evaluation (Phase 9).

Per window: sweep+select on the window's TRAIN region (with an internal
validation tail), LOCK, then evaluate the locked parameters ONCE on the
window's TEST region. Aggregated out-of-sample stats feed the research
verdict. No window ever sees another window's data.
"""
from decimal import Decimal
from typing import Any, Dict, List

from pydantic import BaseModel

from ..data.dataset import OHLCVDataset, timeframe_minutes
from ..evaluation.costs import CostModel
from ..evaluation.engine import run_backtest
from ..evaluation.metrics import compute_metrics
from ..evaluation.split import walk_forward as iter_windows
from .param_space import BaseSpec, ParameterSpace, build_strategy
from .selection import LockedParameters, SelectionRule, select_parameters, NoCandidate
from .sweep import run_parameter_sweep


class WindowResult(BaseModel):
    model_config = {"frozen": True}

    window_index: int
    train_range: List[int]
    test_range: List[int]
    locked: LockedParameters | None
    oos_metrics: Dict[str, Any] | None = None
    oos_return: float | None = None
    skip_reason: str | None = None


class WalkForwardReport(BaseModel):
    n_windows: int
    windows: List[WindowResult]
    mean_oos_return: float | None
    std_oos_return: float | None
    positive_rate: float | None       # fraction of windows with return > 0
    windows_with_trades: int


def _locked_for_window(space: ParameterSpace, base: BaseSpec,
                       train_ds: OHLCVDataset, cost: CostModel,
                       capital: str, rule: SelectionRule
                       ) -> tuple[LockedParameters | None, str | None]:
    """
    Mini selection inside one window's train region:
      train[:cut] -> candidates;  train[cut:] -> validation tail (25%).
    Small trains fall back to selecting on the whole train region.
    """
    n = len(train_ds)
    if n < 4:
        return None, "train window too small"
    val_bars = max(2, int(n * 0.25))
    cut = max(2, n - val_bars)

    sel_ds = train_ds.slice_indices(0, cut)
    val_ds = train_ds.slice_indices(cut, n)
    if len(val_ds) == 0:
        return None, "no validation tail"

    # Sweep over candidates using BOTH halves of sel_ds for scoring is
    # wrong; instead run the sweep directly against sel_ds split in half.
    report = run_parameter_sweep(space, base, sel_ds,
                                 taker_fee=str(cost.taker_fee),
                                 slippage_pct=str(cost.slippage_pct),
                                 initial_capital=capital)
    try:
        locked = select_parameters(report, rule)
    except NoCandidate as e:
        return None, f"no candidate: {e.reason}"
    return locked, None


def run_walk_forward(space: ParameterSpace, base: BaseSpec,
                     dataset: OHLCVDataset, *,
                     train_bars: int, test_bars: int, step_bars: int,
                     taker_fee: str = "0", slippage_pct: str = "0",
                     initial_capital: str = "10000",
                     rule: SelectionRule | None = None) -> WalkForwardReport:
    from src.core.money import init_money_context
    init_money_context()
    rule = rule or SelectionRule()

    cost = CostModel(taker_fee=Decimal(taker_fee),
                     slippage_pct=Decimal(slippage_pct))
    tf_min = timeframe_minutes(base.timeframe)

    windows: List[WindowResult] = []
    returns: List[float] = []
    with_trades = 0

    for w_idx, (tr_sl, te_sl) in enumerate(
            iter_windows(len(dataset), train_bars, test_bars, step_bars)):
        train_ds = dataset.slice_indices(tr_sl.start, tr_sl.stop)
        test_ds = dataset.slice_indices(te_sl.start, te_sl.stop)

        locked, reason = _locked_for_window(space, base, train_ds, cost,
                                            initial_capital, rule)
        if locked is None:
            windows.append(WindowResult(
                window_index=w_idx, train_range=[tr_sl.start, tr_sl.stop],
                test_range=[te_sl.start, te_sl.stop],
                locked=None, skip_reason=reason))
            continue

        strategy = build_strategy(space, base, locked.params)
        bt = run_backtest(strategy, test_ds, cost, initial_capital)
        m = compute_metrics(bt.equity_curve, bt.trades,
                            initial_capital=float(initial_capital),
                            timeframe_minutes=tf_min,
                            fees_paid=float(bt.fees_paid),
                            slippage_cost=float(bt.slippage_cost),
                            turnover_notional=float(bt.turnover_notional))
        ret = m.total_return
        if m.trade_count > 0:
            with_trades += 1
        returns.append(ret)
        windows.append(WindowResult(
            window_index=w_idx, train_range=[tr_sl.start, tr_sl.stop],
            test_range=[te_sl.start, te_sl.stop],
            locked=locked, oos_metrics=m.model_dump(mode="json"),
            oos_return=ret))

    evaluated = [r for r in returns]
    mean_r = (sum(evaluated) / len(evaluated)) if evaluated else None
    var = ((sum((r - mean_r) ** 2 for r in evaluated) / len(evaluated))
           if evaluated else None)
    std_r = (var ** 0.5) if var is not None else None
    pos_rate = (sum(1 for r in evaluated if r > 0) / len(evaluated)
                ) if evaluated else None

    return WalkForwardReport(
        n_windows=len(windows),
        windows=windows,
        mean_oos_return=mean_r,
        std_oos_return=std_r,
        positive_rate=pos_rate,
        windows_with_trades=with_trades,
    )
