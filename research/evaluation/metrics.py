"""
Performance metrics computed FROM equity/trade data (never from signals).

FLOAT DOMAIN (documented): metrics are statistical summaries for humans
and research comparison; the trading path remains Decimal. Annualization
assumes a 24/7 crypto calendar: periods_per_year = 525600 / timeframe_min.
"""
import math
from typing import Any, Dict, List

from pydantic import BaseModel


class Metrics(BaseModel):
    total_return: float
    cagr: float | None             # None when dataset spans < 1 day (annualizing
    calmar: float | None           # shorter windows is meaningless, not merely noisy)
    sharpe: float
    sortino: float
    max_drawdown: float            # fraction, 0..1
    volatility: float              # annualized bar-return std
    win_rate: float
    avg_win: float
    avg_loss: float                # negative number
    profit_factor: float | None    # None when no losing trades
    expectancy: float              # mean trade pnl
    trade_count: int
    turnover: float                # traded notional / initial capital
    fees_paid: float
    slippage_cost: float


def periods_per_year(timeframe_minutes: int) -> float:
    return 525600.0 / float(timeframe_minutes)


def _max_drawdown(equity: List[float]) -> float:
    peak = -math.inf
    dd = 0.0
    for e in equity:
        peak = max(peak, e)
        if peak > 0:
            dd = max(dd, (peak - e) / peak)
    return dd


def compute_metrics(equity_curve: List[Dict[str, Any]],
                    trades: List[Dict[str, Any]],
                    *, initial_capital: float, timeframe_minutes: int,
                    fees_paid: float, slippage_cost: float,
                    turnover_notional: float) -> Metrics:
    eq = [p["equity"] for p in equity_curve]
    if not eq:
        raise ValueError("empty equity curve")

    start, end = float(initial_capital), eq[-1]
    total_return = end / start - 1.0 if start > 0 else 0.0

    ppy = periods_per_year(timeframe_minutes)
    years = len(eq) / ppy
    max_dd = _max_drawdown(eq)

    # CAGR/Calmar require >= 1 day of data: annualizing shorter windows is
    # statistically meaningless (and numerically explosive), so we refuse.
    cagr = None
    calmar = None
    if years >= 1.0:
        if start > 0 and end > 0:
            cagr = (end / start) ** (1.0 / years) - 1.0
        else:
            cagr = -1.0
        calmar = (cagr / max_dd) if max_dd > 0 else 0.0

    rets = [eq[i] / eq[i - 1] - 1.0 for i in range(1, len(eq)) if eq[i - 1] != 0]
    if rets:
        mean_r = sum(rets) / len(rets)
        var = sum((r - mean_r) ** 2 for r in rets) / len(rets)
        vol = var ** 0.5
        sharpe = (mean_r / vol * math.sqrt(ppy)) if vol > 0 else 0.0
        downside = [min(r, 0.0) for r in rets]
        dvar = sum(d ** 2 for d in downside) / len(downside)
        dstd = dvar ** 0.5
        sortino = (mean_r / dstd * math.sqrt(ppy)) if dstd > 0 else 0.0
    else:
        vol = sharpe = sortino = 0.0

    pnls = [float(t["pnl"]) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = (len(wins) / len(pnls)) if pnls else 0.0
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    gross_w, gross_l = sum(wins), abs(sum(losses))
    profit_factor = (gross_w / gross_l) if gross_l > 0 else None
    expectancy = (sum(pnls) / len(pnls)) if pnls else 0.0

    return Metrics(
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        volatility=vol * math.sqrt(ppy),
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        expectancy=expectancy,
        trade_count=len(pnls),
        turnover=turnover_notional / start if start > 0 else 0.0,
        fees_paid=fees_paid,
        slippage_cost=slippage_cost,
    )
