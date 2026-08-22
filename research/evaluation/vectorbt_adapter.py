"""
VectorBT adapter (Phase 8).

VectorBT is an EVALUATION BACKEND ONLY -- never part of the strategy
contract, never imported by strategies. This module translates our
signal stream + cost model into VectorBT's portfolio simulation.

SEMANTIC MAPPING (documented)
-----------------------------
Our engine: decision on close(T), fill at open(T+1) with adverse
slippage baked into the fill price and taker fee on notional.
VectorBT equivalent: shift the signal arrays by one bar (execute next
bar) and supply `price=` as the slippage-adjusted open array with
`fees=taker_rate`. Both engines therefore fill the same trades at the
same prices with the same costs; cross-validation compares equity.
"""
from typing import Dict, List, Tuple

from src.strategies.base import BaseStrategy

from .costs import CostModel


def collect_signal_marks(strategy: BaseStrategy, dataset) -> List[Tuple[int, str]]:
    """
    Replays the dataset through a FRESH strategy instance and records
    (bar_index, "BUY"/"SELL") for every emitted intent. Pure translation
    helper shared by both engines.
    """
    from src.core.types import MarketEvent, Packet  # local: no hard dep at import time

    marks = []
    symbol = strategy.config.symbol
    strategy.reset()
    for i, bar in enumerate(dataset.bars):
        event = MarketEvent(packet=Packet(
            exchange_ts=bar.ts * 1000, local_arrival_ts=bar.ts * 1000,
            drift_us=0, source="backtest", topic=symbol,
            payload={"price": str(bar.close)}, sequence_id=bar.ts))
        intent = strategy.on_market_event(event)
        if intent is not None:
            side = "BUY" if intent.side.value == "BUY" else "SELL"
            marks.append((i, side))
    return marks


def signals_to_entries_exits(marks: List[Tuple[int, str]], n_bars: int,
                             long_short: bool = True):
    """
    -> (entries, exits, short_entries, short_exits) numpy bool arrays,
    shifted by one bar to implement next-bar execution.
    A BUY while short acts as short-exit AND long-entry (mirrors our
    reversal semantics); SELL symmetric.
    """
    import numpy as np

    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)
    s_entries = np.zeros(n_bars, dtype=bool)
    s_exits = np.zeros(n_bars, dtype=bool)

    for bar_idx, side in marks:
        j = bar_idx + 1                    # next-bar execution
        if j >= n_bars:
            continue                       # unfilled final-bar intent
        if side == "BUY":
            entries[j] = True
            s_exits[j] = True
        else:
            if long_short:
                s_entries[j] = True
            exits[j] = True
    return entries, exits, s_entries, s_exits


def slipped_open_prices(dataset, cost_model: CostModel):
    """Per-side adverse-slipped open arrays (fill price series)."""
    import numpy as np

    opens = np.array(dataset.opens(), dtype=float)
    s = float(cost_model.slippage_pct)
    buy_px = opens * (1.0 + s)
    sell_px = opens * (1.0 - s)
    return buy_px, sell_px


def run_vectorbt(strategy: BaseStrategy, dataset, cost_model: CostModel,
                 initial_cash: float) -> Dict[str, float]:
    """
    Cross-validation backend. Returns a metrics subset comparable with
    compute_metrics (total_return / max_dd / trade counts).
    Raises ImportError with guidance when vectorbt is unavailable.
    """
    try:
        import vectorbt as vbt
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "vectorbt is required for this evaluation backend") from e

    marks = collect_signal_marks(strategy, dataset)
    entries, exits, s_entries, s_exits = signals_to_entries_exits(
        marks, len(dataset))

    close = dataset.closes()
    opens = dataset.opens()

    def run_one(entries_arr, exits_arr):
        pf = vbt.Portfolio.from_signals(
            close, entries_arr, exits_arr,
            price=opens,                       # next-bar OPEN fills
            slippage=float(cost_model.slippage_pct),   # adverse, per-order
            fees=float(cost_model.taker_fee),  # fraction of order value
            init_cash=initial_cash,
            # SIZE TRANSLATION: fixed contract quantities, never all-in.
            size=float(strategy.config.trade_size),
            size_type="amount",
        )
        return {
            "total_return": float(pf.total_return()),
            "max_drawdown": float(pf.max_drawdown()),
            "n_trades": int(pf.trades.count()) if hasattr(pf.trades, "count")
                        else int(len(pf.trades)),
        }

    return run_one(entries, exits)
