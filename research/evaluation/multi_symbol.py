"""Multi-symbol portfolio backtest (Phase 15 R2).

Merges N per-symbol bar streams by (timestamp, symbol) and routes each
bar to its strategy. ALL fills mutate ONE shared Portfolio via the
existing run_backtest fill path (_execute) -- next-open fill, costs.

DETERMINISM: merge key is (ts, symbol) — stable sort, insertion-order
independent. Same inputs => identical results.

N=1 REGRESSION: with a single symbol this harness must produce the same
fills and equity as run_backtest on the same dataset.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.core.money import init_money_context, to_decimal, ZERO
from src.core.portfolio import Portfolio
from src.core.types import MarketEvent, OrderIntent, Packet

from ..data.dataset import OHLCVDataset
from .costs import CostModel


def _merge_by_timestamp(datasets: Dict[str, OHLCVDataset]) -> List[Tuple[str, Any]]:
    """
    Deterministic merge: sort by (ts, symbol). Insertion-order independent.
    """
    stream: List[Tuple[int, str, Any]] = []
    for symbol in sorted(datasets.keys()):
        for bar in datasets[symbol]:
            stream.append((bar.ts, symbol, bar))
    stream.sort(key=lambda t: (t[0], t[1]))
    return [(sym, bar) for _, sym, bar in stream]


class MultiSymbolResult:
    def __init__(self):
        self.equity_curve: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        self.fills: List[Dict[str, Any]] = []
        self.filled_intents = 0
        self.unfilled_intents = 0
        self.rejected_small = 0
        self.fees_paid = ZERO
        self.slippage_cost = ZERO
        self.turnover_notional = ZERO
        self._last_open_qty = ZERO
        self._open_entry: Dict[str, Any] | None = None

    def summary(self) -> Dict[str, Any]:
        return {
            "n_bars_marked": len(self.equity_curve),
            "filled_intents": self.filled_intents,
            "unfilled_intents": self.unfilled_intents,
            "trades_closed": len(self.trades),
            "fees_paid": str(self.fees_paid),
            "slippage_cost": str(self.slippage_cost),
            "turnover_notional": str(self.turnover_notional),
        }


def run_multi_symbol_backtest(
    strategies_and_datasets: Dict[str, Tuple[Any, OHLCVDataset]],
    cost_model: CostModel,
    initial_capital: str,
) -> MultiSymbolResult:
    """
    strategies_and_datasets: {symbol: (strategy_instance, dataset)}
    All strategies share ONE Portfolio.
    Each strategy sees only its own symbol's bars (isolation by routing).
    """
    init_money_context()
    pf = Portfolio(starting_cash=to_decimal(initial_capital))

    result = MultiSymbolResult()
    # Per-symbol pending intents (market orders work at next tick of that symbol)
    pending: Dict[str, OrderIntent] = {}

    merged = _merge_by_timestamp({s: d for s, (_, d) in strategies_and_datasets.items()})

    from .engine import _execute  # reuse the exact fill path

    bars_seen = 0
    total_bars = sum(len(d) for _, d in strategies_and_datasets.values())
    # Track remaining bars per symbol so unfilled detection is per-symbol
    remaining_per_symbol = {s: len(d) for s, (_, d) in strategies_and_datasets.items()}

    for symbol, bar in merged:
        strategy, dataset = strategies_and_datasets[symbol]
        remaining_per_symbol[symbol] -= 1

        # 1) Fill pending intent for this symbol at THIS bar's open
        if pending.get(symbol) is not None:
            intent = pending.pop(symbol)
            ref = bar.open
            from .engine import _execute as _ex
            _execute(intent, ref=ref, ts=bar.ts, portfolio=pf,
                     cost_model=cost_model, result=result)

        # 2) Mark-to-market at close
        event_price = to_decimal(str(bar.close))
        pf.mark_price(symbol, event_price, ts_us=bar.ts * 1000)
        eq_result = pf.update_equity(now_us=bar.ts * 1000)
        result.equity_curve.append({"ts": bar.ts, "equity": float(eq_result.equity)})

        # 3) Feed strategy
        event = MarketEvent(packet=Packet(
            exchange_ts=bar.ts * 1000, local_arrival_ts=bar.ts * 1000,
            drift_us=0, source="multi_backtest", topic=symbol,
            payload={"price": str(bar.close)}, sequence_id=bar.ts))
        intent = strategy.on_market_event(event)
        if intent is not None:
            if remaining_per_symbol[symbol] <= 0:
                result.unfilled_intents += 1
            else:
                pending[symbol] = intent
        bars_seen += 1

    return result
