"""
Deterministic bar-backtest engine (Phase 8).

EXECUTION SEMANTICS (explicit contract)
---------------------------------------
1. DECISION: the strategy receives ONE MarketEvent per bar carrying that
   bar's CLOSE (Phase 7 close-based conventions). Decision at bar T uses
   only bars <= T.
2. FILL: an intent emitted at bar T is filled at the OPEN of bar T+1,
   adversely slipped and charged taker fees on the notional.
   Same-bar-close fills are forbidden (information leakage).
3. An intent pending on the final bar can never fill -> counted as
   `unfilled_intents`, never silently dropped.
4. ACCOUNTING: fills flow through core.portfolio.Portfolio -- the SAME
   implementation as live/replay. No parallel accounting exists.
5. MARKING: after each bar, the portfolio is marked at the bar CLOSE and
   equity bookkeeping advances -> deterministic equity curve.

The engine contains zero strategy logic and zero indicator math.
"""
from typing import Any, Dict, List

from src.core.money import init_money_context, to_decimal, ZERO
from src.core.types import ExecutionReport, MarketEvent, OrderIntent, OrderSide, Packet
from src.core.portfolio import Portfolio
from src.strategies.base import BaseStrategy

from .costs import CostModel


class BacktestResult:
    """Container (floats allowed in metrics domain only; money stays Decimal)."""

    def __init__(self):
        self.equity_curve: List[Dict[str, Any]] = []   # [{"ts","equity"}]
        self.trades: List[Dict[str, Any]] = []
        self.fills: List[Dict[str, Any]] = []
        self.filled_intents = 0
        self.unfilled_intents = 0
        self.rejected_small = 0
        self.fees_paid = ZERO
        self.slippage_cost = ZERO
        self.turnover_notional = ZERO
        # internal trade-assembly bookkeeping (not part of the public result)
        self._last_open_qty = ZERO
        self._open_entry: Dict[str, Any] | None = None

    def summary(self) -> Dict[str, Any]:
        return {
            "n_bars_marked": len(self.equity_curve),
            "filled_intents": self.filled_intents,
            "unfilled_intents": self.unfilled_intents,
            "rejected_small": self.rejected_small,
            "trades_closed": len(self.trades),
            "fees_paid": str(self.fees_paid),
            "slippage_cost": str(self.slippage_cost),
            "turnover_notional": str(self.turnover_notional),
        }


def run_backtest(strategy: BaseStrategy, dataset, cost_model: CostModel,
                 initial_capital: str) -> BacktestResult:
    init_money_context()
    symbol = strategy.config.symbol
    portfolio = Portfolio(starting_cash=to_decimal(initial_capital))
    result = BacktestResult()

    bars = dataset.bars
    pending: OrderIntent | None = None

    for i, bar in enumerate(bars):
        # ---- 1) execute intent decided on the PREVIOUS bar at THIS open ----
        if pending is not None:
            _execute(pending, ref=bar.open, ts=bar.ts, portfolio=portfolio,
                     cost_model=cost_model, result=result)
            pending = None

        # ---- 2) decision on this bar's close (same MarketEvent shape as live) --
        event = MarketEvent(packet=Packet(
            exchange_ts=bar.ts * 1000,          # us domain for contracts
            local_arrival_ts=bar.ts * 1000,
            drift_us=0,
            source="backtest",
            topic=symbol,
            payload={
                "price": str(bar.close),
                "open": str(bar.open), "high": str(bar.high),
                "low": str(bar.low), "volume": str(bar.volume),
            },
            sequence_id=bar.ts,
        ))
        intent = strategy.on_market_event(event)
        if intent is not None:
            if i == len(bars) - 1:
                result.unfilled_intents += 1     # no next bar to fill into
                pending = None
            else:
                pending = intent

        # ---- 3) mark-to-market at this bar's close ----
        portfolio.mark_price(symbol, bar.close, ts_us=bar.ts * 1000)
        eq = portfolio.update_equity(now_us=bar.ts * 1000)
        result.equity_curve.append({"ts": bar.ts, "equity": float(eq.equity)})

    return result


def _execute(intent: OrderIntent, ref, ts, portfolio: Portfolio,
             cost_model: CostModel, result: BacktestResult) -> None:
    if intent.quantity < cost_model.min_order_qty or intent.quantity <= ZERO:
        result.rejected_small += 1
        return

    side_str = "BUY" if intent.side == OrderSide.BUY else "SELL"
    fill_px = cost_model.fill_price(side_str, ref)
    actual_notional = intent.quantity * fill_px          # traded value
    slipped = abs(fill_px - ref) * intent.quantity
    fee = cost_model.fee(actual_notional)

    before_realized = portfolio.realized_pnl
    report = ExecutionReport(
        client_order_id=intent.client_order_id,
        exchange_order_id=f"bt-{intent.client_order_id}",
        symbol=intent.symbol,
        side=intent.side,
        status="FILLED",
        filled_quantity=intent.quantity,
        last_filled_price=fill_px,
        remaining_quantity="0",
        timestamp=ts * 1000,
        fee=fee,
    )
    portfolio.apply_report(report)
    realized_delta = portfolio.realized_pnl - before_realized

    result.filled_intents += 1
    result.fees_paid += fee
    result.slippage_cost += slipped
    result.turnover_notional += actual_notional

    fill_rec = {
        "ts": ts, "side": side_str,
        "qty": str(intent.quantity),
        "price": str(fill_px), "fee": str(fee),
        "slippage": str(slipped),
        "cloid": intent.client_order_id,
    }
    result.fills.append(fill_rec)

    # --- trade assembly ---
    pos = portfolio.positions.get(intent.symbol)
    qty_now = pos.quantity if pos else ZERO
    prev_open_qty = result._last_open_qty

    was_flat, now_flat = prev_open_qty == ZERO, qty_now == ZERO
    flipped = (prev_open_qty != ZERO and qty_now != ZERO
               and (qty_now > ZERO) != (prev_open_qty > ZERO))

    if was_flat and not now_flat:
        result._open_entry = {"entry_ts": ts, "entry_fill": fill_rec["cloid"],
                              "side_was": side_str}

    if (prev_open_qty != ZERO and (now_flat or flipped)):
        if result._open_entry is not None:
            result.trades.append({
                **result._open_entry,
                "exit_ts": ts,
                "exit_fill": fill_rec["cloid"],
                "pnl": str(realized_delta),
            })
            result._open_entry = None
        if flipped:
            result._open_entry = {"entry_ts": ts, "entry_fill": fill_rec["cloid"],
                                  "side_was": side_str}
    result._last_open_qty = qty_now
