"""
Buy-and-hold benchmark (Phase 8).

Convention: buy at the FIRST bar open (slipped + taker fee, same costs as
the strategy), hold to the end; terminal exposure is marked-to-market at
the last close WITHOUT liquidation fees (documented -- applies equally
when comparing against strategy equity marked at close).
"""
from src.core.money import Decimal, to_decimal
from src.core.types import ExecutionReport, OrderSide

from .costs import CostModel


def run_buy_and_hold(dataset, cost_model: CostModel,
                     initial_capital: str):
    from src.core.portfolio import Portfolio

    symbol = dataset.symbol
    portfolio = Portfolio(starting_cash=to_decimal(initial_capital))
    bars = dataset.bars
    if not bars:
        raise ValueError("empty dataset")

    first = bars[0]
    entry_ref = first.open
    px = cost_model.fill_price("BUY", entry_ref)
    # spend ALL initial cash on the slipped entry price
    notional_budget = portfolio.cash / (Decimal("1") + cost_model.taker_fee)
    qty = notional_budget / px

    report = ExecutionReport(
        client_order_id="bh-entry", exchange_order_id="bt-bh-entry",
        symbol=symbol, side=OrderSide.BUY, status="FILLED",
        filled_quantity=qty, last_filled_price=px,
        remaining_quantity="0", timestamp=first.ts * 1000,
        fee=cost_model.fee(qty * px),
    )
    portfolio.apply_report(report)

    curve = []
    for b in bars:
        portfolio.mark_price(symbol, b.close, ts_us=b.ts * 1000)
        eq = portfolio.update_equity(now_us=b.ts * 1000)
        curve.append({"ts": b.ts, "equity": float(eq.equity)})

    fees = float(portfolio.fees_paid)
    return curve, fees
