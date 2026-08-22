"""
Phase 5: Portfolio & Accounting tests.

Matrix: long/short lifecycle, reversals through zero, partial/multi fills,
fees, realized+unrealized PnL exactness, cash flows, equity/drawdown,
staleness semantics, safety no-ops, deterministic snapshots.

Every number asserted here is hand-computed Decimal arithmetic.
"""
from decimal import Decimal

import pytest

from src.core.money import init_money_context, to_decimal
from src.core.portfolio import Portfolio
from src.core.types import ExecutionReport, OrderSide
from src.simulator.state_hasher import StateHasher

D = to_decimal


@pytest.fixture(autouse=True)
def _money_ctx():
    init_money_context()


def rpt(cloid, side, qty, price, status="FILLED", fee="0", ts=1000,
        symbol="BTCUSDT"):
    return ExecutionReport(
        client_order_id=cloid,
        exchange_order_id=f"x-{cloid}",
        symbol=symbol,
        side=side,
        status=status,
        filled_quantity=D(qty),
        last_filled_price=D(price),
        remaining_quantity="0",
        timestamp=ts,
        fee=D(fee),
    )


def pf(cash="10000"):
    return Portfolio(starting_cash=cash)


def opened(side, qty, price, fee="0", cash="10000", symbol="BTCUSDT"):
    """Fresh portfolio holding ONE opened position (helper for chained cases)."""
    p = Portfolio(starting_cash=cash)
    p.apply_report(rpt("open-" + symbol, side, qty, price, fee=fee, symbol=symbol))
    return p


# ============================ LONG LIFECYCLE ============================

class TestLongLifecycle:
    def test_open_long(self):
        p = pf()
        assert p.apply_report(rpt("a", OrderSide.BUY, "1.0", "100")) == D("1.0")
        pos = p.positions["BTCUSDT"]
        assert pos.quantity == D("1.0")
        assert pos.avg_entry_price == D("100")
        assert p.cash == D("10000") - D("100")

    def test_add_long_weighted_average(self):
        p = opened(OrderSide.BUY, "1.0", "100")
        p.apply_report(rpt("b", OrderSide.BUY, "1.0", "110"))
        pos = p.positions["BTCUSDT"]
        assert pos.quantity == D("2.0")
        assert pos.avg_entry_price == D("105")  # (1*100 + 1*110)/2 exactly

    def test_partial_close_long_realizes_at_avg(self):
        p = opened(OrderSide.BUY, "2.0", "100")
        p.apply_report(rpt("b", "SELL", "0.5", "120"))
        # realized = (120-100)*0.5 = 10 ; basis stays 100
        assert p.realized_pnl == D("10")
        pos = p.positions["BTCUSDT"]
        assert pos.quantity == D("1.5")
        assert pos.avg_entry_price == D("100")
        assert p.cash == D("10000") - D("200") + D("60")

    def test_full_close_resets_basis_keeps_realized(self):
        p = opened(OrderSide.BUY, "2.0", "100")
        p.apply_report(rpt("b", "SELL", "2.0", "130"))
        pos = p.positions["BTCUSDT"]
        assert pos.quantity == D("0")
        assert pos.avg_entry_price == D("0")
        assert p.realized_pnl == D("60")  # (130-100)*2


# ============================ SHORT LIFECYCLE ===========================

class TestShortLifecycle:
    def test_open_short_cash_increases(self):
        p = opened("SELL", "1.0", "100")
        pos = p.positions["BTCUSDT"]
        assert pos.quantity == D("-1.0")
        assert pos.avg_entry_price == D("100")
        assert p.cash == D("10100")  # proceeds credited

    def test_add_short_weighted_average(self):
        p = opened("SELL", "1.0", "100")
        p.apply_report(rpt("b", "SELL", "1.0", "120"))
        pos = p.positions["BTCUSDT"]
        assert pos.quantity == D("-2.0")
        assert pos.avg_entry_price == D("110")

    def test_partial_close_short_profits_on_decline(self):
        p = opened("SELL", "2.0", "100")
        p.apply_report(rpt("b", OrderSide.BUY, "0.5", "80"))
        # realized = (100-80)*0.5 = 10
        assert p.realized_pnl == D("10")
        pos = p.positions["BTCUSDT"]
        assert pos.quantity == D("-1.5")
        assert pos.avg_entry_price == D("100")

    def test_full_close_short(self):
        p = opened("SELL", "2.0", "100")
        p.apply_report(rpt("b", OrderSide.BUY, "2.0", "70"))
        pos = p.positions["BTCUSDT"]
        assert pos.quantity == D("0")
        assert p.realized_pnl == D("60")  # (100-70)*2


# ============================== REVERSALS ===============================

class TestReversalThroughZero:
    def test_long_reverses_to_short(self):
        p = opened(OrderSide.BUY, "1.0", "100")
        p.apply_report(rpt("b", "SELL", "3.0", "90"))
        pos = p.positions["BTCUSDT"]
        # closing 1.0 @ avg100 vs price90 -> -10 realized;
        # residual 2.0 opens SHORT at fill price 90
        assert p.realized_pnl == D("-10")
        assert pos.quantity == D("-2.0")
        assert pos.avg_entry_price == D("90")

    def test_short_reverses_to_long(self):
        p = opened("SELL", "1.0", "100")
        p.apply_report(rpt("b", OrderSide.BUY, "4.0", "110"))
        pos = p.positions["BTCUSDT"]
        # closing short 1.0 @ avg100 vs price110 -> -10 realized;
        # residual 3.0 opens LONG at 110
        assert p.realized_pnl == D("-10")
        assert pos.quantity == D("3.0")
        assert pos.avg_entry_price == D("110")


# ================================ FILLS =================================

class TestFillSemantics:
    def test_multiple_fills_one_intent_accumulate(self):
        """One intent, three reports: partials then terminal fill.
        Sizes chosen so sequential weighted-averaging stays EXACT under
        the canonical context (divisions by 0.5 and 1 are terminating)."""
        p = pf()
        p.apply_report(rpt("o1", OrderSide.BUY, "0.2", "100", status="PARTIAL_FILL"))
        p.apply_report(rpt("o1", OrderSide.BUY, "0.3", "200", status="PARTIAL_FILL"))
        p.apply_report(rpt("o1", OrderSide.BUY, "0.5", "300", status="FILLED"))
        pos = p.positions["BTCUSDT"]
        assert pos.quantity == D("1.0")
        # step1: avg=100; step2: (0.2*100+0.3*200)/0.5 = 160;
        # step3: (0.5*160+0.5*300)/1.0 = 230 -- exact
        assert pos.avg_entry_price == D("230")

    def test_distinct_execution_prices_track_exact_notional(self):
        p = pf()
        p.apply_report(rpt("a", OrderSide.BUY, "0.1", "117234.52"))
        p.apply_report(rpt("b", "SELL", "0.1", "117300.00"))
        # cash: -11723.452 + 11730 = +6.548 net
        assert p.cash == D("10006.548")
        assert p.realized_pnl == to_decimal("6.548")


# ================================ FEES ==================================

class TestFees:
    def test_fees_charged_both_sides_and_accumulated(self):
        p = pf()
        p.apply_report(rpt("a", OrderSide.BUY, "1.0", "100", fee="0.10"))
        p.apply_report(rpt("b", "SELL", "1.0", "110", fee="0.11"))
        assert p.fees_paid == D("0.21")
        # cash: -100 -0.10 +110 -0.11
        assert p.cash == to_decimal("10009.79")

    def test_fee_survives_snapshot_round_trip(self):
        p = opened(OrderSide.BUY, "1.0", "100", fee="0.125")
        clone = Portfolio.from_snapshot(p.snapshot())
        assert clone.fees_paid == D("0.125")
        assert clone.snapshot() == p.snapshot()


# ================================= PNL ==================================

class TestPnLExactness:
    def test_classic_float_trap_is_exact(self):
        p = opened(OrderSide.BUY, "0.1", "1")
        p.apply_report(rpt("b", "SELL", "0.1", "1.3"))
        # 0.1*(1.3-1) = 0.03 exactly -- floats would give 0.030000000000000002
        assert p.realized_pnl == to_decimal("0.03")
        assert str(p.realized_pnl) == "0.03"

    def test_combined_realized_plus_unrealized(self):
        p = opened(OrderSide.BUY, "1.0", "100")
        p.apply_report(rpt("b", "SELL", "0.5", "120"))       # realized +10
        p.mark_price("BTCUSDT", D("130"), ts_us=1)
        unreal = p.unrealized_pnl("BTCUSDT")
        assert unreal == D("15")                              # (130-105)*... wait avg stays 100 -> (130-100)*0.5
        total = p.realized_pnl + unreal
        assert total == D("25")


# ================================ CASH ==================================

class TestCashAccounting:
    def test_buy_then_sell_cash_flows(self):
        p = pf()
        p.apply_report(rpt("a", OrderSide.BUY, "2.0", "50", fee="1"))
        assert p.cash == to_decimal("9899")     # 10000 - 100 - 1
        p.apply_report(rpt("b", "SELL", "1.0", "60", fee="1"))
        assert p.cash == to_decimal("9958")     # 9899 + 60 - 1

    def test_zero_quantity_fill_has_zero_effect(self):
        p = opened(OrderSide.BUY, "1.0", "100", fee="5")
        before = p.snapshot()
        assert p.apply_report(
            rpt("z", OrderSide.BUY, "0", "100", status="FILLED", fee="7")
        ) is None
        assert p.snapshot() == before  # even a weird zero-fill fee is ignored


# ========================= EQUITY / DRAWDOWN ============================

class TestEquityAndDrawdown:
    def test_equity_marks_to_market(self):
        p = opened(OrderSide.BUY, "1.0", "100")
        # unpriced first: partial valuation, cash-only figure
        eq0 = p.update_equity(now_us=1)
        assert eq0.fully_priced is False
        assert eq0.equity == D("9900")  # cash only; position excluded
        # now price arrives
        p.mark_price("BTCUSDT", D("150"), ts_us=1)
        eq = p.update_equity(now_us=1)
        assert eq.equity == D("10050")  # 9900 cash + 1.0*150
        assert eq.fully_priced is True
        assert p.peak_equity == D("10050")

    def test_drawdown_decline_and_recovery(self):
        p = opened(OrderSide.BUY, "1.0", "100")
        # peak forms: equity 9900+140 = 10040
        p.mark_price("BTCUSDT", D("140"), ts_us=1)
        p.update_equity(now_us=1)
        assert p.drawdown().peak_equity == D("10040")
        # decline: equity 9900+50 = 9950
        p.mark_price("BTCUSDT", D("50"), ts_us=2)
        p.update_equity(now_us=2)
        dd = p.drawdown()
        assert dd.equity == D("9950")
        assert dd.peak_equity == D("10040")
        assert dd.drawdown_abs == D("90")
        expected_pct = (D("90") / D("10040")) * D("100")  # 0.89641...% exactly
        assert dd.drawdown_pct == expected_pct
        # recovery to previous high clears drawdown without a new peak
        p.mark_price("BTCUSDT", D("140"), ts_us=3)
        p.update_equity(now_us=3)
        assert p.drawdown().drawdown_abs == D("0")
        assert p.drawdown().peak_equity == D("10040")

    def test_near_zero_peak_guarded(self):
        p = Portfolio(starting_cash="0")
        info = p.drawdown()
        assert info.peak_equity == D("0")
        assert info.drawdown_pct == D("0")  # ratio undefined; guarded not divided

    def test_stale_mark_becomes_unpriced(self):
        p = Portfolio("10000", max_staleness_us=100)
        p.apply_report(rpt("a", OrderSide.BUY, "1.0", "100"))
        p.mark_price("BTCUSDT", D("150"), ts_us=1000)
        fresh = p.unrealized_pnl("BTCUSDT", now_us=1050)     # within window
        stale = p.unrealized_pnl("BTCUSDT", now_us=2000)     # beyond window
        assert fresh == D("50")
        assert stale is None                                  # never invents prices
        eq = p.update_equity(now_us=2000)
        assert eq.fully_priced is False
        assert eq.unpriced_symbols == ("BTCUSDT",)
        # partial valuation must NOT advance the peak
        assert p.peak_equity == D("10000")

    def test_missing_mark_unpriced_flat_is_zero(self):
        p = opened(OrderSide.BUY, "1.0", "100")
        other = p.unrealized_pnl("ETHUSDT")      # no position -> ZERO
        unmarked = p.unrealized_pnl("BTCUSDT")   # position but no mark -> None
        assert other == D("0")
        assert unmarked is None


# =============================== SAFETY =================================

class TestMutationSafety:
    def test_non_fill_statuses_have_zero_accounting_effect(self):
        p = opened(OrderSide.BUY, "1.0", "100")
        baseline = p.snapshot()
        for status in ("NEW", "CANCELED", "REJECTED"):
            out = p.apply_report(rpt(f"r-{status}", OrderSide.BUY, "99", "1",
                                     status=status, fee="42"))
            assert out is None
            assert p.snapshot() == baseline, f"{status} mutated portfolio!"

    def test_mark_price_never_touches_positions_or_cash(self):
        """THE core invariant: MarketEvents mark only."""
        p = opened(OrderSide.BUY, "1.0", "100")
        before_positions = dict(p.positions)
        before_cash = p.cash
        p.mark_price("BTCUSDT", D("999999"), ts_us=5)
        assert dict(p.positions) == before_positions
        assert p.cash == before_cash
        assert p.marks["BTCUSDT"].price == D("999999")

    def test_portfolio_api_accepts_no_intents_or_decisions(self):
        """Structural guarantee: no mutation door exists for intentions."""
        import inspect
        public = [m for m in dir(Portfolio) if not m.startswith("_")]
        mutators = [m for m in ("apply_report", "mark_price", "update_equity")]
        assert set(mutators).issubset(set(public))
        sigs = {name: str(inspect.signature(getattr(Portfolio, name)))
                for name in mutators}
        assert all("OrderIntent" not in s and "RiskDecision" not in s
                   for s in sigs.values())


# ============================= SNAPSHOTS ================================

class TestSnapshots:
    def test_snapshot_round_trip_exact(self):
        p = opened(OrderSide.BUY, "1.23456789", "98765.4321", fee="0.001")
        p.apply_report(rpt("b", "SELL", "0.5", "99000.5", fee="0.002"))
        p.mark_price("BTCUSDT", D("99500"), ts_us=777)
        p.update_equity(now_us=777)
        clone = Portfolio.from_snapshot(p.snapshot())
        assert clone.snapshot() == p.snapshot()

    def test_snapshot_hash_deterministic_across_instances(self):
        def build():
            x = opened(OrderSide.BUY, "1.0", "100", fee="0.01")
            x.mark_price("BTCUSDT", D("110"), ts_us=9)
            return x

        h1 = StateHasher.hash_state(build().snapshot())
        h2 = StateHasher.hash_state(build().snapshot())
        assert h1 == h2

    def test_snapshot_values_are_fixed_point_strings(self):
        p = opened(OrderSide.BUY, "0.01000000", "117234.52000000")
        snap = p.snapshot()
        # 10000 - 0.01*117234.52 = 8827.6548 (trailing zeros from exponent alignment)
        assert Decimal(snap["cash"]) == Decimal("8827.6548")
        assert snap["positions"]["BTCUSDT"]["quantity"] == "0.01000000"
        json_text = str(sorted(snap.items()))
        assert "E+" not in json_text and "e-" not in json_text

    def test_hash_changes_when_financials_change(self):
        p = pf()
        h0 = StateHasher.hash_state(p.snapshot())
        p.apply_report(rpt("a", OrderSide.BUY, "1.0", "100"))
        h1 = StateHasher.hash_state(p.snapshot())
        assert h0 != h1
