"""
Phase 8: backtest engine tests — next-open execution, costs, unfilled
final intents, trade assembly, determinism, no-look-ahead via the ENGINE.

Fixture series (EMA fast=2 slow=3), hand-traced:
  closes [100,100,100,140,145,150,90,85,80,120]
  -> BUY decided idx3 fills idx4 open=145; SELL decided idx6 fills idx7
     open=85; BUY decided idx9 (last) -> UNFILLED.
Costs: taker 0.001, slippage 0.0005. Exact values (Decimal):
  entry fill 145.07250, exit fill 84.95750, qty 0.5
  fees      = 0.001 * 0.5*(145.0725+84.9575) = 0.115015
  slippage  = 0.5*(0.07250+0.04250)          = 0.057500
  turnover  =                                = 115.015000
  realized pnl = (84.9575-145.0725)*0.5      = -30.057500
"""
import pytest

from research.data import OHLCVDataset
from research.evaluation.costs import CostModel
from research.evaluation.engine import run_backtest
from src.core.money import init_money_context, to_decimal
from src.strategies.ema_crossover import EmaCrossoverConfig, EmaCrossoverStrategy

PRICES = [100, 100, 100, 140, 145, 150, 90, 85, 80, 120]


def make_ds(prices=None, start=1600000000000):
    prices = prices or PRICES
    rows = [[start + i * 60000, float(p), float(p) + 1,
             max(float(p) - 1, 0.5), float(p), 10.0]
            for i, p in enumerate(prices)]
    return OHLCVDataset.from_records(rows, symbol="BTC/USDT", timeframe="1m")


def make_strategy():
    return EmaCrossoverStrategy(EmaCrossoverConfig(
        strategy_name="ema", strategy_version="1.0.0",
        symbol="BTC/USDT", timeframe="1m", trade_size="0.5",
        fast_period=2, slow_period=3))


def run(prices=None, taker="0.001", slip="0.0005", capital="10000"):
    init_money_context()
    return run_backtest(make_strategy(), make_ds(prices),
                        CostModel(taker_fee=to_decimal(taker),
                                  slippage_pct=to_decimal(slip)),
                        initial_capital=capital)


class TestExecutionSemantics:
    def test_fills_at_next_open_with_adverse_slippage(self):
        r = run()
        assert r.fills[0]["ts"] == 1600000240000            # idx4 open
        assert to_decimal(r.fills[0]["price"]) == to_decimal("145.0725")
        assert to_decimal(r.fills[1]["price"]) == to_decimal("84.9575")

    def test_exact_costs_and_turnover(self):
        r = run()
        assert r.fees_paid == to_decimal("0.115015")
        assert r.slippage_cost == to_decimal("0.0575")
        assert r.turnover_notional == to_decimal("115.015")

    def test_final_bar_intent_counted_unfilled(self):
        r = run()
        assert r.filled_intents == 2
        assert r.unfilled_intents == 1                      # last-bar BUY

    def test_trade_assembly_entry_exit(self):
        r = run()
        assert len(r.trades) == 1
        t = r.trades[0]
        assert t["side_was"] == "BUY"
        assert t["entry_ts"] == 1600000240000
        assert t["exit_ts"] == 1600000420000                # idx7
        assert to_decimal(t["pnl"]) == to_decimal("-30.0575")

    def test_min_order_size_rejects_tiny_intents(self):
        small = EmaCrossoverStrategy(EmaCrossoverConfig(
            strategy_name="ema", strategy_version="1.0.0",
            symbol="BTC/USDT", timeframe="1m", trade_size="0.00001",
            fast_period=2, slow_period=3))
        init_money_context()
        r = run_backtest(small, make_ds(),
                         CostModel(taker_fee=to_decimal("0.001"),
                                   min_order_qty=to_decimal("0.001")),
                         "10000")
        # two executable intents rejected on size; last-bar intent unfilled
        assert r.filled_intents == 0
        assert r.rejected_small == 2
        assert r.unfilled_intents == 1


class TestZeroCostIdentity:
    def test_zero_costs_fill_at_exact_open(self):
        r = run(taker="0", slip="0")
        assert to_decimal(r.fills[0]["price"]) == to_decimal("145")
        assert r.fees_paid == to_decimal("0")
        assert r.slippage_cost == to_decimal("0")
        # pnl = (85 - 145) * 0.5 exactly
        assert to_decimal(r.trades[0]["pnl"]) == to_decimal("-30")


class TestEquityCurve:
    def test_curve_marked_every_bar_starting_flat(self):
        r = run()
        assert len(r.equity_curve) == len(PRICES)
        assert r.equity_curve[0] == {"ts": 1600000000000, "equity": 10000.0}

    def test_losing_fixture_ends_below_start(self):
        r = run()
        assert r.equity_curve[-1]["equity"] < 10000.0


class TestDeterminismAndNoLookAhead:
    def test_repeated_runs_identical(self):
        assert run().summary() == run().summary()

    def test_future_prices_cannot_change_past_decisions(self):
        """Phase-7 truncation methodology through the ENGINE.

        Property under test: everything DECIDED/FILLED strictly before the
        cutoff is invariant to arbitrarily different future prices. The
        open trade's EXIT lies after the cutoff by construction and may
        legitimately differ -- it is excluded from the comparison.
        """
        prefix = PRICES[:7]
        cutoff_ts = 1600000000000 + 6 * 60000          # last prefix bar ts

        alt_future = [x * 10 for x in PRICES[7:]]      # dramatic divergence
        ra = run(prefix + PRICES[7:])
        rb = run(prefix + alt_future)

        pre_fills_a = [f for f in ra.fills if f["ts"] <= cutoff_ts]
        pre_fills_b = [f for f in rb.fills if f["ts"] <= cutoff_ts]
        assert pre_fills_a == pre_fills_b and len(pre_fills_a) >= 1

        assert ra.equity_curve[:7] == rb.equity_curve[:7]

    def test_engine_reuses_core_portfolio_accounting(self):
        """Single accounting truth: cash movement matches core.Portfolio math."""
        r = run()
        expected_cash = (to_decimal("10000")
                         - to_decimal("72.53625") - to_decimal("0.072536250")
                         + to_decimal("42.47875") - to_decimal("0.042478750"))
        assert r.equity_curve[-1]["equity"] == pytest.approx(float(expected_cash))
