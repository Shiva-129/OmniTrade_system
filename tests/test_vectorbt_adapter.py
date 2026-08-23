"""
Phase 8: VectorBT adapter tests.

1. Translation correctness: signal marks -> shifted boolean arrays
   (next-bar execution), unfilled final intents dropped.
2. Cross-validation: our deterministic engine and VectorBT must agree on
   the same fixture (same fills, same costs) within tolerance.
"""
import pytest

from research.data import OHLCVDataset
from research.evaluation.costs import CostModel
from research.evaluation.engine import run_backtest
from research.evaluation.vectorbt_adapter import (
    collect_signal_marks, signals_to_entries_exits, run_vectorbt)
from src.core.money import init_money_context, to_decimal
from src.strategies.ema_crossover import EmaCrossoverConfig, EmaCrossoverStrategy

PRICES = [100, 100, 100, 140, 145, 150, 90, 85, 80, 120]


def make_ds(prices=None):
    prices = prices or PRICES
    rows = [[1600000000000 + i * 60000, float(p), float(p) + 1,
             max(float(p) - 1, 0.5), float(p), 10.0]
            for i, p in enumerate(prices)]
    return OHLCVDataset.from_records(rows, symbol="BTC/USDT", timeframe="1m")


def make_strategy():
    return EmaCrossoverStrategy(EmaCrossoverConfig(
        strategy_name="ema", strategy_version="1.0.0",
        symbol="BTC/USDT", timeframe="1m", trade_size="0.5",
        fast_period=2, slow_period=3))


@pytest.fixture(autouse=True)
def _ctx():
    init_money_context()


class TestTranslation:
    def test_signal_marks_match_engine_fills(self):
        marks = collect_signal_marks(make_strategy(), make_ds())
        assert marks == [(3, "BUY"), (6, "SELL"), (9, "BUY")]

    def test_arrays_shifted_one_bar(self):
        marks = [(3, "BUY"), (6, "SELL")]
        e, x, se, sx = signals_to_entries_exits(marks, n_bars=10)
        assert list(e).count(True) == 1 and e[4]           # idx3 -> idx4
        assert list(x).count(True) == 1 and x[7]           # idx6 -> idx7
        # BUY -> short-exit at j, SELL -> short-entry + long-exit
        assert list(se).count(True) == 1 and se[7]
        assert list(sx).count(True) == 1 and sx[4]

    def test_long_short_arrays(self):
        marks = [(2, "BUY"), (5, "SELL")]
        e, x, se, sx = signals_to_entries_exits(marks, n_bars=10, long_short=True)
        assert e[3] and sx[3]   # BUY -> long entry + short exit
        assert x[6] and se[6]   # SELL -> long exit + short entry
        e2, x2, se2, sx2 = signals_to_entries_exits(marks, n_bars=10, long_short=False)
        assert not any(se2)     # short entries disabled
        assert e2[3] and sx2[3] # BUY still gives long entry + short exit
        assert x2[6] and not se2[6]

    def test_final_bar_intent_dropped_from_arrays(self):
        marks = [(9, "BUY")]                                # last bar of 10
        e, x, se, sx = signals_to_entries_exits(marks, n_bars=10)
        assert not any(e) and not any(x)                    # cannot fill


class TestCrossValidationAgainstVectorBT:
    def test_engines_agree_on_total_return(self):
        ds = make_ds()
        cm = CostModel(taker_fee=to_decimal("0.001"),
                       slippage_pct=to_decimal("0"))
        ours = run_backtest(make_strategy(), ds, cm, "10000")

        vbt_res = run_vectorbt(make_strategy(), ds,
                               CostModel(taker_fee=to_decimal("0.001"),
                                         slippage_pct=to_decimal("0")),
                               initial_cash=10000.0)

        ours_ret = ours.equity_curve[-1]["equity"] / 10000.0 - 1.0
        # NOTE: our final-bar intent stays UNFILLED; vbt drops it too.
        assert ours_ret == pytest.approx(vbt_res["total_return"], rel=1e-9)

    def test_engines_agree_with_costs(self):
        ds = make_ds()
        for taker, slip in [("0", "0"), ("0.001", "0"), ("0.001", "0.0005")]:
            cm = CostModel(taker_fee=to_decimal(taker),
                           slippage_pct=to_decimal(slip))
            ours = run_backtest(make_strategy(), ds, cm, "10000")
            theirs = run_vectorbt(make_strategy(), ds, cm, 10000.0)
            ours_ret = ours.equity_curve[-1]["equity"] / 10000.0 - 1.0
            assert ours_ret == pytest.approx(theirs["total_return"], rel=1e-9), \
                f"divergence at taker={taker} slip={slip}"

    def test_trade_counts_match(self):
        ds = make_ds()
        cm = CostModel(taker_fee=to_decimal("0.001"))
        ours = run_backtest(make_strategy(), ds, cm, "10000")
        theirs = run_vectorbt(make_strategy(), ds, cm, 10000.0)
        # one closed round trip in both engines
        assert len(ours.trades) == theirs["n_trades"]
