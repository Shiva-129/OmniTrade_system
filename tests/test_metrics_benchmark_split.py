"""
Phase 8: metrics, benchmark, and time-split tests.
Metrics verified against hand-computed fixtures; annualization assumes a
24/7 crypto calendar (525600 minutes/year).
"""
import math

import pytest

from research.evaluation.benchmark import run_buy_and_hold
from research.evaluation.metrics import compute_metrics, periods_per_year, _max_drawdown
from research.evaluation.split import train_val_test, walk_forward
from research.data import OHLCVDataset


def eq(points):
    return [{"ts": i, "equity": e} for i, e in enumerate(points)]


class TestAnnualization:
    def test_periods_per_year_crypto_calendar(self):
        assert periods_per_year(1) == 525600.0
        assert periods_per_year(60) == 8760.0
        assert periods_per_year(1440) == 365.0


class TestDrawdown:
    def test_known_drawdown(self):
        assert _max_drawdown([1000, 1250, 1000]) == pytest.approx(0.2)

    def test_no_drawdown_monotonic(self):
        assert _max_drawdown([1, 2, 3]) == 0.0


def _metrics(curve, trades=None, tf=60, capital=1000.0, fees=0.0, slip=0.0,
             turnover=0.0):
    return compute_metrics(eq(curve), trades or [],
                           initial_capital=capital, timeframe_minutes=tf,
                           fees_paid=fees, slippage_cost=slip,
                           turnover_notional=turnover)


class TestMetricMath:
    def test_flat_equity_zero_risk_numbers(self):
        m = _metrics([1000] * 10)
        assert m.total_return == 0.0
        assert m.sharpe == 0.0 and m.sortino == 0.0
        assert m.max_drawdown == 0.0
        # sub-day dataset: annualization refused (documented contract)
        assert m.cagr is None and m.calmar is None

    def test_total_return_exact(self):
        m = _metrics([1000, 1100])
        assert m.total_return == pytest.approx(0.10)

    def test_cagr_computed_only_for_full_day_datasets(self):
        curve = [1000 * (1 + 1e-6) ** i for i in range(8760)]  # 365 days hourly
        m = _metrics(curve)
        assert m.cagr is not None and m.cagr > 0
        assert m.calmar is not None

    def test_max_drawdown_and_calmar_consistency(self):
        # one year of hourly bars => years=1 => cagr = total return
        curve = [1000, 1250, 1000] + [1000] * 8758
        m = _metrics(curve)
        assert m.max_drawdown == pytest.approx(0.2)
        assert m.calmar == pytest.approx(m.cagr / 0.2)

    def test_trade_statistics_hand_computed(self):
        trades = [{"pnl": "50"}, {"pnl": "-20"}, {"pnl": "30"}]
        m = _metrics([1000, 1060], trades)
        assert m.trade_count == 3
        assert m.win_rate == pytest.approx(2 / 3)
        assert m.avg_win == pytest.approx(40.0)
        assert m.avg_loss == pytest.approx(-20.0)
        assert m.profit_factor == pytest.approx(80 / 20)
        assert m.expectancy == pytest.approx(20.0)

    def test_profit_factor_none_when_no_losses(self):
        m = _metrics([1000, 1010], [{"pnl": "10"}])
        assert m.profit_factor is None

    def test_sharpe_positive_for_smooth_gains(self):
        curve = [1000 * (1.001 ** i) for i in range(500)]
        m = _metrics(curve)
        assert m.sharpe > 1.0
        assert m.volatility > 0.0

    def test_fees_and_slippage_passthrough(self):
        m = _metrics([1000], fees=12.5, slip=3.25, turnover=200.0)
        assert m.fees_paid == 12.5
        assert m.slippage_cost == 3.25
        assert m.turnover == pytest.approx(0.2)


def _two_bar_ds():
    rows = [
        [1600000000000, 100.0, 105.0, 99.0, 104.0, 10],
        [1600000060000, 104.0, 210.0, 103.0, 200.0, 10],
    ]
    return OHLCVDataset.from_records(rows, symbol="BTC/USDT", timeframe="1m")


class TestBenchmark:
    def test_buy_and_hold_tracks_price_ratio_after_costs(self):
        ds = _two_bar_ds()
        from research.evaluation.costs import CostModel
        from src.core.money import to_decimal
        cm = CostModel(taker_fee=to_decimal("0"))
        curve, fees = run_buy_and_hold(ds, cm, "1000")
        ratio = 200.0 / 100.0
        assert curve[-1]["equity"] == pytest.approx(1000 * ratio)

    def test_costs_reduce_benchmark(self):
        ds = _two_bar_ds()
        from research.evaluation.costs import CostModel
        from src.core.money import to_decimal
        free_curve, free_fees = run_buy_and_hold(ds, CostModel(), "1000")
        paid_curve, paid_fees = run_buy_and_hold(
            ds, CostModel(taker_fee=to_decimal("0.01")), "1000")
        assert paid_curve[-1]["equity"] < free_curve[-1]["equity"]
        assert paid_fees > free_fees == 0.0


class TestSplits:
    def test_contiguous_time_ordered_split(self):
        tr, va, te = train_val_test(100)
        assert (tr.start, tr.stop) == (0, 60)
        assert (va.start, va.stop) == (60, 80)
        assert (te.start, te.stop) == (80, 100)
        assert tr.stop == va.start and va.stop == te.start

    def test_split_rejects_bad_fractions(self):
        with pytest.raises(ValueError):
            train_val_test(100, train_frac=0.9, val_frac=0.2)
        with pytest.raises(ValueError):
            train_val_test(2)                      # slices too small

    def test_walk_forward_windows_slide(self):
        windows = list(walk_forward(100, train_bars=50, test_bars=10, step_bars=10))
        assert len(windows) == 5                    # starts at 0..40
        for i, (tr, te) in enumerate(windows):
            assert tr.start == i * 10
            assert tr.stop == i * 10 + 50
            assert te.start == tr.stop              # test immediately after train
            assert te.stop - te.start == 10

    def test_walk_forward_never_leaks_forward_into_past_windows(self):
        """Each window's test slice must lie entirely AFTER its train."""
        for tr, te in walk_forward(120, 60, 20, 20):
            assert te.start >= tr.stop
