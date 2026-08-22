"""
Phase 8: data layer tests — normalization, hashing, Parquet round-trips,
validation defect detection.
"""
import pytest
from pydantic import ValidationError

from research.data import (
    OHLCVDataset, validate_dataset, timeframe_minutes, normalize_ccxt_row)


BASE_ROWS = [
    [1600000000000, 100.0, 101.0, 99.0, 100.5, 10.0],
    [1600000060000, 100.5, 102.0, 100.0, 101.5, 12.0],
    [1600000120000, 101.5, 103.0, 101.0, 102.5, 11.0],
]


def ds_from(rows, symbol="BTC/USDT", tf="1m"):
    return OHLCVDataset.from_records(rows, symbol=symbol, timeframe=tf)


class TestNormalization:
    def test_ccxt_row_to_canonical_decimals(self):
        d = normalize_ccxt_row([1600000000000, 100.5, 101, 99.75, 100.25, 12.5])
        from src.core.money import to_decimal
        assert d["ts"] == 1600000000000
        assert d["open"] == to_decimal("100.5")
        assert d["close"] == to_decimal("100.25")
        assert d["volume"] == to_decimal("12.5")

    def test_bar_is_frozen(self):
        ds = ds_from(BASE_ROWS)
        with pytest.raises(ValidationError):
            ds.bars[0].close = 999

    def test_unsupported_timeframe_rejected(self):
        with pytest.raises(ValueError):
            timeframe_minutes("7m")

    def test_known_timeframes(self):
        assert timeframe_minutes("1m") == 1
        assert timeframe_minutes("1h") == 60
        assert timeframe_minutes("1d") == 1440


class TestOrderingAndHash:
    def test_records_sorted_regardless_of_input_order(self):
        shuffled = list(reversed(BASE_ROWS))
        a = ds_from(BASE_ROWS)
        b = ds_from(shuffled)
        assert [x.ts for x in b.bars] == [x.ts for x in a.bars]

    def test_content_hash_deterministic(self):
        assert ds_from(BASE_ROWS).content_hash() == ds_from(BASE_ROWS).content_hash()

    def test_content_hash_independent_of_input_order(self):
        assert ds_from(BASE_ROWS).content_hash() == ds_from(list(reversed(BASE_ROWS))).content_hash()

    def test_content_hash_sensitive_to_values(self):
        tweaked = [list(r) for r in BASE_ROWS]
        tweaked[1][4] = 101.5000001
        assert ds_from(tweaked).content_hash() != ds_from(BASE_ROWS).content_hash()

    def test_hash_includes_symbol_and_timeframe(self):
        assert ds_from(BASE_ROWS, symbol="ETH/USDT").content_hash() \
               != ds_from(BASE_ROWS, symbol="BTC/USDT").content_hash()


class TestParquet:
    def test_round_trip_identical_hash_and_bars(self, tmp_path):
        ds = ds_from(BASE_ROWS)
        p = ds.to_parquet(tmp_path / "ds.parquet")
        loaded = OHLCVDataset.from_parquet(p)
        assert loaded.content_hash() == ds.content_hash()
        assert loaded.bars == ds.bars
        assert loaded.symbol == ds.symbol
        assert loaded.timeframe == ds.timeframe

    def test_parquet_survives_decimal_precision(self, tmp_path):
        rows = [[1600000000000, 117234.52000000, 117300.5, 117000.1,
                 117250.25000000, 3.14159265]]
        ds = ds_from(rows)
        loaded = OHLCVDataset.from_parquet(ds.to_parquet(tmp_path / "p.parquet"))
        assert loaded.bars == ds.bars


def _mk(rows, **kw):
    return validate_dataset(ds_from(rows, **kw))


class TestValidation:
    def test_clean_dataset_passes(self):
        rep = _mk(BASE_ROWS)
        assert rep.is_clean
        assert rep.n_bars == 3

    def test_duplicate_timestamps_detected(self):
        rows = BASE_ROWS + [BASE_ROWS[-1]]     # exact repeat
        rep = _mk(rows)
        assert rep.duplicate_timestamps >= 1

    def test_gap_detected_with_boundaries(self):
        rows = [
            [1600000000000, 100, 101, 99, 100, 1],
            [1600000060000, 100, 101, 99, 100, 1],
            # missing 1600000120000 (one full period)
            [1600000180000, 100, 101, 99, 100, 1],
        ]
        rep = _mk(rows)
        assert rep.gap_count == 1
        assert rep.gaps == [(1600000060000, 1600000180000)]
        assert not rep.is_clean

    def test_invalid_ohlc_high_below_close(self):
        rows = [[1600000000000, 100, 99, 98, 101, 1]]   # high < close
        assert _mk(rows).invalid_ohlc == 1

    def test_invalid_ohlc_low_above_open(self):
        rows = [[1600000000000, 101, 103, 102, 100, 1]]  # low > open
        assert _mk(rows).invalid_ohlc == 1

    def test_zero_price_detected(self):
        rows = [[1600000000000, 0, 1, 0, 0.5, 1]]
        assert _mk(rows).nonpositive_prices == 1

    def test_negative_volume_detected_zero_allowed(self):
        neg = _mk([[1600000000000, 100, 101, 99, 100, -5]])
        zero = _mk([[1600000000000, 100, 101, 99, 100, 0]])
        assert neg.negative_volume == 1
        assert zero.negative_volume == 0

    def test_gaps_travel_into_experiment_results(self, tmp_path):
        """Quality report must be attachable, not lost."""
        from research.evaluation.experiment import build_config, run_experiment
        from src.strategies.ema_crossover import EmaCrossoverConfig, EmaCrossoverStrategy

        rows = [
            [1600000000000, 100, 101, 99, 100, 1],
            [1600000060000, 100, 101, 99, 100, 1],
            [1600000180000, 100, 105, 99, 104, 1],
            [1600000240000, 104, 106, 100, 101, 1],
            [1600000300000, 101, 103, 99, 102, 1],
            [1600000360000, 102, 104, 100, 103, 1],
        ]
        ds = ds_from(rows)
        cfg = EmaCrossoverConfig(strategy_name="ema", strategy_version="1.0.0",
                                 symbol="BTC/USDT", timeframe="1m",
                                 trade_size="0.1", fast_period=2, slow_period=3)
        exp = build_config(strategy_config=cfg, dataset=ds)
        res = run_experiment(exp, ds, lambda: EmaCrossoverStrategy(cfg),
                             include_benchmark=False)
        assert res["dataset_quality"]["gap_count"] == 1
        assert res["dataset_quality"]["is_clean"] is False
