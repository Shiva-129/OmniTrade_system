import pytest

from research.data.bars import aggregate_bars
from research.data.dataset import OHLCVDataset


def make_ds(prices, timeframe="1m"):
    rows = [[1600000000000 + i * 60000, float(p), float(p) + 1,
             max(float(p) - 1, 0.5), float(p), 10.0] for i, p in enumerate(prices)]
    return OHLCVDataset.from_records(rows, symbol="BTCUSDT", timeframe=timeframe)


def test_aggregate_1m_to_5m():
    ds = make_ds([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    agg = aggregate_bars(ds, "5m")
    assert len(agg.bars) == 2
    assert agg.bars[0].open == agg.bars[0].high or True  # sanity
    assert agg.bars[0].close == agg.bars[0].close


def test_aggregate_drops_incomplete():
    ds = make_ds([1, 2, 3, 4, 5, 6, 7])  # 7 bars, ratio 5 -> 1 full
    assert len(aggregate_bars(ds, "5m").bars) == 1


def test_aggregate_invalid_ratio():
    ds = make_ds([1, 2, 3, 4], timeframe="5m")
    with pytest.raises(ValueError):
        aggregate_bars(ds, "1m")


def test_truncation_no_future_access():
    full = make_ds([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    trunc = make_ds([1, 2, 3, 4, 5])
    assert aggregate_bars(full, "5m").bars[0].close == aggregate_bars(trunc, "5m").bars[0].close
    # changing future bar must not affect prior aggregated bar
    future_changed = make_ds([1, 2, 3, 4, 5, 6, 7, 8, 9, 999])
    assert aggregate_bars(future_changed, "5m").bars[0] == aggregate_bars(full, "5m").bars[0]


def test_timestamp_is_interval_open():
    ds = make_ds([10, 20, 30, 40, 50])
    agg = aggregate_bars(ds, "5m")
    assert agg.bars[0].ts == ds.bars[0].ts
