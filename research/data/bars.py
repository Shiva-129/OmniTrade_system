"""Deterministic bar aggregation (Phase 14 P2).

Aggregates 1m-equivalent bars into higher timeframes without future
access. Every output bar's timestamp is the open of its interval.
"""
from typing import List

from .dataset import Bar, OHLCVDataset


def aggregate_bars(dataset: OHLCVDataset, target_timeframe: str) -> OHLCVDataset:
    """
    Deterministically aggregates a sorted OHLCV dataset into a coarser
    timeframe. Input must be sorted (enforced by OHLCVDataset).
    No bar ever reads a future bar.
    """
    from .dataset import timeframe_minutes

    src_min = timeframe_minutes(dataset.timeframe)
    dst_min = timeframe_minutes(target_timeframe)
    if dst_min % src_min != 0:
        raise ValueError(f"{target_timeframe} must be multiple of {dataset.timeframe}")
    ratio = dst_min // src_min
    if ratio == 1:
        return dataset

    out: List[Bar] = []
    bars = dataset.bars
    for i in range(0, len(bars), ratio):
        chunk = bars[i: i + ratio]
        if len(chunk) < ratio:
            break  # incomplete trailing bar is dropped deterministically
        # timestamp is open of first bar in chunk
        ts = chunk[0].ts
        open_ = chunk[0].open
        close = chunk[-1].close
        high = max(b.high for b in chunk)
        low = min(b.low for b in chunk)
        volume = sum((b.volume for b in chunk), start=chunk[0].volume * 0)  # Decimal sum
        out.append(Bar(ts=ts, open=open_, high=high, low=low, close=close, volume=volume))
    return OHLCVDataset(bars=out, symbol=dataset.symbol, timeframe=target_timeframe)
