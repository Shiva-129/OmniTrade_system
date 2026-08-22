"""Explicit data-quality reporting. Corruption is REPORTED, never repaired."""
from typing import List, Tuple

from pydantic import BaseModel

from .dataset import OHLCVDataset, timeframe_minutes


class DataQualityReport(BaseModel):
    """Immutable quality summary; travels with experiment results."""
    n_bars: int
    ordering_violations: int          # adjacent pairs out of order (post-sort: 0)
    duplicate_timestamps: int         # repeated ts groups count
    gap_count: int                    # intervals larger than one period
    gaps: List[Tuple[int, int]]       # [(prev_ts, next_ts)] of each gap
    invalid_ohlc: int                 # high<max(o,c) / low>min(o,c) / h<l
    nonpositive_prices: int           # any of o/h/l/c <= 0
    negative_volume: int              # volume < 0 (zero volume allowed: dead bars)
    timezone_note: str = "timestamps are UTC epoch-ms by contract"

    @property
    def is_clean(self) -> bool:
        return not (
            self.ordering_violations or self.duplicate_timestamps
            or self.gap_count or self.invalid_ohlc or self.nonpositive_prices
            or self.negative_volume
        )

    def summary(self) -> dict:
        d = self.model_dump()
        d["gaps"] = [list(g) for g in self.gaps]
        d["is_clean"] = self.is_clean
        return d


def validate_dataset(ds: OHLCVDataset) -> DataQualityReport:
    tf_min = timeframe_minutes(ds.timeframe)
    expected_step_ms = tf_min * 60_000

    ordering_violations = 0
    dup_groups = 0
    gaps: List[Tuple[int, int]] = []
    invalid_ohlc = 0
    nonpositive = 0
    neg_vol = 0

    prev_ts = None
    seen_ts = set()
    dup_seen = set()

    for b in ds.bars:
        if prev_ts is not None:
            if b.ts < prev_ts:
                ordering_violations += 1
            elif b.ts - prev_ts > expected_step_ms:
                gaps.append((prev_ts, b.ts))
        if b.ts in seen_ts and b.ts not in dup_seen:
            dup_seen.add(b.ts)
        seen_ts.add(b.ts)

        if b.high < b.low or b.high < max(b.open, b.close) or b.low > min(b.open, b.close):
            invalid_ohlc += 1
        if min(b.open, b.high, b.low, b.close) <= 0:
            nonpositive += 1
        if b.volume < 0:
            neg_vol += 1
        prev_ts = b.ts

    return DataQualityReport(
        n_bars=len(ds.bars),
        ordering_violations=ordering_violations,
        duplicate_timestamps=len(dup_seen),
        gap_count=len(gaps),
        gaps=gaps,
        invalid_ohlc=invalid_ohlc,
        nonpositive_prices=nonpositive,
        negative_volume=neg_vol,
    )
