"""
Time-ordered data splits + walk-forward windows (Phase 8).

HARD RULES
----------
- Contiguous, time-ordered slices ONLY. No shuffling, ever.
- The TEST slice is produced by the split itself and must never be seen
  during parameter selection (enforced structurally: selection code only
  receives train/val slices).
"""
from typing import Iterator, Tuple


def train_val_test(n_bars: int, train_frac: float = 0.6,
                   val_frac: float = 0.2) -> Tuple[slice, slice, slice]:
    """
    Contiguous split of [0, n_bars): train | validation | test.
    Fractions floor-rounded; test absorbs the remainder (>=1 bar enforced).
    """
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("fractions must satisfy 0<train,val, train+val<1")
    t = int(n_bars * train_frac)
    v = int(n_bars * val_frac)
    if t < 1 or v < 1 or n_bars - t - v < 1:
        raise ValueError("slices too small for this n_bars")
    return slice(0, t), slice(t, t + v), slice(t + v, n_bars)


def walk_forward(n_bars: int, train_bars: int, test_bars: int,
                 step_bars: int) -> Iterator[Tuple[slice, slice]]:
    """
    Rolling windows: [(train_i, test_i)] where train immediately precedes
    its test window and windows advance by `step_bars`. Validation is the
    tail of each train window when callers need one (their choice).
    """
    if train_bars < 1 or test_bars < 1 or step_bars < 1:
        raise ValueError("window sizes must be >= 1")
    start = 0
    while start + train_bars + test_bars <= n_bars:
        yield (
            slice(start, start + train_bars),
            slice(start + train_bars, start + train_bars + test_bars),
        )
        start += step_bars
