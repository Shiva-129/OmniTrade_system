from typing import List, Optional

from .ema import sma


def volume_sma(volumes: List[float], period: int) -> List[Optional[float]]:
    return sma(volumes, period)


def volume_ratio(volumes: List[float], period: int) -> List[Optional[float]]:
    """volume / SMA(volume). None during warm-up or when SMA is 0."""
    ma = sma(volumes, period)
    out: List[Optional[float]] = [None] * len(volumes)
    for i, (v, m) in enumerate(zip(volumes, ma)):
        if m is not None and m != 0:
            out[i] = v / m
    return out
