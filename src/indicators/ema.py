from typing import List, Optional


def sma(values: List[float], period: int) -> List[Optional[float]]:
    if period < 1:
        raise ValueError("period must be >=1")
    out: List[Optional[float]] = [None] * len(values)
    if not values:
        return out
    window_sum = 0.0
    for i, v in enumerate(values):
        window_sum += v
        if i >= period:
            window_sum -= values[i - period]
        if i >= period - 1:
            out[i] = window_sum / period
    return out


def ema(values: List[float], period: int) -> List[Optional[float]]:
    if period < 1:
        raise ValueError("period must be >=1")
    out: List[Optional[float]] = [None] * len(values)
    if not values:
        return out
    alpha = 2.0 / (period + 1.0)
    # seed with SMA at period-1
    s = sma(values, period)
    ema_prev: Optional[float] = None
    for i, v in enumerate(values):
        if i < period - 1:
            continue
        if i == period - 1:
            ema_prev = s[i]
            out[i] = ema_prev
        else:
            assert ema_prev is not None
            ema_prev = alpha * v + (1.0 - alpha) * ema_prev
            out[i] = ema_prev
    return out
