import math
from typing import List, Optional


def rolling_std(values: List[float], period: int) -> List[Optional[float]]:
    if period < 1:
        raise ValueError("period must be >=1")
    n = len(values)
    out: List[Optional[float]] = [None] * n
    if n < period:
        return out
    window_sum = 0.0
    window_sq = 0.0
    for i, v in enumerate(values):
        window_sum += v
        window_sq += v * v
        if i >= period:
            ov = values[i - period]
            window_sum -= ov
            window_sq -= ov * ov
        if i >= period - 1:
            m = window_sum / period
            var = (window_sq / period) - (m * m)
            out[i] = math.sqrt(max(0.0, var))
    return out


def realized_volatility(values: List[float], period: int) -> List[Optional[float]]:
    """Annualized-ish volatility of log returns over period."""
    if period < 1:
        raise ValueError("period must be >=1")
    n = len(values)
    out: List[Optional[float]] = [None] * n
    if n < period + 1:
        return out
    log_rets = [0.0] * n
    for i in range(1, n):
        if values[i - 1] > 0 and values[i] > 0:
            log_rets[i] = math.log(values[i] / values[i - 1])
    # rolling std of log returns
    std = rolling_std(log_rets, period)
    for i in range(n):
        if std[i] is not None:
            out[i] = std[i] * math.sqrt(252)  # trading days annualization
    return out
