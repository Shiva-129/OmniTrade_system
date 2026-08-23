from typing import List, Optional


def rsi(values: List[float], period: int) -> List[Optional[float]]:
    if period < 1:
        raise ValueError("period must be >=1")
    n = len(values)
    out: List[Optional[float]] = [None] * n
    if n <= period:
        return out
    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        delta = values[i] - values[i - 1]
        if delta > 0:
            gains[i] = delta
        else:
            losses[i] = -delta
    # initial averages at i == period
    avg_gain = sum(gains[1: period + 1]) / period
    avg_loss = sum(losses[1: period + 1]) / period
    if avg_loss == 0:
        out[period] = 100.0 if avg_gain != 0 else 50.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100.0 - (100.0 / (1.0 + rs))
    # Wilder smoothing
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i] = 100.0 if avg_gain != 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out
