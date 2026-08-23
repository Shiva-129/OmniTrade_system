from typing import List, Optional


def atr(bars, period: int) -> List[Optional[float]]:
    """
    bars: iterable of objects with .high/.low/.close as float-compatible,
          or dicts with keys 'high','low','close'.
    """
    if period < 1:
        raise ValueError("period must be >=1")
    n = len(bars)
    out: List[Optional[float]] = [None] * n
    if n == 0:
        return out

    def get(b, k):
        if isinstance(b, dict):
            return float(b[k])
        return float(getattr(b, k))

    tr = [0.0] * n
    for i in range(n):
        high = get(bars[i], "high")
        low = get(bars[i], "low")
        if i == 0:
            tr[i] = high - low
        else:
            prev_close = get(bars[i - 1], "close")
            tr[i] = max(high - low, abs(high - prev_close), abs(low - prev_close))
    if n < period:
        return out
    # first ATR is SMA of TR
    out[period - 1] = sum(tr[:period]) / period
    for i in range(period, n):
        assert out[i - 1] is not None
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period  # type: ignore
    return out
