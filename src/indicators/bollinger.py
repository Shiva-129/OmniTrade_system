import math
from typing import List, Optional, Tuple


def bollinger(
    values: List[float], period: int, num_std: float = 2.0
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    if period < 1:
        raise ValueError("period must be >=1")
    n = len(values)
    middle: List[Optional[float]] = [None] * n
    upper: List[Optional[float]] = [None] * n
    lower: List[Optional[float]] = [None] * n
    if n < period:
        return middle, upper, lower
    window_sum = 0.0
    window_sq_sum = 0.0
    for i, v in enumerate(values):
        window_sum += v
        window_sq_sum += v * v
        if i >= period:
            ov = values[i - period]
            window_sum -= ov
            window_sq_sum -= ov * ov
        if i >= period - 1:
            m = window_sum / period
            # population std
            var = (window_sq_sum / period) - (m * m)
            var = max(0.0, var)
            std = math.sqrt(var)
            middle[i] = m
            upper[i] = m + num_std * std
            lower[i] = m - num_std * std
    return middle, upper, lower
