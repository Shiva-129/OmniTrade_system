from typing import List, Optional


def adx(bars, period: int) -> List[Optional[float]]:
    if period < 1:
        raise ValueError("period must be >=1")
    n = len(bars)
    out: List[Optional[float]] = [None] * n
    if n <= period:
        return out

    def get(b, k):
        if isinstance(b, dict):
            return float(b[k])
        return float(getattr(b, k))

    tr = [0.0] * n
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    for i in range(n):
        high = get(bars[i], "high")
        low = get(bars[i], "low")
        if i == 0:
            tr[i] = high - low
        else:
            prev_high = get(bars[i - 1], "high")
            prev_low = get(bars[i - 1], "low")
            prev_close = get(bars[i - 1], "close")
            tr[i] = max(high - low, abs(high - prev_close), abs(low - prev_close))
            up_move = high - prev_high
            down_move = prev_low - low
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
    # Wilder smoothing for TR, +DM, -DM
    # first smoothed at period
    if n <= period:
        return out
    sm_tr = sum(tr[1: period + 1])  # Wilder starts at 1
    sm_plus = sum(plus_dm[1: period + 1])
    sm_minus = sum(minus_dm[1: period + 1])
    # Need DX series
    dx = [None] * n
    # first DX at period
    if sm_tr != 0:
        plus_di = 100.0 * sm_plus / sm_tr
        minus_di = 100.0 * sm_minus / sm_tr
        denom = plus_di + minus_di
        dx[period] = 100.0 * abs(plus_di - minus_di) / denom if denom != 0 else 0.0
    # subsequent
    for i in range(period + 1, n):
        sm_tr = sm_tr - sm_tr / period + tr[i]
        sm_plus = sm_plus - sm_plus / period + plus_dm[i]
        sm_minus = sm_minus - sm_minus / period + minus_dm[i]
        if sm_tr != 0:
            plus_di = 100.0 * sm_plus / sm_tr
            minus_di = 100.0 * sm_minus / sm_tr
            denom = plus_di + minus_di
            dx[i] = 100.0 * abs(plus_di - minus_di) / denom if denom != 0 else 0.0
    # ADX is Wilder SMA of DX: first ADX at 2*period-1 is SMA of DX[period:2*period]
    if n < 2 * period:
        return out
    # SMA of first period DX values
    valid_dx = [v for v in dx[period: 2 * period] if v is not None]
    if len(valid_dx) < period:
        return out
    out[2 * period - 1] = sum(valid_dx) / period
    for i in range(2 * period, n):
        if dx[i] is None or out[i - 1] is None:
            continue
        out[i] = (out[i - 1] * (period - 1) + dx[i]) / period  # type: ignore
    return out
