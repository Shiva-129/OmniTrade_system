from typing import List, Optional, Tuple

from .ema import ema


def macd(
    values: List[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    if fast < 1 or slow < 1 or signal < 1:
        raise ValueError("periods must be >=1")
    if fast >= slow:
        raise ValueError("fast must be < slow")
    n = len(values)
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    macd_line: List[Optional[float]] = [None] * n
    for i in range(n):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line[i] = ema_fast[i] - ema_slow[i]  # type: ignore
    # signal is EMA of macd_line (skip Nones)
    # Build compact macd values for EMA, then map back
    signal_line: List[Optional[float]] = [None] * n
    hist: List[Optional[float]] = [None] * n
    # collect non-None macd values in order
    # For EMA warmup we need signal period of macd values, not bars
    # So signal warmup starts at first valid macd
    first_valid = next((i for i, v in enumerate(macd_line) if v is not None), None)
    if first_valid is None:
        return macd_line, signal_line, hist
    # Extract sequence of macd values from first_valid onward
    seq = [v for v in macd_line[first_valid:] if v is not None]
    # EMA of seq
    from .ema import ema as ema_fn

    sig_seq = ema_fn(seq, signal)  # type: ignore
    # Map back: sig_seq[signal-1] corresponds to macd index first_valid+signal-1
    for j, val in enumerate(sig_seq):
        idx = first_valid + j
        if val is not None:
            signal_line[idx] = val
            if macd_line[idx] is not None:
                hist[idx] = macd_line[idx] - val  # type: ignore
    return macd_line, signal_line, hist
