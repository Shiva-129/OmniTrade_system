import math

import pytest

from src.indicators import (
    sma, ema, rsi, macd, atr, bollinger, adx,
    volume_sma, volume_ratio, rolling_std,
)


def test_sma_hand_computed():
    assert sma([1, 2, 3, 4, 5], 3) == [None, None, 2.0, 3.0, 4.0]
    assert sma([10, 10, 10], 3) == [None, None, 10.0]


def test_sma_period_one():
    assert sma([5, 6, 7], 1) == [5.0, 6.0, 7.0]


def test_sma_invalid():
    with pytest.raises(ValueError):
        sma([1, 2], 0)


def test_ema_hand_computed():
    # period 3: seed SMA 2.0, then 3.0
    assert ema([1, 2, 3, 4], 3) == [None, None, 2.0, 3.0]
    # period 2
    assert ema([10, 10, 10, 20], 2) == [None, 10.0, 10.0, pytest.approx(16.666666, rel=1e-5)]


def test_ema_warmup_none():
    assert ema([1, 2], 3) == [None, None]


def test_rsi_hand_computed_period2():
    # values: 1,2,1,2  period 2
    # gains: [0,1,0,1] losses [0,0,1,0]
    # i=2 avg_gain 0.5 avg_loss 0.5 RS1 RSI 50
    # i=3 avg_gain 0.75 avg_loss 0.25 RS3 RSI 75
    out = rsi([1, 2, 1, 2], 2)
    assert out[0] is None and out[1] is None
    assert out[2] == pytest.approx(50.0)
    assert out[3] == pytest.approx(75.0)


def test_rsi_zero_std_flat():
    out = rsi([100, 100, 100, 100, 100], 3)
    # first valid at 3 is 50 (no gain/loss)
    assert out[3] == pytest.approx(50.0)


def test_macd_small_periods():
    values = [1, 2, 3, 4, 5, 6, 7, 8]
    m, sig, hist = macd(values, fast=2, slow=3, signal=2)
    # fast seeded at 1, slow at 2, first macd at 2
    assert m[2] is not None
    assert sig[3] is not None  # signal needs 2 macd values
    assert hist[3] is not None


def test_macd_invalid_fast_slow():
    with pytest.raises(ValueError):
        macd([1, 2, 3], fast=3, slow=2)


def test_atr_hand_computed():
    bars = [
        {"high": 10, "low": 8, "close": 9},
        {"high": 11, "low": 9, "close": 10},
        {"high": 12, "low": 10, "close": 11},
    ]
    # TR: 2,2,2  ATR period2: [None,2.0,2.0]
    assert atr(bars, 2) == [None, 2.0, 2.0]


def test_bollinger_hand_computed():
    m, u, l = bollinger([1, 2, 3, 4, 5], 3, num_std=1.0)
    # i=2 window [1,2,3] mean2 std ~0.816
    assert m[2] == pytest.approx(2.0)
    assert u[2] == pytest.approx(2.0 + 0.8164965809, rel=1e-6)
    assert l[2] == pytest.approx(2.0 - 0.8164965809, rel=1e-6)


def test_adx_warmup_and_value():
    bars = [
        {"high": 10, "low": 8, "close": 9},
        {"high": 11, "low": 9, "close": 10},
        {"high": 12, "low": 10, "close": 11},
        {"high": 13, "low": 11, "close": 12},
        {"high": 14, "low": 12, "close": 13},
    ]
    out = adx(bars, 2)
    # warmup 3 None, first ADX at 3
    assert out[:3] == [None, None, None]
    assert out[3] is not None


def test_volume_ratio():
    vols = [10, 10, 20, 10]
    assert volume_sma(vols, 2) == [None, 10.0, 15.0, 15.0]
    assert volume_ratio(vols, 2)[2] == pytest.approx(20 / 15)


def test_rolling_std_hand_computed():
    out = rolling_std([1, 2, 3], 3)
    # std of [1,2,3] population
    assert out[2] == pytest.approx(math.sqrt(2 / 3), rel=1e-6)


def test_volume_invalid():
    with pytest.raises(ValueError):
        volume_sma([1, 2], 0)


def test_truncation_no_look_ahead():
    full = [1, 2, 3, 4, 5, 100, 101]
    trunc = [1, 2, 3, 4, 5]
    for fn in [
        lambda v: sma(v, 3),
        lambda v: ema(v, 3),
        lambda v: rsi(v, 3),
    ]:
        assert fn(full)[: len(trunc)] == fn(trunc)
    # bar-based
    bars_full = [{"high": x + 1, "low": x - 1, "close": x} for x in full]
    bars_trunc = [{"high": x + 1, "low": x - 1, "close": x} for x in trunc]
    assert atr(bars_full, 3)[: len(trunc)] == atr(bars_trunc, 3)
    m_full, _, _ = bollinger(full, 3)
    m_trunc, _, _ = bollinger(trunc, 3)
    assert m_full[: len(trunc)] == m_trunc


def test_empty_inputs():
    assert sma([], 3) == []
    assert ema([], 3) == []
    assert rsi([], 3) == []
    assert atr([], 3) == []
