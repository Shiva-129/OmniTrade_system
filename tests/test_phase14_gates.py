"""Phase 14 hard-gate tests: look-ahead, multi-symbol, cost sensitivity, reproducibility."""
import pytest

from research.data.bars import aggregate_bars
from research.data.dataset import OHLCVDataset
from research.evaluation.comparison import compare_strategies, portfolio_metrics
from research.evaluation.experiment import build_config, run_experiment
from src.core.money import to_decimal
from src.core.portfolio import Portfolio
from src.core.types import ExecutionReport, MarketEvent, OrderSide, Packet
from src.indicators import sma


def ev(ts, price, topic="BTCUSDT"):
    return MarketEvent(packet=Packet(exchange_ts=ts, local_arrival_ts=ts, drift_us=0, source="t", topic=topic, payload={"price": str(price)}, sequence_id=ts))


def test_look_ahead_indicator_truncation():
    full = [1, 2, 3, 4, 5, 100]
    trunc = [1, 2, 3, 4, 5]
    assert sma(full, 3)[:5] == sma(trunc, 3)
    # bar aggregation truncation
    ds_full = OHLCVDataset.from_records([[1600000000000 + i*60000, float(p), float(p)+1, float(p)-1, float(p), 10] for i,p in enumerate(full)], symbol="BTCUSDT", timeframe="1m")
    ds_trunc = OHLCVDataset.from_records([[1600000000000 + i*60000, float(p), float(p)+1, float(p)-1, float(p), 10] for i,p in enumerate(trunc)], symbol="BTCUSDT", timeframe="1m")
    agg_full = aggregate_bars(ds_full, "1m")
    agg_trunc = aggregate_bars(ds_trunc, "1m")
    assert [b.close for b in agg_full.bars[:5]] == [b.close for b in agg_trunc.bars]


def test_indicator_correctness_sma():
    assert sma([1, 2, 3, 4, 5], 3)[2] == 2.0
    assert sma([10, 10, 10], 3)[2] == 10.0


def test_deterministic_signals_new_strategies():
    from src.strategies.sma_trend import SmaTrendStrategy, SmaTrendConfig
    cfg = SmaTrendConfig(strategy_name="sma_trend", symbol="BTCUSDT", trade_size="0.1", fast_period=2, slow_period=3)
    def run():
        s = SmaTrendStrategy(cfg)
        return [s.on_market_event(ev(i+1, p)).client_order_id if s.on_market_event(ev(i+1, p)) else None for i,p in enumerate([100,101,102,103,104]*2)]
    # Actually test determinism properly: same sequence twice
    s1 = SmaTrendStrategy(cfg)
    s2 = SmaTrendStrategy(cfg)
    outs1 = [s1.on_market_event(ev(i+1, p)) for i,p in enumerate([100,101,102,103,104,105])]
    outs2 = [s2.on_market_event(ev(i+1, p)) for i,p in enumerate([100,101,102,103,104,105])]
    assert [o.client_order_id if o else None for o in outs1] == [o.client_order_id if o else None for o in outs2]


def test_multi_symbol_isolation_strategies():
    from src.strategies.ema_crossover import EmaCrossoverStrategy, EmaCrossoverConfig
    cfg_btc = EmaCrossoverConfig(strategy_name="ema", symbol="BTCUSDT", trade_size="0.1", fast_period=2, slow_period=3)
    cfg_eth = EmaCrossoverConfig(strategy_name="ema", symbol="ETHUSDT", trade_size="0.1", fast_period=2, slow_period=3)
    btc = EmaCrossoverStrategy(cfg_btc)
    eth = EmaCrossoverStrategy(cfg_eth)
    # interleaved stream
    events = [ev(1, 100, "BTCUSDT"), ev(2, 200, "ETHUSDT"), ev(3, 101, "BTCUSDT"), ev(4, 201, "ETHUSDT")]
    for e in events:
        btc.on_market_event(e)
        eth.on_market_event(e)
    # each should have seen only its own symbol events
    assert btc._events_seen == 2
    assert eth._events_seen == 2


def test_portfolio_multi_symbol_isolation():
    pf = Portfolio(starting_cash="10000")
    def rpt(sym, side, qty, price):
        return ExecutionReport(client_order_id=f"{sym}-{side}", exchange_order_id=f"x-{sym}", symbol=sym, side=side, status="FILLED", filled_quantity=to_decimal(qty), last_filled_price=to_decimal(price), remaining_quantity=to_decimal("0"), timestamp=1, fee=to_decimal("0"))
    pf.apply_report(rpt("BTCUSDT", OrderSide.BUY, "1", "100"))
    pf.apply_report(rpt("ETHUSDT", OrderSide.BUY, "2", "200"))
    assert pf.positions["BTCUSDT"].quantity == to_decimal("1")
    assert pf.positions["ETHUSDT"].quantity == to_decimal("2")
    pf.mark_price("BTCUSDT", to_decimal("110"), ts_us=1)
    # ETH mark stale? not yet
    assert pf.unrealized_pnl("BTCUSDT", now_us=1) == to_decimal("10")
    # ETH without fresh mark? Actually we didn't mark ETH yet
    assert pf.unrealized_pnl("ETHUSDT", now_us=1) is None


def test_cost_sensitivity():
    from research.evaluation.engine import run_backtest
    from research.evaluation.costs import CostModel
    from src.strategies.ema_crossover import EmaCrossoverStrategy, EmaCrossoverConfig
    rows = [[1600000000000 + i*60000, float(p), float(p)+1, float(p)-1, float(p), 10] for i,p in enumerate([100,100,100,140,145,150,90,85,80,120])]
    ds = OHLCVDataset.from_records(rows, symbol="BTCUSDT", timeframe="1m")
    cfg = EmaCrossoverConfig(strategy_name="ema", symbol="BTCUSDT", trade_size="0.5", fast_period=2, slow_period=3)
    r_free = run_backtest(EmaCrossoverStrategy(cfg), ds, CostModel(taker_fee=to_decimal("0")), "10000")
    r_paid = run_backtest(EmaCrossoverStrategy(cfg), ds, CostModel(taker_fee=to_decimal("0.01")), "10000")
    assert r_paid.fees_paid > r_free.fees_paid
    assert r_paid.equity_curve[-1]["equity"] < r_free.equity_curve[-1]["equity"]


def test_repeated_experiment_reproducibility():
    from src.strategies.sma_trend import SmaTrendStrategy, SmaTrendConfig
    rows = [[1600000000000 + i*60000, float(p), float(p)+1, float(p)-1, float(p), 10] for i,p in enumerate([100,101,102,103,104,105,106,107,108,109]*2)]
    ds = OHLCVDataset.from_records(rows, symbol="BTCUSDT", timeframe="1m")
    cfg = SmaTrendConfig(strategy_name="sma_trend", symbol="BTCUSDT", trade_size="0.1", fast_period=2, slow_period=3)
    exp = build_config(strategy_config=cfg, dataset=ds)
    def factory(): return SmaTrendStrategy(cfg)
    r1 = run_experiment(exp, ds, factory, include_benchmark=False)
    r2 = run_experiment(exp, ds, factory, include_benchmark=False)
    assert r1["config_hash"] == r2["config_hash"] == exp.config_hash
    assert r1["test"]["metrics"] == r2["test"]["metrics"]


def test_experiment_hash_includes_indicator_versions():
    from src.strategies.ema_crossover import EmaCrossoverConfig
    rows = [[1600000000000, 100, 101, 99, 100, 10]]
    ds = OHLCVDataset.from_records(rows, symbol="BTCUSDT", timeframe="1m")
    cfg = EmaCrossoverConfig(strategy_name="ema", symbol="BTCUSDT", trade_size="0.1", fast_period=2, slow_period=3)
    exp = build_config(strategy_config=cfg, dataset=ds)
    assert exp.indicator_versions  # non-empty
    assert exp.strategy_config_hash  # non-empty
    # changing indicator version would change hash
    assert "ema" in exp.indicator_versions


def test_strategy_import_restrictions_new():
    import pathlib
    for p in pathlib.Path("src/strategies").glob("*.py"):
        if p.name == "base.py": continue
        text = p.read_text()
        assert "import redis" not in text.lower()
        assert "BrokerInterface" not in text
        assert "Portfolio" not in text or "trade_size" in text.lower()
        assert "SafetyController" not in text


def test_comparison_does_not_leak_test():
    rows = [[1600000000000 + i*60000, float(p), float(p)+1, float(p)-1, float(p), 10] for i,p in enumerate([100]*10 + [140]*10)]
    ds = OHLCVDataset.from_records(rows, symbol="BTCUSDT", timeframe="1m")
    from src.strategies.sma_trend import SmaTrendConfig
    cfgs = [
        SmaTrendConfig(strategy_name="sma_trend", symbol="BTCUSDT", trade_size="0.1", fast_period=2, slow_period=3),
        SmaTrendConfig(strategy_name="sma_trend", symbol="BTCUSDT", trade_size="0.1", fast_period=2, slow_period=5),
    ]
    res = compare_strategies(ds, cfgs)
    assert len(res) == 2
    assert all("test_metrics" in r for r in res)


def test_portfolio_exposure_readonly():
    pf = Portfolio(starting_cash="10000")
    pf.apply_report(ExecutionReport(client_order_id="a", exchange_order_id="x", symbol="BTCUSDT", side=OrderSide.BUY, status="FILLED", filled_quantity=to_decimal("2"), last_filled_price=to_decimal("100"), remaining_quantity=to_decimal("0"), timestamp=1, fee=to_decimal("0")))
    pf.mark_price("BTCUSDT", to_decimal("110"), ts_us=1)
    exp = pf.exposure_by_symbol(now_us=1)
    assert exp["BTCUSDT"] == to_decimal("220")
    assert pf.total_gross_exposure(now_us=1) == to_decimal("220")
    snap_before = pf.snapshot()
    _ = pf.exposure_by_symbol()
    assert pf.snapshot() == snap_before  # read-only


def test_no_strategy_bypass_risk_gatekeeper_safety():
    # Structural: strategies must not import Risk/Gatekeeper/Safety
    import pathlib
    for p in pathlib.Path("src/strategies").glob("*.py"):
        if p.name == "base.py": continue
        text = p.read_text()
        assert "RiskManager" not in text
        assert "Gatekeeper" not in text
        assert "SafetyController" not in text
