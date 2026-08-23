"""
Reproducible experiments (Phase 8).

An ExperimentConfig pins EVERYTHING that influences an outcome:
strategy identity+version+params, dataset CONTENT hash, date range,
cost assumptions, capital, software tag. config_hash = sha256(canonical
json). Two identical configs + same dataset => byte-identical results.

SPLIT DISCIPLINE: run_experiment computes train/val/test metrics from
their own slices; parameter selection code must consume ONLY the
train/val entries (test is reported, never fed back).
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict

from pydantic import BaseModel

from src.core.money import init_money_context

from ..data.dataset import OHLCVDataset
from ..data.validate import validate_dataset
from .benchmark import run_buy_and_hold
from .costs import CostModel
from .engine import run_backtest
from .metrics import Metrics, compute_metrics
from .split import train_val_test


INDICATOR_VERSIONS = {
    "ema": "1.0", "sma": "1.0", "rsi": "1.0", "macd": "1.0",
    "atr": "1.0", "bollinger": "1.0", "adx": "1.0",
    "volume": "1.0", "volatility": "1.0",
}


class ExperimentConfig(BaseModel):
    model_config = {"frozen": True}

    strategy_name: str
    strategy_version: str
    parameters: Dict[str, str]           # canonical string params only
    symbol: str
    timeframe: str
    dataset_hash: str
    start_ts: int
    end_ts: int
    maker_fee: str = "0"
    taker_fee: str = "0"
    slippage_pct: str = "0"
    initial_capital: str = "10000"
    seed: int | None = None              # strategies are deterministic; kept for audit
    software_version: str = "phase14"
    strategy_config_hash: str = ""       # hash of full strategy config for reproducibility
    indicator_versions: Dict[str, str] = {}

    def canonical(self) -> str:
        payload = self.model_dump(mode="json")
        return json.dumps(payload, sort_keys=True)

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()[:16]

    def cost_model(self) -> CostModel:
        from src.core.money import to_decimal
        return CostModel(
            maker_fee=to_decimal(self.maker_fee),
            taker_fee=to_decimal(self.taker_fee),
            slippage_pct=to_decimal(self.slippage_pct),
        )


def build_config(*, strategy_config, dataset: OHLCVDataset,
                 taker_fee: str = "0", maker_fee: str = "0",
                 slippage_pct: str = "0",
                 initial_capital: str = "10000") -> ExperimentConfig:
    """Derives an ExperimentConfig from a live strategy-config instance."""
    raw_params = {
        k: v for k, v in strategy_config.model_dump(mode="json").items()
        if k not in ("strategy_name", "strategy_version",
                     "symbol", "timeframe")
    }
    # strategy_config_hash captures full config for reproducibility
    cfg_payload = json.dumps(strategy_config.model_dump(mode="json"), sort_keys=True)
    cfg_hash = hashlib.sha256(cfg_payload.encode("utf-8")).hexdigest()[:16]
    return ExperimentConfig(
        strategy_name=strategy_config.strategy_name,
        strategy_version=strategy_config.strategy_version,
        parameters={k: str(v) for k, v in raw_params.items()},   # canonical strings
        symbol=strategy_config.symbol,
        timeframe=strategy_config.timeframe,
        dataset_hash=dataset.content_hash(),
        start_ts=dataset.start_ts,
        end_ts=dataset.end_ts,
        taker_fee=taker_fee, maker_fee=maker_fee, slippage_pct=slippage_pct,
        initial_capital=initial_capital,
        strategy_config_hash=cfg_hash,
        indicator_versions=dict(INDICATOR_VERSIONS),
    )


def _slice_metrics(result, tf_min: int, capital: float) -> Dict[str, Any]:
    m: Metrics = compute_metrics(
        result.equity_curve, result.trades,
        initial_capital=capital, timeframe_minutes=tf_min,
        fees_paid=float(result.fees_paid),
        slippage_cost=float(result.slippage_cost),
        turnover_notional=float(result.turnover_notional),
    )
    return {
        "metrics": m.model_dump(mode="json"),
        "execution": result.summary(),
        "equity_curve": result.equity_curve,
        "trades": result.trades,
        "fills": result.fills,
    }


def run_experiment(config: ExperimentConfig, dataset: OHLCVDataset,
                   strategy_factory: Callable[[], Any],
                   include_benchmark: bool = True) -> Dict[str, Any]:
    """
    strategy_factory: zero-arg callable returning a FRESH strategy per
    split (no state bleed between splits -- enforced by construction).
    """
    init_money_context()
    if dataset.content_hash() != config.dataset_hash:
        raise ValueError("dataset hash mismatch vs experiment config")

    tf_min = {"1m": 1}.get(config.timeframe)
    from ..data.dataset import timeframe_minutes as tfm
    tf_min = tfm(config.timeframe)

    t_sl, v_sl, te_sl = train_val_test(len(dataset))
    out: Dict[str, Any] = {
        "config": config.canonical(),
        "config_hash": config.config_hash,
        "dataset_quality": validate_dataset(dataset).summary(),
        "splits": {"train": [t_sl.start, t_sl.stop],
                   "validation": [v_sl.start, v_sl.stop],
                   "test": [te_sl.start, te_sl.stop]},
    }

    for name, sl in (("train", t_sl), ("validation", v_sl), ("test", te_sl)):
        sub = dataset.slice_indices(sl.start, sl.stop)
        bt = run_backtest(strategy_factory(), sub, config.cost_model(),
                          config.initial_capital)
        out[name] = _slice_metrics(bt, tf_min, float(config.initial_capital))

    if include_benchmark:
        curve, fees = run_buy_and_hold(dataset.slice_indices(te_sl.start, te_sl.stop),
                                       config.cost_model(), config.initial_capital)
        bm = compute_metrics(curve, [],
                             initial_capital=float(config.initial_capital),
                             timeframe_minutes=tf_min, fees_paid=fees,
                             slippage_cost=0.0, turnover_notional=0.0)
        out["benchmark_test"] = {"metrics": bm.model_dump(mode="json"),
                                 "equity_curve": curve}

    return out


def save_results(results: Dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, sort_keys=True), encoding="utf-8")
    return path
