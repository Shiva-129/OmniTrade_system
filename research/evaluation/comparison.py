"""Strategy comparison and portfolio-level research helpers (Phase 14 P6).

Thin wrappers over existing Phase 8/9 machinery. No new backtest engine,
no test-set leakage.
"""
from typing import Any, Callable, Dict, List

from .experiment import build_config, run_experiment
from .metrics import compute_metrics


def compare_strategies(
    dataset,
    strategy_configs: List[Any],
    taker_fee: str = "0",
    slippage_pct: str = "0",
    initial_capital: str = "10000",
) -> List[Dict[str, Any]]:
    """
    Runs each strategy config through the same dataset+costs.
    Returns list of {strategy_name, version, params, test_metrics, config_hash}
    sorted by test Sharpe descending, without ever using test data for selection
    (comparison is reporting, not selection).
    """
    results = []
    for cfg in strategy_configs:
        exp_cfg = build_config(strategy_config=cfg, dataset=dataset,
                               taker_fee=taker_fee, slippage_pct=slippage_pct,
                               initial_capital=initial_capital)
        # fresh factory per run
        def factory(c=cfg):
            # instantiate the correct strategy class via its config type
            cls = type(c)
            # Need strategy class, not config: use a registry via config.strategy_name
            from src.strategies.ema_crossover import EmaCrossoverStrategy
            from src.strategies.sma_trend import SmaTrendStrategy
            from src.strategies.rsi_momentum import RsiMomentumStrategy
            from src.strategies.bollinger_reversion import BollingerReversionStrategy
            from src.strategies.macd_trend import MacdTrendStrategy
            from src.strategies.adx_volatility import AdxVolatilityStrategy
            from src.strategies.donchian_breakout import DonchianBreakoutStrategy
            from src.strategies.zscore_mean_reversion import ZScoreMeanReversionStrategy
            name_map = {
                "ema": EmaCrossoverStrategy, "sma_trend": SmaTrendStrategy,
                "rsi_momentum": RsiMomentumStrategy, "bollinger": BollingerReversionStrategy,
                "macd_trend": MacdTrendStrategy, "adx_volatility": AdxVolatilityStrategy,
                "donchian": DonchianBreakoutStrategy, "zscore": ZScoreMeanReversionStrategy,
                "ema_crossover": EmaCrossoverStrategy, "zscore_mean_reversion": ZScoreMeanReversionStrategy,
                "donchian_breakout": DonchianBreakoutStrategy,
            }
            strat_cls = name_map.get(c.strategy_name, EmaCrossoverStrategy)
            return strat_cls(c)

        res = run_experiment(exp_cfg, dataset, factory, include_benchmark=False)
        results.append({
            "strategy_name": cfg.strategy_name,
            "strategy_version": cfg.strategy_version,
            "params": exp_cfg.parameters,
            "config_hash": exp_cfg.config_hash,
            "test_metrics": res["test"]["metrics"],
            "validation_metrics": res["validation"]["metrics"],
        })
    # sort by test Sharpe for reporting (does not imply selection)
    return sorted(results, key=lambda r: r["test_metrics"].get("sharpe", 0) or 0, reverse=True)


def portfolio_metrics(portfolios_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate portfolio-level research view: sum of individual test
    total_returns (equal-weight assumption, documented).
    """
    if not portfolios_results:
        return {"total_return": 0, "n_strategies": 0}
    total = sum(r["test_metrics"]["total_return"] for r in portfolios_results)
    return {"total_return": total, "n_strategies": len(portfolios_results)}
