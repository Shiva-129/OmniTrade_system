"""
Parameter spaces (Phase 9).

A ParameterSpace declares an ORDERED grid of canonical-string values per
axis. Enumeration is a deterministic cartesian product in declared order.

VALIDATION = STRATEGY CONTRACT: every candidate is materialized through
the REAL pydantic strategy-config class (single source of truth).
Contract-violating combos (fast>=slow, window<2, exit>=entry, ...) are
REJECTED EXPLICITLY with the validator's reason -- never silently
accepted, never silently dropped.
"""
import itertools
from typing import Any, Callable, Dict, List, Tuple

from pydantic import BaseModel, ValidationError

# Real config constructors -- the contract is the validator.
from src.strategies.base import BaseStrategy
from src.strategies.donchian_breakout import DonchianBreakoutStrategy
from src.strategies.donchian_breakout import DonchianConfig
from src.strategies.ema_crossover import EmaCrossoverConfig
from src.strategies.ema_crossover import EmaCrossoverStrategy
from src.strategies.zscore_mean_reversion import ZScoreConfig
from src.strategies.zscore_mean_reversion import ZScoreMeanReversionStrategy

REQUIRED_AXES: Dict[str, Tuple[str, ...]] = {
    "ema_crossover": ("fast_period", "slow_period", "cooldown_events"),
    "zscore_mean_reversion": ("window", "entry_z", "exit_z"),
    "donchian_breakout": ("lookback", "atr_period", "atr_stop_multiplier"),
}

_CONFIG_CLS: Dict[str, type] = {
    "ema_crossover": EmaCrossoverConfig,
    "zscore_mean_reversion": ZScoreConfig,
    "donchian_breakout": DonchianConfig,
}

_STRATEGY_CLS: Dict[str, type] = {
    "ema_crossover": EmaCrossoverStrategy,
    "zscore_mean_reversion": ZScoreMeanReversionStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
}

_INT_AXES = {"fast_period", "slow_period", "cooldown_events",
             "window", "lookback", "atr_period"}
_FLOAT_AXES = {"entry_z", "exit_z", "atr_stop_multiplier"}


class BaseSpec(BaseModel):
    """Non-swept strategy identity fields shared by every candidate."""
    model_config = {"frozen": True}

    strategy_name: str
    symbol: str
    timeframe: str
    trade_size: str
    strategy_version: str = "1.0.0"


class ParameterSpace(BaseModel):
    model_config = {"frozen": True}

    strategy_name: str
    grid: Dict[str, Tuple[str, ...]]

    def iter_params(self) -> List[Dict[str, str]]:
        axes = list(self.grid.keys())
        for axis in REQUIRED_AXES[self.strategy_name]:
            if axis not in axes:
                raise ValueError(f"missing required axis {axis!r} "
                                 f"for {self.strategy_name}")
        value_lists = [list(self.grid[a]) for a in axes]
        return [dict(zip(axes, combo)) for combo in itertools.product(*value_lists)]


def build_strategy_config(space: ParameterSpace, base: BaseSpec,
                          params: Dict[str, str]):
    """
    Materializes ONE candidate through the real strategy config.
    Raises pydantic.ValidationError for contract violations.
    """
    cls = _CONFIG_CLS[space.strategy_name]
    kwargs: Dict[str, Any] = dict(base.model_dump())
    for k, v in params.items():
        if k in _INT_AXES:
            kwargs[k] = int(v)
        elif k in _FLOAT_AXES:
            kwargs[k] = float(v)
        else:
            raise ValueError(f"unknown axis {k!r} for {space.strategy_name}")
    return cls(**kwargs)


def build_strategy(space: ParameterSpace, base: BaseSpec,
                   params: Dict[str, str]) -> BaseStrategy:
    """Config + strategy instance in one contract-checked step."""
    config = build_strategy_config(space, base, params)
    return _STRATEGY_CLS[space.strategy_name](config)


def evaluate_candidates(space: ParameterSpace, base: BaseSpec
                        ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Splits the cartesian product into (valid_params, rejected) where each
    rejection carries the strategy contract's own error message.
    Deterministic order preserved.
    """
    valid, rejected = [], []
    for params in space.iter_params():
        try:
            build_strategy_config(space, base, params)
        except ValidationError as e:
            rejected.append({"params": params,
                             "reason": e.errors()[0]["msg"]})
        except ValueError as e:
            rejected.append({"params": params, "reason": str(e)})
        else:
            valid.append(params)
    return valid, rejected
