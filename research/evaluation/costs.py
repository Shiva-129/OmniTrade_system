"""Backward-compat shim: CostModel now lives in core so paper trading,
research and future real brokers share one fee/slippage policy."""
from src.core.costs import CostModel  # noqa: F401

__all__ = ["CostModel"]
