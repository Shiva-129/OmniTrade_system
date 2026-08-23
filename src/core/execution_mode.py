"""
Execution Mode -- explicit, no implicit fallback (Phase 12).

PAPER    : simulated fills only
TESTNET  : Binance testnet via BinanceTestnetBroker (max allowed)
DISABLED : no orders at all (dry-run / maintenance)

Production is intentionally impossible in this phase. Any attempt to
configure it raises ValueError.
"""
from __future__ import annotations

from enum import Enum


class ExecutionMode(str, Enum):
    PAPER = "PAPER"
    TESTNET = "TESTNET"
    DISABLED = "DISABLED"


def parse_execution_mode(value: str) -> ExecutionMode:
    normalized = value.strip().upper()
    if normalized == "PAPER":
        return ExecutionMode.PAPER
    if normalized == "TESTNET":
        return ExecutionMode.TESTNET
    if normalized == "DISABLED":
        return ExecutionMode.DISABLED
    if normalized in ("PROD", "PRODUCTION", "LIVE", "REAL"):
        raise ValueError(
            f"ExecutionMode {value!r} is not allowed in Phase 12 -- "
            "production trading is not implemented. Use PAPER, TESTNET, or DISABLED."
        )
    raise ValueError(f"Unknown ExecutionMode {value!r}. Expected PAPER, TESTNET, or DISABLED.")


def is_production_mode(value: str) -> bool:
    return value.strip().upper() in ("PROD", "PRODUCTION", "LIVE", "REAL", "BINANCE", "BINANCE_LIVE")
