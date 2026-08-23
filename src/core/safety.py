"""
Central Safety Mechanism (Phase 12) -- one HALT path.

All failure domains converge here: reconciliation mismatch, unknown
execution, stale stream, journal/Redis/clock failures, etc.

State machine: HEALTHY -> DEGRADED -> HALT (terminal, no auto-recovery).
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from ..core.logger import get_logger

logger = get_logger("SafetyController")


class SafetyState(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    HALT = "HALT"


class SafetyController:
    """
    Single authority for safety transitions. Records reasons and
    prevents ad-hoc HALT implementations elsewhere.
    """

    def __init__(self):
        self.state = SafetyState.HEALTHY
        self.reasons: List[str] = []
        self._halt_reason: Optional[str] = None

    def degrade(self, reason: str) -> None:
        if self.state == SafetyState.HALT:
            return  # terminal
        self.state = SafetyState.DEGRADED
        self.reasons.append(f"DEGRADED: {reason}")
        logger.warning("safety_degraded", reason=reason)

    def halt(self, reason: str) -> None:
        if self.state == SafetyState.HALT:
            return
        self.state = SafetyState.HALT
        self._halt_reason = reason
        self.reasons.append(f"HALT: {reason}")
        logger.error("safety_halt", reason=reason)

    def is_halted(self) -> bool:
        return self.state == SafetyState.HALT

    def is_degraded(self) -> bool:
        return self.state == SafetyState.DEGRADED

    def can_submit_new_position(self) -> bool:
        return self.state == SafetyState.HEALTHY

    def can_submit_reducing(self) -> bool:
        return self.state in (SafetyState.HEALTHY, SafetyState.DEGRADED)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "halt_reason": self._halt_reason,
            "reasons": list(self.reasons),
        }
