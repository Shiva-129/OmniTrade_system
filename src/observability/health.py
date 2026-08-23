"""
Health Monitor (Phase 12) -- machine-readable system health.

Aggregates: system status, heartbeat, drift, gaps, WebSocket state,
reconciliation, portfolio, risk, and safety controller.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ..core.safety import SafetyController, SafetyState


class HealthMonitor:
    def __init__(self, safety: Optional[SafetyController] = None):
        self.safety = safety or SafetyController()
        self._fields: Dict[str, Any] = {}
        self._start_time = time.time()

    def set(self, key: str, value: Any) -> None:
        self._fields[key] = value

    def update_from_safety(self) -> None:
        self._fields["safety_state"] = self.safety.state.value
        if self.safety._halt_reason:
            self._fields["halt_reason"] = self.safety._halt_reason

    def snapshot(self) -> Dict[str, Any]:
        snap: Dict[str, Any] = {
            "uptime_s": round(time.time() - self._start_time, 3),
            "status": self.safety.state.value if self.safety.state == SafetyState.HALT else "HEALTHY",
            "safety_state": self.safety.state.value,
        }
        # Map safety to health status
        if self.safety.state == SafetyState.HALT:
            snap["status"] = "HALT"
        elif self.safety.state == SafetyState.DEGRADED:
            snap["status"] = "DEGRADED"
        else:
            snap["status"] = "HEALTHY"
        snap.update(self._fields)
        # Never expose secrets
        for k in list(snap.keys()):
            if "secret" in k.lower() or "api_key" in k.lower():
                del snap[k]
        return snap

    def is_healthy(self) -> bool:
        return self.safety.state == SafetyState.HEALTHY

    def is_halted(self) -> bool:
        return self.safety.is_halted()
