"""
Alert Manager (Phase 12) -- deterministic alert conditions.

No notification platform; a structured event stream. Callers poll
`pending()` or subscribe via callback.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List

from ..core.logger import get_logger

logger = get_logger("AlertManager")


class AlertManager:
    """
    Evaluates named conditions and emits Alert events. Conditions are
    pure functions: (snapshot) -> bool. When a condition flips from
    False -> True, an alert is emitted.
    """

    def __init__(self):
        self._conditions: Dict[str, Callable[[Dict[str, Any]], bool]] = {}
        self._active: Dict[str, bool] = {}
        self._events: List[Dict[str, Any]] = []
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []

    def register(self, name: str, condition: Callable[[Dict[str, Any]], bool]) -> None:
        self._conditions[name] = condition
        self._active[name] = False

    def on_alert(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._callbacks.append(callback)

    def evaluate(self, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        newly: List[Dict[str, Any]] = []
        for name, cond in self._conditions.items():
            try:
                triggered = bool(cond(snapshot))
            except Exception as e:
                logger.warning("alert_condition_failed", alert=name, error=str(e))
                continue
            was_active = self._active.get(name, False)
            if triggered and not was_active:
                event = {"alert": name, "ts": int(time.time() * 1000), "snapshot": dict(snapshot)}
                self._events.append(event)
                newly.append(event)
                for cb in self._callbacks:
                    try:
                        cb(event)
                    except Exception:
                        pass
                logger.warning("alert_triggered", alert=name)
            self._active[name] = triggered
        return newly

    def pending(self) -> List[Dict[str, Any]]:
        return list(self._events)

    def clear(self) -> None:
        self._events.clear()

    @staticmethod
    def standard_conditions() -> Dict[str, Callable[[Dict[str, Any]], bool]]:
        return {
            "WS_DISCONNECTED": lambda s: s.get("ws_state") in ("DISCONNECTED", "STALE"),
            "HEARTBEAT_STALE": lambda s: s.get("heartbeat_age_s", 0) > 5,
            "CLOCK_DRIFT": lambda s: abs(s.get("clock_drift_us", 0)) > 500_000,
            "DATA_GAP": lambda s: s.get("gap_count", 0) > 0,
            "RECONCILIATION_MISMATCH": lambda s: s.get("reconciliation_state") == "MISMATCH",
            "UNKNOWN_EXECUTION": lambda s: s.get("unknown_executions", 0) > 0,
            "DUPLICATE_EXECUTION": lambda s: s.get("duplicate_executions", 0) > 0,
            "BROKER_FAILURE": lambda s: s.get("broker_failures", 0) > 0,
            "REDIS_FAILURE": lambda s: s.get("redis_failures", 0) > 0,
            "JOURNAL_FAILURE": lambda s: s.get("journal_failures", 0) > 0,
            "DRAWDOWN_THRESHOLD": lambda s: s.get("drawdown_pct", 0) > 10,
            "UNEXPECTED_POSITION": lambda s: s.get("unexpected_position", False) is True,
            "UNEXPECTED_OPEN_ORDER": lambda s: s.get("unexpected_open_order", False) is True,
        }
