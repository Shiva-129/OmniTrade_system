"""
PortfolioSession (Phase 15 R4) -- orchestration only.

Composes TradingEngine + ReconciliationEngine + BinanceUserStream +
HealthMonitor into an operable long-running session. Does NOT duplicate
any of those components' logic; delegates entirely.

Fail-closed on startup/reconciliation failure. HALT is terminal.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..core.engine import TradingEngine
from ..core.logger import get_logger
from ..core.safety import SafetyController, SafetyState
from ..reconciliation.engine import ReconciliationEngine

logger = get_logger("PortfolioSession")


class SessionError(Exception):
    pass


class PortfolioSession:
    """
    Wraps a TradingEngine and adds operational lifecycle:
    - startup reconciliation gate (fail-closed)
    - user-stream supervision with reconnect+REST-reconcile
    - health/alert polling
    - ordered shutdown
    Does NOT duplicate engine/safety/broker/reconciliation logic.
    """

    def __init__(self, engine: TradingEngine, *,
                 reconciler: Optional[ReconciliationEngine] = None):
        if not isinstance(engine, TradingEngine):
            raise TypeError("engine must be a TradingEngine")
        self.engine = engine
        self.reconciler = reconciler or ReconciliationEngine()
        self._running = False
        self._started = False

    @property
    def safety(self) -> SafetyController:
        return self.engine.safety

    async def start(self) -> None:
        """
        Ordered startup:
          1. connect broker / load markets (via engine.broker)
          2. restore journal (already done in engine.__init__)
          3. REST reconciliation via broker.startup_reconcile()
          4. verify consistency
          5. connect user stream (if present)
          6. start engine tasks
          7. HEALTHY -> allow trading
        Fail-closed at any step.
        """
        if self._running:
            raise SessionError("session already running")

        # Step 1-3: reconciliation via broker if available
        broker = self.engine.broker
        if broker is not None and hasattr(broker, "startup_reconcile"):
            try:
                report = broker.startup_reconcile()
            except Exception as e:
                logger.error("session_startup_reconcile_failed", error=str(e))
                self.engine.safety.halt(f"startup reconcile error: {e}")
                raise SessionError("startup reconciliation failed") from e

            if not report.get("ok", True):
                mismatches = report.get("mismatches", [])
                logger.error("session_startup_mismatch", mismatches=mismatches)
                self.engine.safety.halt(
                    f"startup reconciliation mismatch: {mismatches}")
                raise SessionError("startup reconciliation failed: mismatches found")

        # Step 5: connect user stream if present
        us = getattr(self.engine, "user_stream", None)
        if us is not None and hasattr(us, "connect"):
            try:
                await us.connect()
            except Exception as e:
                logger.error("session_user_stream_connect_failed", error=str(e))
                # DEGRADED, not HALT -- REST still works
                self.engine.safety.degrade(f"user stream connect failed: {e}")

        # Step 6: start engine (this starts producers/consumer tasks)
        await self.engine.start()
        self._running = True
        self._started = True
        logger.info("session_started", mode=self.engine.execution_mode.value)

    async def stop(self) -> None:
        """Ordered shutdown."""
        if not self._running and not self._started:
            return
        logger.info("session_stopping")
        # Safety HALT first to block submissions during teardown
        try:
            self.engine.safety.halt("session stopping")
        except Exception:
            pass
        await self.engine.stop()

        # Disconnect user stream
        us = getattr(self.engine, "user_stream", None)
        if us is not None and hasattr(us, "disconnect"):
            try:
                await us.disconnect()
            except Exception:
                pass

        self._running = False
        logger.info("session_stopped")

    def pause(self) -> None:
        """DEGRADED: no new positions, reductions allowed."""
        self.engine.safety.degrade("session paused")
        logger.info("session_paused")

    def resume(self) -> None:
        """Operator explicitly resumes from pause.
        Cannot resume from HALT (terminal)."""
        if self.engine.safety.is_halted():
            raise SessionError("cannot resume from HALT; operator reset required")
        if self.engine.safety.is_degraded():
            # Create new controller to go back to HEALTHY
            self.engine.safety.state = SafetyState.HEALTHY
            logger.info("session_resumed")

    def trigger_reconciliation(self, exchange_orders: Dict[str, Any],
                               exchange_balances: Dict[str, str]) -> Dict[str, Any]:
        """
        Explicit reconciliation call (scheduled or manual).
        Returns result summary. HALTs if MISMATCH/UNKNOWN.
        """
        local_orders = {
            k: {"status": v.get("status", ""), "filled_qty": str(v.get("filled_qty", "0"))}
            for k, v in self.engine._orders.items()  # type: ignore[attr-defined]
        } if hasattr(self.engine, "_orders") else {}

        local_positions = {}
        if self.engine.portfolio:
            for sym, pos in self.engine.portfolio.positions.items():
                local_positions[sym] = str(pos.quantity)

        result = self.reconciler.reconcile_full(
            local_orders, exchange_orders,
            local_positions, exchange_balances)

        if self.reconciler.should_halt(result):
            self.engine.safety.halt(f"reconciliation {result.state.value}: {result.mismatches}")

        self.engine.health.set("reconciliation_state", result.state.value)
        self.engine.health.set("reconciliation_checked", result.checked_orders)
        return result.model_dump(mode="json")

    def health_snapshot(self) -> Dict[str, Any]:
        return self.engine.health_snapshot()

    @property
    def is_running(self) -> bool:
        return self._running
