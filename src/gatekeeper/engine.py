from .state_controller import StateController
from .command_registry import CommandRegistry
from .guard import ExecutionGuard
from .reconciliation import ReconciliationEngine
from ..core.types import OrderIntent, ExecutionReport
from ..core.logger import get_logger

logger = get_logger("GatekeeperMain")

class Gatekeeper:
    """
    The Single Authority.
    Integrates Authority, Guard, and Reconciliation.
    """
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.state_controller = StateController(redis_url)
        self.command_registry = CommandRegistry()
        self.guard = ExecutionGuard(redis_url)
        self.reconciliation = ReconciliationEngine(self.state_controller, self.guard)

    def submit_intent(self, intent: OrderIntent) -> str:
        """
        Entry point for Strategy Engine.
        Returns:
            - "ACCEPTED"
            - "DUPLICATE"
            - Raises HardBlockError on rejection
        """
        # 1. Idempotency Check
        if not self.command_registry.register(intent):
            logger.info("duplicate_intent_ignored", cloid=intent.client_order_id)
            return "DUPLICATE"

        # 2. Guard Validation (Risk, Connectivity, SafeMode)
        self.guard.validate_intent()

        # 3. Success (Pass through to Execution Adapter - Out of Scope for internal logic but logically next)
        # Note: We do NOT mutate state here.
        logger.info("intent_accepted", cloid=intent.client_order_id)
        return "ACCEPTED"

    def process_execution_report(self, report: ExecutionReport):
        """
        Entry point for Exchange Adapters.
        """
        # 1. Mutate State (Authority)
        self.state_controller.process_execution_report(report)

        # 2. Check Reconciliation Triggers (Optional: Reconcile on every fill?)
        # Typically Reconciliation is async/periodic, but we could do partial checks here.
