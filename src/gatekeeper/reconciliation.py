from typing import Dict
from .state_controller import StateController
from .guard import ExecutionGuard
from ..core.logger import get_logger

logger = get_logger("ReconciliationEngine")

class ReconciliationEngine:
    """
    Component 3: Truth Enforcement.
    Periodically checks Internal Redis State vs Exchange REST Snapshot.
    """
    def __init__(self, state_controller: StateController, guard: ExecutionGuard):
        self.state = state_controller
        self.guard = guard

    def reconcile(self, exchange_snapshot: Dict[str, float]):
        """
        exchange_snapshot: {symbol: position_qty}
        """
        logger.info("starting_reconciliation")
        
        for symbol, exchange_qty in exchange_snapshot.items():
            internal_qty = self.state.get_position(symbol)
            
            # Simple equality check appropriate strictly for integers/scaled
            # Float warning: if using floats, need tolerance. 
            # Spec says "No floating point nondeterminism", implied usage of Decimals or close tolerance
            
            if abs(internal_qty - exchange_qty) > 1e-9: # Epsilon for now if float
                logger.error("CRITICAL_STATE_DRIFT", 
                             symbol=symbol, 
                             internal=internal_qty, 
                             exchange=exchange_qty)
                
                self.guard.enter_safe_mode(f"Drift detected for {symbol}")
                return # Stop checks on first failure

        logger.info("reconciliation_passed")
