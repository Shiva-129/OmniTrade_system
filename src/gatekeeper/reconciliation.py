from decimal import Decimal
from typing import Dict
from .state_controller import StateController
from .guard import ExecutionGuard
from ..core.logger import get_logger
from ..core.money import to_decimal

logger = get_logger("ReconciliationEngine")

TOLERANCE = Decimal("1e-8")

class ReconciliationEngine:
    """
    Component 3: Truth Enforcement.
    Periodically checks Internal Redis State vs Exchange REST Snapshot.
    Uses Decimal tolerance and checks both directions (ghost positions).
    """
    def __init__(self, state_controller: StateController, guard: ExecutionGuard):
        self.state = state_controller
        self.guard = guard

    def reconcile(self, exchange_snapshot: Dict[str, str]):
        """
        exchange_snapshot: {symbol: position_qty as string/Decimal}
        Checks all symbols in union of internal and exchange keys.
        """
        logger.info("starting_reconciliation")
        # Collect all symbols from both sides to detect ghost positions
        internal_symbols = set()
        try:
            if hasattr(self.state, "redis"):
                for key in self.state.redis.scan_iter(match=f"{self.state.PREFIX_POS}:*"):
                    sym = key.split(":")[-1]
                    internal_symbols.add(sym)
        except Exception:
            pass
        all_symbols = set(exchange_snapshot.keys()) | internal_symbols
        # Always check exchange-provided symbols plus any internal we can enumerate
        # Fallback: at minimum check exchange keys; also check ghost by iterating exchange plus explicit internal fetch
        drift_found = False
        for symbol in set(exchange_snapshot.keys()):
            internal_qty = self.state.get_position(symbol)
            try:
                ex_qty = to_decimal(str(exchange_snapshot[symbol]))
                int_qty = to_decimal(str(internal_qty))
            except Exception:
                ex_qty = exchange_snapshot[symbol]
                int_qty = internal_qty
                if ex_qty != int_qty:
                    logger.error("CRITICAL_STATE_DRIFT", symbol=symbol, internal=int_qty, exchange=ex_qty)
                    self.guard.enter_safe_mode(f"Drift detected for {symbol}")
                    drift_found = True
                continue
            if abs(int_qty - ex_qty) > TOLERANCE:
                logger.error("CRITICAL_STATE_DRIFT", symbol=symbol, internal=int_qty, exchange=ex_qty)
                self.guard.enter_safe_mode(f"Drift detected for {symbol}")
                drift_found = True
        # Ghost check: internal has position but exchange missing/zero
        for symbol in list(all_symbols):
            if symbol not in exchange_snapshot:
                internal_qty = self.state.get_position(symbol)
                try:
                    if abs(to_decimal(str(internal_qty))) > TOLERANCE:
                        logger.error("CRITICAL_STATE_DRIFT_GHOST", symbol=symbol, internal=internal_qty, exchange="0")
                        self.guard.enter_safe_mode(f"Ghost position for {symbol}")
                        drift_found = True
                except Exception:
                    if str(internal_qty) not in ("0", "0.0", ""):
                        logger.error("CRITICAL_STATE_DRIFT_GHOST", symbol=symbol, internal=internal_qty, exchange="0")
                        self.guard.enter_safe_mode(f"Ghost position for {symbol}")
                        drift_found = True
        if drift_found:
            return
        logger.info("reconciliation_passed")
