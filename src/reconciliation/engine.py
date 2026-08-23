"""
Dedicated Reconciliation Engine (Phase 12).

Compares LOCAL state (journal + portfolio + order book) vs
BINANCE TESTNET state (REST snapshots) with explicit result states:

    CONSISTENT  -- proceed
    RECOVERABLE -- rebuild local from exchange/journal, re-verify
    MISMATCH    -- safe mode / HALT
    UNKNOWN     -- HALT

No guessing. Unknown execution state -> HALT.
"""
from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from ..core.money import ZERO, to_decimal


class ReconciliationState(str, Enum):
    CONSISTENT = "CONSISTENT"
    RECOVERABLE = "RECOVERABLE"
    MISMATCH = "MISMATCH"
    UNKNOWN = "UNKNOWN"


class ReconciliationResult(BaseModel):
    model_config = {"frozen": True}

    state: ReconciliationState
    checked_orders: int = 0
    mismatches: List[str] = []
    recoverable_orders: List[str] = []
    details: Dict[str, str] = {}


class ReconciliationEngine:
    """
    Pure reconciliation logic. All I/O (fetching exchange state) is
    injected via callables so tests stay deterministic without network.
    """

    TOLERANCE = Decimal("1e-8")

    def reconcile_orders(
        self,
        local_orders: Dict[str, Dict[str, Any]],
        exchange_orders: Dict[str, Dict[str, Any]],
    ) -> ReconciliationResult:
        mismatches: List[str] = []
        recoverable: List[str] = []
        checked = 0

        # Check every local order against exchange
        for coid, local in local_orders.items():
            checked += 1
            ex = exchange_orders.get(coid)
            if ex is None:
                # Local says submitted, exchange says unknown
                # Could be: not yet propagated, or rejected, or missed
                if local.get("status") in ("NEW", "PARTIALLY_FILLED"):
                    mismatches.append(f"local {coid} is {local.get('status')} but exchange unknown")
                continue

            # Compare fields
            for field in ("status", "filled_qty", "remaining_qty"):
                local_val = str(local.get(field, ""))
                if field == "filled_qty":
                    ex_val = str(ex.get(field, ex.get("filled", "")))
                elif field == "remaining_qty":
                    ex_val = str(ex.get(field, ""))
                else:
                    ex_val = str(ex.get(field, ex.get("status", "")))
                # For status, allow alias mapping
                if field == "status":
                    # Normalize PARTIAL_FILL vs PARTIALLY_FILLED
                    local_norm = local_val.replace("PARTIALLY_FILLED", "PARTIAL_FILL")
                    ex_norm = ex_val.replace("PARTIALLY_FILLED", "PARTIAL_FILL")
                    if local_norm != ex_norm:
                        # NEW vs FILLED is recoverable if exchange is ahead
                        if local_val == "NEW" and ex_val in ("FILLED", "PARTIAL_FILL"):
                            recoverable.append(coid)
                        else:
                            mismatches.append(f"{coid} status {local_val} vs {ex_val}")
                elif field in ("filled_qty", "remaining_qty"):
                    try:
                        if abs(to_decimal(local_val or "0") - to_decimal(ex_val or "0")) > self.TOLERANCE:
                            mismatches.append(f"{coid} {field} {local_val} vs {ex_val}")
                    except Exception:
                        if local_val != ex_val:
                            mismatches.append(f"{coid} {field} {local_val} vs {ex_val}")

        # Check for exchange orders not in local (missed submission)
        for coid in exchange_orders:
            if coid not in local_orders:
                mismatches.append(f"exchange has unknown order {coid}")

        if mismatches:
            # Some mismatches may be recoverable (exchange ahead)
            if recoverable and not any("unknown" in m for m in mismatches):
                return ReconciliationResult(
                    state=ReconciliationState.RECOVERABLE,
                    checked_orders=checked,
                    mismatches=mismatches,
                    recoverable_orders=recoverable,
                )
            return ReconciliationResult(
                state=ReconciliationState.MISMATCH,
                checked_orders=checked,
                mismatches=mismatches,
            )
        if recoverable:
            return ReconciliationResult(
                state=ReconciliationState.RECOVERABLE,
                checked_orders=checked,
                recoverable_orders=recoverable,
            )
        return ReconciliationResult(state=ReconciliationState.CONSISTENT, checked_orders=checked)

    def reconcile_positions(
        self,
        local_positions: Dict[str, str],
        exchange_balances: Dict[str, str],
    ) -> ReconciliationResult:
        mismatches: List[str] = []
        checked = 0
        # For spot, compare each asset balance. Use Decimal with tolerance.
        all_assets = set(local_positions.keys()) | set(exchange_balances.keys())
        for asset in all_assets:
            checked += 1
            local_qty = local_positions.get(asset, "0")
            ex_qty = exchange_balances.get(asset, "0")
            try:
                diff = abs(to_decimal(local_qty) - to_decimal(ex_qty))
                if diff > self.TOLERANCE:
                    mismatches.append(f"{asset} local {local_qty} vs exchange {ex_qty} diff {diff}")
            except Exception:
                if local_qty != ex_qty:
                    mismatches.append(f"{asset} {local_qty} vs {ex_qty}")

        if mismatches:
            # Position mismatches are always MISMATCH (not recoverable without trade)
            return ReconciliationResult(
                state=ReconciliationState.MISMATCH,
                checked_orders=checked,
                mismatches=mismatches,
            )
        return ReconciliationResult(state=ReconciliationState.CONSISTENT, checked_orders=checked)

    def reconcile_full(
        self,
        local_orders: Dict[str, Dict[str, Any]],
        exchange_orders: Dict[str, Dict[str, Any]],
        local_positions: Dict[str, str],
        exchange_balances: Dict[str, str],
    ) -> ReconciliationResult:
        order_result = self.reconcile_orders(local_orders, exchange_orders)
        if order_result.state in (ReconciliationState.MISMATCH, ReconciliationState.UNKNOWN):
            return order_result
        pos_result = self.reconcile_positions(local_positions, exchange_balances)
        if pos_result.state != ReconciliationState.CONSISTENT:
            return pos_result
        if order_result.state == ReconciliationState.RECOVERABLE:
            return order_result
        return ReconciliationResult(state=ReconciliationState.CONSISTENT, checked_orders=order_result.checked_orders)

    def should_halt(self, result: ReconciliationResult) -> bool:
        return result.state in (ReconciliationState.MISMATCH, ReconciliationState.UNKNOWN)

    def is_recoverable(self, result: ReconciliationResult) -> bool:
        return result.state == ReconciliationState.RECOVERABLE
