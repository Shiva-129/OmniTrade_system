"""
BrokerInterface -- the ONE execution contract.

Implementations: PaperBroker now; BinanceTestnet/Zerodha later.
The exact same OrderIntent travels every implementation unchanged, and
every implementation speaks ONLY in ExecutionReports.

ACCESS RULES
------------
- Strategies NEVER see a broker instance (structural: they hold none).
- Intents reach the broker ONLY after RiskManager approval AND
  Gatekeeper acceptance (engine-enforced pipeline order).
- Implementations MUST NOT mutate Portfolio state directly. They emit
  ExecutionReports; the engine routes them through the single mutation
  funnel core.engine.TradingEngine.apply_execution_report().
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

from ..core.types import OrderIntent


class BrokerInterface(ABC):

    @abstractmethod
    def submit_order(self, intent: OrderIntent) -> str:
        """
        Submits one Gatekeeper-approved intent.
        Returns an outcome string: "ACCEPTED" | "REJECTED" | "DUPLICATE".
        Any resulting ExecutionReports surface via the broker's report
        outbox (implementation-defined), never via Portfolio writes.
        """

    @abstractmethod
    def cancel_order(self, client_order_id: str) -> str:
        """Returns "CANCELED" | "UNKNOWN". Invalid lifecycle transitions
        raise loudly rather than being swallowed."""

    @abstractmethod
    def get_order(self, client_order_id: str) -> Dict[str, Any] | None:
        """Execution-state view of one order (or None)."""

    @abstractmethod
    def get_open_orders(self) -> list[Dict[str, Any]]:
        """All orders not in a terminal state."""

    @abstractmethod
    def get_positions(self) -> Dict[str, str]:
        """EXECUTION VIEW of net filled quantities per symbol.
        Deliberately distinct from Portfolio accounting state."""

    @abstractmethod
    def get_account_state(self) -> Dict[str, Any]:
        """EXECUTION counters (submitted/filled/fees/...). NOT PnL --
        financial truth lives exclusively in Portfolio."""

    @abstractmethod
    def close(self) -> None:
        """Stops the venue; further submissions raise."""
