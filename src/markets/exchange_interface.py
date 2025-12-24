from abc import ABC, abstractmethod
from typing import AsyncGenerator
from ..core.types import Packet

class ExchangeInterface(ABC):
    """
    Abstract Base Class for Exchange Observers.
    Must provide an async stream of Packets.
    """
    
    @abstractmethod
    async def connect(self):
        """
        Establish connection to the exchange (WS/REST).
        """
        pass

    @abstractmethod
    async def listen(self) -> AsyncGenerator[Packet, None]:
        """
        Yields Packets indefinitely.
        Must handle its own reconnection logic internally or via a manager.
        """
        pass

    @abstractmethod
    async def close(self):
        """
        Graceful shutdown.
        """
        pass
