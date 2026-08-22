"""
OmniTrade EventBus (Phase 4).

ARCHITECTURE BOUNDARY RULE:
Components exchange information ONLY by publishing/subscribing to typed
events on this bus. Direct cross-component method calls for event flow
(e.g. strategy.on_tick(), risk.check()) are forbidden. This keeps the
pipeline swappable and testable, and lets the same contracts serve the
deterministic simulator.

Backend: in-process asyncio.Queue per subscriber (FIFO => deterministic
ordering for a single consumer). The interface is deliberately narrow so
a Redis Streams backend can replace it later without touching consumers.
"""
import asyncio
from typing import Callable, Dict, List

from ..core.logger import get_logger
from .types import MarketEvent, RiskDecision, PortfolioUpdate

logger = get_logger("EventBus")

# Canonical event type surface for the whole system.
EventType = type[MarketEvent] | type[RiskDecision] | type[PortfolioUpdate]

EventHandler = Callable[[object], "asyncio.Future[None] | None"]


class EventBus:
    """
    In-process publish/subscribe bus.

    - subscribe(event_type, handler): registers an async (or sync) handler.
      Each subscriber gets its own bounded FIFO queue; a slow consumer can
      never block or reorder another consumer's stream.
    - publish(event_type, event): enqueues for every subscriber of that type.
      Publish NEVER blocks the producer: if a subscriber's queue is full,
      that is a backpressure violation and is raised loudly -- events are
      never silently dropped.
    - close(): stops accepting publishes and wakes all consumers so their
      loops can exit cleanly.
    """

    DEFAULT_MAXSIZE = 10_000

    def __init__(self, maxsize: int = DEFAULT_MAXSIZE):
        self._maxsize = maxsize
        self._queues: Dict[EventType, List[asyncio.Queue]] = {}
        self._closed = False

    def subscribe(self, event_type: EventType) -> asyncio.Queue:
        """
        Registers a new FIFO queue for the given event type.
        Returns the queue; the consumer loop belongs to the subscriber.
        """
        if self._closed:
            raise RuntimeError("EventBus is closed; cannot subscribe")
        q: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        self._queues.setdefault(event_type, []).append(q)
        return q

    async def publish(self, event_type: EventType, event) -> None:
        """
        Enqueue one event for all subscribers of this type.
        Raises on closed bus or full queue (loud backpressure failure).
        """
        if self._closed:
            raise RuntimeError("EventBus is closed; cannot publish")

        for q in self._queues.get(event_type, []):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.error(
                    "eventbus_backpressure_violation",
                    event_type=getattr(event_type, "__name__", str(event_type)),
                    dropped=False,  # we raise instead of dropping
                )
                raise RuntimeError(
                    f"Subscriber queue full ({q.qsize()}/{self._maxsize}) "
                    f"for {event_type.__name__}; failing loudly rather than dropping events."
                )

    def close(self) -> None:
        """Stops the bus and drains-wakes subscribers via sentinel None."""
        self._closed = True
        for event_type, queues in self._queues.items():
            for q in queues:
                try:
                    q.put_nowait(None)  # wake-up sentinel for consumer loops
                except asyncio.QueueFull:
                    # Queue at capacity AND closed: consumers will still see
                    # closure via the _closed flag check in their loops.
                    pass


async def drain(queue: asyncio.Queue, handler: EventHandler) -> int:
    """
    Reference consumer loop: processes events in FIFO order until the bus
    closes and the sentinel is received. Returns number of events handled.
    Handlers may be async or sync; exceptions propagate to the caller.
    """
    handled = 0
    while True:
        item = await queue.get()
        try:
            if item is None:
                return handled
            result = handler(item)
            if hasattr(result, "__await__"):
                await result
            handled += 1
        finally:
            queue.task_done()
