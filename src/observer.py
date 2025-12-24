import asyncio
import signal
import sys
from typing import List
from .core.clock import Clock
from .core.state import ObserverState
from .core.journal import RawJournal
from .core.logger import list_logger, configure_logging, get_logger
from .core.types import JournalEntry, Packet
from .markets.exchange_interface import ExchangeInterface
from .markets.binance_observer import BinanceObserver
from .markets.kite_observer import KiteObserver

# Configure logging
configure_logging()
logger = get_logger("ObserverMain")

class ObserverSystem:
    def __init__(self):
        self.state = ObserverState()
        self.journal = RawJournal()
        self.running = True
        self.exchanges: List[ExchangeInterface] = []
        self.packet_queue: asyncio.Queue[Packet] = asyncio.Queue()

    async def start(self):
        logger.info("system_startup", version="phase-1-observer")
        
        # Initialize Exchanges
        # TODO: Load config from env or args
        self.exchanges.append(BinanceObserver(symbols=["BTC/USDT"]))
        # self.exchanges.append(KiteObserver())

        # Connect
        for ex in self.exchanges:
            await ex.connect()

        # Start Producers
        producers = [asyncio.create_task(self._ingest_loop(ex)) for ex in self.exchanges]
        
        # Start Consumer
        consumer = asyncio.create_task(self._process_loop())

        # Signal Handling
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.shutdown(s)))

        self.state.set_system_status("CONNECTED") # Optimistic start

        try:
            await asyncio.gather(*producers, consumer)
        except asyncio.CancelledError:
            logger.info("tasks_cancelled")
        finally:
            await self.shutdown(signal.SIGTERM)

    async def _ingest_loop(self, exchange: ExchangeInterface):
        """
        Consumes packets from an exchange and puts them in the queue.
        """
        async for packet in exchange.listen():
            if not self.running:
                break
            await self.packet_queue.put(packet)

    async def _process_loop(self):
        """
        Main Event Loop: Pops from queue, logs, updates state.
        """
        logger.info("processing_loop_started")
        while self.running:
            packet = await self.packet_queue.get()
            
            # 1. Authoritative Time Check
            # packet.local_arrival_ts is set at ingest
            # Calculate drift again if needed or trust packet
            
            # 2. Append to Raw Journal (Immutable, blocking/fast)
            # Serialize
            entry = JournalEntry(
                event_type="PACKET",
                timestamp=packet.local_arrival_ts,
                data=packet.model_dump()
            )
            self.journal.append(entry)

            # 3. Update State (Redis)
            stats = self.state.update_drift(packet.drift_us)
            
            # 4. Check Constraints
            if abs(stats.mean_us) > 500_000: # 500ms
                logger.error("SYSTEM_HALT_DRIFT_VIOLATION", mean_drift_us=stats.mean_us)
                self.state.set_system_status("HALT")
                # In Phase 1 "Observer never dies; it freezes"
                # So we might stop processing new packets or just mark status
                # Spec says: "Observer never dies; it freezes".
                # We will continue to log but mark HALT.
            
            # 5. Emit Structured Log
            logger.info("packet_processed", 
                        drift_us=packet.drift_us, 
                        source=packet.source,
                        rolling_mean_drift=stats.mean_us)
            
            self.packet_queue.task_done()

    async def shutdown(self, sig):
        if not self.running:
            return
        logger.info("shutdown_signal_received", signal=sig.name)
        self.running = False
        
        self.state.set_system_status("HALT")

        for ex in self.exchanges:
            await ex.close()
        
        self.journal.close()
        
        # Cancel all tasks? 
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [t.cancel() for t in tasks]
        logger.info("shutdown_complete")

if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(ObserverSystem().start()))
    except KeyboardInterrupt:
        pass
