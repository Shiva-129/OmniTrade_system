import asyncio
import signal
import sys
from typing import List, Dict, Optional
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

        # Sequence tracking: {source_topic: last_sequence_id}
        self.sequence_tracker: Dict[str, int] = {}

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

        self._transition_status("CONNECTED", "Startup complete", {})

        try:
            # If any producer fails (raises exception), gather will raise immediately
            # (if we don't return_exceptions=True). We want to fail loudly.
            await asyncio.gather(*producers, consumer)
        except asyncio.CancelledError:
            logger.info("tasks_cancelled")
        except Exception as e:
            # Catch producer failures (e.g. Binance stream died)
            logger.critical("system_critical_failure", error=str(e))
            self._transition_status("HALT", "Critical Failure: " + str(e), {"error": str(e)})
        finally:
            await self.shutdown(signal.SIGTERM)

    def _transition_status(self, new_status: str, reason: str, payload: dict):
        """
        Updates system status atomically: Redis + Journal.
        Audit Requirement: State & Status Journaling
        """
        # 1. Update Redis
        self.state.set_system_status(new_status)

        # 2. Journal
        entry = JournalEntry(
            event_type="STATUS_CHANGE",
            timestamp=Clock.now_epoch_us(),
            data={
                "status": new_status,
                "reason": reason,
                "payload": payload
            }
        )
        self.journal.append(entry)

        logger.info("status_change", status=new_status, reason=reason)

    async def _ingest_loop(self, exchange: ExchangeInterface):
        """
        Consumes packets from an exchange, Journals them (Write-Ahead), and puts them in the queue.
        """
        async for packet in exchange.listen():
            if not self.running:
                break

            # Audit Requirement: Journal Atomicity (Write-ahead)
            # We journal BEFORE queuing. If we crash after this, the event is recorded.
            entry = JournalEntry(
                event_type="PACKET",
                timestamp=packet.local_arrival_ts,
                data=packet.model_dump()
            )
            self.journal.append(entry)

            await self.packet_queue.put(packet)

    async def _process_loop(self):
        """
        Main Event Loop: Pops from queue, logs, updates state.
        """
        logger.info("processing_loop_started")
        while self.running:
            packet = await self.packet_queue.get()
            
            # 1. Sequence & Gap Detection
            # Audit Requirement: Sequence & Gap Detection
            key = f"{packet.source}:{packet.topic}"
            if packet.sequence_id is not None:
                # Assuming sequence_id is int for trade IDs in this context
                # If it's str, we might need parsing. Binance trade ID is int.
                try:
                    seq_id = int(packet.sequence_id)
                    last_seq = self.sequence_tracker.get(key)

                    if last_seq is not None:
                        expected = last_seq + 1
                        if seq_id > expected:
                            gap_size = seq_id - expected
                            msg = f"Sequence Gap: Expected {expected}, Got {seq_id}"
                            logger.error("sequence_gap_detected", source=key, gap=gap_size)

                            # Journal GAP
                            self.journal.append(JournalEntry(
                                event_type="GAP",
                                timestamp=Clock.now_epoch_us(),
                                data={"source": key, "expected": expected, "got": seq_id}
                            ))

                            # Transition to DEGRADED
                            self.state.record_gap()
                            current_status = self.state.get_system_status()
                            if current_status == "CONNECTED":
                                self._transition_status("DEGRADED", msg, {"gap": gap_size})

                        elif seq_id < last_seq:
                            # Out of order or duplicate?
                            logger.warning("out_of_order_packet", source=key, seq=seq_id, last=last_seq)

                    self.sequence_tracker[key] = seq_id
                except ValueError:
                    pass # Non-integer sequence ID, skip check for now

            # 2. Update State (Redis)
            stats = self.state.update_drift(packet.drift_us)
            
            # 3. Check Constraints
            if abs(stats.mean_us) > 500_000: # 500ms
                logger.error("SYSTEM_HALT_DRIFT_VIOLATION", mean_drift_us=stats.mean_us)
                self._transition_status("HALT", "Drift Violation", {"mean_drift_us": stats.mean_us})
            
            # 4. Emit Structured Log
            logger.info("packet_processed", 
                        drift_us=packet.drift_us, 
                        source=packet.source,
                        rolling_mean_drift=stats.mean_us)
            
            self.packet_queue.task_done()

    async def shutdown(self, sig):
        if not self.running:
            return
        logger.info("shutdown_signal_received", signal=sig.name if hasattr(sig, 'name') else str(sig))
        self.running = False
        
        self._transition_status("HALT", "Shutdown Initiated", {"signal": str(sig)})

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
