"""
OmniTrade TradingEngine (Phase 4): central orchestrator.

Responsibilities:
- Owns the EventBus and the lifecycle of the Observer (market data).
- Consumes MarketEvents in strict FIFO order (deterministic ordering).
- Runs the integrity pipeline stage: sequence/gap detection, drift
  statistics + heartbeat, HALT/DEGRADED transitions (moved verbatim from
  the old observer._process_loop -- same audit semantics, journaled).
- Provides register_stage(): the extension point where later phases plug
  in Strategy -> Risk -> Gatekeeper -> Execution consumers. The engine
  itself contains NO strategy/risk/portfolio logic by design.

The strategy must never know whether it is live, paper, or replay:
everything downstream of MarketEvent is identical in all modes.
"""
import asyncio
from typing import List, Dict

from .clock import Clock
from .events import EventBus, EventType, drain
from .money import to_decimal
from .portfolio import Portfolio
from .state import ObserverState
from .journal import RawJournal
from .logger import configure_logging, get_logger
from .types import ExecutionReport, JournalEntry, MarketEvent, PortfolioUpdate
from ..gatekeeper.state_controller import StateController
from ..markets.exchange_interface import ExchangeInterface

configure_logging()
logger = get_logger("TradingEngine")


class TradingEngine:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        journal_path: str = "journal.jsonl",
        portfolio: Portfolio | None = None,
    ):
        self.bus = EventBus()
        self.state = ObserverState(redis_url=redis_url)
        self.journal = RawJournal(filepath=journal_path)
        # Gatekeeper mirror of positions (Phase 2 semantics, unchanged).
        self.state_controller = StateController(redis_url)
        # Phase 5: optional financial truth. None => observation-only mode.
        self.portfolio = portfolio
        self.exchanges: List[ExchangeInterface] = []
        self.sequence_tracker: Dict[str, int] = {}
        self.processed_count = 0
        self.running = False
        self._tasks: List[asyncio.Task] = []

    # ---------- stage registration (extension point for Phase 5+) ----------

    def register_stage(self, event_type: EventType) -> asyncio.Queue:
        """Later phases call this to receive typed events off the bus."""
        return self.bus.subscribe(event_type)

    # ------------------------- execution fan-out ----------------------------

    async def apply_execution_report(self, report: ExecutionReport) -> None:
        """
        THE canonical entry point for fills (Phase 5).

        Order is load-bearing:
          1. WRITE-AHEAD journal (audit: record before applying anywhere)
          2. Gatekeeper StateController (Phase 2 semantics, unchanged)
          3. Portfolio (single financial truth)
          4. PortfolioUpdate published for downstream stages

        Rejections/cancellations flow through too -- they journal and
        update gatekeeper order state, but the Portfolio no-ops on them.
        """
        self.journal.append(JournalEntry(
            event_type="PACKET",
            timestamp=Clock.now_epoch_us(),
            data={"source": "execution_report", **report.model_dump(mode="json")},
        ))

        self.state_controller.process_execution_report(report)

        if self.portfolio is not None:
            delta = self.portfolio.apply_report(report)
            if delta is not None:
                await self.bus.publish(PortfolioUpdate, PortfolioUpdate(
                    symbol=report.symbol, quantity_delta=delta,
                ))

    # ------------------------------ lifecycle ------------------------------

    async def start(self):
        init_money_context_once()
        logger.info("engine_startup", version="phase-4-engine")
        self.running = True

        market_q = self.register_stage(MarketEvent)
        consumer = asyncio.create_task(
            drain(market_q, self._handle_market_event), name="integrity-stage"
        )

        producers = [
            asyncio.create_task(self._ingest_loop(ex), name=f"ingest-{i}")
            for i, ex in enumerate(self.exchanges)
        ]
        self._tasks = [*producers, consumer]

        self._transition_status("CONNECTED", "Engine startup complete", {})

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("engine_tasks_cancelled")
        except Exception as e:
            logger.critical("engine_critical_failure", error=str(e))
            self._transition_status("HALT", f"Critical Failure: {e}", {"error": str(e)})
            raise
        finally:
            # Cancellation-safe: shield cleanup so Ctrl+C on Windows still
            # closes exchanges, wakes consumers, and flushes the journal.
            try:
                await asyncio.shield(self.shutdown())
            except asyncio.CancelledError:
                await self._finalize()

    async def _finalize(self):
        """Last-resort synchronous-safe cleanup (journal is line-buffered)."""
        if not self.journal.is_closed:
            self.journal.close()

    def add_exchange(self, exchange: ExchangeInterface):
        self.exchanges.append(exchange)

    async def stop(self):
        """
        Ordered shutdown: stop ingesting first so no new events enter after
        consumers are cancelled => zero silent drops.
        """
        if not self.running:
            return
        self.running = False
        for ex in self.exchanges:
            await ex.close()
        self.bus.close()          # wakes consumers with sentinel
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self.journal.close()

    async def shutdown(self):
        if not self.running:
            return
        logger.info("shutdown_signal_received")
        self._transition_status("HALT", "Shutdown Initiated", {})
        await self.stop()
        logger.info("shutdown_complete")

    # ---------------------------- pipeline stages ---------------------------

    async def _ingest_loop(self, exchange: ExchangeInterface):
        """
        Producer: consume packets from one exchange, WRITE-AHEAD journal,
        then publish onto the bus. Journal-before-publish preserves the
        Phase 1 audit invariant (an event recorded before it is processed).
        Failures propagate loudly to trigger HALT -- no silent drops.
        """
        async for packet in exchange.listen():
            entry = JournalEntry(
                event_type="PACKET",
                timestamp=packet.local_arrival_ts,
                data=packet.model_dump(),
            )
            self.journal.append(entry)
            await self.bus.publish(MarketEvent, MarketEvent(packet=packet))

    async def _handle_market_event(self, event: MarketEvent):
        """
        Integrity stage (FIFO, single consumer => deterministic order).
        Same logic as legacy observer._process_loop, event-shaped.
        """
        packet = event.packet

        # 1. Sequence & gap detection
        key = f"{packet.source}:{packet.topic}"
        if packet.sequence_id is not None:
            try:
                seq_id = int(packet.sequence_id)
                last_seq = self.sequence_tracker.get(key)
                if last_seq is not None:
                    expected = last_seq + 1
                    if seq_id > expected:
                        gap_size = seq_id - expected
                        msg = f"Sequence Gap: Expected {expected}, Got {seq_id}"
                        logger.error("sequence_gap_detected", source=key, gap=gap_size)
                        self.journal.append(JournalEntry(
                            event_type="GAP",
                            timestamp=Clock.now_epoch_us(),
                            data={"source": key, "expected": expected, "got": seq_id},
                        ))
                        self.state.record_gap()
                        if self.state.get_system_status() == "CONNECTED":
                            self._transition_status("DEGRADED", msg, {"gap": gap_size})
                    elif seq_id < last_seq:
                        logger.warning("out_of_order_packet", source=key, seq=seq_id, last=last_seq)
                self.sequence_tracker[key] = seq_id
            except ValueError:
                pass  # non-integer sequence id; skip check

        # 2. Mark-to-market (Phase 5): prices ONLY -- never positions/cash.
        #    Venue price arrives as a wire float; the str() hop is the
        #    single sanctioned ingestion boundary into Decimal.
        #    NAMESPACE RULE: packet.topic MUST equal the trading symbol used
        #    on intents/reports -- marks are keyed by it.
        if self.portfolio is not None:
            raw_price = packet.payload.get("price") if packet.payload else None
            if raw_price is not None:
                try:
                    self.portfolio.mark_price(
                        packet.topic,
                        to_decimal(str(raw_price)),
                        ts_us=packet.exchange_ts,
                    )
                    self.portfolio.update_equity(now_us=packet.exchange_ts)
                except Exception as e:
                    logger.warning("mark_price_skipped", error=str(e),
                                   topic=packet.topic)  # never invent prices

        # 3. Drift stats + per-packet heartbeat (Phase 4 fix lives here now)
        stats = self.state.update_drift(packet.drift_us)

        # 4. Constraint enforcement
        if abs(stats.mean_us) > 500_000:
            logger.error("SYSTEM_HALT_DRIFT_VIOLATION", mean_drift_us=stats.mean_us)
            self._transition_status("HALT", "Drift Violation", {"mean_drift_us": stats.mean_us})

        # 5. Structured log
        logger.info(
            "market_event_processed",
            drift_us=packet.drift_us,
            source=packet.source,
            rolling_mean_drift=stats.mean_us,
        )
        self.processed_count += 1

    # ------------------------------- status ---------------------------------

    def _transition_status(self, new_status: str, reason: str, payload: dict):
        """Atomic status update: Redis + Journal (audit requirement)."""
        self.state.set_system_status(new_status)
        self.journal.append(JournalEntry(
            event_type="STATUS_CHANGE",
            timestamp=Clock.now_epoch_us(),
            data={"status": new_status, "reason": reason, "payload": payload},
        ))
        logger.info("status_change", status=new_status, reason=reason)


def init_money_context_once():
    """Installs canonical Decimal policy at engine entry."""
    from .money import init_money_context
    init_money_context()


if __name__ == "__main__":  # pragma: no cover
    try:
        asyncio.run(TradingEngine().start())
    except KeyboardInterrupt:
        pass
