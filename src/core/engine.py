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
from .execution_mode import ExecutionMode, parse_execution_mode
from ..observability.alerts import AlertManager
from ..observability.health import HealthMonitor
from ..observability.metrics import MetricsRegistry
from .portfolio import Portfolio
from .safety import SafetyController, SafetyState
from .state import ObserverState
from .journal import RawJournal
from .logger import configure_logging, get_logger
from .types import ExecutionReport, JournalEntry, MarketEvent, PortfolioUpdate, RiskDecision, RiskCheck
from ..gatekeeper.engine import Gatekeeper
from ..gatekeeper.state_controller import StateController
from ..markets.exchange_interface import ExchangeInterface
from ..strategies.base import BaseStrategy
from ..core.risk_manager import RiskManager

configure_logging()
logger = get_logger("TradingEngine")


class TradingEngine:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        journal_path: str = "journal.jsonl",
        portfolio: Portfolio | None = None,
        strategy: BaseStrategy | None = None,
        risk_manager: RiskManager | None = None,
        gatekeeper: Gatekeeper | None = None,
        broker=None,
        safety: SafetyController | None = None,
        execution_mode: str | ExecutionMode = ExecutionMode.TESTNET,
        metrics: MetricsRegistry | None = None,
        health: HealthMonitor | None = None,
        alerts: AlertManager | None = None,
        user_stream=None,
    ):
        self.bus = EventBus()
        self.state = ObserverState(redis_url=redis_url)
        self.journal = RawJournal(filepath=journal_path)
        # Gatekeeper mirror of positions (Phase 2 semantics, unchanged).
        self.state_controller = StateController(redis_url)
        # Phase 5: optional financial truth. None => observation-only mode.
        self.portfolio = portfolio
        # Phase 12: SafetyController is the single authority.
        # ObserverState remains the telemetry source; Safety owns transitions.
        self.safety: SafetyController = safety or SafetyController()
        # D2: ExecutionMode fail-closed (PAPER/TESTNET/DISABLED only)
        if isinstance(execution_mode, str):
            self.execution_mode = parse_execution_mode(execution_mode)
        else:
            # Validate enum value is not production
            if execution_mode not in (ExecutionMode.PAPER, ExecutionMode.TESTNET, ExecutionMode.DISABLED):
                raise ValueError(f"ExecutionMode {execution_mode!r} not allowed")
            self.execution_mode = execution_mode
        if self.execution_mode == ExecutionMode.DISABLED:
            self.safety.halt("DISABLED execution mode")
        # D4: Observability (vendor-agnostic, swappable)
        self.metrics: MetricsRegistry = metrics or MetricsRegistry()
        self.health: HealthMonitor = health or HealthMonitor(self.safety)
        self.alerts: AlertManager = alerts or AlertManager()
        for _n, _c in AlertManager.standard_conditions().items():
            try:
                self.alerts.register(_n, _c)
            except Exception as e:
                logger.debug("observability_failed", error=str(e), context="alerts_register")
        # D5: User-data stream (optional, testnet only)
        self.user_stream = user_stream
        # Wire gatekeeper's guard to same safety (single state machine).
        if gatekeeper is not None:
            # Ensure gatekeeper's guard delegates to this safety
            gatekeeper.safety = self.safety  # type: ignore[attr-defined]
            if hasattr(gatekeeper, "guard") and gatekeeper.guard is not None:
                gatekeeper.guard.safety = self.safety  # type: ignore[attr-defined]
        self.gatekeeper = gatekeeper
        # Phase 7 pipeline: MarketEvent -> Strategy -> Risk -> Gatekeeper.
        self.strategy = strategy
        self.risk_manager = risk_manager
        # Phase 10: optional execution venue (PaperBroker today).
        self.broker = broker
        # Duplicate-execution guard: every applied report's exchange id.
        self._seen_exec_ids: set[str] = set()
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

        Phase 10: duplicate ExecutionReports (same exchange_order_id) are
        dropped before any mutation -- a replayed/duplicated fill can
        never double-count.
        """
        exec_key = report.exchange_order_id
        if exec_key in self._seen_exec_ids:
            logger.warning("duplicate_execution_report_dropped",
                           exchange_order_id=exec_key,
                           client_order_id=report.client_order_id)
            return
        self._seen_exec_ids.add(exec_key)

        # Preserve causal order for replay: use the report's own
        # timestamp (market time, not wall clock) so it sorts interleaved
        # with the market packet that triggered it.
        self.journal.append(JournalEntry(
            event_type="PACKET",
            timestamp=int(report.timestamp),
            data={"source": "execution_report", **report.model_dump(mode="json")},
        ))

        self.state_controller.process_execution_report(report)

        if self.portfolio is not None:
            delta = self.portfolio.apply_report(report)
            if delta is not None:
                await self.bus.publish(PortfolioUpdate, PortfolioUpdate(
                    symbol=report.symbol, quantity_delta=delta,
                ))
        # D4: metrics for executions
        try:
            if report.status == "FILLED":
                self.metrics.inc("fills")
                self.metrics.inc("execution_trades_total")
            elif report.status in ("PARTIAL_FILL", "PARTIALLY_FILLED"):
                self.metrics.inc("partial_fills")
        except Exception as e:
            logger.debug("observability_failed", error=str(e), context="metrics_fills")
        # Update health with portfolio state
        try:
            if self.portfolio is not None:
                self.health.set("equity", float(self.portfolio.last_equity))
                self.health.set("fees_paid", float(self.portfolio.fees_paid))
        except Exception as e:
            logger.debug("observability_failed", error=str(e), context="health_equity")

    def seed_execution_ids(self, ids) -> None:
        """Restart recovery: pre-load applied report ids from a previous
        session so replayed reports cannot double-count."""
        self._seen_exec_ids.update(ids)

    def _is_intent_reducing(self, intent) -> bool:
        """Determines if an intent would reduce absolute exposure.
        Uses Portfolio._evolve_position for exact arithmetic when possible.
        Fail-closed: if portfolio or price is missing, returns False."""
        if self.portfolio is None:
            return False
        pos = self.portfolio.positions.get(intent.symbol)
        if pos is None:
            return False
        from .money import ZERO

        if pos.quantity == ZERO:
            return False
        # Determine signed delta and reference price
        from .types import OrderSide

        signed = intent.quantity if intent.side == OrderSide.BUY else -intent.quantity
        # Reference price for prospective calculation
        ref_price = intent.price
        if ref_price is None:
            # MARKET order: use current mark if available
            mark = self.portfolio.marks.get(intent.symbol)
            if mark is None:
                return False
            # mark is _Mark with price field
            try:
                ref_price = mark.price
            except Exception:
                return False
        try:
            evolved = self.portfolio._evolve_position(pos, signed, ref_price)
            # Reducing iff absolute quantity shrinks and not flipping through zero
            return abs(evolved.quantity) < abs(pos.quantity)
        except Exception:
            return False

    async def perform_startup_reconciliation(self) -> bool:
        """D6: 9-step startup ordering. Returns True if consistent (HEALTHY)."""
        try:
            # 2. Connect / load markets (if broker is Binance)
            if self.broker is not None and hasattr(self.broker, "_exchange"):
                try:
                    if hasattr(self.broker._exchange, "load_markets"):
                        self.broker._exchange.load_markets()
                except Exception as e:
                    logger.error("startup_load_markets_failed", error=str(e))
                    self.safety.halt(f"startup load_markets failed: {e}")
                    return False
            # 3. Load local journal/state (ensure file exists)
            try:
                # Journal is already opened in __init__; just verify it
                if self.journal.is_closed:
                    raise RuntimeError("journal is closed at startup")
            except Exception as e:
                self.safety.halt(f"journal load failed: {e}")
                return False
            # 4-8. REST reconciliation if broker supports it
            if self.broker is not None and hasattr(self.broker, "startup_reconcile"):
                try:
                    result = self.broker.startup_reconcile()
                    self.health.set("reconciliation_state", "MISMATCH" if not result.get("ok", True) else "CONSISTENT")
                    self.health.set("reconciliation_mismatches", len(result.get("mismatches", [])))
                    if not result.get("ok", True):
                        self.safety.halt(f"startup reconciliation mismatch: {result.get('mismatches')}")
                        try:
                            self.metrics.inc("reconciliation_mismatch")
                        except Exception as e:
                            logger.debug("observability_failed", error=str(e), context="reconciliation_mismatch")
                        return False
                    try:
                        self.metrics.inc("reconciliation_success")
                    except Exception as e:
                        logger.debug("observability_failed", error=str(e), context="reconciliation_success")
                except Exception as e:
                    logger.error("startup_reconcile_failed", error=str(e))
                    self.safety.halt(f"startup reconcile failed: {e}")
                    return False
            # 9. Establish WS (if present)
            if self.user_stream is not None:
                try:
                    await self.user_stream.connect()
                    self.health.set("ws_state", self.user_stream.connection_state())
                except Exception as e:
                    logger.error("user_stream_connect_failed", error=str(e))
                    self.safety.degrade(f"WS connect failed: {e}")
                    # WS failure is DEGRADED, not HALT, per spec
            # Only now HEALTHY
            return not self.safety.is_halted()
        except Exception as e:
            logger.error("startup_reconciliation_error", error=str(e))
            self.safety.halt(f"startup error: {e}")
            return False

    # ------------------------------ lifecycle ------------------------------

    async def start(self):
        init_money_context_once()
        if self.strategy is not None and self.risk_manager is None:
            raise ValueError(
                "Strategy stage requires a RiskManager: "
                "strategies must never bypass risk evaluation."
            )
        # D6: Startup ordering (fail-closed)
        if not await self.perform_startup_reconciliation():
            logger.error("startup_reconciliation_failed_halt", safety_state=self.safety.state.value)
            # Still proceed to start but safety is HALT, so no orders will be submitted
        logger.info("engine_startup", version="phase-13-engine")
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
        Ordered shutdown (D6): stop submissions -> cancel/settle -> flush
        journal -> close WS -> close broker -> close Redis.
        """
        if not self.running:
            return
        # 1. Stop submissions
        try:
            self.safety.halt("shutdown: stop submissions")
        except Exception:
            pass
        self.running = False
        # 2. Cancel open orders per policy (best-effort)
        if self.broker is not None and hasattr(self.broker, "get_open_orders"):
            try:
                for order in list(self.broker.get_open_orders()):
                    coid = order.get("client_order_id") or order.get("clientOrderId", "")
                    if coid:
                        try:
                            self.broker.cancel_order(coid)
                        except Exception:
                            pass
            except Exception:
                pass
        # 3. Stop ingesting first so no new events enter after consumers are cancelled
        for ex in self.exchanges:
            try:
                await ex.close()
            except Exception:
                pass
        self.bus.close()          # wakes consumers with sentinel
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        # 4. Flush journal
        try:
            if not self.journal.is_closed:
                self.journal.close()
        except Exception:
            pass
        # 5. Close WS
        if self.user_stream is not None:
            try:
                await self.user_stream.disconnect()
            except Exception:
                pass
        # 6. Close broker
        if self.broker is not None and hasattr(self.broker, "close"):
            try:
                self.broker.close()
            except Exception:
                pass
        # 7. Close Redis (via state)
        try:
            if hasattr(self.state, "redis") and hasattr(self.state.redis, "close"):
                self.state.redis.close()
        except Exception:
            pass

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

    def health_snapshot(self) -> dict:
        """Machine-readable health (Phase 12)."""
        snap = self.health.snapshot()
        # Enrich with live telemetry — fail-closed on unknown/stale heartbeat via flag (not auto-DEGRADED to keep fresh startup HEALTHY)
        try:
            last = self.state.redis.get("observer:last_update")
            if last:
                snap["heartbeat_age_s"] = __import__("time").time() - int(last) / 1_000_000
                snap["heartbeat_stale"] = snap["heartbeat_age_s"] > 30
            else:
                snap["heartbeat_age_s"] = 999
                snap["heartbeat_stale"] = True
        except Exception:
            snap["heartbeat_age_s"] = 999
            snap["heartbeat_stale"] = True
        try:
            snap["gap_count"] = self.state.get_gap_count()
        except Exception:
            pass
        if self.portfolio is not None:
            try:
                snap["equity"] = float(self.portfolio.last_equity)
                snap["drawdown_pct"] = float(self.portfolio.drawdown().drawdown_pct)
            except Exception:
                pass
        # Evaluate alerts (vendor-agnostic, no side effects in core)
        try:
            self.alerts.evaluate(snap)
        except Exception:
            pass
        return snap

    async def _handle_market_event(self, event: MarketEvent):
        """
        Integrity stage (FIFO, single consumer => deterministic order).
        Same logic as legacy observer._process_loop, event-shaped.
        """
        packet = event.packet
        # D4: health heartbeat
        try:
            self.health.set("last_market_ts", packet.exchange_ts)
            self.health.set("last_source", packet.source)
        except Exception as e:
            logger.debug("observability_failed", error=str(e), context="health_heartbeat")

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
                        try:
                            self.metrics.inc("gaps")
                            self.health.set("gap_count", self.state.get_gap_count())
                            self.health.set("last_gap_source", key)
                        except Exception as e:
                            logger.debug("observability_failed", error=str(e), context="gap_metrics")
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
        mark: Optional = None
        raw_price = packet.payload.get("price") if packet.payload else None
        if raw_price is not None:
            try:
                mark = to_decimal(str(raw_price))
            except Exception:
                mark = None
            if mark is not None:
                if self.portfolio is not None:
                    self.portfolio.mark_price(
                        packet.topic, mark, ts_us=packet.exchange_ts)
                    self.portfolio.update_equity(now_us=packet.exchange_ts)

                # Phase 10: broker works RESTING orders at this price
                # BEFORE the strategy sees it (deterministic ordering).
                if self.broker is not None:
                    self.broker.on_market_price(
                        packet.topic, mark, ts_us=packet.exchange_ts)
                    for rep in self.broker.drain_reports():
                        await self.apply_execution_report(rep)
            else:
                logger.warning("mark_price_skipped", error="unparseable price",
                               topic=packet.topic)  # never invent prices

        # 3. Drift stats + per-packet heartbeat (Phase 4 fix lives here now)
        stats = self.state.update_drift(packet.drift_us)
        try:
            self.health.set("clock_drift_us", stats.mean_us)
            self.health.set("heartbeat_age_s", 0)
            self.metrics.gauge("clock_drift_us", float(stats.mean_us))
        except Exception as e:
            logger.debug("observability_failed", error=str(e), context="drift_health_gauge")

        # 4. Constraint enforcement
        if abs(stats.mean_us) > 500_000:
            logger.error("SYSTEM_HALT_DRIFT_VIOLATION", mean_drift_us=stats.mean_us)
            try:
                self.metrics.inc("halt_total", tags={"reason": "drift"})
            except Exception as e:
                logger.debug("observability_failed", error=str(e), context="halt_total_drift")
            self._transition_status("HALT", "Drift Violation", {"mean_drift_us": stats.mean_us})

        # 5. Strategy stage (Phase 7 + 13): signal -> SAFETY -> risk -> gatekeeper.
        #    SafetyController is the single authority: no order may reach
        #    Risk/Gatekeeper/Broker if safety says no. Fail-closed.
        if self.strategy is not None:
            intent = self.strategy.on_market_event(event)
            if intent is not None:
                try:
                    self.metrics.inc("orders_submitted")
                except Exception as e:
                    logger.debug("observability_failed", error=str(e), context="orders_submitted")
                # D1: authoritative safety check (before risk)
                is_reducing = self._is_intent_reducing(intent)
                blocked_reason = None
                if self.safety.is_halted():
                    blocked_reason = f"safety HALT: {self.safety.snapshot().get('halt_reason','')}"
                elif self.safety.is_degraded() and not is_reducing:
                    blocked_reason = "safety DEGRADED: only reducing orders allowed"

                if blocked_reason is not None:
                    try:
                        self.metrics.inc("safety_blocks", tags={"state": self.safety.state.value})
                    except Exception as e:
                        logger.debug("observability_failed", error=str(e), context="safety_blocks")
                    # Synthesize a safety-blocked decision (audited, no risk call)
                    decision = RiskDecision(
                        client_order_id=intent.client_order_id,
                        symbol=intent.symbol,
                        approved=False,
                        rule="SAFETY",
                        reason=blocked_reason,
                        checks=(RiskCheck(rule="SAFETY", passed=False, detail=blocked_reason),),
                        details={"safety_state": self.safety.state.value, "reducing": str(is_reducing)},
                    )
                    self.journal.append(JournalEntry(
                        event_type="PACKET",
                        timestamp=int(packet.exchange_ts),
                        data={"source": "risk_decision", **decision.model_dump(mode="json")},
                    ))
                    await self.bus.publish(RiskDecision, decision)
                    logger.warning("safety_blocked_submit",
                                   cloid=intent.client_order_id, reason=blocked_reason)
                else:
                    decision = self.risk_manager.evaluate(
                        intent, now_us=packet.exchange_ts
                    )
                    try:
                        if decision.approved:
                            self.metrics.inc("risk_approvals")
                        else:
                            self.metrics.inc("risk_rejections", tags={"rule": decision.rule})
                    except Exception as e:
                        logger.debug("observability_failed", error=str(e), context="risk_metrics")
                    self.journal.append(JournalEntry(
                        event_type="PACKET",
                        timestamp=int(packet.exchange_ts),
                        data={"source": "risk_decision",
                              **decision.model_dump(mode="json")},
                    ))
                    await self.bus.publish(RiskDecision, decision)
                    # RESEARCH-ONLY contract: Gatekeeper=None is allowed only for unit/research;
                    # for PAPER/TESTNET it is fail-closed (no broker). Log explicitly.
                    if decision.approved and self.gatekeeper is None:
                        logger.warning("gatekeeper_missing_fail_closed",
                                       cloid=intent.client_order_id, mode=self.execution_mode.value)
                        try:
                            self.metrics.inc("gatekeeper_missing_blocked")
                        except Exception:
                            pass
                    if decision.approved and self.gatekeeper is not None:
                        outcome = self.gatekeeper.submit_intent(intent)
                        logger.info("intent_submitted",
                                    cloid=intent.client_order_id,
                                    outcome=outcome)
                        try:
                            if outcome == "DUPLICATE":
                                self.metrics.inc("duplicate_suppressed")
                            elif outcome == "ACCEPTED":
                                self.metrics.inc("gatekeeper_accepted")
                            self.health.set("last_gatekeeper_outcome", outcome)
                        except Exception as e:
                            logger.debug("observability_failed", error=str(e), context="gatekeeper_metrics")
                        # Phase 10: Gatekeeper-approved intents may now reach
                        # the venue. Reports drain through the same funnel.
                        if outcome == "ACCEPTED" and self.broker is not None:
                            try:
                                self.metrics.inc("broker_submissions")
                            except Exception as e:
                                logger.debug("observability_failed", error=str(e), context="broker_submissions")
                            broker_outcome = self.broker.submit_order(intent)
                            logger.info("broker_submission",
                                        cloid=intent.client_order_id,
                                        outcome=broker_outcome)
                            for rep in self.broker.drain_reports():
                                await self.apply_execution_report(rep)

        # 6. Structured log
        logger.info(
            "market_event_processed",
            drift_us=packet.drift_us,
            source=packet.source,
            rolling_mean_drift=stats.mean_us,
        )
        self.processed_count += 1

    # ------------------------------- status ---------------------------------

    def _transition_status(self, new_status: str, reason: str, payload: dict):
        """Atomic status update: Redis + Journal (audit requirement).
        SafetyController is the single authority: ObserverState is the
        telemetry source, Safety owns the state machine. Mapping:
          CONNECTED -> HEALTHY, DEGRADED -> DEGRADED, HALT -> HALT.
        Preserve backward-compatible Redis observer:status behavior.
        """
        self.state.set_system_status(new_status)
        # Sync SafetyController (single authority, no second state machine)
        if new_status == "HALT":
            self.safety.halt(reason)
            try:
                self.metrics.inc("halt_total")
                self.health.set("last_halt_reason", reason)
            except Exception as e:
                logger.debug("observability_failed", error=str(e), context="halt_total")
        elif new_status == "DEGRADED":
            self.safety.degrade(reason)
            try:
                self.metrics.inc("degraded_total")
                self.health.set("last_degraded_reason", reason)
            except Exception as e:
                logger.debug("observability_failed", error=str(e), context="degraded_total")
        elif new_status == "CONNECTED":
            # Fail-closed: do not auto-recover HALT/DEGRADED via observer.
            # HEALTHY stays HEALTHY; DEGRADED/HALT require explicit recovery.
            pass
        self.journal.append(JournalEntry(
            event_type="STATUS_CHANGE",
            timestamp=Clock.now_epoch_us(),
            data={"status": new_status, "reason": reason, "payload": payload},
        ))
        logger.info("status_change", status=new_status, reason=reason)
        # D4: evaluate alerts after every status change
        try:
            self.alerts.evaluate(self.health_snapshot())
        except Exception as e:
            logger.debug("observability_failed", error=str(e), context="alerts_evaluate")


def init_money_context_once():
    """Installs canonical Decimal policy at engine entry."""
    from .money import init_money_context
    init_money_context()


if __name__ == "__main__":  # pragma: no cover
    try:
        asyncio.run(TradingEngine().start())
    except KeyboardInterrupt:
        pass
