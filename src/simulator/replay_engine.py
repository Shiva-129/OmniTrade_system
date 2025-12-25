"""
OmniTrade Simulator: Replay Engine

The core deterministic replay engine.
Processes events ONE AT A TIME, synchronously.
NO ASYNC. NO CONCURRENCY. NO PARALLELISM.
"""
from decimal import Decimal
from typing import Optional, Dict, Any, List
from .context import SimulatorConfig, DeterministicRNG, init_decimal_context
from .journal_reader import JournalReader, OrderedEvent
from .state_store import SimulatedStateStore
from .verdict import ReplayVerdict, VerdictStatus, DivergencePoint
from ..core.types import JournalEntry

class ReplayEngine:
    """
    Deterministic Replay Engine.
    
    Invariant: One event in → one state change out → persist hash → next event.
    """
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.rng = DeterministicRNG(config.rng_seed)
        self.state = SimulatedStateStore()
        self.journal = JournalReader(config.journal_path)
        
        # Hash log: event_index -> state_hash
        self.hash_log: Dict[int, str] = {}
        
        # For causal tracking
        self.causal_parent: Optional[int] = None
        
        # Reference hashes for verification (loaded separately)
        self.reference_hashes: Dict[int, str] = {}

    def load_reference_hashes(self, reference_path: str):
        """
        Load reference hash log from a previous (live) run.
        """
        import json
        with open(reference_path, 'r') as f:
            data = json.load(f)
            self.reference_hashes = {int(k): v for k, v in data.items()}

    def run(self) -> ReplayVerdict:
        """
        Execute the replay.
        Returns verdict indicating PASS or FAIL.
        """
        # Initialize decimal context FIRST
        init_decimal_context()
        
        # Load journal
        try:
            total_events = self.journal.load()
        except Exception as e:
            return ReplayVerdict(
                status=VerdictStatus.ERROR,
                events_processed=0,
                events_total=0,
                config_hash=self.config.config_hash,
                rng_seed=self.config.rng_seed,
                error_message=f"Journal load failed: {e}"
            )

        processed = 0
        
        # Process events ONE AT A TIME
        for ordered_event in self.journal:
            # Step 1: Process the event
            try:
                self._process_single_event(ordered_event)
            except Exception as e:
                return ReplayVerdict(
                    status=VerdictStatus.ERROR,
                    events_processed=processed,
                    events_total=total_events,
                    config_hash=self.config.config_hash,
                    rng_seed=self.config.rng_seed,
                    error_message=f"Event {ordered_event.index} failed: {e}"
                )

            # Step 2: Compute and store state hash
            state_hash = self.state.get_state_hash()
            self.hash_log[ordered_event.index] = state_hash

            # Step 3: Verify against reference (if available)
            if ordered_event.index in self.reference_hashes:
                expected = self.reference_hashes[ordered_event.index]
                if state_hash != expected:
                    # DIVERGENCE DETECTED
                    divergence = DivergencePoint(
                        event_index=ordered_event.index,
                        expected_hash=expected,
                        actual_hash=state_hash,
                        event_data=ordered_event.event.data,
                        causal_chain=self._build_causal_chain(ordered_event.index)
                    )
                    return ReplayVerdict(
                        status=VerdictStatus.FAIL,
                        events_processed=processed,
                        events_total=total_events,
                        config_hash=self.config.config_hash,
                        rng_seed=self.config.rng_seed,
                        divergence=divergence
                    )

            # Update causal parent
            self.causal_parent = ordered_event.index
            processed += 1

        # All events processed successfully
        return ReplayVerdict(
            status=VerdictStatus.PASS,
            events_processed=processed,
            events_total=total_events,
            config_hash=self.config.config_hash,
            rng_seed=self.config.rng_seed
        )

    def _process_single_event(self, ordered_event: OrderedEvent):
        """
        Process exactly ONE event.
        This method MUST be synchronous and deterministic.
        """
        event = ordered_event.event
        event_type = event.event_type
        data = event.data

        if event_type == "PACKET":
            self._handle_packet(data)
        elif event_type == "STATUS_CHANGE":
            self._handle_status_change(data)
        elif event_type == "GAP":
            self._handle_gap(data)
        elif event_type == "ERROR":
            self._handle_error(data)
        else:
            # Unknown event type - log but continue
            pass

        # Update last seen timestamp
        self.state.last_seen_ts = event.timestamp

    def _handle_packet(self, data: Dict[str, Any]):
        """
        Handle a market data packet.
        Reuses Phase 1 logic path.
        """
        # Extract execution report if present (Phase 2 logic)
        if "status" in data and data.get("source") == "execution_report":
            self._handle_execution_report(data)
        
        # Update drift tracking (simulation mode)
        drift = data.get("drift_us", 0)
        # In simulation, we just track; no actual alerts

    def _handle_execution_report(self, data: Dict[str, Any]):
        """
        Handle execution report - update state.
        Mirrors StateController.process_execution_report
        """
        status = data.get("status")
        symbol = data.get("symbol", "")
        client_order_id = data.get("client_order_id", "")
        filled_qty = Decimal(str(data.get("filled_quantity", 0)))
        side = data.get("side", "BUY")

        # Store order state
        self.state.set_order(client_order_id, data)

        # Update position on fills
        if status in ["PARTIAL_FILL", "FILLED"]:
            delta = filled_qty if side == "BUY" else -filled_qty
            self.state.update_position(symbol, delta)

    def _handle_status_change(self, data: Dict[str, Any]):
        """Handle system status change."""
        new_status = data.get("status", "CONNECTED")
        self.state.set_system_status(new_status)

    def _handle_gap(self, data: Dict[str, Any]):
        """Handle gap detection event."""
        self.state.increment_gap_count()
        # Gap may trigger DEGRADED status
        if self.state.gap_count > 5:
            self.state.set_system_status("DEGRADED")

    def _handle_error(self, data: Dict[str, Any]):
        """Handle error event - may trigger HALT."""
        error_type = data.get("error_type", "")
        if error_type == "CRITICAL":
            self.state.set_system_status("HALT")

    def _build_causal_chain(self, current_index: int) -> List[int]:
        """
        Build the causal chain leading to current event.
        Simple linear chain for now.
        """
        chain = []
        idx = current_index - 1
        while idx >= 0 and len(chain) < 10:
            chain.append(idx)
            idx -= 1
        return list(reversed(chain))

    def save_hash_log(self, output_path: str):
        """Save the hash log for future reference comparison."""
        import json
        with open(output_path, 'w') as f:
            json.dump(self.hash_log, f, indent=2)
