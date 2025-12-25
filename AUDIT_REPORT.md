# AUDIT REPORT: OmniTrade Phase 1 (Observer)

**AUDITOR**: Jules (Lead Infrastructure Auditor)
**DATE**: 2024-05-21
**SUBJECT**: Specification & Implementation Audit of `src/observer.py`

---

## CRITICAL FAILURES

### 1. Time Base Incompatibility (The 1.7 Billion Second Lag)
- **Failure Timeline**: Immediate upon first packet ingestion.
- **Observable Damage**: The system will **IMMEDIATELY HALT** and never enter a functional state.
- **Root Cause**: `drift_us` is calculated as `Packet.exchange_ts` (Epoch Microseconds, ~1.76e18) minus `Clock.now_us()` (Monotonic Microseconds, ~Time Since Boot, e.g., 1e11).
- **Detail**: The difference is ~54 years (1.7e15 us). The safety threshold is 500ms (5e5 us). `1.7e15 > 5e5` is always true.
- **Location**: `src/markets/binance_observer.py` calls `Clock.calculate_drift` with mismatched time bases.

### 2. Silent Disconnection (The Zombie Observer)
- **Failure Timeline**: When the Exchange API becomes unreachable (Network Partition) or Authentication fails.
- **Observable Damage**: The system remains in `CONNECTED` state, but ingests no data. Forensic logs show a sudden stop in packets with no "ERROR" or "DISCONNECTED" event.
- **Root Cause**: `BinanceObserver.listen` contains a `try...except` block that catches *all* exceptions, logs them as `binance_stream_error`, and sleeps. It never propagates the error to `ObserverSystem` to trigger a state change to `DEGRADED` or `HALT`.

---

## DETERMINISM BREAKS

### 1. The Shadow Gap (Queue Volatility)
- **Scenario**: The process crashes (OOM, SigKill) after `packet = await self.packet_queue.get()` but before `self.journal.append(entry)`.
- **Exact Replay Failure**: The live system (and the exchange) "saw" and acknowledged the packet. The Journal does not contain it. A replay of the Journal will create a state *skipping* that packet.
- **Impact**: Numerical state (e.g., VWAP, Volatility) in Phase 2 will diverge from the "Truth" that the live system momentarily acted upon.

### 2. Unlogged State Mutations
- **Scenario**: The system triggers a `HALT` due to drift.
- **Exact Replay Failure**: The Journal records `PACKET` events but does **not** record `STATUS_CHANGE` events (despite `JournalEntry` supporting them).
- **Impact**: It is impossible to reconstruct *when* the system decided to Halt during a forensic replay. You cannot verify if the system reacted correctly to a specific drift event.

---

## REQUIRED FUSES

1.  **Sequence Continuity Check**:
    - **Invariant**: `current_packet.sequence_id == last_packet.sequence_id + 1`
    - **Purpose**: To distinguish a "Price Gap" (Market moved 10% in 1ms) from a "Data Loss Event" (We missed the messages in between).
    - **Status**: `Packet` contains `sequence_id`, but `ObserverSystem` ignores it.

2.  **Epoch Alignment**:
    - **Invariant**: Internal clock must rely on `time.time_ns()` (Epoch) for drift calculations, OR `Clock` must calibrate Monotonic-to-Epoch offset at startup.
    - **Purpose**: To make `drift` calculation physically meaningful.

3.  **Heartbeat & Status Journaling**:
    - **Invariant**: All calls to `self.state.set_system_status` must be mirrored to `self.journal.append`.
    - **Purpose**: To enable deterministic replay of system behavior (not just market data).

---

## VERDICT

**FAIL**

The system is currently non-functional due to the Time Base Mismatch. Even if patched, it suffers from Critical Silent Failures and Determinism Breaks that render it unsafe for Phase 2 (Trading). It requires significant revision.
