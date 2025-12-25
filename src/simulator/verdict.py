"""
OmniTrade Simulator: Replay Verdict

Defines the result structure for replay verification.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

class VerdictStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"

@dataclass
class DivergencePoint:
    """
    Details of the first divergence detected during replay.
    """
    event_index: int
    expected_hash: str
    actual_hash: str
    event_data: Dict[str, Any]
    causal_chain: List[int] = field(default_factory=list)  # Parent event indices

@dataclass
class ReplayVerdict:
    """
    Final verdict of a replay run.
    """
    status: VerdictStatus
    events_processed: int
    events_total: int
    config_hash: str
    rng_seed: int
    divergence: Optional[DivergencePoint] = None
    error_message: Optional[str] = None

    def is_pass(self) -> bool:
        return self.status == VerdictStatus.PASS

    def summary(self) -> str:
        if self.status == VerdictStatus.PASS:
            return f"PASS: {self.events_processed}/{self.events_total} events replayed identically"
        elif self.status == VerdictStatus.FAIL:
            return f"FAIL: Divergence at event {self.divergence.event_index}"
        else:
            return f"ERROR: {self.error_message}"
