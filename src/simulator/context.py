"""
OmniTrade Simulator: Global Deterministic Context

Phase 4: the Decimal policy is now OWNED by src/core/money.py (canonical
prec=28 / ROUND_HALF_EVEN) so the live path and the replay path are
guaranteed to share one arithmetic universe. This module re-exports it
under the simulator's historical names for compatibility.
"""
import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from ..core.money import (
    CANONICAL_CONTEXT as DECIMAL_CONTEXT,
    init_money_context,
)

# Compatibility alias -- do not define a second context here.
__all__ = [
    "DECIMAL_CONTEXT",
    "init_decimal_context",
    "SimulatorConfig",
    "DeterministicRNG",
]


def init_decimal_context():
    """
    Sets the global decimal context. Call ONCE at simulator startup.
    Delegates to the canonical project-wide money policy.
    """
    init_money_context()

@dataclass(frozen=True)
class SimulatorConfig:
    """
    Immutable configuration for a simulation run.

    Phase 5: initial_cash (fixed-point string) optionally activates
    portfolio participation during replay. None preserves pre-Phase-5
    behavior AND legacy config hashes (deliberately excluded from
    _compute_hash).
    """
    config_hash: str            # Hash of the config snapshot
    rng_seed: int               # Fixed RNG seed
    journal_path: str           # Path to raw event journal
    dependency_versions: Dict[str, str] = field(default_factory=dict)
    initial_cash: Optional[str] = None

    def verify_hash(self) -> bool:
        """
        Recomputes config hash and verifies integrity.
        """
        computed = self._compute_hash()
        return computed == self.config_hash

    def _compute_hash(self) -> str:
        data = f"{self.rng_seed}:{self.journal_path}:{sorted(self.dependency_versions.items())}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

class DeterministicRNG:
    """
    Wrapper around random.Random with explicit seed.
    Provides reproducible randomness.
    """
    def __init__(self, seed: int):
        self._seed = seed
        self._rng = random.Random(seed)

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def random(self) -> float:
        return self._rng.random()

    def choice(self, seq):
        return self._rng.choice(seq)

    def get_seed(self) -> int:
        return self._seed
