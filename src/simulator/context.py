"""
OmniTrade Simulator: Global Deterministic Context

This module defines the SINGLE global execution context for the simulator.
All decimal math, RNG, and configuration MUST flow through here.
"""
import decimal
import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# --- GLOBAL DECIMAL CONTEXT (IMMUTABLE AFTER INIT) ---
# This MUST match production execution exactly.
DECIMAL_CONTEXT = decimal.Context(
    prec=28,                          # Precision
    rounding=decimal.ROUND_HALF_EVEN, # Banker's rounding
    Emin=-999999,
    Emax=999999,
    capitals=1,
    clamp=0,
    flags=[],
    traps=[decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow]
)

def init_decimal_context():
    """
    Sets the global decimal context. Call ONCE at simulator startup.
    """
    decimal.setcontext(DECIMAL_CONTEXT)

@dataclass(frozen=True)
class SimulatorConfig:
    """
    Immutable configuration for a simulation run.
    """
    config_hash: str            # Hash of the config snapshot
    rng_seed: int               # Fixed RNG seed
    journal_path: str           # Path to raw event journal
    dependency_versions: Dict[str, str] = field(default_factory=dict)

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
