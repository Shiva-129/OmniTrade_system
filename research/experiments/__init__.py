"""Experiment registry package (Phase 15 R1)."""
from .registry import DuplicateExperimentError, ExperimentRegistry

__all__ = ["ExperimentRegistry", "DuplicateExperimentError"]
