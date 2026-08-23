"""Experiment Registry (Phase 15 R1).

Append-only JSONL store of experiment results. The experiment_id is the
config_hash (immutable by construction). No update or delete API exists.

TRADING-PLANE ISOLATION: registry failures are contained here. Nothing in
src/core imports this module; a disk failure loses research records but
can never affect order submission, safety state, or Portfolio accounting.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional


class DuplicateExperimentError(Exception):
    """Raised when recording an experiment_id that already exists."""


class ExperimentRegistry:
    """
    Append-only JSONL registry.

    - record(payload) requires payload["config_hash"] (the experiment_id).
      Writing an ID that already exists raises DuplicateExperimentError.
    - No update/delete methods exist. Immutability is structural.
    - load_all()/get() re-read the file; corruption in one line is
      skipped loudly (printed), never repaired.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _iter_file(self) -> Iterator[Dict[str, Any]]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"[registry] corrupt line skipped in {self.path}")

    def exists(self, experiment_id: str) -> bool:
        for rec in self._iter_file():
            if rec.get("config_hash") == experiment_id:
                return True
        return False

    def record(self, payload: Dict[str, Any]) -> str:
        experiment_id = payload.get("config_hash")
        if not experiment_id:
            raise ValueError("payload must contain config_hash as experiment_id")
        if self.exists(experiment_id):
            raise DuplicateExperimentError(
                f"experiment_id {experiment_id} already recorded; "
                "registry is append-only"
            )
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return experiment_id

    def get(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        for rec in self._iter_file():
            if rec.get("config_hash") == experiment_id:
                return rec
        return None

    def load_all(self) -> List[Dict[str, Any]]:
        return list(self._iter_file())

    def count(self) -> int:
        return len(list(self._iter_file()))

    def query(self, predicate: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
        return [r for r in self._iter_file() if predicate(r)]
