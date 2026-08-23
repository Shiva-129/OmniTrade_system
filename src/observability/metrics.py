"""
Metrics Registry (Phase 12) -- vendor-agnostic counters/gauges/histograms.

No core trading logic imports a vendor SDK directly. This registry is
the only metrics surface; swapping vendors means swapping this file.
"""
from __future__ import annotations

from typing import Any, Dict
from collections import defaultdict


class MetricsRegistry:
    """
    In-memory metrics store. In production, flush() would push to
    Prometheus/Datadog/etc. For tests, inspect directly.
    """

    def __init__(self):
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, list[float]] = defaultdict(list)

    def inc(self, name: str, value: float = 1.0, tags: Dict[str, str] | None = None) -> None:
        key = self._key(name, tags)
        self.counters[key] += value

    def gauge(self, name: str, value: float, tags: Dict[str, str] | None = None) -> None:
        key = self._key(name, tags)
        self.gauges[key] = value

    def histogram(self, name: str, value: float, tags: Dict[str, str] | None = None) -> None:
        key = self._key(name, tags)
        self.histograms[key].append(value)

    def get_counter(self, name: str, tags: Dict[str, str] | None = None) -> float:
        return self.counters.get(self._key(name, tags), 0.0)

    def get_gauge(self, name: str, tags: Dict[str, str] | None = None) -> float | None:
        return self.gauges.get(self._key(name, tags))

    def snapshot(self) -> Dict[str, Any]:
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {k: list(v) for k, v in self.histograms.items()},
        }

    def reset(self) -> None:
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()

    @staticmethod
    def _key(name: str, tags: Dict[str, str] | None) -> str:
        if not tags:
            return name
        suffix = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{{suffix}}}"
