"""
OmniTrade Simulator: Journal Reader

Reads the raw event journal from Phase 1 for replay.
Enforces strict ordering rules.
"""
import json
from dataclasses import dataclass
from typing import Iterator, List, Optional
from ..core.types import JournalEntry

@dataclass
class OrderedEvent:
    """
    Event with ordering metadata for deterministic replay.
    """
    index: int
    local_arrival_ts: int
    sequence_id: Optional[int]
    source_priority: int  # Lower = higher priority (WS=1, REST=2)
    event: JournalEntry

    def ordering_key(self):
        """
        Returns tuple for stable sorting.
        Order: (local_arrival_ts, sequence_id or MAX, source_priority)
        """
        seq = self.sequence_id if self.sequence_id is not None else 2**63
        return (self.local_arrival_ts, seq, self.source_priority)

class JournalReader:
    """
    Reads and orders events from the raw journal file.
    """
    SOURCE_PRIORITY = {
        "binance_ws": 1,
        "binance_ccxt": 1,
        "kite_ws": 1,
        "binance_rest": 2,
        "kite_rest": 2,
    }
    DEFAULT_PRIORITY = 3

    def __init__(self, journal_path: str):
        self.journal_path = journal_path
        self._events: List[OrderedEvent] = []

    def load(self) -> int:
        """
        Loads all events from the journal.
        Returns the count of events loaded.
        """
        self._events = []
        index = 0
        
        with open(self.journal_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                entry = JournalEntry.model_validate_json(line)
                
                # Extract ordering metadata from the event data
                data = entry.data
                local_ts = entry.timestamp
                seq_id = data.get("sequence_id")
                source = data.get("source", "unknown")
                priority = self.SOURCE_PRIORITY.get(source, self.DEFAULT_PRIORITY)
                
                ordered = OrderedEvent(
                    index=index,
                    local_arrival_ts=local_ts,
                    sequence_id=seq_id,
                    source_priority=priority,
                    event=entry
                )
                self._events.append(ordered)
                index += 1
        
        # Sort by deterministic ordering rules
        self._events.sort(key=lambda e: e.ordering_key())
        return len(self._events)

    def __iter__(self) -> Iterator[OrderedEvent]:
        """Yields events in deterministic order."""
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def get_event(self, index: int) -> Optional[OrderedEvent]:
        """Get event by its sorted index."""
        if 0 <= index < len(self._events):
            return self._events[index]
        return None
