import json
import os
from .types import JournalEntry
from .clock import Clock

class RawJournal:
    """
    Append-only journal for raw event recording.
    Writes newline-delimited JSON.
    """
    def __init__(self, filepath: str = "journal.jsonl"):
        self.filepath = filepath
        # Open in append mode, text
        self._file = open(self.filepath, "a", buffering=1) # Line buffered

    def append(self, entry: JournalEntry):
        """
        Writes a single entry to the journal.
        """
        payload = entry.model_dump()
        # Ensure we write a single line
        line = json.dumps(payload) + "\n"
        self._file.write(line)
        # Flush handled by buffering=1 for newlines, but can force if critical
        # self._file.flush()

    def close(self):
        self._file.close()

    @staticmethod
    def replay(filepath: str):
        """
        Generator to replay entries from a journal file.
        """
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    yield JournalEntry.model_validate_json(line)
