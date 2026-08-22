# Root conftest: ensures repo root is on sys.path so `src.*` imports resolve
# regardless of how pytest is invoked.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
