"""
Historical ingestion layer — deterministic, validated, no silent repair.

Supports local CSV/Parquet OHLCV files:
  timestamp,open,high,low,close,volume  (timestamp = UTC epoch ms or ISO8601)

- Validates ordering, duplicates, OHLC, volume, timeframe step
- Produces deterministic content_hash (symbol|timeframe|n + canonical rows)
- Never silently repairs: malformed rows are counted in rejected, not fixed
- Same file → identical hash; different data → different hash
- No future data enters earlier bars (bars sorted, no look-ahead)
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .dataset import OHLCVDataset, Bar, timeframe_minutes, normalize_ccxt_row
from .validate import validate_dataset, DataQualityReport
from src.core.money import to_decimal


class IngestionResult:
    def __init__(self, dataset: OHLCVDataset | None, report: DataQualityReport | None,
                 rejected_rows: List[Dict[str, Any]], source_path: str, error: str | None = None):
        self.dataset = dataset
        self.report = report
        self.rejected_rows = rejected_rows
        self.source_path = source_path
        self.error = error
        self.is_ok = dataset is not None and error is None


def _parse_timestamp(raw: Any) -> int:
    # Accept int epoch ms, or ISO8601 string
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    s = str(raw).strip()
    if s.isdigit():
        return int(s)
    # Try ISO8601 -> epoch ms (assume UTC)
    try:
        from datetime import datetime, timezone
        # Handle "2023-01-01 00:00:00" or "2023-01-01T00:00:00Z"
        s2 = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception as e:
        raise ValueError(f"unparseable timestamp {raw!r}: {e}")


def _validate_row(ts: int, o: Any, h: Any, low: Any, c: Any, v: Any, line_no: int) -> tuple | None:
    # Returns normalized row or None if rejected (with reason)
    try:
        # Use same ingestion boundary as live: str() hop
        ts_i = _parse_timestamp(ts)
        # Validate OHLC via Decimal (will raise if not numeric)
        to_decimal(str(o)), to_decimal(str(h)), to_decimal(str(low)), to_decimal(str(c)), to_decimal(str(v))
        # OHLC consistency will be caught by validate_dataset, but reject obvious high<low here
        if float(h) < float(low):
            return None
        return (ts_i, o, h, low, c, v)
    except Exception:
        return None


def load_csv_dataset(path: str | Path, symbol: str, timeframe: str) -> IngestionResult:
    p = Path(path)
    if not p.exists():
        return IngestionResult(None, None, [], str(p), error="DATA_REQUIRED: file not found")
    rejected: List[Dict[str, Any]] = []
    records: List[tuple] = []
    try:
        with open(p, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Normalize header lower
            if reader.fieldnames is None:
                return IngestionResult(None, None, [], str(p), error="DATA_REQUIRED: empty file")
            lower_fields = {k.lower(): k for k in reader.fieldnames}
            required = ["timestamp", "open", "high", "low", "close", "volume"]
            # Alternative column names: ts, time
            aliases = {"ts": "timestamp", "time": "timestamp", "datetime": "timestamp"}
            for i, row in enumerate(reader, start=2):
                # Map aliases
                norm = {}
                for k, v in row.items():
                    kl = k.lower()
                    kl = aliases.get(kl, kl)
                    norm[kl] = v
                # Check required
                if not all(col in norm for col in required):
                    rejected.append({"line": i, "reason": "missing columns", "row": row})
                    continue
                ts_raw = norm["timestamp"]
                rec = _validate_row(ts_raw, norm["open"], norm["high"], norm["low"], norm["close"], norm["volume"], i)
                if rec is None:
                    rejected.append({"line": i, "reason": "malformed OHLC/volume/timestamp", "row": row})
                    continue
                records.append(rec)
    except Exception as e:
        return IngestionResult(None, None, [], str(p), error=f"DATA_REQUIRED: read error {e}")

    if not records:
        return IngestionResult(None, None, rejected, str(p), error="DATA_REQUIRED: no valid rows")

    # Sort deterministically by ts (dataset does sort, but we want to detect ordering violations)
    # Keep original order for detection: we will create dataset then validate will count ordering_violations
    # But we must ensure no future data enters earlier bars — bars are sorted, no look-ahead
    ds = OHLCVDataset.from_records(records, symbol=symbol, timeframe=timeframe)
    report = validate_dataset(ds)

    # If duplicate timestamps or ordering violations, they are reported but not silently repaired
    # The dataset is still returned with sorted bars; caller can check report.is_clean
    return IngestionResult(ds, report, rejected, str(p), error=None)


def load_parquet_dataset(path: str | Path, symbol: str, timeframe: str) -> IngestionResult:
    p = Path(path)
    if not p.exists():
        return IngestionResult(None, None, [], str(p), error="DATA_REQUIRED: file not found")
    try:
        import pandas as pd
        df = pd.read_parquet(p)
        # Expect columns timestamp/open/high/low/close/volume (or ts)
        cols = {c.lower(): c for c in df.columns}
        # Normalize
        records = []
        rejected = []
        for idx, row in df.iterrows():
            try:
                ts_col = cols.get("timestamp") or cols.get("ts") or cols.get("time")
                if ts_col is None:
                    raise ValueError("missing timestamp column")
                ts_raw = row[ts_col]
                rec = _validate_row(ts_raw, row[cols["open"]], row[cols["high"]], row[cols["low"]], row[cols["close"]], row[cols["volume"]], int(idx)+2)
                if rec is None:
                    rejected.append({"line": int(idx)+2, "reason": "malformed", "row": dict(row)})
                    continue
                records.append(rec)
            except Exception as e:
                rejected.append({"line": int(idx)+2, "reason": str(e), "row": dict(row)})
        if not records:
            return IngestionResult(None, None, rejected, str(p), error="DATA_REQUIRED: no valid rows")
        ds = OHLCVDataset.from_records(records, symbol=symbol, timeframe=timeframe)
        report = validate_dataset(ds)
        return IngestionResult(ds, report, rejected, str(p), error=None)
    except Exception as e:
        return IngestionResult(None, None, [], str(p), error=f"DATA_REQUIRED: parquet read error {e}")


def ingest_historical_file(path: str | Path, symbol: str, timeframe: str) -> IngestionResult:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return load_csv_dataset(p, symbol, timeframe)
    elif p.suffix.lower() in (".parquet", ".pq"):
        return load_parquet_dataset(p, symbol, timeframe)
    else:
        # Try CSV then Parquet
        r = load_csv_dataset(p, symbol, timeframe)
        if r.error and "file not found" not in r.error:
            return r
        return load_parquet_dataset(p, symbol, timeframe)
