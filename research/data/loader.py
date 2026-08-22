"""Historical data loading. Provider adapters normalize BEFORE storage."""
from typing import List

from .dataset import OHLCVDataset


def load_from_ccxt(exchange_id: str = "binance", symbol: str = "BTC/USDT",
                   timeframe: str = "1h", since_ms: int | None = None,
                   limit: int = 1000) -> OHLCVDataset:
    """
    Live fetch path (network required). Kept OUT of tests; the backtest
    layer consumes datasets, never providers.
    """
    import ccxt  # deferred: research-only dependency path

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    try:
        raw: List[list] = exchange.fetch_ohlcv(symbol, timeframe,
                                               since=since_ms, limit=limit)
    finally:
        try:
            exchange.close()
        except Exception:
            pass
    return OHLCVDataset.from_records(raw, symbol=symbol, timeframe=timeframe)
