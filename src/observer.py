"""
Standalone Market Observer mode (Phase 4).

Since the TradingEngine now owns ingestion, journaling, gap/drift
integrity, and status transitions, this module is a thin composition
root: configure exchange observers, hand them to an engine WITHOUT any
downstream stages registered => observation-only operation.

Run:  python -m src.observer
"""
import asyncio
import sys

from .core.engine import TradingEngine
from .core.logger import get_logger
from .markets.binance_observer import BinanceObserver

logger = get_logger("ObserverMain")


def build_observation_engine() -> TradingEngine:
    """Phase 4 default topology: Binance spot/futures trades, no stages."""
    # TODO(phase-11): load symbols/credentials from env/config
    engine = TradingEngine()
    engine.add_exchange(BinanceObserver(symbols=["BTC/USDT"]))
    return engine


async def _amain() -> int:
    engine = build_observation_engine()
    try:
        await engine.start()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception:
        logger.critical("observer_mode_failed", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_amain()))
