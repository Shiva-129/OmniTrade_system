import asyncio
import random
from typing import AsyncGenerator, Dict, Any, List, Optional

import ccxt.pro as ccxt  # Use .pro for WebSocket support

from ..core.types import Packet
from ..core.clock import Clock
from .exchange_interface import ExchangeInterface
from ..core.logger import get_logger

logger = get_logger("BinanceObserver")

# Reconnect policy (unit-tested in test_binance_observer.py)
MAX_RECONNECT_ATTEMPTS = 8
BASE_BACKOFF_S = 0.5
MAX_BACKOFF_S = 30.0


def backoff_delay(attempt: int, rng: Optional[random.Random] = None) -> float:
    """
    Exponential backoff with jitter: min(MAX, BASE * 2^attempt) +-25% jitter.
    Deterministic when an rng seed is supplied (simulator parity).
    """
    base = min(MAX_BACKOFF_S, BASE_BACKOFF_S * (2 ** max(attempt - 1, 0)))
    jitter = 1.0 if rng is None else 0.75 + 0.5 * rng.random()
    return base * jitter


class BinanceObserver(ExchangeInterface):
    """
    Binance Observer using CCXT Pro.

    Phase 4 additions:
    - Multi-symbol fan-out: one internal task per symbol feeds a shared
      queue; listen() merges them into a single Packet stream. Each symbol
      reconnects independently (isolation).
    - Automatic reconnection with capped exponential backoff + jitter.
      After MAX_RECONNECT_ATTEMPTS consecutive failures the stream raises,
      propagating to the engine to trigger HALT -- never silent death.
    """

    def __init__(self, symbols: List[str]):
        self.symbols = list(symbols) or []
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        self.running = False
        self._queue: "asyncio.Queue[Packet]" = asyncio.Queue()
        self._tasks: List[asyncio.Task] = []
        self._pending_get: Optional[asyncio.Task] = None

    async def connect(self):
        logger.info("connecting_to_binance", symbols=self.symbols)
        # CCXT lazy-connects on first watch call.

    async def listen(self) -> AsyncGenerator[Packet, None]:
        """
        Merge per-symbol streams into one packet stream.

        Failure propagation: while waiting for the next packet we also wait
        on the per-symbol tasks -- if ANY symbol exhausts its reconnect
        budget, its exception is re-raised here so the engine HALTs loudly
        instead of silently stalling.
        """
        self.running = True
        logger.info("starting_binance_stream", symbols=self.symbols)

        try:
            for symbol in self.symbols:
                t = asyncio.create_task(
                    self._symbol_loop(symbol), name=f"binance-{symbol}"
                )
                self._tasks.append(t)

            while self.running:
                get_task = asyncio.create_task(self._queue.get())
                self._pending_get = get_task
                done, _ = await asyncio.wait(
                    {get_task, *self._tasks},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if get_task in done:
                    yield get_task.result()
                    self._queue.task_done()

                # Re-raise any dead symbol stream (fail loudly).
                for t in self._tasks:
                    if t.done() and not t.cancelled():
                        exc = t.exception()
                        if exc is not None:
                            raise exc
        finally:
            if self._pending_get and not self._pending_get.done():
                self._pending_get.cancel()
            await self.close()

    async def _symbol_loop(self, symbol: str):
        """
        One symbol => one resilient watch loop. Failures here never affect
        other symbols' tasks (isolation requirement).
        """
        attempt = 0
        while self.running:
            try:
                trades = await self.exchange.watch_trades(symbol)
                local_ts = Clock.now_epoch_us()
                attempt = 0  # success resets the failure counter
                for trade in trades:
                    if not self.running:
                        return
                    self._queue.put_nowait(
                        self._wrap_packet(trade, local_ts, symbol)
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if not self.running:
                    return
                attempt += 1
                if attempt >= MAX_RECONNECT_ATTEMPTS:
                    # Audit rule: fail loudly, never silently.
                    logger.critical(
                        "binance_stream_gave_up",
                        symbol=symbol,
                        attempts=attempt,
                        error=str(e),
                    )
                    raise
                delay = backoff_delay(attempt)
                logger.warning(
                    "binance_stream_reconnecting",
                    symbol=symbol,
                    attempt=attempt,
                    delay_s=round(delay, 3),
                    error=str(e),
                )
                await asyncio.sleep(delay)

    def _wrap_packet(self, data: Dict[str, Any], local_ts: int, topic: str) -> Packet:
        exchange_ts_ms = data.get("timestamp")
        exchange_ts_us = exchange_ts_ms * 1000 if exchange_ts_ms else local_ts
        drift = Clock.calculate_drift(exchange_ts_us, local_ts)
        return Packet(
            exchange_ts=exchange_ts_us,
            local_arrival_ts=local_ts,
            drift_us=drift,
            source="binance_ccxt",
            topic=topic,
            payload=data,
            sequence_id=data.get("id"),
        )

    async def close(self):
        was_running = self.running
        self.running = False
        if self._pending_get and not self._pending_get.done():
            self._pending_get.cancel()
        for t in self._tasks:
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
        if was_running:
            await self.exchange.close()
