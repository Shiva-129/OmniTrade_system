import ccxt.pro as ccxt  # Use .pro for WebSocket support
import asyncio
import logging
from typing import AsyncGenerator, Dict, Any
from ..core.types import Packet
from ..core.clock import Clock
from .exchange_interface import ExchangeInterface
from ..core.logger import get_logger

logger = get_logger("BinanceObserver")

class BinanceObserver(ExchangeInterface):
    """
    Binance Observer using CCXT Pro (or async support).
    Observes specified symbols.
    """
    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        # Initialize CCXT exchange in async mode
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'} # Observing futures by default, configurable
        })
        self.running = False

    async def connect(self):
        logger.info("connecting_to_binance", symbols=self.symbols)
        # CCXT lazy connects on first watch
        pass

    async def listen(self) -> AsyncGenerator[Packet, None]:
        self.running = True
        logger.info("starting_binance_stream")
        
        try:
            # We loop over symbols or use watch_ticker for all? 
            # For simplicity in this observer phase, let's watch recent trades for a single symbol
            # handling multiple symbols usually requires asyncio.gather or watch_multiple
            # optimizing for one symbol first as proof of concept if list has 1, else gather
            
            # Using watch_trades which gives a stream
            if len(self.symbols) == 1:
                symbol = self.symbols[0]
                # CCXT watch_trades returns a list of trades, but we want the loop to just yield per packet
                # Actually watch_trades blocks until NEW trades arrive
                while self.running:
                    trades = await self.exchange.watch_trades(symbol)
                    # Capture arrival time ASAP
                    local_ts = Clock.now_epoch_us()
                    
                    for trade in trades:
                        # Convert to Packet
                        yield self._wrap_packet(trade, local_ts, symbol)
            else:
                # Multi-stream support (naive for now)
                # Ideally we spawn tasks for each symbol
                # keeping it simple: just warn
                logger.warning("multi_symbol_not_fully_implemented_using_first", symbol=self.symbols[0])
                symbol = self.symbols[0]
                while self.running:
                    trades = await self.exchange.watch_trades(symbol)
                    local_ts = Clock.now_epoch_us()
                    for trade in trades:
                         yield self._wrap_packet(trade, local_ts, symbol)

        except Exception as e:
            # Audit Requirement: "Observer must fail loudly, never silently."
            logger.error("binance_stream_failure", error=str(e))
            self.running = False
            raise # Propagate to system to trigger HALT/Journaling

    def _wrap_packet(self, data: Dict[str, Any], local_ts: int, topic: str) -> Packet:
        # Binance/CCXT standard fields
        # timestamp is ms int
        exchange_ts_ms = data.get('timestamp')
        exchange_ts_us = exchange_ts_ms * 1000 if exchange_ts_ms else local_ts
        
        # Audit Requirement: Time Base Alignment
        drift = Clock.calculate_drift(exchange_ts_us, local_ts)
        
        return Packet(
            exchange_ts=exchange_ts_us,
            local_arrival_ts=local_ts,
            drift_us=drift,
            source="binance_ccxt",
            topic=topic,
            payload=data,
            sequence_id=data.get('id') # Trade ID
        )

    async def close(self):
        self.running = False
        await self.exchange.close()
