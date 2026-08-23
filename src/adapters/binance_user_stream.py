"""
Binance Testnet User-Data Stream (Phase 12).

WEBSOCKET = low-latency transport
REST      = recovery authority
JOURNAL   = historical record
PORTFOLIO = accounting via ExecutionReports only

All Binance-specific event shapes stay inside this module.
Outbound: normalized ExecutionReports via callback, deduplicated by
exchange executionId.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Dict, Optional, Set

from ..core.journal import RawJournal
from ..core.logger import get_logger
from ..core.money import ZERO, to_decimal

logger = get_logger("BinanceUserStream")

# How long without a message before we consider the stream stale.
STALE_THRESHOLD_S = 30.0


class BinanceUserStream:
    """
    Authenticated user-data stream for Binance testnet.

    The stream is created via REST (POST /api/v3/userDataStream) to
    obtain a listenKey, then connected via WebSocket. For tests, pass
    `ws_factory` and `rest_factory` to inject fakes.
    """

    def __init__(
        self,
        config,
        *,
        journal: Optional[RawJournal] = None,
        on_execution_report: Optional[Callable] = None,
        ws_factory=None,
        rest_factory=None,
        keepalive_interval_s: float = 30 * 60,
    ):
        # Fail-closed: only testnet config is accepted (reuse Phase 11 validation).
        if getattr(config, "binance_env", None) != "testnet":
            raise ValueError("BinanceUserStream requires BINANCE_ENV='testnet'")
        if "testnet" not in getattr(config, "base_url", "").lower():
            raise ValueError("User stream base_url must be testnet")
        self.config = config
        self.journal = journal
        self.on_execution_report = on_execution_report
        self._ws_factory = ws_factory
        self._rest_factory = rest_factory
        self._ws = None
        self._listen_key: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_msg_ts: float = 0.0
        self._last_keepalive_ts: float = 0.0
        self._seen_exec_ids: Set[str] = set()
        self._state: str = "DISCONNECTED"  # DISCONNECTED -> CONNECTED -> STALE -> RECONNECTING
        self.keepalive_interval_s = keepalive_interval_s

    # -- lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        if self._running:
            return
        self._running = True
        self._state = "CONNECTING"
        self._listen_key = await self._create_listen_key()
        self._ws = await self._connect_ws(self._listen_key)
        self._last_msg_ts = time.time()
        self._last_keepalive_ts = time.time()
        self._state = "CONNECTED"
        self._task = asyncio.create_task(self._recv_loop(), name="binance-user-stream")
        self._keepalive_task = asyncio.create_task(self._keepalive_loop(), name="binance-keepalive")
        logger.info("user_stream_connected", listen_key=self._listen_key[:8] + "...")

    async def disconnect(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._keepalive_task:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            self._keepalive_task = None
        if self._ws and hasattr(self._ws, "close"):
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        self._state = "DISCONNECTED"
        logger.info("user_stream_disconnected")

    async def reconnect(self) -> None:
        logger.info("user_stream_reconnecting")
        self._state = "RECONNECTING"
        await self.disconnect()
        # Brief backoff before reconnect (deterministic for tests via sleep mock).
        await asyncio.sleep(0.1)
        await self.connect()

    def is_stale(self, now: Optional[float] = None) -> bool:
        if not self._running or self._state != "CONNECTED":
            return True
        now = now if now is not None else time.time()
        return (now - self._last_msg_ts) > STALE_THRESHOLD_S

    def connection_state(self) -> str:
        return self._state

    # -- internals -----------------------------------------------------------

    async def _create_listen_key(self) -> str:
        if self._rest_factory is not None:
            rest = self._rest_factory(self.config)
            if hasattr(rest, "create_listen_key"):
                return await rest.create_listen_key() if asyncio.iscoroutinefunction(rest.create_listen_key) else rest.create_listen_key()
            if hasattr(rest, "post_user_data_stream"):
                return rest.post_user_data_stream()
        # Real path: use ccxt or direct REST. For testnet, ccxt's create_listen_key.
        # Lazy import to keep unit tests lightweight.
        try:
            import ccxt  # type: ignore
            ex = ccxt.binance({
                "apiKey": self.config.api_key,
                "secret": self.config.api_secret,
                "enableRateLimit": True,
            })
            ex.set_sandbox_mode(True)
            # ccxt's privatePostUserDataStream
            result = ex.private_post_user_data_stream()  # type: ignore
            return result["listenKey"]
        except Exception as e:
            raise RuntimeError("Failed to create user-data listenKey") from e

    async def _connect_ws(self, listen_key: str):
        if self._ws_factory is not None:
            return self._ws_factory(listen_key, self.config)
        # Real WebSocket via websockets library
        try:
            import websockets  # type: ignore
        except ImportError as e:
            raise ImportError("websockets is required for user-data stream") from e
        url = f"{self._get_ws_base()}/ws/{listen_key}"
        ws = await websockets.connect(url)  # type: ignore
        return ws

    def _get_ws_base(self) -> str:
        # Derive WS base from REST base (testnet only)
        base = self.config.base_url.lower()
        if "testnet.binance.vision" in base:
            return "wss://testnet.binance.vision"
        return "wss://testnet.binance.vision"

    async def _keepalive_loop(self) -> None:
        """Periodically PUT listenKey to prevent 60-min expiry.
        Failures mark STALE and will be handled via reconnect+reconcile."""
        try:
            while self._running:
                await asyncio.sleep(self.keepalive_interval_s)
                if not self._running or self._listen_key is None:
                    break
                try:
                    await self._keepalive_listen_key()
                    self._last_keepalive_ts = time.time()
                    logger.info("user_stream_keepalive", listen_key=self._listen_key[:8] + "...")
                except Exception as e:
                    logger.error("user_stream_keepalive_failed", error=str(e))
                    self._state = "STALE"
                    # Trigger reconnect via the same path as WS failure
                    # (caller will do REST reconcile)
                    break
        except asyncio.CancelledError:
            pass

    async def _keepalive_listen_key(self) -> None:
        if self._rest_factory is not None:
            rest = self._rest_factory(self.config)
            if hasattr(rest, "keepalive_listen_key"):
                r = rest.keepalive_listen_key(self._listen_key)
                if asyncio.iscoroutine(r):
                    await r
                return
            if hasattr(rest, "put_user_data_stream"):
                r = rest.put_user_data_stream(self._listen_key)
                if asyncio.iscoroutine(r):
                    await r
                return
        # Real path: ccxt PUT
        try:
            import ccxt  # type: ignore
            ex = ccxt.binance({
                "apiKey": self.config.api_key,
                "secret": self.config.api_secret,
                "enableRateLimit": True,
            })
            ex.set_sandbox_mode(True)
            ex.private_put_user_data_stream({"listenKey": self._listen_key})  # type: ignore
        except Exception as e:
            raise RuntimeError("keepalive failed") from e

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                self._last_msg_ts = time.time()
                await self._handle_raw(raw)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("user_stream_recv_error", error=str(e))
            self._state = "STALE"
            # Do not auto-reconnect here; caller (reconciliation layer)
            # decides to reconnect + REST-reconcile.

    async def _handle_raw(self, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        except Exception:
            logger.warning("user_stream_malformed", raw=str(raw)[:200])
            return

        # Binance user-data wraps executionReport under "e" == "executionReport"
        event_type = msg.get("e", "")
        if event_type != "executionReport":
            # Ignore non-execution events (outboundAccountPosition etc.)
            return

        # Deduplicate by execution id: t = tradeId or X = execution id
        exec_id = str(msg.get("t", msg.get("X", msg.get("i", ""))))
        dedup_key = f"{msg.get('c', '')}:{exec_id}"
        if dedup_key in self._seen_exec_ids:
            self._journal("execution_duplicate_suppressed", client_order_id=msg.get("c", ""), exec_id=exec_id)
            return
        self._seen_exec_ids.add(dedup_key)

        report = self._normalize_execution_report(msg)
        if report is None:
            return
        # Journal before callback (write-ahead)
        self._journal("execution_report_ws", client_order_id=report.client_order_id,
                      exchange_order_id=report.exchange_order_id, status=report.status)
        if self.on_execution_report:
            try:
                result = self.on_execution_report(report)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("execution_report_callback_failed", error=str(e))

    def _normalize_execution_report(self, msg: Dict[str, Any]):
        """
        Converts a Binance executionReport JSON into an ExecutionReport.
        Returns None if the message is not a fill we care about.
        """
        from ..core.types import ExecutionReport, OrderSide

        try:
            coid = str(msg.get("c", msg.get("C", "")))
            if not coid:
                return None
            symbol = str(msg.get("s", ""))
            side_raw = str(msg.get("S", "BUY")).upper()
            side = OrderSide.BUY if side_raw == "BUY" else OrderSide.SELL
            # X is execution type: NEW, PARTIAL_FILL, FILLED, CANCELED, REJECTED
            x = str(msg.get("X", msg.get("x", "NEW"))).upper()
            status_map = {
                "NEW": "NEW", "PARTIALLY_FILLED": "PARTIAL_FILL",
                "PARTIAL_FILL": "PARTIAL_FILL", "FILLED": "FILLED",
                "CANCELED": "CANCELED", "CANCELLED": "CANCELED",
                "REJECTED": "REJECTED", "EXPIRED": "CANCELED",
            }
            status = status_map.get(x)
            if status is None:
                logger.error("unknown_execution_status", status=x)
                return None
            # l = last executed qty, L = last price, z = cumulative filled, n = fee
            qty = to_decimal(str(msg.get("l", msg.get("z", 0)) or 0))
            # For NEW/CANCELED with no fill, qty will be 0
            price_raw = msg.get("L", msg.get("p", 0))
            price = to_decimal(str(price_raw or 0))
            fee = ZERO
            # Fee: try commission field 'n' if present (may be in different asset)
            if "n" in msg and msg["n"]:
                try:
                    fee = to_decimal(str(msg["n"]))
                except Exception:
                    fee = ZERO
            # Use orderId as exchange_order_id, executionId as dedup-augment
            ex_id = str(msg.get("i", msg.get("t", coid)))
            # Preserve executionId in exchange_order_id for dedup
            t = msg.get("t", "")
            if t:
                ex_id = f"{ex_id}:{t}"
            return ExecutionReport(
                client_order_id=coid,
                exchange_order_id=ex_id,
                symbol=symbol,
                side=side,
                status=status,  # type: ignore
                filled_quantity=qty,
                last_filled_price=price,
                remaining_quantity=ZERO,  # will be computed by Portfolio from filled
                timestamp=int(msg.get("E", int(time.time() * 1000))),
                fee=fee,
            )
        except Exception as e:
            logger.error("normalize_execution_report_failed", error=str(e))
            return None

    def _journal(self, event: str, **data) -> None:
        if self.journal is None:
            return
        try:
            from ..core.types import JournalEntry
            self.journal.append(JournalEntry(
                event_type="PACKET",
                timestamp=int(time.time() * 1000000),
                data={"source": "binance_user_stream", "event": event, **data},
            ))
        except Exception:
            pass

    def seed_seen_ids(self, ids) -> None:
        self._seen_exec_ids.update(ids)
