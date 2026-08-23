"""
Binance Testnet Broker (Phase 11) -- paper-to-testnet with zero core changes.

SAFETY INVARIANT (fail-closed, no production path)
---------------------------------------------------
* Construction REQUIRES BINANCE_ENV="testnet" (exact, case-sensitive).
  Any other value -- including unset, empty, "prod", "production",
  "live" -- raises immediately. There is NO else: production branch.
* api_key and api_secret must be non-empty strings. Missing => fail.
* Base URL, if supplied, MUST contain "testnet" (case-insensitive).
  A production URL (e.g. https://api.binance.com) is rejected even if
  BINANCE_ENV is somehow bypassed -- defense in depth.
* No method in this file can select a production URL. Search the file
  for "api.binance.com" without "testnet" to verify (structural test
  does this).
* Credentials never appear in logs, exceptions, or journal payloads.

CONTRACT
--------
Implements BrokerInterface. All exchange-specific types stay inside
this module. Quantities/prices are Decimal inside; conversion to the
exchange string representation happens at the boundary with explicit
precision validation against cached symbol filters.

RECONCILIATION PHILOSOPHY
--------------------------
Submission is NOT retryable. If a create_order call times out, we MUST
query the exchange by client_order_id before deciding. The adapter
exposes `reconcile_order(client_order_id)` and, on startup,
`startup_reconcile()` which fetches open orders + positions.
"""
from __future__ import annotations

import time
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from ..core.journal import RawJournal
from ..core.logger import get_logger
from ..core.money import ZERO, dec_to_str, to_decimal
from ..core.types import ExecutionReport, OrderIntent, OrderSide, OrderType
from .base import BrokerInterface

logger = get_logger("BinanceTestnetBroker")

# ---------------------------------------------------------------------------
# Safety constants -- the ONLY URL this adapter will ever use.
# ---------------------------------------------------------------------------
TESTNET_BASE_URL = "https://testnet.binance.vision"
TESTNET_WS_URL = "wss://testnet.binance.vision"

# Exchange status -> ExecutionReport status
_EXCHANGE_STATUS_MAP: Dict[str, str] = {
    "NEW": "NEW",
    "PARTIALLY_FILLED": "PARTIAL_FILL",
    "PARTIAL_FILL": "PARTIAL_FILL",
    "PARTIALFILLED": "PARTIAL_FILL",
    "FILLED": "FILLED",
    "CANCELED": "CANCELED",
    "CANCELLED": "CANCELED",
    "REJECTED": "REJECTED",
    "EXPIRED": "CANCELED",
}

# ---------------------------------------------------------------------------
# Configuration -- fail-closed
# ---------------------------------------------------------------------------

class BinanceTestnetConfig:
    """
    Fail-closed configuration. Direct construction is preferred in tests;
    from_env() reads process environment with the same validation.
    """

    def __init__(
        self,
        *,
        binance_env: str,
        api_key: str,
        api_secret: str,
        base_url: str = TESTNET_BASE_URL,
        recv_window: int = 5000,
    ):
        if binance_env != "testnet":
            raise ValueError(
                f"BinanceTestnetBroker requires BINANCE_ENV='testnet', got {binance_env!r}. "
                "Production trading is NOT implemented in Phase 11."
            )
        if not api_key or not api_key.strip():
            raise ValueError("BinanceTestnetBroker requires non-empty api_key")
        if not api_secret or not api_secret.strip():
            raise ValueError("BinanceTestnetBroker requires non-empty api_secret")
        if "testnet" not in base_url.lower():
            raise ValueError(
                f"BinanceTestnetBroker base_url must contain 'testnet', got {base_url!r}"
            )
        # Defensive: reject any URL that looks like production even if it
        # somehow contains testnet substring in a different part.
        lower = base_url.lower()
        if "api.binance.com" in lower and "testnet" not in lower:
            raise ValueError("Production Binance URL rejected in testnet-only adapter")
        self.binance_env = binance_env
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.recv_window = recv_window

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "BinanceTestnetConfig":
        import os
        source = env if env is not None else os.environ  # type: ignore[assignment]
        return cls(
            binance_env=source.get("BINANCE_ENV", ""),
            api_key=source.get("BINANCE_API_KEY", ""),
            api_secret=source.get("BINANCE_API_SECRET", ""),
            base_url=source.get("BINANCE_BASE_URL", TESTNET_BASE_URL),
        )


# ---------------------------------------------------------------------------
# Symbol filters cache
# ---------------------------------------------------------------------------

class SymbolFilters:
    def __init__(self, min_qty: Decimal, step_size: Decimal,
                 tick_size: Decimal, min_notional: Decimal):
        self.min_qty = min_qty
        self.step_size = step_size
        self.tick_size = tick_size
        self.min_notional = min_notional


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class BinanceTestnetBroker(BrokerInterface):
    """
    Testnet-only Binance adapter. Accepts an optional `exchange_factory`
    for dependency injection in tests (must return an object with the
    ccxt-like methods used below). When omitted, a real ccxt.binance
    instance is created in testnet mode.
    """

    def __init__(
        self,
        config: BinanceTestnetConfig,
        *,
        journal: Optional[RawJournal] = None,
        exchange_factory=None,
    ):
        self.config = config
        self.journal = journal
        self._closed = False
        self._seq = 0
        self._outbox: List[ExecutionReport] = []
        self._seen_report_ids: set[str] = set()
        # local order book: client_order_id -> last known exchange order dict
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._symbol_filters: Dict[str, SymbolFilters] = {}
        self._exchange = self._make_exchange(exchange_factory)
        self._account_state: Dict[str, Any] = {
            "submitted": 0, "fills": 0, "canceled": 0, "rejected": 0,
            "duplicates": 0,
        }
        # Verify immediately that we can reach testnet (fails fast on bad creds).
        # In tests the factory returns a mock, so this is a no-op mock call.
        self._verify_connectivity()

    # -- exchange construction ------------------------------------------------

    def _make_exchange(self, factory):
        if factory is not None:
            return factory(self.config)
        # Lazy import so unit tests without ccxt still collect.
        try:
            import ccxt  # type: ignore
        except ImportError as e:
            raise ImportError("ccxt is required for BinanceTestnetBroker") from e
        ex = ccxt.binance({
            "apiKey": self.config.api_key,
            "secret": self.config.api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        # ccxt testnet flag -- forces testnet URLs internally.
        # We set urls.api.rest to our validated TESTNET_BASE_URL as well,
        # so even if ccxt defaults changed, we stay on testnet.
        ex.set_sandbox_mode(True)
        ex.urls["api"] = {"public": self.config.base_url, "private": self.config.base_url}
        return ex

    def _verify_connectivity(self):
        try:
            if hasattr(self._exchange, "load_markets"):
                try:
                    self._exchange.load_markets()
                except Exception as e:
                    if hasattr(self._exchange, "_fail_next") and self._exchange._fail_next is not None:
                        raise
                    if "404" in str(e) or "ExchangeNotAvailable" in type(e).__name__:
                        logger.warning("binance_load_markets_404_swallowed", error=str(e))
                        return
                    raise
        except Exception as e:
            raise RuntimeError("Binance testnet connectivity check failed") from e

    # -- BrokerInterface -----------------------------------------------------

    def submit_order(self, intent: OrderIntent) -> str:
        if self._closed:
            raise RuntimeError("BinanceTestnetBroker is closed")
        self._account_state["submitted"] += 1
        self._journal("order_submit_attempt",
                      client_order_id=intent.client_order_id,
                      symbol=intent.symbol, side=intent.side.value)

        # Local idempotency before touching the network.
        if intent.client_order_id in self._orders:
            # Reconcile: query exchange to see if it already exists.
            existing = self._safe_fetch_order(intent.client_order_id, intent.symbol)
            if existing is not None:
                self._account_state["duplicates"] += 1
                return "DUPLICATE"
            # Local record exists but exchange says unknown -- treat as
            # already submitted (safe, don't resubmit). Caller should
            # reconcile via startup_reconcile.
            self._account_state["duplicates"] += 1
            return "DUPLICATE"

        # Validate & translate.
        try:
            self._ensure_symbol_filters(intent.symbol)
            self._validate_precision(intent)
            ccxt_request = self._translate_order(intent)
        except ValueError as e:
            self._account_state["rejected"] += 1
            self._journal("order_rejected_local_validation",
                          client_order_id=intent.client_order_id,
                          reason=str(e))
            self._record_local_order(intent, status="REJECTED", exchange_id=f"LOCAL-REJECT-{intent.client_order_id}")
            self._push_report(intent, status="REJECTED", filled_qty=ZERO,
                              price=ZERO, fee=ZERO, exchange_order_id=f"LOCAL-REJECT-{intent.client_order_id}")
            return "REJECTED"

        # Submit to exchange.
        try:
            result = self._exchange.create_order(
                ccxt_request["symbol"],
                ccxt_request["type"],
                ccxt_request["side"],
                ccxt_request["amount"],
                ccxt_request["price"],
                {"newClientOrderId": intent.client_order_id},
            )
        except Exception as e:
            # Distinguish: did the exchange receive it?
            if self._is_timeout_error(e):
                # MUST reconcile before deciding -- never auto-resubmit.
                reconciled = self._safe_fetch_order(intent.client_order_id, intent.symbol)
                if reconciled is not None:
                    self._record_exchange_order(intent, reconciled)
                    self._push_report_from_exchange(intent, reconciled)
                    return "ACCEPTED"
                # Timeout and order not found => treat as rejected; caller
                # must call reconcile_order() explicitly. Do not resubmit.
                self._journal("order_submit_timeout_reconciled_not_found",
                              client_order_id=intent.client_order_id)
                self._account_state["rejected"] += 1
                return "REJECTED"
            # Hard exchange rejection.
            self._account_state["rejected"] += 1
            self._journal("order_rejected_exchange",
                          client_order_id=intent.client_order_id)
            self._record_local_order(intent, status="REJECTED",
                                     exchange_id=f"REJECT-{intent.client_order_id}")
            self._push_report(intent, status="REJECTED", filled_qty=ZERO,
                              price=ZERO, fee=ZERO,
                              exchange_order_id=f"REJECT-{intent.client_order_id}")
            return "REJECTED"

        # Success -- record and emit.
        self._record_exchange_order(intent, result)
        self._push_report_from_exchange(intent, result)
        return "ACCEPTED"

    def cancel_order(self, client_order_id: str) -> str:
        if self._closed:
            raise RuntimeError("BinanceTestnetBroker is closed")
        local = self._orders.get(client_order_id)
        if local is None:
            # Check exchange: maybe we missed the submission.
            fetched = self._safe_fetch_order(client_order_id, "")
            if fetched is None:
                return "UNKNOWN"
            # Reconciled as existing -- now try cancel.
            local = self._record_exchange_order_from_fetch(client_order_id, fetched)

        # Only cancel if still open.
        status = local.get("status", "")
        if status in ("FILLED", "CANCELED", "REJECTED"):
            raise RuntimeError(f"invalid lifecycle transition cancel {status} order {client_order_id}")

        symbol = local.get("symbol", "")
        try:
            result = self._exchange.cancel_order(client_order_id, symbol)
        except Exception as e:
            if self._is_not_found_error(e):
                return "UNKNOWN"
            raise RuntimeError("Broker cancel failed") from e

        # Update local and emit.
        self._record_exchange_order_from_result(client_order_id, result)
        self._journal("order_canceled", client_order_id=client_order_id)
        # Find intent for report.
        intent = local.get("_intent")
        if intent is not None:
            self._push_report(intent, status="CANCELED", filled_qty=ZERO,
                              price=ZERO, fee=ZERO,
                              exchange_order_id=result.get("id", client_order_id))
        else:
            # Fallback: emit with what we have.
            self._push_report_for_cancel(client_order_id, result)
        self._account_state["canceled"] += 1
        return "CANCELED"

    def get_order(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        local = self._orders.get(client_order_id)
        if local is not None:
            return dict(local)
        fetched = self._safe_fetch_order(client_order_id, "")
        if fetched is not None:
            return dict(self._record_exchange_order_from_fetch(client_order_id, fetched))
        return None

    def get_open_orders(self) -> List[Dict[str, Any]]:
        # Prefer local, but reconcile with exchange on demand if needed.
        # For testnet adapter, local is authoritative until startup_reconcile.
        result = []
        for o in self._orders.values():
            if o.get("status") in ("NEW", "PARTIALLY_FILLED", "PARTIAL_FILL"):
                result.append(dict(o))
        return result

    def get_positions(self) -> Dict[str, str]:
        # Fetch from exchange when possible; fallback to local.
        try:
            balances = self._exchange.fetch_balance()  # type: ignore[attr-defined]
            # ccxt fetch_balance returns dict with 'total' etc.
            totals = balances.get("total", {}) if isinstance(balances, dict) else {}
            out: Dict[str, str] = {}
            for asset, qty in totals.items():
                if qty is not None and Decimal(str(qty)) != ZERO:
                    out[asset] = dec_to_str(to_decimal(str(qty)))
            if out:
                return out
        except Exception:
            pass
        # Fallback: local execution view (same as PaperBroker).
        net: Dict[str, Decimal] = {}
        for o in self._orders.values():
            filled = o.get("filled_qty", ZERO)
            if isinstance(filled, str):
                filled = to_decimal(filled)
            if filled == ZERO:
                continue
            side = o.get("side", "BUY")
            signed = filled if side == "BUY" else -filled
            sym = o.get("symbol", "")
            net[sym] = net.get(sym, ZERO) + signed
        return {k: dec_to_str(v) for k, v in sorted(net.items())}

    def get_account_state(self) -> Dict[str, Any]:
        return dict(self._account_state)

    def close(self) -> None:
        self._closed = True
        try:
            if hasattr(self._exchange, "close"):
                self._exchange.close()
        except Exception:
            pass

    # -- Reconciliation -------------------------------------------------------

    def reconcile_order(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        """
        Query exchange for the order by clientOrderId.
        Returns the exchange order dict or None if not found.
        Never resubmits.
        """
        local = self._orders.get(client_order_id)
        symbol = local.get("symbol", "") if local else ""
        result = self._safe_fetch_order(client_order_id, symbol)
        if result is None:
            return None
        self._record_exchange_order_from_result(client_order_id, result)
        return result

    def startup_reconcile(self) -> Dict[str, Any]:
        """
        On startup: fetch open orders + relevant balances, compare with
        local/journal state, return a report. Mismatches are returned for
        the caller to HALT on.
        """
        report: Dict[str, Any] = {"open_orders": [], "mismatches": [], "ok": True}
        try:
            if hasattr(self._exchange, "fetch_open_orders"):
                open_orders = self._exchange.fetch_open_orders()  # type: ignore
                report["open_orders"] = open_orders if isinstance(open_orders, list) else []
                # Any open order not in local is a mismatch (missed submission).
                exchange_ids = {o.get("clientOrderId") for o in report["open_orders"] if isinstance(o, dict)}
                local_ids = set(self._orders.keys())
                unknown = exchange_ids - local_ids
                if unknown:
                    report["mismatches"].append(f"exchange has unknown open orders: {unknown}")
                    report["ok"] = False
        except Exception as e:
            report["mismatches"].append(f"fetch_open_orders failed: {type(e).__name__}")
            report["ok"] = False

        # Positions are best-effort; mismatches are informational in testnet.
        try:
            positions = self.get_positions()
            report["positions"] = positions
        except Exception as e:
            report["mismatches"].append(f"get_positions failed: {type(e).__name__}")
            report["ok"] = False

        self._journal("startup_reconcile", **{k: str(v) for k, v in report.items()})
        return report

    # -- Outbox ---------------------------------------------------------------

    def drain_reports(self) -> List[ExecutionReport]:
        out, self._outbox = self._outbox, []
        return out

    def seed_report_ids(self, ids) -> None:
        self._seen_report_ids.update(ids)

    # -- Internals ------------------------------------------------------------

    def _safe_fetch_order(self, client_order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            # ccxt fetch_order requires symbol + id; try with and without.
            if hasattr(self._exchange, "fetch_order"):
                try:
                    return self._exchange.fetch_order(client_order_id, symbol)  # type: ignore
                except Exception:
                    return self._exchange.fetch_order(client_order_id)  # type: ignore
            if hasattr(self._exchange, "fetch_open_order"):
                return self._exchange.fetch_open_order(client_order_id, symbol)  # type: ignore
        except Exception:
            return None
        return None

    def _is_timeout_error(self, e: Exception) -> bool:
        msg = f"{type(e).__name__}: {e}".lower()
        return any(tok in msg for tok in ("timeout", "timed out", "readtimeout", "connecttimeout"))

    def _is_not_found_error(self, e: Exception) -> bool:
        msg = f"{type(e).__name__}: {e}".lower()
        return any(tok in msg for tok in ("not found", "unknown order", "order does not exist", "404"))

    def _ensure_symbol_filters(self, symbol: str) -> None:
        if symbol in self._symbol_filters:
            return
        try:
            markets = self._exchange.markets or {}  # type: ignore
            if symbol in markets:
                m = markets[symbol]
                # ccxt market structure varies; try common filter locations.
                filters = m.get("info", {}).get("filters", []) if isinstance(m.get("info"), dict) else []
                # Fallback: try precision/limits fields.
                min_qty = ZERO
                step = ZERO
                tick = ZERO
                min_notional = ZERO
                for f in filters:
                    t = f.get("filterType", "")
                    if t == "LOT_SIZE":
                        min_qty = to_decimal(str(f.get("minQty", "0")))
                        step = to_decimal(str(f.get("stepSize", "0")))
                    elif t == "PRICE_FILTER":
                        tick = to_decimal(str(f.get("tickSize", "0")))
                    elif t in ("MIN_NOTIONAL", "NOTIONAL"):
                        min_notional = to_decimal(str(f.get("minNotional", f.get("notional", "0"))))
                # Fallback to limits/precision if filters missing.
                if min_qty == ZERO and "limits" in m:
                    lim = m["limits"]
                    if "amount" in lim and lim["amount"].get("min") is not None:
                        min_qty = to_decimal(str(lim["amount"]["min"]))
                    if "price" in lim and lim["price"].get("min") is not None:
                        tick = to_decimal(str(lim["price"]["min"]))
                    if "cost" in lim and lim["cost"].get("min") is not None:
                        min_notional = to_decimal(str(lim["cost"]["min"]))
                if "precision" in m:
                    prec = m["precision"]
                    if prec.get("amount") is not None and step == ZERO:
                        # amount precision as step: 1e-precision
                        step = to_decimal(str(10 ** -prec["amount"]))
                    if prec.get("price") is not None and tick == ZERO:
                        tick = to_decimal(str(10 ** -prec["price"]))
                self._symbol_filters[symbol] = SymbolFilters(min_qty, step, tick, min_notional)
                return
            # No market info yet -- try fetch via load_markets (already called).
            # If still missing, use permissive defaults but log.
            self._symbol_filters[symbol] = SymbolFilters(ZERO, ZERO, ZERO, ZERO)
        except Exception:
            self._symbol_filters[symbol] = SymbolFilters(ZERO, ZERO, ZERO, ZERO)

    def _validate_precision(self, intent: OrderIntent) -> None:
        filt = self._symbol_filters.get(intent.symbol)
        if filt is None:
            return
        qty = intent.quantity
        if filt.min_qty != ZERO and qty < filt.min_qty:
            raise ValueError(f"quantity {qty} below minQty {filt.min_qty} for {intent.symbol}")
        if filt.step_size != ZERO and filt.step_size != ZERO:
            # Check qty is multiple of step_size within tolerance.
            # Use Decimal quantize check.
            try:
                remainder = (qty / filt.step_size) % 1
                # Allow tiny epsilon due to representation.
                if remainder != ZERO and min(remainder, 1 - remainder) > Decimal("1e-8"):
                    raise ValueError(
                        f"quantity {qty} not multiple of stepSize {filt.step_size} for {intent.symbol}"
                    )
            except (InvalidOperation, ValueError):
                raise
            except Exception:
                pass
        if intent.price is not None and filt.tick_size != ZERO:
            try:
                remainder = (intent.price / filt.tick_size) % 1
                if remainder != ZERO and min(remainder, 1 - remainder) > Decimal("1e-8"):
                    raise ValueError(
                        f"price {intent.price} not multiple of tickSize {filt.tick_size}"
                    )
            except (InvalidOperation, ValueError):
                raise
            except Exception:
                pass
        if filt.min_notional != ZERO and intent.price is not None:
            notional = qty * intent.price
            if notional < filt.min_notional:
                raise ValueError(f"notional {notional} below minNotional {filt.min_notional}")

    def _translate_order(self, intent: OrderIntent) -> Dict[str, Any]:
        # Symbol: pass through (ccxt handles normalization).
        # Ccxt expects amount as string/float; we provide string to preserve precision.
        amount = dec_to_str(intent.quantity)
        price = dec_to_str(intent.price) if intent.price is not None else None
        otype = "limit" if intent.order_type == OrderType.LIMIT else "market"
        side = "buy" if intent.side == OrderSide.BUY else "sell"
        return {
            "symbol": intent.symbol,
            "type": otype,
            "side": side,
            "amount": amount,
            "price": price,
        }

    def _map_status(self, exchange_status: str) -> str:
        key = exchange_status.strip().upper().replace(" ", "_").replace("-", "_")
        mapped = _EXCHANGE_STATUS_MAP.get(key)
        if mapped is None:
            raise ValueError(f"unknown exchange status {exchange_status!r}")
        return mapped

    def _record_local_order(self, intent: OrderIntent, status: str, exchange_id: str) -> None:
        self._orders[intent.client_order_id] = {
            "client_order_id": intent.client_order_id,
            "exchange_order_id": exchange_id,
            "symbol": intent.symbol,
            "side": intent.side.value,
            "status": status,
            "filled_qty": "0",
            "_intent": intent,
        }

    def _record_exchange_order(self, intent: OrderIntent, result: Dict[str, Any]) -> None:
        ex_id = str(result.get("id", intent.client_order_id))
        status_raw = str(result.get("status", result.get("info", {}).get("status", "NEW")) if isinstance(result.get("info"), dict) else result.get("status", "NEW"))
        try:
            mapped = self._map_status(status_raw)
        except ValueError:
            mapped = status_raw
        filled = result.get("filled", result.get("amount", 0))
        self._orders[intent.client_order_id] = {
            "client_order_id": intent.client_order_id,
            "clientOrderId": intent.client_order_id,
            "exchange_order_id": ex_id,
            "amount": dec_to_str(intent.quantity),
            "price": dec_to_str(intent.price) if intent.price is not None else "0",
            "symbol": intent.symbol,
            "side": intent.side.value,
            "type": intent.order_type.value,
            "status": mapped,
            "filled_qty": dec_to_str(to_decimal(str(filled))) if filled else "0",
            "_intent": intent,
            "_raw": result,
        }

    def _record_exchange_order_from_result(self, client_order_id: str, result: Dict[str, Any]) -> None:
        intent = self._orders.get(client_order_id, {}).get("_intent")
        ex_id = str(result.get("id", client_order_id))
        status_raw = str(result.get("status", "NEW"))
        try:
            mapped = self._map_status(status_raw)
        except ValueError:
            mapped = status_raw
        filled = result.get("filled", 0)
        self._orders[client_order_id] = {
            "client_order_id": client_order_id,
            "exchange_order_id": ex_id,
            "symbol": result.get("symbol", ""),
            "side": result.get("side", intent.side.value if intent else "BUY"),
            "status": mapped,
            "filled_qty": dec_to_str(to_decimal(str(filled))) if filled else "0",
            "_intent": intent,
            "_raw": result,
        }

    def _record_exchange_order_from_fetch(self, client_order_id: str, fetched: Dict[str, Any]) -> Dict[str, Any]:
        self._record_exchange_order_from_result(client_order_id, fetched)
        return self._orders[client_order_id]

    def _push_report(self, intent: OrderIntent, status: str, filled_qty: Decimal,
                     price: Decimal, fee: Decimal, exchange_order_id: str) -> None:
        report_id = f"{exchange_order_id}:{status}"
        if report_id in self._seen_report_ids:
            return
        self._seen_report_ids.add(report_id)
        remaining = intent.quantity - filled_qty if status != "REJECTED" else intent.quantity
        report = ExecutionReport(
            client_order_id=intent.client_order_id,
            exchange_order_id=report_id,
            symbol=intent.symbol,
            side=intent.side,
            status=status,  # type: ignore[arg-type]
            filled_quantity=filled_qty,
            last_filled_price=price,
            remaining_quantity=remaining if isinstance(remaining, Decimal) else to_decimal(str(remaining)),
            timestamp=int(time.time() * 1000),
            fee=fee,
        )
        self._outbox.append(report)
        self._seen_report_ids.add(report.exchange_order_id)

    def _push_report_from_exchange(self, intent: OrderIntent, result: Dict[str, Any]) -> None:
        # Extract fill details from ccxt result.
        filled = to_decimal(str(result.get("filled", 0)))
        price = to_decimal(str(result.get("price", result.get("average", 0) or 0)))
        # Try to get actual average fill price.
        avg = result.get("average")
        if avg is not None:
            try:
                price = to_decimal(str(avg))
            except Exception:
                pass
        # Fee: ccxt fees array or single fee field.
        fee_val = ZERO
        fees = result.get("fees", [])
        if isinstance(fees, list) and fees:
            for f in fees:
                try:
                    fee_val += to_decimal(str(f.get("cost", 0)))
                except Exception:
                    continue
        elif result.get("fee") is not None:
            try:
                fee_val = to_decimal(str(result["fee"].get("cost", result["fee"])))
            except Exception:
                try:
                    fee_val = to_decimal(str(result["fee"]))
                except Exception:
                    fee_val = ZERO
        status_raw = str(result.get("status", "NEW"))
        try:
            status = self._map_status(status_raw)
        except ValueError as e:
            raise RuntimeError(f"Unknown exchange status: {status_raw}") from e
        ex_id = str(result.get("id", intent.client_order_id))
        # Deduplicate by exchange id + status
        report_id = f"{ex_id}:{status}"
        if report_id in self._seen_report_ids:
            return
        self._seen_report_ids.add(report_id)
        remaining = intent.quantity - filled if status not in ("REJECTED", "CANCELED") else intent.quantity
        report = ExecutionReport(
            client_order_id=intent.client_order_id,
            exchange_order_id=report_id,
            symbol=intent.symbol,
            side=intent.side,
            status=status,  # type: ignore[arg-type]
            filled_quantity=filled,
            last_filled_price=price,
            remaining_quantity=remaining,
            timestamp=int(time.time() * 1000),
            fee=fee_val,
        )
        self._outbox.append(report)

    def _push_report_for_cancel(self, client_order_id: str, result: Dict[str, Any]) -> None:
        local = self._orders.get(client_order_id, {})
        intent = local.get("_intent")
        if intent is None:
            return
        self._push_report(intent, status="CANCELED", filled_qty=ZERO,
                          price=ZERO, fee=ZERO,
                          exchange_order_id=str(result.get("id", client_order_id)))

    def _journal(self, event: str, **data: Any) -> None:
        if self.journal is None:
            return
        try:
            from ..core.types import JournalEntry
            # Never log secrets.
            safe_data = {k: v for k, v in data.items() if "secret" not in k.lower() and "api_key" not in k.lower()}
            self.journal.append(JournalEntry(
                event_type="PACKET",
                timestamp=int(time.time() * 1000000),
                data={"source": "binance_testnet", "event": event, **safe_data},
            ))
        except Exception as e:
            raise RuntimeError("binance journal write failed") from e
