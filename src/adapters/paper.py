"""
PaperBroker (Phase 10) -- a deterministic simulated execution venue.

BEHAVIOR CONTRACT (documented, tested)
--------------------------------------
LIFECYCLE
  NEW -> PARTIALLY_FILLED -> FILLED
  NEW -> CANCELED ; PARTIALLY_FILLED -> CANCELED
  -> REJECTED (submission-time only)
Invalid transitions raise RuntimeError (fail loudly, never swallowed).

FILL MODEL
  MARKET orders: work at the NEXT on_market_price() call after
    submission (one-tick latency, modeled explicitly). Fill price =
    adverse-slipped reference via CostModel.fill_price (taker fee).
  LIMIT BUY: rests until market price <= limit; fills at
    min(price, limit) -- NEVER above the limit. Maker fee, no slippage.
  LIMIT SELL: rests until market price >= limit; fills at
    max(price, limit) -- NEVER below the limit. Maker fee, no slippage.
  PARTIALS: an explicit fill schedule of absolute quantities per order
    (e.g. 1.0 -> [0.3, 0.4, 0.3]); one chunk per price event; when the
    schedule is exhausted any remainder sweeps on the next event.

IDEMPOTENCY
  - Duplicate client_order_id submission returns "DUPLICATE" and never
    re-executes.
  - Every fill carries a unique exec id "<exchange_order_id>:<fill_no>";
    a repeated report id is journaled as duplicate_suppressed and
    dropped -- it can never double-count anywhere downstream.

MONEY RULES
  The broker NEVER holds a Portfolio reference and cannot mutate one.
  It emits ExecutionReports into an outbox; the engine drains them
  through the single mutation funnel apply_execution_report().
  get_positions()/get_account_state() are EXECUTION VIEWS only.

JOURNAL
  Every transition is appended (PACKET / source="paper_broker") so a
  session is fully reconstructable: rebuild_from_journal() restores
  order states, dedup registries and the id counter.
"""
import json
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel

from ..core.clock import Clock
from ..core.costs import CostModel
from ..core.journal import JournalEntry, RawJournal
from ..core.money import ZERO, dec_to_str, to_decimal
from ..core.types import ExecutionReport, OrderIntent, OrderSide, OrderType
from .base import BrokerInterface


class PaperOrderState(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


_ALLOWED_TRANSITIONS = {
    PaperOrderState.NEW: {
        PaperOrderState.PARTIALLY_FILLED, PaperOrderState.FILLED,
        PaperOrderState.CANCELED},
    PaperOrderState.PARTIALLY_FILLED: {
        PaperOrderState.PARTIALLY_FILLED, PaperOrderState.FILLED,
        PaperOrderState.CANCELED},
    PaperOrderState.FILLED: set(),
    PaperOrderState.CANCELED: set(),
    PaperOrderState.REJECTED: set(),
}


@dataclass
class _PaperOrder:
    intent: OrderIntent
    status: PaperOrderState
    exchange_order_id: str
    filled_qty: Decimal = ZERO
    fees_paid: Decimal = ZERO
    next_chunk: int = 0            # index into fill schedule
    report_ids: set = field(default_factory=set)

    @property
    def remaining(self) -> Decimal:
        return self.intent.quantity - self.filled_qty


class FillSchedule(BaseModel):
    """Absolute per-event fill quantities (canonical strings).
    Empty tuple => single full fill on first eligible event. Once the
    schedule is exhausted, any remainder sweeps on the next event."""
    chunks: Tuple[str, ...] = ()

    def chunk_qty(self, order: _PaperOrder) -> Optional[Decimal]:
        if self.chunks:
            if order.next_chunk < len(self.chunks):
                return to_decimal(self.chunks[order.next_chunk])
            return order.remaining          # schedule done -> sweep remainder
        return order.remaining              # default: full immediate fill


class PaperBroker(BrokerInterface):

    def __init__(self, cost_model: CostModel,
                 journal: Optional[RawJournal] = None,
                 fill_schedule: FillSchedule | None = None):
        self.cost_model = cost_model
        self.journal = journal
        self.fill_schedule = fill_schedule or FillSchedule()
        self._orders: Dict[str, _PaperOrder] = {}
        self._outbox: List[ExecutionReport] = []
        self._seen_report_ids: set[str] = set()
        self._seq = 0
        self._closed = False
        # execution-view counters (NOT accounting)
        self.stats = {"submitted": 0, "accepted": 0, "rejected": 0,
                      "duplicates": 0, "fills": 0, "partial_fills": 0,
                      "canceled": 0, "fees_charged": ZERO,
                      "notional_executed": ZERO}

    # ------------------------------------------------------------------
    # BrokerInterface
    # ------------------------------------------------------------------

    def submit_order(self, intent: OrderIntent) -> str:
        if self._closed:
            raise RuntimeError("PaperBroker is closed")
        if intent.client_order_id in self._orders:
            self.stats["duplicates"] += 1
            self._journal("duplicate_suppressed",
                          client_order_id=intent.client_order_id)
            return "DUPLICATE"

        self.stats["submitted"] += 1
        self._journal("order_submitted", client_order_id=intent.client_order_id,
                      symbol=intent.symbol, side=intent.side.value,
                      quantity=dec_to_str(intent.quantity),
                      order_type=intent.order_type.value,
                      intent=intent.model_dump(mode="json"))

        if intent.quantity <= ZERO or intent.quantity < self.cost_model.min_order_qty:
            return self._reject(intent, f"invalid quantity "
                                        f"{dec_to_str(intent.quantity)}")

        self._seq += 1
        ex_id = f"PAPER-{self._seq}"
        order = _PaperOrder(intent=intent, status=PaperOrderState.NEW,
                            exchange_order_id=ex_id)
        self._orders[intent.client_order_id] = order
        self.stats["accepted"] += 1
        self._journal("order_accepted", client_order_id=intent.client_order_id,
                      exchange_order_id=ex_id)
        self._emit(order, status="NEW", qty=ZERO,
                   px=ZERO, fee=ZERO, ts_us=intent.timestamp)
        return "ACCEPTED"

    def cancel_order(self, client_order_id: str) -> str:
        order = self._orders.get(client_order_id)
        if order is None:
            return "UNKNOWN"
        if order.status not in (PaperOrderState.NEW,
                                PaperOrderState.PARTIALLY_FILLED):
            raise RuntimeError(
                f"invalid transition: cancel {order.status.value} "
                f"order {client_order_id}")
        self._transition(order, PaperOrderState.CANCELED)
        self.stats["canceled"] += 1
        self._journal("order_canceled", client_order_id=client_order_id)
        self._emit(order, status="CANCELED", qty=ZERO, px=ZERO, fee=ZERO,
                   ts_us=Clock.now_epoch_us())
        return "CANCELED"

    def get_order(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        o = self._orders.get(client_order_id)
        return self._order_view(o) if o else None

    def get_open_orders(self) -> List[Dict[str, Any]]:
        return [self._order_view(o) for o in self._orders.values()
                if o.status in (PaperOrderState.NEW,
                                PaperOrderState.PARTIALLY_FILLED)]

    def get_positions(self) -> Dict[str, str]:
        """EXECUTION VIEW: net filled quantity per symbol from THIS venue."""
        net: Dict[str, Decimal] = {}
        for o in self._orders.values():
            if o.status in (PaperOrderState.FILLED,
                            PaperOrderState.PARTIALLY_FILLED):
                signed = (o.filled_qty if o.intent.side == OrderSide.BUY
                          else -o.filled_qty)
                net[o.intent.symbol] = net.get(o.intent.symbol, ZERO) + signed
        return {k: dec_to_str(v) for k, v in sorted(net.items())}

    def get_account_state(self) -> Dict[str, Any]:
        s = dict(self.stats)
        for k in ("fees_charged",):
            s[k] = dec_to_str(s[k])
        s["open_orders"] = len(self.get_open_orders())
        return s

    def close(self):
        self._closed = True

    # ------------------------------------------------------------------
    # Market-data driven fills
    # ------------------------------------------------------------------

    def on_market_price(self, symbol: str, price: Decimal, ts_us: int) -> int:
        """
        Works all open orders on `symbol` against ONE observed price.
        Returns number of fills emitted this tick.
        """
        if self._closed:
            raise RuntimeError("PaperBroker is closed")
        fills = 0
        for cloid, order in list(self._orders.items()):
            if order.intent.symbol != symbol:
                continue
            if order.status not in (PaperOrderState.NEW,
                                    PaperOrderState.PARTIALLY_FILLED):
                continue                              # terminal: nothing to do
            if order.status == PaperOrderState.NEW \
                    and order.intent.order_type == OrderType.MARKET:
                fills += self._work_market(order, price, ts_us)
            elif order.intent.order_type == OrderType.LIMIT:
                fills += self._work_limit(order, price, ts_us)
        return fills

    # ------------------------------------------------------------------
    # Outbox + introspection helpers used by the engine
    # ------------------------------------------------------------------

    def drain_reports(self) -> List[ExecutionReport]:
        out, self._outbox = self._outbox, []
        return out

    def seed_report_ids(self, ids) -> None:
        """Restart recovery: pre-load known exec ids so replayed reports
        can never double-count."""
        self._seen_report_ids.update(ids)

    @classmethod
    def rebuild_from_journal(cls, path: str, cost_model: CostModel,
                             fill_schedule: "FillSchedule | None" = None):
        """
        Reconstructs broker state from its journal entries:
        open/partial orders (resting work continues after restart),
        terminal states, dedup registry, id counter and stats.
        Previously emitted reports are NOT re-emitted.
        """
        broker = cls(cost_model, fill_schedule=fill_schedule)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                d = entry.get("data", {})
                if d.get("source") != "paper_broker":
                    continue
                kind = d.get("event")
                cloid = d.get("client_order_id")

                if kind == "order_submitted":
                    broker.stats["submitted"] += 1
                    intent = OrderIntent.model_validate_json(
                        json.dumps(d["intent"]))
                    broker._orders[cloid] = _PaperOrder(
                        intent=intent, status=PaperOrderState.NEW,
                        exchange_order_id="PENDING")
                elif kind == "order_accepted":
                    broker.stats["accepted"] += 1
                    ex = d["exchange_order_id"]
                    o = broker._orders.get(cloid)
                    if o is not None:
                        o.exchange_order_id = ex
                    seq = int(ex.split("-")[1])
                    broker._seq = max(broker._seq, seq)
                elif kind == "partial_fill":
                    broker.stats["partial_fills"] += 1
                    o = broker._orders.get(cloid)
                    if o is not None:
                        broker._transition(o, PaperOrderState.PARTIALLY_FILLED)
                        o.filled_qty += to_decimal(d["quantity"])
                        o.fees_paid += to_decimal(d["fee"])
                        o.next_chunk += 1
                elif kind == "full_fill":
                    broker.stats["fills"] += 1
                    o = broker._orders.get(cloid)
                    if o is not None:
                        broker._transition(o, PaperOrderState.FILLED)
                        o.filled_qty += to_decimal(d["quantity"])
                        o.fees_paid += to_decimal(d["fee"])
                        o.next_chunk += 1
                elif kind == "order_rejected":
                    broker.stats["rejected"] += 1
                    if cloid in broker._orders:
                        broker._transition(broker._orders[cloid],
                                           PaperOrderState.REJECTED)
                elif kind == "order_canceled":
                    broker.stats["canceled"] += 1
                    o = broker._orders.get(cloid)
                    if o is not None and o.status != PaperOrderState.FILLED:
                        broker._transition(o, PaperOrderState.CANCELED)
                elif kind == "report_emitted":
                    rid = d["report_id"]
                    if rid in broker._seen_report_ids:
                        broker.stats["duplicates"] += 1
                        continue
                    broker._seen_report_ids.add(rid)

        # fees_charged from per-order accumulation (exact Decimal replay)
        total_fees = sum((o.fees_paid for o in broker._orders.values()),
                         ZERO)
        broker.stats["fees_charged"] = total_fees
        return broker

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _work_market(self, order: _PaperOrder, price: Decimal,
                     ts_us: int) -> int:
        side = order.intent.side.value
        px = self.cost_model.fill_price(side, price)          # adverse slip
        return self._apply_fill(order, px, ts_us, maker=False)

    def _work_limit(self, order: _PaperOrder, price: Decimal,
                    ts_us: int) -> int:
        limit = order.intent.price
        if limit is None:
            return 0
        if order.intent.side == OrderSide.BUY:
            if price > limit:
                return 0                                      # not crossed
            px = min(price, limit)                            # <= limit
        else:
            if price < limit:
                return 0
            px = max(price, limit)                            # >= limit
        return self._apply_fill(order, px, ts_us, maker=True)

    def _apply_fill(self, order: _PaperOrder, px: Decimal, ts_us: int,
                    maker: bool) -> int:
        chunk = self.fill_schedule.chunk_qty(order)
        if chunk is None or order.remaining <= ZERO:
            return 0
        qty = min(chunk, order.remaining)

        self._transition(order, PaperOrderState.PARTIALLY_FILLED)
        order.filled_qty += qty
        order.next_chunk += 1
        notional = qty * px
        fee = self.cost_model.fee(notional, side_is_maker=maker)
        order.fees_paid += fee

        terminal = order.remaining == ZERO
        if terminal:
            self._transition(order, PaperOrderState.FILLED)

        self.stats["fills" if terminal else "partial_fills"] += 1
        self.stats["fees_charged"] += fee
        self.stats["notional_executed"] += notional

        status = ("FILLED" if terminal else "PARTIAL_FILL")
        self._emit(order, status=status, qty=qty, px=px, fee=fee,
                   ts_us=ts_us, maker=maker)
        self._journal("full_fill" if terminal else "partial_fill",
                      client_order_id=order.intent.client_order_id,
                      quantity=dec_to_str(qty), price=dec_to_str(px),
                      fee=dec_to_str(fee))
        return 1

    def _reject(self, intent: OrderIntent, reason: str) -> str:
        self._seq += 1
        ex_id = f"PAPER-{self._seq}"
        order = _PaperOrder(intent=intent, status=PaperOrderState.REJECTED,
                            exchange_order_id=ex_id)
        self._orders[intent.client_order_id] = order
        self.stats["rejected"] += 1
        self._journal("order_rejected", client_order_id=intent.client_order_id,
                      reason=reason)
        self._emit(order, status="REJECTED", qty=ZERO, px=ZERO, fee=ZERO,
                   ts_us=intent.timestamp)
        return "REJECTED"

    def _transition(self, order: _PaperOrder, target: PaperOrderState):
        if target not in _ALLOWED_TRANSITIONS[order.status]:
            raise RuntimeError(
                f"invalid lifecycle transition {order.status.value} -> "
                f"{target.value} for {order.intent.client_order_id}")
        order.status = target

    def _emit(self, order: _PaperOrder, *, status: str, qty: Decimal,
              px: Decimal, fee: Decimal, ts_us: int, maker: bool = False):
        fill_no = order.next_chunk if qty > ZERO else 0
        report_id = f"{order.exchange_order_id}:{status}:{fill_no}"
        if report_id in self._seen_report_ids:
            self.stats["duplicates"] += 1
            self._journal("duplicate_suppressed", report_id=report_id)
            return
        self._seen_report_ids.add(report_id)

        report = ExecutionReport(
            client_order_id=order.intent.client_order_id,
            exchange_order_id=report_id,
            symbol=order.intent.symbol,
            side=order.intent.side,
            status=status,
            filled_quantity=qty,
            last_filled_price=px,
            remaining_quantity=dec_to_str(order.remaining),
            timestamp=ts_us,
            fee=fee,
        )
        self._outbox.append(report)
        self._seen_report_ids.add(report.exchange_order_id)
        self._journal("report_emitted", report_id=report.exchange_order_id,
                      status=status, client_order_id=report.client_order_id)

    def _journal(self, event: str, **data):
        if self.journal is None:
            return
        try:
            self.journal.append(JournalEntry(
                event_type="PACKET",
                timestamp=Clock.now_epoch_us(),
                data={"source": "paper_broker", "event": event, **data},
            ))
        except Exception as e:                     # journal failure is fatal
            raise RuntimeError(f"paper-broker journal write failed: {e}") from e

    @staticmethod
    def _order_view(o: _PaperOrder) -> Dict[str, Any]:
        return {
            "client_order_id": o.intent.client_order_id,
            "exchange_order_id": o.exchange_order_id,
            "symbol": o.intent.symbol,
            "side": o.intent.side.value,
            "type": o.intent.order_type.value,
            "quantity": dec_to_str(o.intent.quantity),
            "price": None if o.intent.price is None else dec_to_str(o.intent.price),
            "status": o.status.value,
            "filled_quantity": dec_to_str(o.filled_qty),
        }
