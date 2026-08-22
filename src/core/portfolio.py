"""
OmniTrade Portfolio & Accounting (Phase 5).

SINGLE financial truth for every execution mode (replay / backtest /
paper / live). One implementation -- never forked per mode.

CORE INVARIANTS
---------------
1. Positions and cash change ONLY via ExecutionReport fills.
   OrderIntent, RiskDecision, Strategy signals, rejections, cancellations,
   and MarketEvents can NEVER mutate them. MarketEvent may only refresh
   price MARKS.
2. All arithmetic runs under the canonical money policy (core.money):
   Decimal everywhere, floats forbidden, fixed-point strings on the wire.
3. State is event-sourced: applying the same report sequence to the same
   starting state yields byte-identical snapshots/hashes (replay-safe).

ACCOUNTING MODEL (authoritative definitions)
--------------------------------------------
BUY fill : cash -= (qty * price) + fee      ; position increases
SELL fill: cash += (qty * price) - fee      ; position decreases
fees     : absolute quote-currency amounts carried ON the report;
           accumulated in fees_paid on every fill.
Avg entry: weighted average across adds (cost basis).
Close    : long  -> realized += (price - avg) * closed_qty
           short -> realized += (avg - price) * closed_qty
Reversal : closing portion realizes at the OLD avg; residual opens the
           opposite side at the FILL price. Going flat resets avg to 0.
NEW / CANCELED / REJECTED / zero-qty fills: ZERO accounting effect.

MARK-TO-MARKET
--------------
Marks come exclusively from MarketEvents (engine extracts venue prices).
A mark older than max_staleness_us (evaluated against an EXPLICIT,
caller-supplied now_us -- never wall-clock) is treated as UNPRICED:
unrealized PnL returns None, equity is flagged partial, peak/drawdown do
not update. Prices are never invented; flat positions are priced at 0.
"""
from typing import Dict, Optional, Tuple

from pydantic import BaseModel

from .money import Decimal, to_decimal, dec_to_str, ZERO
from .types import ExecutionReport, OrderSide


class EquityResult(BaseModel):
    """Outcome of an equity computation."""
    model_config = {"frozen": True}

    equity: Decimal
    fully_priced: bool
    unpriced_symbols: Tuple[str, ...]


class DrawdownInfo(BaseModel):
    """Current drawdown relative to peak equity."""
    model_config = {"frozen": True}

    peak_equity: Decimal
    equity: Decimal
    drawdown_abs: Decimal
    drawdown_pct: Decimal  # 0..100; 0 when peak <= 0 (undefined ratio)


class PositionState(BaseModel):
    """Per-symbol accounting state."""
    model_config = {"frozen": True}

    quantity: Decimal = ZERO            # signed; +long / -short / 0 flat
    avg_entry_price: Decimal = ZERO     # weighted cost basis; 0 when flat
    realized_pnl: Decimal = ZERO        # cumulative realized for THIS symbol


class _Mark(BaseModel):
    model_config = {"frozen": True}

    price: Decimal
    ts_us: int


class Portfolio:
    """
    Mutable aggregate root. Mutations enter through exactly two doors:
      - apply_report(report)  : fills (the ONLY position/cash mutator)
      - mark_price(...)       : market marks (never touches positions/cash)

    Derived views (unrealized/equity/drawdown/snapshot) are read-only and
    side-effect free except update_equity(), which advances peak/drawdown
    state -- call it explicitly from the engine/replay loop.
    """

    def __init__(self, starting_cash: Decimal, max_staleness_us: int = 30_000_000):
        self.starting_cash = to_decimal(starting_cash)
        self.cash = self.starting_cash
        self.max_staleness_us = int(max_staleness_us)

        self.positions: Dict[str, PositionState] = {}
        self.marks: Dict[str, _Mark] = {}

        self.fees_paid = ZERO
        self.realized_pnl = ZERO          # portfolio-wide accumulator
        self.peak_equity = self.starting_cash  # equity at t=0
        self.last_equity = self.starting_cash
        self.last_equity_fully_priced = True

    # ------------------------------------------------------------------
    # MUTATION DOOR 1: executions (the only position/cash mutator)
    # ------------------------------------------------------------------

    def apply_report(self, report: ExecutionReport) -> Optional[Decimal]:
        """
        Applies one ExecutionReport. Returns the signed quantity delta
        applied, or None when the report has NO accounting effect
        (non-fill statuses and zero-quantity fills).

        This is the single funnel through which fills touch the portfolio.
        """
        if report.status not in ("PARTIAL_FILL", "FILLED"):
            return None  # intentions/rejections never move money
        if report.filled_quantity == 0:
            return None  # zero-quantity rejection semantics

        delta = self._apply_fill(
            symbol=report.symbol,
            side=report.side,
            quantity=report.filled_quantity,
            price=report.last_filled_price,
            fee=report.fee,
        )
        return delta

    def _apply_fill(self, symbol: str, side: OrderSide,
                    quantity: Decimal, price: Decimal, fee: Decimal) -> Decimal:
        signed = quantity if side == OrderSide.BUY else -quantity

        # --- cash flow (quote currency) ---
        notional = quantity * price
        if side == OrderSide.BUY:
            self.cash -= notional + fee
        else:
            self.cash += notional - fee
        self.fees_paid += fee

        pos = self.positions.get(symbol, PositionState())
        realized_before = pos.realized_pnl
        new_pos = self._evolve_position(pos, signed, price)
        self.positions[symbol] = new_pos

        self.realized_pnl += new_pos.realized_pnl - realized_before
        return signed

    @staticmethod
    def _evolve_position(pos: PositionState, signed_qty: Decimal,
                         price: Decimal) -> PositionState:
        """
        Weighted-average cost basis with through-zero reversal support.
        Pure function -> trivially deterministic.
        """
        q = pos.quantity
        avg = pos.avg_entry_price
        realized = pos.realized_pnl

        if q == ZERO:
            # Opening (either direction) from flat.
            return PositionState(quantity=signed_qty, avg_entry_price=price,
                                 realized_pnl=realized)

        same_direction = (q > ZERO) == (signed_qty > ZERO)
        if same_direction:
            # Adding: new weighted average (basis is always positive;
            # divide by |total| -- signed total broke short-side adds).
            total = q + signed_qty
            new_avg = (abs(q) * avg + abs(signed_qty) * price) / abs(total)
            return PositionState(quantity=total, avg_entry_price=new_avg,
                                 realized_pnl=realized)

        # Reducing / closing / reversing.
        closed = min(abs(q), abs(signed_qty))
        pnl_per_unit = (price - avg) if q > ZERO else (avg - price)
        realized += pnl_per_unit * closed

        residual = signed_qty + q  # signed arithmetic through zero
        if residual == ZERO:
            # Fully flat: reset basis.
            return PositionState(quantity=ZERO, avg_entry_price=ZERO,
                                 realized_pnl=realized)
        if (residual > ZERO) == (q > ZERO):
            # Still same side, smaller position.
            return PositionState(quantity=residual, avg_entry_price=avg,
                                 realized_pnl=realized)
        # Reversed through zero: residual opens opposite side AT FILL PRICE.
        return PositionState(quantity=residual, avg_entry_price=price,
                             realized_pnl=realized)

    # ------------------------------------------------------------------
    # MUTATION DOOR 2: market marks (never touches positions/cash)
    # ------------------------------------------------------------------

    def mark_price(self, symbol: str, price: Decimal, ts_us: int) -> None:
        self.marks[symbol] = _Mark(price=to_decimal(price), ts_us=int(ts_us))

    def _is_mark_fresh(self, symbol: str, now_us: Optional[int]) -> bool:
        mark = self.marks.get(symbol)
        if mark is None:
            return False
        if now_us is None:
            return True  # no clock supplied => staleness check waived
        return (now_us - mark.ts_us) <= self.max_staleness_us

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    def unrealized_pnl(self, symbol: str, now_us: Optional[int] = None) -> Optional[Decimal]:
        """
        Mark-to-market PnL for one symbol.
        None when the position is open but UNPRICED (missing/stale mark).
        Exactly ZERO when flat -- never invented.
        """
        pos = self.positions.get(symbol)
        if pos is None or pos.quantity == ZERO:
            return ZERO
        if not self._is_mark_fresh(symbol, now_us):
            return None
        mark = self.marks[symbol].price
        if pos.quantity > ZERO:
            return (mark - pos.avg_entry_price) * pos.quantity
        return (pos.avg_entry_price - mark) * abs(pos.quantity)

    def equity(self, now_us: Optional[int] = None) -> EquityResult:
        """
        cash + market value of all OPEN, PRICED positions.
        Open positions without a fresh mark are excluded from the figure
        and reported via unpriced_symbols (partial valuation).
        """
        unpriced = []
        market_value = ZERO
        for sym, pos in self.positions.items():
            if pos.quantity == ZERO:
                continue
            pnl_or_none = self.unrealized_pnl(sym, now_us)
            if pnl_or_none is None:
                unpriced.append(sym)
                continue
            market_value += self.positions[sym].quantity * self.marks[sym].price
        unpriced_tuple = tuple(sorted(unpriced))
        return EquityResult(
            equity=self.cash + market_value,
            fully_priced=len(unpriced_tuple) == 0,
            unpriced_symbols=unpriced_tuple,
        )

    def update_equity(self, now_us: Optional[int] = None) -> EquityResult:
        """
        Computes equity and advances peak/drawdown bookkeeping.
        Peak updates ONLY on fully-priced observations so a partial
        valuation can never corrupt the high-water mark.
        """
        result = self.equity(now_us)
        self.last_equity = result.equity
        self.last_equity_fully_priced = result.fully_priced
        if result.fully_priced and result.equity > self.peak_equity:
            self.peak_equity = result.equity
        return result

    def drawdown(self) -> DrawdownInfo:
        dd_abs = self.peak_equity - self.last_equity
        if dd_abs < ZERO:
            dd_abs = ZERO
        if self.peak_equity > ZERO:
            dd_pct = (dd_abs / self.peak_equity) * 100
        else:
            dd_pct = ZERO  # undefined ratio near/under zero equity; guarded
        return DrawdownInfo(
            peak_equity=self.peak_equity,
            equity=self.last_equity,
            drawdown_abs=dd_abs,
            drawdown_pct=dd_pct,
        )

    # ------------------------------------------------------------------
    # Deterministic persistence
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, object]:
        """
        Fully deterministic serializable state.
        - every Decimal as canonical fixed-point string
        - sorted keys when dumped with sort_keys=True
        - hashable via StateHasher.hash_state(snapshot())
        - exact round-trip via from_snapshot()
        """
        return {
            "starting_cash": dec_to_str(self.starting_cash),
            "cash": dec_to_str(self.cash),
            "fees_paid": dec_to_str(self.fees_paid),
            "realized_pnl": dec_to_str(self.realized_pnl),
            "peak_equity": dec_to_str(self.peak_equity),
            "last_equity": dec_to_str(self.last_equity),
            "last_equity_fully_priced": bool(self.last_equity_fully_priced),
            "max_staleness_us": self.max_staleness_us,
            "positions": {
                sym: {
                    "quantity": dec_to_str(p.quantity),
                    "avg_entry_price": dec_to_str(p.avg_entry_price),
                    "realized_pnl": dec_to_str(p.realized_pnl),
                }
                for sym, p in sorted(self.positions.items())
            },
            "marks": {
                sym: {"price": dec_to_str(m.price), "ts_us": m.ts_us}
                for sym, m in sorted(self.marks.items())
            },
        }

    @classmethod
    def from_snapshot(cls, snap: Dict[str, object]) -> "Portfolio":
        """Exact reconstruction -- precision-loss-free round trip."""
        pf = cls(
            starting_cash=to_decimal(snap["starting_cash"]),  # type: ignore[arg-type]
            max_staleness_us=snap["max_staleness_us"],         # type: ignore[arg-type]
        )
        pf.cash = to_decimal(snap["cash"])                     # type: ignore[arg-type]
        pf.fees_paid = to_decimal(snap["fees_paid"])           # type: ignore[arg-type]
        pf.realized_pnl = to_decimal(snap["realized_pnl"])     # type: ignore[arg-type]
        pf.peak_equity = to_decimal(snap["peak_equity"])       # type: ignore[arg-type]
        pf.last_equity = to_decimal(snap["last_equity"])       # type: ignore[arg-type]
        pf.last_equity_fully_priced = snap["last_equity_fully_priced"]  # type: ignore[assignment]
        pf.positions = {
            sym: PositionState(
                quantity=to_decimal(d["quantity"]),            # type: ignore[arg-type]
                avg_entry_price=to_decimal(d["avg_entry_price"]),  # type: ignore[arg-type]
                realized_pnl=to_decimal(d["realized_pnl"]),    # type: ignore[arg-type]
            )
            for sym, d in snap["positions"].items()            # type: ignore[union-attr]
        }
        pf.marks = {
            sym: _Mark(price=to_decimal(m["price"]), ts_us=m["ts_us"])  # type: ignore[arg-type]
            for sym, m in snap["marks"].items()                # type: ignore[union-attr]
        }
        return pf
