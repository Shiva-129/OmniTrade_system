"""
OmniTrade Risk Engine (Phase 6).

Deterministic rule pipeline: OrderIntent + Portfolio snapshot + system
status -> explicit, auditable RiskDecision.

HARD RULES
----------
1. evaluate() is PURE w.r.t. the Portfolio: it reads positions, marks,
   realized PnL, equity bookkeeping -- it NEVER mutates cash, positions,
   PnL, fees, peaks or drawdown state. (Tested by snapshot-hash equality.)
2. Risk keeps its OWN explicitly-fed state (day anchors, loss events).
   Feeding happens through record_* / start_day doors only.
3. Every evaluation returns a FULL ordered audit trail of all rules --
   never just True/False.

REDUCING-ORDER POLICY (explicit)
--------------------------------
An order "reduces" iff the symbol holds an open position, the signed
delta is OPPOSITE to it, and |result| <= |current| (strict reduction,
NO flip through zero -- flips open new exposure and count as increasing).
Restrictive gates (DEGRADED, stale data, cooldown, daily-loss cap,
drawdown cap) block increasing orders but ALWAYS permit pure reductions:
a trader must never be trapped in a position. HALT blocks everything,
including reductions ("block all trading activity").

BOUNDARY CONVENTIONS (tested)
-----------------------------
- Size/count limits: PASS exactly AT the limit, FAIL one unit above.
- Loss/drawdown caps: BLOCK exactly AT the cap (reached = no new risk),
  PASS below it.
- Cooldown window: active while now - last_loss < cooldown_us;
  expired AT exactly cooldown_us.

RULE PIPELINE ORDER (fixed -> deterministic reason selection)
-------------------------------------------------------------
1 SYSTEM_STATUS  2 STALE_DATA  3 COOLDOWN  4 MAX_DAILY_LOSS
5 MAX_DRAWDOWN   6 MAX_ORDER_SIZE  7 MAX_POSITION_SIZE  8 MAX_OPEN_POSITIONS

All rules are always evaluated (complete audit trail); the FIRST failure
in this order becomes the decision's primary rule/reason.

REFERENCE PRICE POLICY
----------------------
LIMIT intents use intent.price. MARKET intents require a FRESH mark on
the intent symbol; missing/stale reference => rejected under STALE_DATA.
Prices are never invented.
"""
from typing import Callable, Dict, Optional, Tuple

from pydantic import BaseModel

from .money import Decimal, ZERO, dec_to_str
from .portfolio import PositionState
from .types import OrderIntent, OrderSide, RiskCheck, RiskDecision


class RiskLimits(BaseModel):
    """All risk knobs in one frozen object. Decimals via canonical policy."""
    model_config = {"frozen": True}

    max_order_size: Decimal          # abs qty per order (inclusive)
    max_position_size: Decimal       # abs qty per symbol (inclusive)
    max_open_positions: int          # distinct symbols with nonzero qty
    max_daily_loss: Decimal          # quote ccy; daily PnL drop >= cap blocks
    max_drawdown_pct: Decimal        # 0..100; dd_pct >= cap blocks new risk
    stale_data_us: int               # mark freshness budget
    cooldown_us: int                 # post-loss-event window


class RiskState(BaseModel):
    """
    Risk-owned state, mutated ONLY through its feed doors.
    Kept separate from Portfolio so evaluation stays pure.
    """
    day_anchor_pnl: Optional[Decimal] = None
    last_loss_event_ts: Optional[int] = None


# Fixed pipeline order -- the single source of truth for reason selection.
RULE_SYSTEM_STATUS = "SYSTEM_STATUS"
RULE_STALE_DATA = "STALE_DATA"
RULE_COOLDOWN = "COOLDOWN"
RULE_DAILY_LOSS = "MAX_DAILY_LOSS"
RULE_DRAWDOWN = "MAX_DRAWDOWN"
RULE_ORDER_SIZE = "MAX_ORDER_SIZE"
RULE_POSITION_SIZE = "MAX_POSITION_SIZE"
RULE_OPEN_POSITIONS = "MAX_OPEN_POSITIONS"

PIPELINE: Tuple[str, ...] = (
    RULE_SYSTEM_STATUS, RULE_STALE_DATA, RULE_COOLDOWN, RULE_DAILY_LOSS,
    RULE_DRAWDOWN, RULE_ORDER_SIZE, RULE_POSITION_SIZE, RULE_OPEN_POSITIONS,
)

_RESTRICTED_RULES = frozenset({
    RULE_STALE_DATA, RULE_COOLDOWN, RULE_DAILY_LOSS, RULE_DRAWDOWN,
})
_FAIL_CLOSED_STATUSES = frozenset({"HALT", "UNKNOWN"})


class RiskManager:
    """
    Pure evaluator. Construct with the portfolio REFERENCE (reads only),
    frozen limits, and a system-status provider (engine wires
    ObserverState.get_system_status; tests inject constants).
    """

    def __init__(self, portfolio, limits: RiskLimits,
                 status_provider: Callable[[], str]):
        self.portfolio = portfolio
        self.limits = limits
        self._status_of = status_provider
        self.state = RiskState()

    # ------------------------- feed doors ------------------------------

    def start_day(self, now_us: int) -> None:
        """
        Anchors the daily-loss baseline at current TOTAL pnl
        (realized + unrealized). Call once per trading day boundary.
        """
        self.state = self.state.model_copy(
            update={"day_anchor_pnl": self._total_pnl()}
        )

    def record_fill_outcome(self, ts_us: int, realized_delta: Decimal) -> None:
        """
        Engine calls after a fill is APPLIED to the portfolio.
        Negative realized delta registers a loss event (cooldown trigger).
        """
        if realized_delta < ZERO:
            self.state = self.state.model_copy(
                update={"last_loss_event_ts": int(ts_us)}
            )

    # ------------------------- helpers (pure) ---------------------------

    def _total_pnl(self) -> Decimal:
        """realized + sum(unrealized); staleness waived; unpriced skipped."""
        total = self.portfolio.realized_pnl
        for sym in self.portfolio.positions:
            u = self.portfolio.unrealized_pnl(sym)  # no clock -> waive staleness
            if u is not None:
                total += u
        return total

    def _is_fresh(self, symbol: str, now_us: int) -> bool:
        mark = self.portfolio.marks.get(symbol)
        return (mark is not None
                and (now_us - mark.ts_us) <= self.limits.stale_data_us)

    @staticmethod
    def _signed_qty(intent: OrderIntent) -> Decimal:
        return intent.quantity if intent.side == OrderSide.BUY else -intent.quantity

    def _reference_price(self, intent: OrderIntent, now_us: int) -> Optional[Decimal]:
        if intent.price is not None:
            return intent.price
        # MARKET order: borrow the freshest mark as execution estimate.
        mark = self.portfolio.marks.get(intent.symbol)
        if mark is not None and (now_us - mark.ts_us) <= self.limits.stale_data_us:
            return mark.price
        return None

    def classify(self, intent: OrderIntent, ref_price: Decimal) -> Tuple[bool, Dict[str, str]]:
        """
        Classifies the intent against the CURRENT position.
        Returns (is_reducing, context).
        Reducing: opposite-signed delta that shrinks absolute exposure
        WITHOUT flipping through zero (flips open new risk).
        """
        pos = self.portfolio.positions.get(intent.symbol)
        ctx: Dict[str, str] = {}
        if pos is None or pos.quantity == ZERO:
            return False, ctx

        signed = self._signed_qty(intent)
        evolved_qty = self._prospective_quantity(pos, signed, ref_price)
        ctx["current_position"] = dec_to_str(pos.quantity)
        ctx["prospective_position"] = dec_to_str(evolved_qty)

        opposite = (pos.quantity > ZERO) != (signed > ZERO)
        shrinking = abs(evolved_qty) <= abs(pos.quantity)
        is_reducing = bool(opposite and shrinking)
        return is_reducing, ctx

    def _prospective_quantity(self, pos, signed_delta: Decimal,
                              ref_price: Decimal) -> Decimal:
        """Pure reuse of portfolio fill arithmetic (no mutation)."""
        evolved = self.portfolio._evolve_position(pos, signed_delta, ref_price)
        return evolved.quantity

    # ------------------------- the pipeline -----------------------------

    def evaluate(self, intent: OrderIntent, now_us: int) -> RiskDecision:
        """
        THE evaluation door. Pure: identical inputs -> identical decision;
        portfolio untouched (hash-verified in tests).
        """
        checks = []
        details: Dict[str, str] = {
            "system_status": self._status_of(),
            "order_qty": dec_to_str(intent.quantity),
            "max_order_size": dec_to_str(self.limits.max_order_size),
            "max_position_size": dec_to_str(self.limits.max_position_size),
        }

        status = self._status_of()
        reducing = False
        ref_price = self._reference_price(intent, now_us)
        has_ref = ref_price is not None
        if has_ref:
            reducing, rctx = self.classify(intent, ref_price)
            details.update(rctx)
        details["reducing"] = str(reducing)

        # ---- 1 SYSTEM_STATUS -------------------------------------------
        passed = status == "CONNECTED" or (status == "DEGRADED" and reducing)
        detail = ""
        if status in _FAIL_CLOSED_STATUSES:
            detail = f"status={status}; all trading blocked (fail-closed)"
        elif status == "DEGRADED":
            detail = ("DEGRADED: only strictly position-reducing orders "
                      "permitted" if not reducing else
                      "DEGRADED: reduction permitted by policy")
        elif status != "CONNECTED":
            detail = f"unknown status={status}"
        checks.append(RiskCheck(rule=RULE_SYSTEM_STATUS, passed=passed, detail=detail))

        # ---- 2 STALE_DATA ----------------------------------------------
        fresh = self._is_fresh(intent.symbol, now_us)
        passed = True
        detail = f"mark_age_ok(symbol={intent.symbol})"
        if not fresh and not reducing:
            passed = False
            mark = self.portfolio.marks.get(intent.symbol)
            if mark is None:
                detail = (f"no market data for {intent.symbol}; "
                          "new-risk order blocked")
            else:
                age = now_us - mark.ts_us
                detail = (f"stale mark age_us={age} > budget_us="
                          f"{self.limits.stale_data_us}")
        elif not fresh:
            detail = "data stale but order reduces risk; permitted by policy"
        checks.append(RiskCheck(rule=RULE_STALE_DATA, passed=passed, detail=detail))

        # ---- 3 COOLDOWN --------------------------------------------------
        last_loss = self.state.last_loss_event_ts
        passed = True
        detail = "no recent loss event"
        if last_loss is not None and not reducing:
            elapsed = now_us - last_loss
            if elapsed < self.limits.cooldown_us:
                passed = False
                detail = (f"cooldown active: {elapsed}us since loss < "
                          f"{self.limits.cooldown_us}us window")
        checks.append(RiskCheck(rule=RULE_COOLDOWN, passed=passed, detail=detail))

        # ---- 4 MAX_DAILY_LOSS --------------------------------------------
        anchor = self.state.day_anchor_pnl
        passed = True
        detail = "day not anchored; rule inactive"
        daily_pnl = None
        if anchor is not None:
            daily_pnl = self._total_pnl() - anchor
            details["daily_pnl"] = dec_to_str(daily_pnl)
            details["max_daily_loss"] = dec_to_str(self.limits.max_daily_loss)
            passed = daily_pnl > -self.limits.max_daily_loss or reducing
            detail = (f"daily_pnl={dec_to_str(daily_pnl)} vs cap=-"
                      f"{dec_to_str(self.limits.max_daily_loss)}"
                      ) if not passed else f"daily_pnl={dec_to_str(daily_pnl)}"
        checks.append(RiskCheck(rule=RULE_DAILY_LOSS, passed=passed, detail=detail))

        # ---- 5 MAX_DRAWDOWN ------------------------------------------------
        dd = self.portfolio.drawdown()
        details["drawdown_pct"] = dec_to_str(dd.drawdown_pct)
        details["max_drawdown_pct"] = dec_to_str(self.limits.max_drawdown_pct)
        passed = dd.drawdown_pct < self.limits.max_drawdown_pct or reducing
        detail = (f"drawdown={dec_to_str(dd.drawdown_pct)}% vs cap="
                  f"{dec_to_str(self.limits.max_drawdown_pct)}%")
        checks.append(RiskCheck(rule=RULE_DRAWDOWN, passed=passed, detail=detail))

        # ---- 6 MAX_ORDER_SIZE -----------------------------------------------
        qty_ok = ZERO < intent.quantity <= self.limits.max_order_size
        checks.append(RiskCheck(
            rule=RULE_ORDER_SIZE, passed=qty_ok,
            detail=f"|qty|={dec_to_str(intent.quantity)} limit="
                   f"{dec_to_str(self.limits.max_order_size)}",
        ))

        # ---- 7 MAX_POSITION_SIZE ---------------------------------------------
        if has_ref:
            pos = self.portfolio.positions.get(intent.symbol) or PositionState()
            current = pos.quantity
            prospective = self._prospective_quantity(
                pos, self._signed_qty(intent), ref_price,
            )
            size_ok = abs(prospective) <= self.limits.max_position_size
            details["prospective_position"] = dec_to_str(prospective)
            checks.append(RiskCheck(
                rule=RULE_POSITION_SIZE, passed=size_ok,
                detail=f"prospective={dec_to_str(abs(prospective))} from "
                       f"{dec_to_str(current)}; limit="
                       f"{dec_to_str(self.limits.max_position_size)}",
            ))
        else:
            checks.append(RiskCheck(
                rule=RULE_POSITION_SIZE, passed=False,
                detail="no reference price available to size the order",
            ))

        # ---- 8 MAX_OPEN_POSITIONS ----------------------------------------------
        open_syms = [s for s, p in self.portfolio.positions.items()
                     if p.quantity != ZERO]
        opens_new = (open_syms.count(intent.symbol) == 0)
        count_ok = True
        if opens_new and len(open_syms) >= self.limits.max_open_positions:
            count_ok = False
        checks.append(RiskCheck(
            rule=RULE_OPEN_POSITIONS, passed=count_ok,
            detail=f"open={len(open_syms)} limit={self.limits.max_open_positions}"
                   + ("; new symbol" if opens_new else "; existing symbol"),
        ))

        # ---------------- verdict (first failure wins, fixed order) ----------
        by_rule = {c.rule: c for c in checks}
        failed = next((r for r in PIPELINE if not by_rule[r].passed), None)
        approved = failed is None

        return RiskDecision(
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            approved=approved,
            rule=failed or "PASS",
            reason="" if approved else by_rule[failed].detail,
            checks=tuple(checks),
            details=details,
        )
