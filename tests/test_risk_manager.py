"""
Phase 6: Risk Engine tests.

Matrix: every rule isolated + boundaries (at/below/above), long & short,
reducing vs increasing, DEGRADED/HALT interlock, combinations with
deterministic reason selection, portfolio PURITY, repeated-evaluation
determinism, decision contract serialization.

All numbers hand-computed Decimal arithmetic.
"""
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.core.money import init_money_context, to_decimal, dec_to_str
from src.core.portfolio import Portfolio
from src.core.risk_manager import (
    RiskManager,
    RiskLimits,
    RULE_SYSTEM_STATUS, RULE_STALE_DATA, RULE_COOLDOWN,
    RULE_DAILY_LOSS, RULE_DRAWDOWN, RULE_ORDER_SIZE,
    RULE_POSITION_SIZE, RULE_OPEN_POSITIONS, PIPELINE,
)
from src.core.types import (
    ExecutionReport, OrderIntent, OrderSide, OrderType,
)
from src.simulator.state_hasher import StateHasher

D = to_decimal
NOW = 2000          # standard evaluation instant
MARK_TS = 1000      # standard mark instant (fresh: 2000-1000 << 60s budget)


@pytest.fixture(autouse=True)
def _money_ctx():
    init_money_context()


# ----------------------------- helpers -----------------------------------

def limits(**over):
    base = dict(
        max_order_size=D("10"),
        max_position_size=D("20"),
        max_open_positions=3,
        max_daily_loss=D("100"),
        max_drawdown_pct=D("10"),
        stale_data_us=60_000_000,
        cooldown_us=30_000_000,
    )
    base.update(over)
    return RiskLimits(**base)


def intent(cloid="i1", side=OrderSide.BUY, qty="1.0", price="100",
           symbol="BTCUSDT", otype=OrderType.LIMIT):
    return OrderIntent(
        client_order_id=cloid, symbol=symbol, side=side,
        order_type=otype, quantity=D(qty),
        price=None if price is None else D(price),
        timestamp=MARK_TS,
    )


def fill(cloid, side, qty, price, symbol="BTCUSDT", fee="0"):
    """Direct portfolio fill (position setup -- bypasses risk by design)."""
    return ExecutionReport(
        client_order_id=cloid, exchange_order_id=f"x-{cloid}",
        symbol=symbol, side=side, status="FILLED",
        filled_quantity=D(qty), last_filled_price=D(price),
        remaining_quantity="0", timestamp=MARK_TS, fee=D(fee),
    )


def system(status="CONNECTED", cash="10000", **lim_over):
    box = {"v": status}
    pf = Portfolio(starting_cash=cash)
    rm = RiskManager(pf, limits(**lim_over), status_provider=lambda: box["v"])
    return pf, rm, box


def mark(pf, symbol="BTCUSDT", price="100", ts=MARK_TS):
    pf.mark_price(symbol, D(price), ts_us=ts)


def rule_of(decision):
    return decision.rule


def check(decision, rule):
    return next(c for c in decision.checks if c.rule == rule)


# ------------------- 1. SYSTEM_STATUS interlock --------------------------

class TestSystemStatusInterlock:
    def test_connected_approves_clean_order(self):
        pf, rm, _ = system()
        mark(pf)
        d = rm.evaluate(intent(), now_us=NOW)
        assert d.approved is True
        assert d.rule == "PASS"

    def test_halt_blocks_increasing(self):
        pf, rm, box = system(status="HALT")
        mark(pf)
        d = rm.evaluate(intent(), now_us=NOW)
        assert d.approved is False
        assert d.rule == RULE_SYSTEM_STATUS

    def test_halt_blocks_even_reducing(self):
        pf, rm, box = system(status="HALT")
        pf.apply_report(fill("o1", OrderSide.BUY, "5", "100"))
        d = rm.evaluate(intent(side=OrderSide.SELL, qty="2"), now_us=NOW)
        assert d.approved is False
        assert d.rule == RULE_SYSTEM_STATUS
        assert check(d, RULE_SYSTEM_STATUS).detail.startswith("status=HALT")

    def test_degraded_blocks_new_position(self):
        pf, rm, box = system(status="DEGRADED")
        mark(pf)
        d = rm.evaluate(intent(), now_us=NOW)  # flat -> any buy is new risk
        assert d.approved is False
        assert d.rule == RULE_SYSTEM_STATUS

    def test_degraded_allows_strict_reduction(self):
        pf, rm, box = system(status="DEGRADED")
        pf.apply_report(fill("o1", OrderSide.BUY, "5", "100"))
        mark(pf)
        d = rm.evaluate(intent(side=OrderSide.SELL, qty="2"), now_us=NOW)
        assert d.approved is True
        assert "reduction permitted" in check(d, RULE_SYSTEM_STATUS).detail

    def test_degraded_rejects_flip_through_zero(self):
        pf, rm, box = system(status="DEGRADED")
        pf.apply_report(fill("o1", OrderSide.BUY, "1", "100"))
        mark(pf)
        # SELL 3 on a 1-long flips to -2 short => increases exposure
        d = rm.evaluate(intent(side=OrderSide.SELL, qty="3"), now_us=NOW)
        assert d.approved is False
        assert d.rule == RULE_SYSTEM_STATUS
        assert d.details["reducing"] == "False"

    def test_unknown_status_fail_closed(self):
        pf, rm, box = system(status="WARPED")
        mark(pf)
        d = rm.evaluate(intent(), now_us=NOW)
        assert d.approved is False
        assert d.rule == RULE_SYSTEM_STATUS


# -------------------- 2. STALE_DATA protection ---------------------------

class TestStaleDataProtection:
    def test_missing_mark_blocks_opening(self):
        pf, rm, _ = system()
        d = rm.evaluate(intent(price="100"), now_us=NOW)  # no mark anywhere
        assert d.approved is False
        assert d.rule == RULE_STALE_DATA
        assert "no market data" in d.reason

    def test_stale_mark_blocks_increasing(self):
        pf, rm, _ = system()
        mark(pf, ts=1000)
        late = 1000 + 60_000_001  # one microsecond past budget
        d = rm.evaluate(intent(), now_us=late)
        assert d.approved is False
        assert d.rule == RULE_STALE_DATA
        assert "stale mark age_us=" in d.reason

    def test_mark_within_budget_is_fresh(self):
        pf, rm, _ = system()
        mark(pf, ts=1000)
        edge = 1000 + 60_000_000  # exactly at budget -> still fresh
        d = rm.evaluate(intent(), now_us=edge)
        assert d.approved is True

    def test_stale_mark_allows_reduction_by_policy(self):
        pf, rm, _ = system()
        pf.apply_report(fill("o1", OrderSide.BUY, "5", "100"))
        mark(pf, ts=1000)
        d = rm.evaluate(intent(side=OrderSide.SELL, qty="2"),
                        now_us=1000 + 61_000_000)
        assert d.approved is True
        assert "reduces risk" in check(d, RULE_STALE_DATA).detail

    def test_market_order_without_reference_price_rejected(self):
        pf, rm, _ = system()
        d = rm.evaluate(intent(otype=OrderType.MARKET, price=None), now_us=NOW)
        assert d.approved is False
        assert d.rule == RULE_STALE_DATA  # first failing gate in pipeline

    def test_market_order_with_fresh_mark_approved(self):
        pf, rm, _ = system()
        mark(pf, price="101")
        d = rm.evaluate(intent(otype=OrderType.MARKET, price=None), now_us=NOW)
        assert d.approved is True
        assert d.details["prospective_position"] == dec_to_str(D("1.0"))


# --------------------------- 3. COOLDOWN ---------------------------------

class TestCooldown:
    def test_blocks_inside_window(self):
        pf, rm, _ = system()
        mark(pf)
        rm.record_fill_outcome(ts_us=1000, realized_delta=D("-5"))
        d = rm.evaluate(intent(), now_us=1000 + 29_999_999)
        assert d.approved is False
        assert d.rule == RULE_COOLDOWN

    def test_expires_exactly_at_boundary(self):
        pf, rm, _ = system()
        mark(pf)
        rm.record_fill_outcome(ts_us=1000, realized_delta=D("-5"))
        d = rm.evaluate(intent(), now_us=1000 + 30_000_000)
        assert d.approved is True

    def test_reduction_during_cooldown_allowed(self):
        pf, rm, _ = system()
        pf.apply_report(fill("o1", OrderSide.BUY, "5", "100"))
        mark(pf)
        rm.record_fill_outcome(ts_us=1000, realized_delta=D("-5"))
        d = rm.evaluate(intent(side=OrderSide.SELL, qty="2"),
                        now_us=1000 + 1)
        assert d.approved is True

    def test_profitable_fill_does_not_arm_cooldown(self):
        pf, rm, _ = system()
        mark(pf)
        rm.record_fill_outcome(ts_us=1000, realized_delta=D("+7"))
        d = rm.evaluate(intent(), now_us=1000 + 1)
        assert d.approved is True


# ------------------------- 4. MAX_DAILY_LOSS ------------------------------

class TestMaxDailyLoss:
    def _realize(self, pf, close_price):
        pf.apply_report(fill("d1", OrderSide.BUY, "2", "100"))
        pf.apply_report(fill("d2", OrderSide.SELL, "2", close_price))

    def test_unanchored_day_rule_inactive(self):
        pf, rm, _ = system(max_daily_loss=D("1"))
        self._realize(pf, "40")            # realized -120, no anchor set
        mark(pf)
        d = rm.evaluate(intent(), now_us=NOW)
        assert d.approved is True
        assert "day not anchored" in check(d, RULE_DAILY_LOSS).detail

    def test_below_cap_passes(self):
        pf, rm, _ = system()
        rm.start_day(now_us=NOW)
        self._realize(pf, "51")            # realized -98 > -100
        mark(pf)
        d = rm.evaluate(intent(), now_us=NOW)
        assert d.approved is True
        assert d.details["daily_pnl"] == dec_to_str(D("-98"))

    def test_exactly_at_cap_blocks(self):
        pf, rm, _ = system()
        rm.start_day(now_us=NOW)
        self._realize(pf, "50")            # realized exactly -100 == cap
        mark(pf)
        d = rm.evaluate(intent(), now_us=NOW)
        assert d.approved is False
        assert d.rule == RULE_DAILY_LOSS
        assert d.details["daily_pnl"] == dec_to_str(D("-100"))

    def test_beyond_cap_blocks(self):
        pf, rm, _ = system()
        rm.start_day(now_us=NOW)
        self._realize(pf, "40")            # realized -120
        mark(pf)
        d = rm.evaluate(intent(), now_us=NOW)
        assert d.approved is False
        assert d.rule == RULE_DAILY_LOSS

    def test_breach_allows_reduction_but_not_increase(self):
        pf, rm, _ = system(max_daily_loss=D("40"))
        rm.start_day(now_us=NOW)
        pf.apply_report(fill("d1", OrderSide.BUY, "3", "100"))
        pf.apply_report(fill("d2", OrderSide.SELL, "1", "50"))  # -50 realized
        mark(pf, price="100")              # unrealized 0 -> clean numbers
        d_sell = rm.evaluate(intent(side=OrderSide.SELL, qty="1"), now_us=NOW)
        assert d_sell.approved is True     # reducing permitted by policy
        d_buy = rm.evaluate(intent(side=OrderSide.BUY, qty="1"), now_us=NOW)
        assert d_buy.approved is False
        assert d_buy.rule == RULE_DAILY_LOSS

    def test_anchor_resets_baseline(self):
        pf, rm, _ = system()
        self._realize_loss_50_then_reanchor(pf, rm)
        mark(pf, price="100")
        d = rm.evaluate(intent(), now_us=NOW)
        assert d.approved is True  # daily pnl measured from NEW anchor

    def _realize_loss_50_then_reanchor(self, pf, rm):
        pf.apply_report(fill("d1", OrderSide.BUY, "2", "100"))
        pf.apply_report(fill("d2", OrderSide.SELL, "1", "50"))  # -50, hold 1
        rm.start_day(now_us=NOW)           # anchor := current total (-50)
        pf.apply_report(fill("d3", OrderSide.SELL, "1", "50"))  # another -50


# ------------------------- 5. MAX_DRAWDOWN --------------------------------

class TestMaxDrawdown:
    def _breached_state(self, pf):
        """peak 10200 -> equity 10010 => dd_pct ~1.8627%."""
        pf.apply_report(fill("dd1", OrderSide.BUY, "1", "100"))  # cash 9900
        pf.mark_price("BTCUSDT", D("300"), ts_us=500)
        pf.update_equity(now_us=500)       # equity 10200 -> peak
        pf.mark_price("BTCUSDT", D("110"), ts_us=600)
        pf.update_equity(now_us=600)       # equity 10010 -> dd

    def test_at_or_above_cap_blocks_new_risk(self):
        pf, rm, _ = system(max_drawdown_pct=D("1"))
        self._breached_state(pf)
        mark(pf, symbol="ETHUSDT")
        d = rm.evaluate(intent(symbol="ETHUSDT"), now_us=NOW)
        assert d.approved is False
        assert d.rule == RULE_DRAWDOWN

    def test_below_cap_passes(self):
        pf, rm, _ = system(max_drawdown_pct=D("2"))
        self._breached_state(pf)
        mark(pf, symbol="ETHUSDT")
        d = rm.evaluate(intent(symbol="ETHUSDT"), now_us=NOW)
        assert d.approved is True

    def test_breach_allows_reduction(self):
        pf, rm, _ = system(max_drawdown_pct=D("1"))
        self._breached_state(pf)
        pf.apply_report(fill("e1", OrderSide.BUY, "1", "100",
                             symbol="ETHUSDT"))
        mark(pf, symbol="ETHUSDT")
        d = rm.evaluate(intent(symbol="ETHUSDT", side=OrderSide.SELL,
                               qty="0.5"), now_us=NOW)
        assert d.approved is True


# ------------------------- 6. MAX_ORDER_SIZE ------------------------------

class TestMaxOrderSize:
    def test_at_limit_passes(self):
        pf, rm, _ = system(max_order_size=D("10"))
        mark(pf)
        d = rm.evaluate(intent(qty="10"), now_us=NOW)
        assert d.approved is True

    def test_one_unit_above_fails(self):
        pf, rm, _ = system(max_order_size=D("10"))
        mark(pf)
        d = rm.evaluate(intent(qty="10.1"), now_us=NOW)
        assert d.approved is False
        assert d.rule == RULE_ORDER_SIZE

    def test_below_limit_passes(self):
        pf, rm, _ = system(max_order_size=D("10"))
        mark(pf)
        d = rm.evaluate(intent(qty="9.9"), now_us=NOW)
        assert d.approved is True

    def test_zero_quantity_fails(self):
        pf, rm, _ = system()
        mark(pf)
        d = rm.evaluate(intent(qty="0"), now_us=NOW)
        assert d.approved is False
        assert d.rule == RULE_ORDER_SIZE

    def test_size_cap_applies_even_when_reducing(self):
        pf, rm, _ = system(max_order_size=D("10"), max_position_size=D("20"))
        pf.apply_report(fill("o1", OrderSide.BUY, "15", "100"))
        mark(pf)
        d = rm.evaluate(intent(side=OrderSide.SELL, qty="11"), now_us=NOW)
        assert d.approved is False         # reducing, but oversized
        assert d.rule == RULE_ORDER_SIZE


# ----------------------- 7. MAX_POSITION_SIZE ------------------------------

class TestMaxPositionSize:
    def test_prospective_at_limit_passes_long(self):
        pf, rm, _ = system(max_position_size=D("2.0"))
        pf.apply_report(fill("o1", OrderSide.BUY, "1.5", "100"))
        mark(pf)
        d = rm.evaluate(intent(qty="0.5"), now_us=NOW)
        assert d.approved is True
        assert d.details["prospective_position"] == dec_to_str(D("2.0"))

    def test_above_limit_fails_long(self):
        pf, rm, _ = system(max_position_size=D("2.0"))
        pf.apply_report(fill("o1", OrderSide.BUY, "1.5", "100"))
        mark(pf)
        d = rm.evaluate(intent(qty="0.6"), now_us=NOW)
        assert d.approved is False
        assert d.rule == RULE_POSITION_SIZE

    def test_short_add_above_limit_fails(self):
        pf, rm, _ = system(max_position_size=D("2.0"))
        pf.apply_report(fill("o1", OrderSide.SELL, "1.5", "100"))
        mark(pf)
        d = rm.evaluate(intent(side=OrderSide.SELL, qty="0.6"), now_us=NOW)
        assert d.approved is False
        assert d.rule == RULE_POSITION_SIZE

    def test_flip_through_zero_counts_full_exposure(self):
        pf, rm, _ = system(max_position_size=D("2.0"))
        pf.apply_report(fill("o1", OrderSide.BUY, "1.5", "100"))
        mark(pf)
        # SELL 4 flips long 1.5 -> short 2.5 ; |2.5| > limit
        d = rm.evaluate(intent(side=OrderSide.SELL, qty="4"), now_us=NOW)
        assert d.approved is False
        assert d.rule == RULE_POSITION_SIZE


# ---------------------- 8. MAX_OPEN_POSITIONS ------------------------------

class TestMaxOpenPositions:
    def _two_open(self, pf):
        pf.apply_report(fill("e1", OrderSide.BUY, "1", "100", symbol="ETHUSDT"))
        pf.apply_report(fill("s1", OrderSide.BUY, "1", "100", symbol="SOLUSDT"))

    def test_new_symbol_at_limit_fails(self):
        pf, rm, _ = system(max_open_positions=2)
        self._two_open(pf)
        mark(pf, symbol="BTCUSDT")
        d = rm.evaluate(intent(symbol="BTCUSDT"), now_us=NOW)
        assert d.approved is False
        assert d.rule == RULE_OPEN_POSITIONS

    def test_existing_symbol_exempt(self):
        pf, rm, _ = system(max_open_positions=2)
        self._two_open(pf)
        mark(pf, symbol="ETHUSDT")
        d = rm.evaluate(intent(symbol="ETHUSDT", qty="0.5"), now_us=NOW)
        assert d.approved is True

    def test_one_below_limit_allows_new(self):
        pf, rm, _ = system(max_open_positions=3)
        self._two_open(pf)
        mark(pf, symbol="BTCUSDT")
        d = rm.evaluate(intent(symbol="BTCUSDT"), now_us=NOW)
        assert d.approved is True


# ------------- 9. combinations / determinism / purity ----------------------

class TestCombinationsAndDeterminism:
    def test_first_pipeline_failure_wins(self):
        pf, rm, box = system(status="HALT", max_order_size=D("5"))
        mark(pf)
        d = rm.evaluate(intent(qty="99"), now_us=NOW)
        assert d.rule == RULE_SYSTEM_STATUS          # rule 1 beats rule 6
        assert check(d, RULE_SYSTEM_STATUS).passed is False
        assert check(d, RULE_ORDER_SIZE).passed is False  # full trail kept

    def test_stale_and_cooldown_stale_reported_first(self):
        pf, rm, _ = system()
        mark(pf, ts=1000)
        rm.record_fill_outcome(ts_us=1500, realized_delta=D("-9"))
        late = 1000 + 61_000_000                     # stale...
        # ...but cooldown window (30M us) already expired at 'late'
        d = rm.evaluate(intent(), now_us=late + 0)
        assert d.rule in (RULE_STALE_DATA,)          # stale gate hit first
        assert check(d, RULE_STALE_DATA).passed is False

    def test_multiple_failures_listed_in_fixed_order(self):
        pf, rm, box = system(max_daily_loss=D("80"), max_drawdown_pct=D("0.01"))
        box["v"] = "DEGRADED"
        pf.apply_report(fill("x", OrderSide.BUY, "1", "100"))
        mark(pf, ts=1000)
        rm.start_day(now_us=NOW)
        pf.apply_report(fill("y", OrderSide.SELL, "1", "10"))  # realized -90
        pf.update_equity(now_us=NOW)                # equity 9910 -> dd 0.9%
        mark(pf, ts=1000)
        d = rm.evaluate(intent(), now_us=1000 + 61_000_000)    # stale too
        assert d.approved is False
        failed_rules = [c.rule for c in d.checks if not c.passed]
        assert failed_rules[0] == d.rule             # primary = first in trail
        # DEGRADED + STALE_DATA + DAILY_LOSS(-90<=-80) + DRAWDOWN(0.9%>=0.01%)
        assert failed_rules == [
            RULE_SYSTEM_STATUS, RULE_STALE_DATA,
            RULE_DAILY_LOSS, RULE_DRAWDOWN,
        ]

    def test_repeated_evaluation_identical_decision(self):
        pf, rm, _ = system()
        pf.apply_report(fill("o1", OrderSide.BUY, "5", "100"))
        mark(pf)
        rm.record_fill_outcome(ts_us=1000, realized_delta=D("-5"))
        i1 = rm.evaluate(intent(side=OrderSide.SELL, qty="2"), now_us=NOW)
        i2 = rm.evaluate(intent(side=OrderSide.SELL, qty="2"), now_us=NOW)
        assert i1 == i2
        assert i1.model_dump_json() == i2.model_dump_json()


class TestPortfolioPurity:
    def test_evaluation_never_mutates_portfolio(self):
        pf, rm, box = system()
        pf.apply_report(fill("o1", OrderSide.BUY, "5", "100", fee="1"))
        pf.mark_price("BTCUSDT", D("110"), ts_us=MARK_TS)
        pf.update_equity(now_us=MARK_TS)
        rm.start_day(now_us=MARK_TS)
        rm.record_fill_outcome(ts_us=MARK_TS, realized_delta=D("-3"))
        before_hash = StateHasher.hash_state(pf.snapshot())
        before_cash, before_peak = pf.cash, pf.peak_equity

        cases = [
            intent("p1"),
            intent("p2", side=OrderSide.SELL, qty="2"),
            intent("p3", otype=OrderType.MARKET, price=None),
            intent("p4", qty="99"),                       # oversized
            intent("p5", price=None, otype=OrderType.MARKET),  # unpriced
        ]
        for status in ("CONNECTED", "DEGRADED", "HALT"):
            box["v"] = status
            for i in cases:
                rm.evaluate(i, now_us=NOW + 70_000_000)   # stale too

        assert StateHasher.hash_state(pf.snapshot()) == before_hash
        assert pf.cash == before_cash
        assert pf.peak_equity == before_peak
        assert pf.fees_paid == D("1")

    def test_risk_state_also_untouched_by_evaluate(self):
        pf, rm, _ = system()
        mark(pf)
        rm.record_fill_outcome(ts_us=1000, realized_delta=D("-5"))
        rm.start_day(now_us=NOW)
        snap_before = rm.state.model_dump()
        rm.evaluate(intent(), now_us=NOW + 1)
        rm.evaluate(intent(qty="99"), now_us=NOW + 1)
        assert rm.state.model_dump() == snap_before


class TestDecisionContract:
    def test_decision_is_frozen(self):
        pf, rm, _ = system()
        mark(pf)
        d = rm.evaluate(intent(), now_us=NOW)
        with pytest.raises(ValidationError):
            d.approved = False

    def test_decision_serializes_round_trip(self):
        pf, rm, _ = system()
        pf.apply_report(fill("o1", OrderSide.BUY, "5", "100"))
        mark(pf)
        d = rm.evaluate(intent(side=OrderSide.SELL, qty="2"), now_us=NOW)
        clone = type(d).model_validate_json(d.model_dump_json())
        assert clone == d
        assert len(clone.checks) == len(PIPELINE)

    def test_allowed_property_mirrors_approved(self):
        pf, rm, _ = system()
        mark(pf)
        d_ok = rm.evaluate(intent(), now_us=NOW)
        assert d_ok.allowed is True and d_ok.approved is True
        d_bad = rm.evaluate(intent(qty="999"), now_us=NOW)
        assert d_bad.allowed is False and d_bad.approved is False

    def test_every_check_has_rule_name_and_detail(self):
        pf, rm, _ = system()
        mark(pf)
        d = rm.evaluate(intent(), now_us=NOW)
        assert [c.rule for c in d.checks] == list(PIPELINE)
        assert all(isinstance(c.detail, str) for c in d.checks)
