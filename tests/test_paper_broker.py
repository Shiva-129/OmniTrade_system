"""
Phase 10: PaperBroker unit tests.

Covers: interface contract, market-order simulation (adverse slippage,
both sides), limit-order protection (never above/below limit), partial
fill schedules, fees (maker/taker), rejections, duplicate orders and
reports, cancellation lifecycle, journal behavior incl. loud failure,
restart recovery via rebuild_from_journal, and the Portfolio-purity
guarantee.
"""
import pytest
from pydantic import ValidationError

from src.adapters.base import BrokerInterface
from src.adapters.paper import FillSchedule, PaperBroker, PaperOrderState
from src.core.costs import CostModel
from src.core.journal import RawJournal
from src.core.money import init_money_context, to_decimal
from src.core.types import OrderIntent, OrderSide, OrderType


@pytest.fixture(autouse=True)
def _ctx():
    init_money_context()


def cm(**kw):
    base = dict(taker_fee=to_decimal("0.001"), maker_fee=to_decimal("0.0004"),
                slippage_pct=to_decimal("0.0005"))
    base.update(kw)
    return CostModel(**base)


def intent(cloid, side=OrderSide.BUY, qty="1", px=None,
           otype=OrderType.MARKET, ts=1000, symbol="BTCUSDT"):
    return OrderIntent(
        client_order_id=cloid, symbol=symbol, side=side,
        order_type=otype, quantity=to_decimal(qty),
        price=None if px is None else to_decimal(px), timestamp=ts)


class TestInterfaceContract:
    def test_interface_is_abstract(self):
        with pytest.raises(TypeError):
            BrokerInterface()

    def test_paper_broker_satisfies_interface(self):
        assert isinstance(PaperBroker(cm()), BrokerInterface)


class TestMarketOrders:
    def test_rests_until_first_price_tick(self):
        b = PaperBroker(cm())
        assert b.submit_order(intent("m1")) == "ACCEPTED"
        o = b.get_order("m1")
        assert o["status"] == "NEW"
        b.drain_reports()                       # NEW report emitted on accept
        assert b.drain_reports() == []          # nothing filled without a price

    def test_buy_fills_adverse_up(self):
        b = PaperBroker(cm())
        b.submit_order(intent("m1"))
        b.drain_reports()
        assert b.on_market_price("BTCUSDT", to_decimal("100"), 1000) == 1
        r = [x for x in b.drain_reports() if x.status == "FILLED"][0]
        assert r.last_filled_price == to_decimal("100.05")   # 100 * 1.0005
        assert r.fee == to_decimal("0.10005")                # taker on notional
        assert r.remaining_quantity == to_decimal("0")

    def test_sell_fills_adverse_down(self):
        b = PaperBroker(cm())
        b.submit_order(intent("s1", side=OrderSide.SELL))
        b.on_market_price("BTCUSDT", to_decimal("100"), 1000)
        r = [x for x in b.drain_reports() if x.status == "FILLED"][0]
        assert r.last_filled_price == to_decimal("99.95")    # adverse for seller

    def test_zero_slippage_exact_fill(self):
        b = PaperBroker(cm(slippage_pct=to_decimal("0")))
        b.submit_order(intent("z1"))
        b.on_market_price("BTCUSDT", to_decimal("100"), 1000)
        r = [x for x in b.drain_reports() if x.status == "FILLED"][0]
        assert r.last_filled_price == to_decimal("100")

    def test_execution_view_positions(self):
        b = PaperBroker(cm())
        b.submit_order(intent("m1"))
        b.on_market_price("BTCUSDT", to_decimal("100"), 1000)
        b.drain_reports()
        assert b.get_positions() == {"BTCUSDT": "1"}
        st = b.get_account_state()
        assert st["fills"] == 1 and st["submitted"] == 1


class TestLimitOrders:
    def test_buy_never_fills_above_limit(self):
        b = PaperBroker(cm())
        b.submit_order(intent("l1", px="100", otype=OrderType.LIMIT))
        assert b.on_market_price("BTCUSDT", to_decimal("110"), 1000) == 0
        assert b.get_order("l1")["status"] == "NEW"          # still resting

    def test_buy_cross_fills_at_better_price_maker_fee(self):
        b = PaperBroker(cm())
        b.submit_order(intent("l1", px="100", otype=OrderType.LIMIT))
        b.on_market_price("BTCUSDT", to_decimal("95"), 2000)
        r = b.drain_reports()[-1]
        assert r.status == "FILLED"
        assert r.last_filled_price == to_decimal("95")       # <= limit, never above
        assert r.fee == to_decimal("0.038")                  # maker 4bps on 95

    def test_sell_never_fills_below_limit(self):
        b = PaperBroker(cm())
        b.submit_order(intent("l2", side=OrderSide.SELL, px="90",
                              otype=OrderType.LIMIT))
        assert b.on_market_price("BTCUSDT", to_decimal("80"), 1000) == 0
        b.on_market_price("BTCUSDT", to_decimal("92"), 2000)
        r = b.drain_reports()[-1]
        assert r.last_filled_price == to_decimal("92")       # >= limit


class TestPartialFills:
    def _broker_with_schedule(self):
        b = PaperBroker(cm(), fill_schedule=FillSchedule(
            chunks=("0.3", "0.4", "0.3")))
        b.submit_order(intent("p1", qty="1", px="100",
                              otype=OrderType.LIMIT))
        return b

    def test_three_chunks_exact_position(self):
        b = self._broker_with_schedule()
        outs = []
        for ts, px in [(10, "95"), (20, "90"), (30, "80")]:
            assert b.on_market_price("BTCUSDT", to_decimal(px), ts) == 1
            outs += b.drain_reports()
        statuses = [r.status for r in outs if r.status != "NEW"]
        assert statuses == ["PARTIAL_FILL", "PARTIAL_FILL", "FILLED"]
        qtys = [str(r.filled_quantity) for r in outs if r.status != "NEW"]
        assert qtys == ["0.3", "0.4", "0.3"]
        assert b.get_positions() == {"BTCUSDT": "1.0"}       # EXACTLY
        # fees accumulate across reports
        total_fee = sum((r.fee for r in outs), to_decimal("0"))
        assert to_decimal(b.get_account_state()["fees_charged"]) == total_fee

    def test_schedule_exhaustion_sweeps_remainder(self):
        b = PaperBroker(cm(), fill_schedule=FillSchedule(chunks=("0.5",)))
        b.submit_order(intent("p2", qty="1", px="100",
                              otype=OrderType.LIMIT))
        b.on_market_price("BTCUSDT", to_decimal("99"), 10)
        b.on_market_price("BTCUSDT", to_decimal("98"), 20)
        fills = [r for r in b.drain_reports() if r.status == "FILLED"]
        assert len(fills) == 1 and str(fills[0].filled_quantity) == "0.5"


class TestRejections:
    def test_zero_quantity_rejected_loudly(self):
        b = PaperBroker(cm())
        out = b.submit_order(intent("bad", qty="0"))
        assert out == "REJECTED"
        assert b.get_order("bad")["status"] == "REJECTED"
        rejected = [r for r in b.drain_reports() if r.status == "REJECTED"]
        assert len(rejected) == 1

    def test_below_min_order_qty_rejected(self):
        b = PaperBroker(cm(min_order_qty=to_decimal("0.01")))
        assert b.submit_order(intent("tiny", qty="0.001")) == "REJECTED"


class TestDuplicates:
    def test_duplicate_submission_suppressed(self):
        b = PaperBroker(cm())
        i = intent("dup")
        assert b.submit_order(i) == "ACCEPTED"
        assert b.submit_order(i) == "DUPLICATE"
        assert b.get_account_state()["duplicates"] == 1
        assert b.get_account_state()["submitted"] == 1       # not re-counted

    def test_report_ids_registry_blocks_replay(self):
        b = PaperBroker(cm())
        rid = "PAPER-9:FILLED:1"
        b.seed_report_ids([rid])
        # a rebuilt broker would never re-emit this id; verify registry holds
        assert rid in b._seen_report_ids


class TestCancellation:
    def test_new_order_cancels_cleanly(self):
        b = PaperBroker(cm())
        b.submit_order(intent("c1", px="50", otype=OrderType.LIMIT))
        assert b.cancel_order("c1") == "CANCELED"
        canceled = [r for r in b.drain_reports() if r.status == "CANCELED"]
        assert len(canceled) == 1
        assert b.cancel_order("nope") == "UNKNOWN"

    def test_cancel_filled_order_fails_loudly(self):
        b = PaperBroker(cm())
        b.submit_order(intent("f1"))
        b.drain_reports()
        b.on_market_price("BTCUSDT", to_decimal("100"), 1000)
        with pytest.raises(RuntimeError, match="invalid transition"):
            b.cancel_order("f1")


class TestJournal:
    def test_transitions_journaled_append_only(self, tmp_path):
        jpath = tmp_path / "j.jsonl"
        journal = RawJournal(str(jpath))
        b = PaperBroker(cm(), journal=journal)
        b.submit_order(intent("jm", px="100", otype=OrderType.LIMIT))
        b.on_market_price("BTCUSDT", to_decimal("95"), 2000)
        b.drain_reports()
        journal.close()

        text = jpath.read_text(encoding="utf-8")
        for event in ("order_submitted", "order_accepted",
                      "report_emitted", "full_fill"):
            assert f'"{event}"' in text
        assert '"paper_broker"' in text

    def test_journal_failure_fails_loudly(self, tmp_path):
        journal = RawJournal(str(tmp_path / "j.jsonl"))
        b = PaperBroker(cm(), journal=journal)
        journal.close()                       # simulate I/O failure
        with pytest.raises(RuntimeError, match="journal write failed"):
            b.submit_order(intent("x"))

    def test_close_stops_submissions(self):
        b = PaperBroker(cm())
        b.close()
        with pytest.raises(RuntimeError, match="closed"):
            b.submit_order(intent("late"))


class TestRestartRecovery:
    def test_rebuild_restores_counters_ids_and_dedup(self, tmp_path):
        jpath = tmp_path / "j.jsonl"
        journal = RawJournal(str(jpath))
        b = PaperBroker(cm(), journal=journal)
        b.submit_order(intent("r1", px="100", otype=OrderType.LIMIT))
        b.on_market_price("BTCUSDT", to_decimal("95"), 2000)
        b.drain_reports()
        before_state = b.get_account_state()
        journal.close()

        rebuilt = PaperBroker.rebuild_from_journal(str(jpath), cm())
        after = rebuilt.get_account_state()
        for key in ("submitted", "accepted", "fills", "partial_fills",
                    "rejected", "canceled"):
            assert after[key] == before_state[key]

        # id counter continues (never reuses PAPER-1)
        nxt = rebuilt.submit_order(intent("r2"))
        assert nxt == "ACCEPTED"
        assert rebuilt._orders["r2"].exchange_order_id == "PAPER-2"

        # dedup registry carried over: replayed report ids are suppressed
        assert any(rid.startswith("PAPER-1:") for rid in rebuilt._seen_report_ids)


class TestPortfolioPurity:
    def test_broker_holds_no_portfolio_reference(self):
        b = PaperBroker(cm())
        assert not hasattr(b, "portfolio")
        assert not hasattr(b, "_portfolio")

    def test_execution_views_are_not_accounting(self):
        """get_positions/get_account_state describe EXECUTIONS only."""
        b = PaperBroker(cm(slippage_pct=to_decimal("0")))
        b.submit_order(intent("v1", qty="2"))
        b.drain_reports()
        b.on_market_price("BTCUSDT", to_decimal("10"), 1000)
        b.drain_reports()
        assert b.get_positions() == {"BTCUSDT": "2"}     # executed net qty
        st = b.get_account_state()
        assert "equity" not in st and "cash" not in st and "pnl" not in st

    def test_frozen_fill_schedule_rejects_bad_values(self):
        with pytest.raises(ValidationError):
            FillSchedule(chunks="not-a-list")

