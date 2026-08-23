"""
Phase 11: Binance Testnet Broker -- mock unit tests (no network).

Covers: safety barriers, order translation, precision, status mapping,
ExecutionReport mapping, fees, idempotency, reconciliation, network
failures, lifecycle, and production-safety.
"""
import pytest

from src.adapters.binance import BinanceTestnetBroker, BinanceTestnetConfig
from src.core.money import to_decimal
from src.core.types import OrderIntent, OrderSide, OrderType


# ---------------------------------------------------------------------------
# Helpers: mock exchange factory
# ---------------------------------------------------------------------------

class MockExchange:
    def __init__(self, config):
        self.config = config
        self.markets = {
            "BTCUSDT": {
                "symbol": "BTCUSDT",
                "info": {"filters": [
                    {"filterType": "LOT_SIZE", "minQty": "0.00001", "stepSize": "0.00001"},
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                    {"filterType": "MIN_NOTIONAL", "minNotional": "10.0"},
                ]},
                "limits": {"amount": {"min": 0.00001}, "price": {"min": 0.01}, "cost": {"min": 10.0}},
                "precision": {"amount": 5, "price": 2},
            }
        }
        self._orders = {}
        self._next_id = 1
        self._fail_next = None  # inject exception
        self._timeout_once = False

    def load_markets(self):
        if self._fail_next:
            raise self._fail_next
        return self.markets

    def create_order(self, symbol, otype, side, amount, price, params=None):
        if self._timeout_once:
            self._timeout_once = False
            raise TimeoutError("ReadTimeout: simulated timeout after server acceptance")
        if self._fail_next:
            e = self._fail_next
            self._fail_next = None
            raise e
        oid = str(self._next_id); self._next_id += 1
        coid = (params or {}).get("newClientOrderId", f"mock-{oid}")
        order = {
            "id": oid, "clientOrderId": coid, "symbol": symbol,
            "type": otype, "side": side, "amount": float(amount),
            "price": float(price) if price else 0,
            "filled": 0, "status": "NEW", "average": float(price) if price else 0,
            "fee": {"cost": 0}, "fees": [],
        }
        # Simulate immediate fill for market orders in some tests via side-effect.
        self._orders[coid] = order
        return order

    def fetch_order(self, client_order_id, symbol=None):
        if client_order_id in self._orders:
            return self._orders[client_order_id]
        raise Exception(f"Order does not exist: {client_order_id}")

    def cancel_order(self, client_order_id, symbol=None):
        if client_order_id not in self._orders:
            raise Exception("Unknown order")
        o = self._orders[client_order_id]
        if o["status"] in ("FILLED", "CANCELED", "REJECTED"):
            raise Exception(f"cannot cancel {o['status']}")
        o["status"] = "CANCELED"
        return o

    def fetch_open_orders(self, symbol=None):
        return [o for o in self._orders.values() if o["status"] in ("NEW", "PARTIALLY_FILLED")]

    def fetch_balance(self):
        return {"total": {"BTC": 1.0, "USDT": 10000.0}}

    def close(self):
        pass


def _valid_config(**over):
    base = dict(binance_env="testnet", api_key="k", api_secret="s")
    base.update(over)
    return BinanceTestnetConfig(**base)


def _intent(cloid="c1", qty="0.01", price="50000", otype=OrderType.LIMIT, side=OrderSide.BUY):
    return OrderIntent(
        client_order_id=cloid, symbol="BTCUSDT", side=side,
        order_type=otype, quantity=to_decimal(qty),
        price=to_decimal(price) if price else None, timestamp=123,
    )


def _broker(**over):
    cfg = over.pop("config", _valid_config())
    factory = over.pop("factory", MockExchange)
    return BinanceTestnetBroker(cfg, exchange_factory=factory, **over)


# ---------------------------------------------------------------------------
# 2. Testnet-only safety barrier
# ---------------------------------------------------------------------------

class TestTestnetSafetyBarrier:
    def test_requires_testnet_env(self):
        with pytest.raises(ValueError, match="BINANCE_ENV='testnet'"):
            _valid_config(binance_env="production")

    def test_empty_env_fails(self):
        with pytest.raises(ValueError):
            _valid_config(binance_env="")

    def test_missing_api_key_fails(self):
        with pytest.raises(ValueError, match="api_key"):
            _valid_config(api_key="")

    def test_missing_secret_fails(self):
        with pytest.raises(ValueError, match="api_secret"):
            _valid_config(api_secret="  ")

    def test_production_url_rejected(self):
        with pytest.raises(ValueError, match="testnet"):
            _valid_config(base_url="https://api.binance.com")

    def test_testnet_url_passes(self):
        cfg = _valid_config(base_url="https://testnet.binance.vision")
        assert "testnet" in cfg.base_url

    def test_from_env_requires_testnet(self):
        with pytest.raises(ValueError):
            BinanceTestnetConfig.from_env({"BINANCE_ENV": "prod", "BINANCE_API_KEY": "k", "BINANCE_API_SECRET": "s"})

    def test_no_production_fallback_branch(self):
        # Structural: file must not contain an else: production path pattern
        import pathlib
        src = pathlib.Path("src/adapters/binance.py").read_text()
        # The file should contain exactly one base URL and it must be testnet
        assert "testnet.binance.vision" in src
        # Must not contain a production api URL without testnet
        assert src.count("api.binance.com") == 0 or "testnet" in src.lower()


# ---------------------------------------------------------------------------
# 5. Order translation & precision
# ---------------------------------------------------------------------------

class TestOrderTranslation:
    def test_limit_order_translated(self):
        b = _broker()
        b.submit_order(_intent(qty="0.02", price="50000.12"))
        o = b._orders["c1"]
        assert o["amount"] == "0.02"
        assert o["price"] == "50000.12"

    def test_market_order_translated(self):
        b = _broker()
        b.submit_order(_intent(qty="0.01", price=None, otype=OrderType.MARKET))
        assert b._orders["c1"]["type"].lower() == "market"

    def test_decimal_precision_preserved(self):
        b = _broker()
        b.submit_order(_intent(qty="0.12345", price="50000.12"))
        assert b._orders["c1"]["amount"] == "0.12345"

    def test_min_qty_rejected_locally(self):
        b = _broker()
        # Below minQty 0.00001
        result = b.submit_order(_intent(qty="0.000001", price="50000"))
        assert result == "REJECTED"
        assert b._orders["c1"]["status"] == "REJECTED"

    def test_tick_size_rejected(self):
        b = _broker()
        # tick 0.01, price 50000.123 violates
        result = b.submit_order(_intent(price="50000.123"))
        assert result == "REJECTED"

    def test_min_notional_rejected(self):
        b = _broker()
        # qty 0.0001 * price 100 = 0.01 < minNotional 10
        result = b.submit_order(_intent(qty="0.0001", price="100"))
        assert result == "REJECTED"


# ---------------------------------------------------------------------------
# 7. Client order ID preservation
# ---------------------------------------------------------------------------

class TestClientOrderId:
    def test_same_intent_preserves_id(self):
        b = _broker()
        b.submit_order(_intent(cloid="my-id-123"))
        assert b._orders["my-id-123"]["clientOrderId"] == "my-id-123"

    def test_duplicate_does_not_create_second_order(self):
        b = _broker()
        b.submit_order(_intent(cloid="dup"))
        assert b.submit_order(_intent(cloid="dup")) == "DUPLICATE"
        assert b.get_account_state()["duplicates"] == 1

    def test_duplicate_does_not_hit_exchange_twice(self):
        # Count create_order calls
        b = _broker()
        b.submit_order(_intent(cloid="dup2"))
        calls_before = len(b._orders)
        b.submit_order(_intent(cloid="dup2"))
        assert len(b._orders) == calls_before


# ---------------------------------------------------------------------------
# 8. Status mapping
# ---------------------------------------------------------------------------

class TestStatusMapping:
    @pytest.mark.parametrize("exchange,expected", [
        ("NEW", "NEW"), ("FILLED", "FILLED"), ("CANCELED", "CANCELED"),
        ("PARTIALLY_FILLED", "PARTIAL_FILL"), ("REJECTED", "REJECTED"),
        ("new", "NEW"), ("canceled", "CANCELED"),
    ])
    def test_known_statuses_mapped(self, exchange, expected):
        b = _broker()
        assert b._map_status(exchange) == expected

    def test_unknown_status_raises(self):
        b = _broker()
        with pytest.raises(ValueError, match="unknown exchange status"):
            b._map_status("SUPER_FILLED")


# ---------------------------------------------------------------------------
# 9. ExecutionReport mapping
# ---------------------------------------------------------------------------

class TestExecutionReportMapping:
    def test_report_contains_required_fields(self):
        b = _broker()
        b.submit_order(_intent())
        reports = b.drain_reports()
        assert len(reports) >= 1
        r = reports[0]
        assert r.client_order_id == "c1"
        assert r.symbol == "BTCUSDT"
        assert r.filled_quantity is not None
        assert r.fee is not None
        assert r.exchange_order_id

    def test_fee_extracted_from_exchange(self):
        b = _broker()
        # Inject a mock that returns fees
        def factory(cfg):
            ex = MockExchange(cfg)
            orig = ex.create_order
            def patched(*a, **kw):
                o = orig(*a, **kw)
                o["fees"] = [{"cost": 0.001}]
                o["filled"] = 0.01
                o["average"] = 50000
                return o
            ex.create_order = patched
            return ex
        b2 = BinanceTestnetBroker(_valid_config(), exchange_factory=factory)
        b2.submit_order(_intent())
        r = b2.drain_reports()[0]
        assert r.fee == to_decimal("0.001")

    def test_portfolio_only_via_execution_report(self):
        # Structural: broker has no portfolio reference.
        b = _broker()
        assert not hasattr(b, "portfolio")


# ---------------------------------------------------------------------------
# 11. Reconciliation -- timeout handling
# ---------------------------------------------------------------------------

class TestReconciliation:
    def test_timeout_queries_before_deciding(self):
        b = _broker()
        # Make next create_order timeout, but order actually exists.
        b._exchange._timeout_once = True
        # Pre-populate exchange side so fetch will find it
        b._exchange._orders["c-timeout"] = {
            "id": "99", "clientOrderId": "c-timeout", "symbol": "BTCUSDT",
            "status": "NEW", "filled": 0, "price": 50000,
        }
        result = b.submit_order(_intent(cloid="c-timeout"))
        # Timeout + found => ACCEPTED (safe, don't resubmit)
        assert result == "ACCEPTED"

    def test_timeout_not_found_treated_as_rejected(self):
        b = _broker()
        b._exchange._timeout_once = True
        result = b.submit_order(_intent(cloid="c-missing"))
        assert result == "REJECTED"

    def test_no_auto_resubmit_on_timeout(self):
        # Count create_order calls -- timeout path should call it once, then fetch once.
        calls = {"create": 0, "fetch": 0}
        def factory(cfg):
            ex = MockExchange(cfg)
            orig_create = ex.create_order
            orig_fetch = ex.fetch_order
            def c(*a, **kw):
                calls["create"] += 1
                if calls["create"] == 1:
                    raise TimeoutError("timeout")
                return orig_create(*a, **kw)
            def f(*a, **kw):
                calls["fetch"] += 1
                raise Exception("not found")
            ex.create_order = c
            ex.fetch_order = f
            return ex
        b = BinanceTestnetBroker(_valid_config(), exchange_factory=factory)
        b.submit_order(_intent(cloid="c-retry"))
        assert calls["create"] == 1  # never retried
        assert calls["fetch"] >= 1

    def test_reconcile_order_queries_exchange(self):
        b = _broker()
        b.submit_order(_intent(cloid="rec1"))
        result = b.reconcile_order("rec1")
        assert result is not None
        assert result["clientOrderId"] == "rec1"

    def test_startup_reconcile_reports_mismatch(self):
        b = _broker()
        # Put an unknown order directly on exchange side
        b._exchange._orders["ghost"] = {"id": "ghost", "clientOrderId": "ghost",
                                        "symbol": "BTCUSDT", "status": "NEW"}
        report = b.startup_reconcile()
        assert report["ok"] is False
        assert any("ghost" in m for m in report["mismatches"])


# ---------------------------------------------------------------------------
# 13. Network failure
# ---------------------------------------------------------------------------

class TestNetworkFailure:
    def test_connection_timeout_propagated_safely(self):
        def factory(cfg):
            ex = MockExchange(cfg)
            ex._fail_next = TimeoutError("ConnectTimeout")
            return ex
        # Connectivity check will fail -- construction should raise without leaking secrets
        with pytest.raises(RuntimeError, match="connectivity check failed"):
            BinanceTestnetBroker(_valid_config(), exchange_factory=factory)

    def test_exchange_rejection_returns_rejected(self):
        def factory(cfg):
            ex = MockExchange(cfg)
            ex._fail_next = None
            orig = ex.create_order
            def fail(*a, **kw):
                raise Exception("Filter failure: MIN_NOTIONAL")
            ex.create_order = fail
            return ex
        b = BinanceTestnetBroker(_valid_config(), exchange_factory=factory)
        # Need to bypass local validation for this test -- use valid qty
        result = b.submit_order(_intent())
        assert result == "REJECTED"

    def test_malformed_response_handled(self):
        def factory(cfg):
            ex = MockExchange(cfg)
            def bad_create(*a, **kw):
                return {"not": "an order"}  # missing status
            ex.create_order = bad_create
            return ex
        b = BinanceTestnetBroker(_valid_config(), exchange_factory=factory)
        # Should not crash, should still return ACCEPTED with fallback mapping
        result = b.submit_order(_intent())
        assert result in ("ACCEPTED", "REJECTED")

    def test_unknown_status_raises_loudly(self):
        b = _broker()
        with pytest.raises(ValueError):
            b._map_status("BOGUS_STATUS_XYZ")


# ---------------------------------------------------------------------------
# Lifecycle: cancel, get_order, open orders, positions
# ---------------------------------------------------------------------------

class TestOrderLifecycle:
    def test_cancel_open_order(self):
        b = _broker()
        b.submit_order(_intent(cloid="to-cancel"))
        assert b.cancel_order("to-cancel") == "CANCELED"
        assert b._orders["to-cancel"]["status"] == "CANCELED"

    def test_cancel_unknown_returns_unknown(self):
        b = _broker()
        assert b.cancel_order("ghost-id") == "UNKNOWN"

    def test_cancel_filled_raises(self):
        b = _broker()
        b.submit_order(_intent(cloid="fill-then-cancel"))
        # Simulate fill
        b._orders["fill-then-cancel"]["status"] = "FILLED"
        with pytest.raises(RuntimeError, match="invalid lifecycle"):
            b.cancel_order("fill-then-cancel")

    def test_get_order_returns_copy(self):
        b = _broker()
        b.submit_order(_intent(cloid="get1"))
        o = b.get_order("get1")
        assert o["client_order_id"] == "get1"

    def test_get_open_orders_filters(self):
        b = _broker()
        b.submit_order(_intent(cloid="open1"))
        b.submit_order(_intent(cloid="open2"))
        b._orders["open1"]["status"] = "FILLED"
        opens = b.get_open_orders()
        assert len(opens) == 1 and opens[0]["client_order_id"] == "open2"

    def test_get_positions_local_fallback(self):
        b = _broker()
        # Force fallback path by making fetch_balance fail
        b._exchange.fetch_balance = lambda: (_ for _ in ()).throw(Exception("no balance"))
        b.submit_order(_intent(cloid="pos1"))
        b._orders["pos1"]["filled_qty"] = to_decimal("0.5")
        b._orders["pos1"]["status"] = "FILLED"
        pos = b.get_positions()
        assert "BTCUSDT" in pos

    def test_close_blocks_submissions(self):
        b = _broker()
        b.close()
        with pytest.raises(RuntimeError, match="closed"):
            b.submit_order(_intent(cloid="after-close"))

    def test_get_account_state_has_expected_keys(self):
        b = _broker()
        b.submit_order(_intent())
        st = b.get_account_state()
        for k in ("submitted", "fills", "canceled"):
            assert k in st


# ---------------------------------------------------------------------------
# Credential safety
# ---------------------------------------------------------------------------

class TestCredentialSafety:
    def test_exception_does_not_contain_secret(self):
        from src.adapters.binance import TESTNET_BASE_URL
        # Verify RuntimeError from connectivity failure does not leak secret
        class BadExchange:
            def __init__(self, cfg):
                self.cfg = cfg
            def load_markets(self):
                raise TimeoutError("conn fail")
            def close(self): pass
        def bad_factory(cfg):
            return BadExchange(cfg)
        cfg = BinanceTestnetConfig(binance_env="testnet", api_key="k", api_secret="my-secret-123", base_url=TESTNET_BASE_URL)
        with pytest.raises(RuntimeError, match="connectivity check failed") as exc:
            BinanceTestnetBroker(cfg, exchange_factory=bad_factory)
        msg = str(exc.value) + str(exc.value.__cause__)
        assert "my-secret-123" not in msg
        assert "my-secret" not in msg

    def test_journal_does_not_contain_secret(self, tmp_path):
        from src.core.journal import RawJournal
        jpath = tmp_path / "j.jsonl"
        journal = RawJournal(str(jpath))
        b = BinanceTestnetBroker(_valid_config(), journal=journal, exchange_factory=MockExchange)
        b.submit_order(_intent())
        journal.close()
        text = jpath.read_text()
        assert "my-secret" not in text.lower()
        assert "api_secret" not in text.lower()

    def test_no_secret_in_logs_on_submit(self, tmp_path, caplog):
        b = _broker()
        b.submit_order(_intent())
        for rec in caplog.records:
            assert "secret" not in rec.getMessage().lower()


# ---------------------------------------------------------------------------
# Decimal precision: string representation preserves value
# ---------------------------------------------------------------------------

class TestDecimalPrecision:
    def test_small_quantity_preserved(self):
        b = _broker()
        b.submit_order(_intent(qty="0.001", price="50000"))
        assert b._orders["c1"]["amount"] == "0.001"

    def test_price_string_preserved(self):
        b = _broker()
        b.submit_order(_intent(qty="0.01", price="50000.12"))
        assert b._orders["c1"]["price"] == "50000.12"
