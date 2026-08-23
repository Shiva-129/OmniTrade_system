"""
Phase 11: Contract parity -- PaperBroker vs BinanceTestnetBroker (mock).

The same OrderIntent must produce semantically equivalent outcomes
through both venues (ACCEPTED/REJECTED/DUPLICATE, lifecycle, fees).
No exchange-specific behavior may leak into core.
"""
import pytest

from src.adapters.binance import BinanceTestnetBroker, BinanceTestnetConfig
from src.adapters.paper import FillSchedule, PaperBroker
from src.core.costs import CostModel
from src.core.money import to_decimal
from src.core.types import OrderIntent, OrderSide, OrderType


class MockExchange:
    def __init__(self, config):
        self.markets = {
            "BTCUSDT": {
                "symbol": "BTCUSDT",
                "info": {"filters": []},
                "limits": {"amount": {"min": 0.00001}},
                "precision": {"amount": 5, "price": 2},
            }
        }
        self._orders = {}
        self._id = 0
    def load_markets(self): return self.markets
    def create_order(self, symbol, otype, side, amount, price, params=None):
        self._id += 1
        coid = (params or {}).get("newClientOrderId", f"c{self._id}")
        o = {"id": str(self._id), "clientOrderId": coid, "symbol": symbol,
             "type": otype, "side": side, "amount": float(amount),
             "price": float(price) if price else 0, "filled": 0, "status": "NEW"}
        self._orders[coid] = o
        return o
    def fetch_order(self, cid, sym=None):
        if cid in self._orders: return self._orders[cid]
        raise Exception("not found")
    def cancel_order(self, cid, sym=None):
        o = self._orders[cid]
        if o["status"] in ("FILLED", "CANCELED"): raise Exception("cannot cancel")
        o["status"] = "CANCELED"; return o
    def fetch_open_orders(self, sym=None): return [o for o in self._orders.values() if o["status"] == "NEW"]
    def fetch_balance(self): return {"total": {}}
    def close(self): pass


def _cfg():
    return BinanceTestnetConfig(binance_env="testnet", api_key="k", api_secret="s")


def _intent(cloid="c1", qty="0.01", price="50000", side=OrderSide.BUY, otype=OrderType.LIMIT):
    return OrderIntent(client_order_id=cloid, symbol="BTCUSDT", side=side,
                       order_type=otype, quantity=to_decimal(qty),
                       price=to_decimal(price) if price else None, timestamp=1)


def _paper():
    return PaperBroker(CostModel())


def _binance():
    return BinanceTestnetBroker(_cfg(), exchange_factory=MockExchange)


class TestBrokerContractParity:
    @pytest.mark.parametrize("cloid,qty", [("a1", "0.01"), ("a2", "0.1")])
    def test_submit_accepted_both_venues(self, cloid, qty):
        for broker in (_paper(), _binance()):
            assert broker.submit_order(_intent(cloid=cloid, qty=qty)) == "ACCEPTED"

    def test_rejected_invalid_qty_both(self):
        for broker in (_paper(), _binance()):
            b = broker
            if isinstance(b, PaperBroker):
                b.cost_model = CostModel(min_order_qty=to_decimal("1"))
            # 0.000001 is below Binance minQty 0.00001 and Paper min 1
            result = b.submit_order(_intent(cloid="bad", qty="0.000001"))
            assert result == "REJECTED"

    def test_duplicate_idempotent_both(self):
        for broker in (_paper(), _binance()):
            b = broker
            b.submit_order(_intent(cloid="dup"))
            assert b.submit_order(_intent(cloid="dup")) == "DUPLICATE"

    def test_cancel_open_order_both(self):
        for broker in (_paper(), _binance()):
            b = broker
            b.submit_order(_intent(cloid="to-cancel"))
            assert b.cancel_order("to-cancel") == "CANCELED"

    def test_cancel_unknown_both(self):
        for broker in (_paper(), _binance()):
            assert broker.cancel_order("ghost") == "UNKNOWN"

    def test_cancel_filled_raises_both(self):
        for broker in (_paper(), _binance()):
            b = broker
            b.submit_order(_intent(cloid="fill-cancel"))
            # Simulate fill
            if isinstance(b, PaperBroker):
                b.on_market_price("BTCUSDT", to_decimal("50000"), 1)
                b.drain_reports()
            else:
                b._orders["fill-cancel"]["status"] = "FILLED"
            with pytest.raises(RuntimeError, match="invalid"):
                b.cancel_order("fill-cancel")

    def test_get_positions_execution_view_both(self):
        for broker in (_paper(), _binance()):
            b = broker
            b.submit_order(_intent(cloid="pos1", qty="0.01"))
            # For paper, need a price tick to fill LIMIT
            if isinstance(b, PaperBroker):
                b.on_market_price("BTCUSDT", to_decimal("50000"), 1)
                b.drain_reports()
            else:
                b._orders["pos1"]["filled"] = 0.01
                b._orders["pos1"]["status"] = "FILLED"
            # Both should now have a position view (paper via filled_qty, binance via fetch)
            # Just check the method doesn't crash and returns a dict
            pos = b.get_positions()
            assert isinstance(pos, dict)

    def test_no_production_leak_in_core(self):
        import pathlib, re
        for path in pathlib.Path("src/core").rglob("*.py"):
            text = path.read_text()
            # Check for actual imports, not comments mentioning binance_ws
            assert not re.search(r"^\s*(import|from).*binance", text, re.MULTILINE | re.IGNORECASE), f"{path} leaks binance import"
        for path in pathlib.Path("src/strategies").rglob("*.py"):
            text = path.read_text()
            assert not re.search(r"^\s*(import|from).*binance", text, re.MULTILINE | re.IGNORECASE)
