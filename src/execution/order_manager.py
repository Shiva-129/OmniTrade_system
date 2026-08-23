"""
OrderManager placeholder — not yet implemented.
Any use must fail loud rather than silently no-op.
"""
class OrderManager:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("OrderManager is not implemented — use src.core.engine.TradingEngine + PaperBroker/BinanceTestnetBroker directly")

    def submit_order(self, *args, **kwargs):
        raise NotImplementedError("OrderManager not implemented")

    def cancel_order(self, *args, **kwargs):
        raise NotImplementedError("OrderManager not implemented")
