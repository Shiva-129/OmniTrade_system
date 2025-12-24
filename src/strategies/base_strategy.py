from abc import ABC, abstractmethod
import datetime

class BaseStrategy(ABC):
    def __init__(self, name, symbol, timeframe):
        self.name = name
        self.symbol = symbol
        self.timeframe = timeframe
        self.is_active = False
        self.current_position = 0 # 0: Flat, 1: Long, -1: Short

    @abstractmethod
    def generate_signal(self, data):
        """
        Logic to decide if we should buy or sell. 
        Must return: 'BUY', 'SELL', or 'HOLD'
        """
        pass

    @abstractmethod
    def calculate_position_size(self, account_balance):
        """
        Standardizes how much capital to risk. 
        Never leave this to 'feeling'.
        """
        pass

    def check_risk(self, signal, risk_manager):
        """
        The Filter: Checks with the Risk Manager before sending the order.
        """
        print(f"[{datetime.datetime.now()}] {self.name}: Validating {signal} via Risk Manager...")
        return risk_manager.validate_trade(self.symbol, signal)

    def execute_trade(self, signal, execution_engine):
        """
        The Hand: Sends the validated signal to the actual broker adapter.
        """
        if signal != 'HOLD':
            print(f"[{datetime.datetime.now()}] {self.name}: Executing {signal} for {self.symbol}")
            execution_engine.place_order(self.symbol, signal)