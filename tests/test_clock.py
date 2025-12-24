import unittest
import time
from src.core.clock import Clock

class TestClock(unittest.TestCase):
    def test_precision(self):
        t1 = Clock.now_us()
        time.sleep(0.001)
        t2 = Clock.now_us()
        self.assertIsInstance(t1, int)
        self.assertGreater(t2, t1)
        
    def test_drift(self):
        # Fake exchange drift
        local = 1000
        exchange = 1500
        drift = Clock.calculate_drift(exchange, local)
        self.assertEqual(drift, 500)

if __name__ == '__main__':
    unittest.main()
