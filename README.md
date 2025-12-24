# Omni-Trade System

A production-ready modular algorithmic trading system.

## Refined Structure
- `src/`: Core source code.
    - `adapters/`: Broker connectivity (Zerodha, Binance, etc.).
    - `core/`: Heartbeat, risk management, and portfolio tracking.
    - `data/`: Data loaders and cleaning processors.
    - `strategies/`: Trading logic and signal templates.
    - `execution/`: Order management and funded account gateways.
- `research/`: Jupyter notebooks and backtesting analysis.
- `deployments/`: Infrastructure and deployment configs (Docker, Cloud).
- `tests/`: Unit and integration testing suites.
- `main.py`: Primary entry point for the system.
