# OmniTrade System

A deterministic backtesting and research platform for testing trading strategies on historical data. Give it market data, a strategy, and a parameter space — it runs backtests with realistic costs, validates out-of-sample, and tells you if the improvement is actually robust.

This is not a system that magically finds profitable strategies. It's a research and execution platform designed to test whether a strategy is worth trading.

## What does it do?

Historical OHLCV
→ data validation
→ strategy
→ backtest (realistic fees and slippage)
→ train / validation / test split
→ parameter search (validation only)
→ walk-forward testing
→ cost and robustness checks
→ accept / reject / inconclusive
→ immutable experiment record
→ paper / testnet execution (only if accepted)

Test data is never used for parameter selection. If no candidate beats the baseline under realistic assumptions, the correct result is `NO ROBUST IMPROVEMENT FOUND`.

## Why I built it

I wanted more than a simple backtesting script that shows a nice equity curve by overfitting. Most scripts can be made to look profitable by tuning parameters on the same data they are tested on. I wanted a platform that enforces the boring but important parts: chronological splits, realistic execution (close → open next bar), Decimal accounting, and strict validation before anything reaches paper trading.

## Strategies

Implemented in `src/strategies/`:

- EMA crossover
- SMA trend
- Z-score mean reversion
- Donchian breakout
- RSI momentum
- Bollinger mean reversion
- MACD trend
- ADX / volatility
- EMA + RSI filtered (research variant, example of a controlled modification)

Currently supported by the automated V2 parameter-search grid (`research/validation/param_space.py`):

- `ema_crossover`, `ema_rsi_filtered`, `zscore_mean_reversion`, `donchian_breakout`

The other strategies exist as deterministic `BaseStrategy` examples but need a `REQUIRED_AXES` entry to be grid-optimized. That's intentional — not every strategy is automatically optimized.

## Research pipeline

```
Historical Data (CSV/Parquet)
    ↓
Validation (ordering, gaps, OHLC, dataset hash)
    ↓
Strategy (frozen config, deterministic)
    ↓
Backtest (Decimal, close → open T+1, adverse slippage + taker fee)
    ↓
Parameter search (train → validation)
    ↓
Walk-forward (train → test rolling, test untouched)
    ↓
Cost sensitivity (1× vs 2× slippage)
    ↓
Robustness (neighborhood, regime, trade count)
    ↓
Decision: ACCEPT / REJECT / INCONCLUSIVE + reasons
```

Each step is deterministic. Same dataset + same config + same seed = same `experiment_id`, same `config_hash`, same metrics.

## Example result

The V2 demo on a 60-bar synthetic oscillating sample is **explicitly synthetic** (not real market performance):

- EMA `fast 2 / slow 5` baseline Sharpe `-77`, candidate `fast 2 / slow 5` validation Sharpe `-222` → rejected
- Z-score and Donchian also rejected (insufficient trades, cost sensitivity failed, walk-forward `positive_rate < 0.5`)

On real Binance `BTCUSDT 1h` 2020-2024 (43,816 bars, gaps 15 from exchange downtime, `f630861c`):

- EMA, Z-score, Donchian — all **REJECT** (`validation Sharpe not > baseline +0.1`, `2× slippage Sharpe <=0`, `<5` trades on test)
- EMA + RSI filtered also **REJECT**

This is not a failure of the platform. The platform correctly rejected ideas that didn't survive costs and walk-forward. No experiment has yet been promoted to `ACCEPT`.

## Safety

- **SafetyController** is the only safety authority: `HEALTHY → DEGRADED → HALT` (terminal).
- `HALT` blocks both increasing and reducing orders. `DEGRADED` blocks increasing, allows reducing via `is_reducing` (exact `Decimal`).
- Every order follows `Strategy → Safety → RiskManager → Gatekeeper → Broker`. Order: `["risk","gatekeeper","broker"]` is mutation-tested.
- `Gatekeeper=None` is research-only. For `PAPER`/`TESTNET` it fails closed (`gatekeeper_missing_blocked` metric, no `Broker`).
- Duplicate `ExecutionReport` suppressed (`engine._seen_exec_ids`), journal write-ahead `flush+fsync`.
- Secrets never logged (`with pytest.raises` outside `except:pass` covers it).

No hidden `submit_order` path. Only `src/core/engine.py:615` `broker.submit_order` is legitimate.

## Determinism

- Financial accounting uses `Decimal` (`src/core/money.py` `to_decimal(str())`, `float` only at documented indicator math boundary).
- Event ordering `sorted(ts, symbol)` insertion-order independent.
- Journal append-only `RawJournal` + `JournalReader` skips corrupt `not json` `n==2`.
- Replay `live snapshot == replay snapshot` plus hand `Decimal("4971.85")` financial.
- Experiments `experiment_id = sha256(sort_keys)` deterministic, `registry append-only`.

## Tech stack

- Python 3.13
- Pytest
- Pydantic, structlog
- Pandas, NumPy, PyArrow
- Redis (live `DB 15` for tests)
- CCXT, websockets (Binance testnet, deferred)
- VectorBT (research backend, long-only)
- Decimal

## Project structure

```
src/        core (engine, portfolio, safety, risk, journal, money)
            gatekeeper (guard, state_controller)
            adapters (paper, binance testnet, user stream)
            strategies (base + 5 + filtered)
            indicators, markets, observability, simulator
research/   data (ingestion, dataset, bars, validate)
            evaluation (backtest engine, metrics, costs, multi-symbol)
            validation (param_space, sweep, walkforward, robustness, verdict)
            experiments (registry)
            v2 (experiment engine, CLI)
tests/      behavioral + structural (see Testing)
```

## Running it

```bash
# setup
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt

# normal test gate
py -m pytest -q
# → 602 passed, 2 skipped (opt-in Binance testnet)

# historical ingestion (real CSV/Parquet, not synthetic)
from research.data.ingestion import ingest_historical_file
res = ingest_historical_file("research/data/raw/BTCUSDT_1h.csv", "BTC/USDT", "1h")
print(res.dataset.content_hash()[:8], res.report.is_clean)  # missing file → DATA_REQUIRED

# V2 research (synthetic demo, labelled synthetic)
py -m research.v2.cli optimize --strategy ema_crossover --dataset BTC/USDT --train-bars 30 --test-bars 10 --step-bars 10
py -m research.v2.cli list
py -m research.v2.cli inspect <experiment_id>
py -m research.v2.cli reproduce <experiment_id>
py -m research.v2.cli best
```

## Data

Raw historical market data is **not committed** (large, `.gitignore` `research/data/raw/*`). Only `research/data/metadata/*.json` (`symbol`, `timeframe`, `source`, `retrieved_at`, `content_hash`, `validation`) is committed.

To use real data:

1. Download `BTCUSDT 1h` (or `ETHUSDT`/`SOLUSDT`) from Binance official `https://api.binance.com/api/v3/klines` (UTC, paginated `lastClose+1`, raw `[[openTime,open,high,low,close,volume,...]]` preserved in `research/data/raw/*.json` + normalized `*.csv`).
2. Current real sets in this repo (if present locally): `BTCUSDT 1h 2020-01-01→2024-12-31 43,816 rows f630861c`, `ETHUSDT` same, `SOLUSDT` `2020-08-11→2024-12-31 38,470 rows ada0c483` (gaps 15/10 from exchange downtime, reported not repaired).
3. Same file → same `content_hash` (re-ingestion `hash1==hash2`).

If `research/data/raw/BTCUSDT_1h.csv` does not exist, ingestion returns `DATA_REQUIRED: file not found` — no synthetic fallback.

## Testing

- `602 passed, 2 skipped` (2 = `tests/test_binance_testnet_integration.py` `skipif RUN_BINANCE_TESTNET!="1"`).
- `10` ingestion hard-gate `tests/test_data_ingestion.py`, `10` V2 `tests/test_v2_engine.py`, `9` hardening `tests/test_phase15_6_hardening.py`, `6` `tests/test_j1_gate.py` (including `HALT` blocks both vs `DEGRADED` allows reducing) — all `FAIL when broken` mutation-proven.

No `if False`/`or True`/`assert True` theatrical tests remain.

## Current limitations

- No profitability guarantee. Research on synthetic and on real `1h` 2020-2024 so far correctly `REJECT`ed candidates (cost/robustness).
- Historical `1h` only for `BTC/ETH/SOL`; no `1d`/`4h` full history yet. `2025` beyond `2024-12-31` not downloaded.
- V2 grid only `ema`, `zscore`, `donchian`, `ema_rsi_filtered`; other strategies need `REQUIRED_AXES` extension.
- `vectorbt` backend long-only (arrays `se/sx` proven, engine not short) — research only.
- `journal` `filelock` for concurrent writers not needed (single-process `flush+fsync`).
- Live `RUN_BINANCE_TESTNET` not run in CI (opt-in, correctly not claimed as mock).

## Roadmap

- Real `1h`/`4h` history for `BTC`/`ETH`/`SOL` 2021-2025 already acquired locally, next: larger `train 500 test 100` V2 runs on full `43k` set with non-synthetic regime splits
- Extend `ParameterSpace` for remaining strategies if needed
- Paper/testnet validation only for `ACCEPT` candidates (none yet)
- Stronger `except:pass` observability → `logger.debug` already done for 20 handlers (remaining 8 `stop()` best-effort `pass` documented)

## License

No license file currently in repository.

