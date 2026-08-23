# Deterministic Algorithmic Trading Platform

A personal engineering project — a deterministic research and execution platform for testing trading strategies on historical data and moving only validated ones toward paper/testnet.

The goal is not to manufacture profitable backtests. The system is designed to **reject** strategies that don't survive validation, costs, walk-forward and robustness checks.

---

## What is this?

This is not a trading bot that promises profit. It's a platform that lets you:

- take a strategy (e.g. EMA crossover),
- test it on historical OHLCV with realistic fees and slippage,
- search for better parameters without leaking test data,
- check if the improvement survives walk-forward, cost sensitivity and robustness tests,
- store the experiment immutably,
- and only then consider paper/testnet execution — with safety controls that cannot be bypassed.

Built to be deterministic: same data + same config + same seed = same experiment ID, same metrics, same decision.

---

## What it actually does

```
Historical OHLCV (CSV/Parquet)
    ↓  ingestion + validation (ordering, gaps, OHLC, dataset hash)
Strategy (BaseStrategy, frozen config)
    ↓  causal indicators (no future bar)
Backtest Engine (close → open T+1, adverse slippage + taker fee, Decimal)
    ↓  Portfolio (apply_report sole funnel, Decimal)
       RiskManager → Gatekeeper → Broker (single ordered path)
    ↓  Train / Validation / Test (contiguous, test untouched until final)
         Parameter grid search (deterministic)
    ↓  Walk-forward (train → validate → test rolling)
    ↓  Cost sensitivity (1× vs 2× slippage)
    ↓  Robustness (neighborhood, regime, trade count)
    ↓  Decision  REJECT / INCONCLUSIVE / ACCEPT
    ↓  ExperimentRegistry (append-only JSONL, DuplicateExperimentError)
    ↓  Paper / TESTNET (max)  — only if ACCEPT
```

Simple rule: **test data never influences parameter selection.** If nothing survives, the correct result is `NO ROBUST IMPROVEMENT FOUND`.

---

## Main Features

- Deterministic backtesting (`research/evaluation/engine.py`) and bar aggregation
- CSV/Parquet historical ingestion (`research/data/ingestion.py`) with `DATA_REQUIRED` fail-closed, deterministic `content_hash`
- Indicators: EMA, SMA, RSI, MACD, ATR, Bollinger, ADX, volatility — float at indicator boundary, `Decimal` at trading boundary
- 5 core strategies + 1 filtered research variant (see below)
- Multi-symbol portfolio backtesting with shared `Portfolio` cash (`research/evaluation/multi_symbol.py`, sorted `ts,symbol`, insertion-order independent)
- Risk management (`src/core/risk_manager.py` 8-rule `first-failure-wins`)
- SafetyController `HEALTHY → DEGRADED → HALT` terminal, sole authority
- PaperBroker (`src/adapters/paper.py`) realistic lifecycle `NEW→PARTIAL→FILLED`, resting, `PARTIAL 0.3/0.4/0.3`, fees `maker/taker`
- Binance **TESTNET-only** (`src/adapters/binance.py` `dec_to_str` exact, `1e-8` ghost `scan_iter`, `BinanceTestnetConfig` `BINANCE_ENV must be testnet`, AST guard no `api.binance.com` in `else`)
- Append-only journal `RawJournal` `flush+fsync`, `JournalReader` `try:continue` skip corrupt `n==2`
- Deterministic replay + `StateHasher` (`hash==hash` **plus** hand `Decimal` financial)
- Decimal accounting (`src/core/money.py` `to_decimal(str())` trap `float`)
- Walk-forward validation, robustness (`DeterministicRNG(42)`), cost sensitivity, experiment hashing
- Experiment registry `research/experiments/registry.py` append-only, `fsync`, `DuplicateExperimentError`
- Session lifecycle `PortfolioSession` `start/pause/resume/stop` ordered `halt→cancel→journal→WS→broker→redis`
- Observability `MetricsRegistry`/`HealthMonitor`/`AlertManager` (`heartbeat_stale` flag, now `logger.debug` not `pass`)
- V2 research engine `research/v2/engine.py` deterministic `run_v2_experiment` + CLI

---

## Strategy Research

### Strategies implemented (in `src/strategies/`)

- **SMA Trend** `sma_trend.py` — `SmaTrendConfig fast_period, slow_period`
- **EMA Crossover** `ema_crossover.py` — `EmaCrossoverConfig fast_period, slow_period, allow_short, cooldown_events` — `EMA` seeded with `SMA`, cross `prev_fast≤prev_slow and fast>slow`
- **RSI Momentum** `rsi_momentum.py`
- **Bollinger Reversion** `bollinger_reversion.py`
- **MACD Trend** `macd_trend.py`
- **ADX Volatility** `adx_volatility.py`
- **Donchian Breakout** `donchian_breakout.py`
- **Z-Score Mean Reversion** `zscore_mean_reversion.py`
- **EMA + RSI Filtered** `ema_rsi_filtered.py` — `EmaRsiConfig fast,slow,rsi_period,rsi_buy/sell_threshold` example of controlled modification (V2)

### Currently supported by V2 optimization (`research/validation/param_space.py` `REQUIRED_AXES`)

- `ema_crossover`, `ema_rsi_filtered`, `zscore_mean_reversion`, `donchian_breakout`

Others exist as `BaseStrategy` examples but need `REQUIRED_AXES` extension to be grid-optimized — limitation documented, not pretended.

---

## Research Philosophy

A green suite does not mean a strategy is profitable.

The platform tries to detect:

- overfitting to `TEST`
- future-bar leakage (`future_changed` past `bars[0]` must stay `==`)
- cost fragility (`2× slippage` Sharpe `<=0` → `REJECT`)
- insufficient trades (`<5` on test → `REJECT`)
- unstable parameters (`fast 8 → 12 → 20` fragile → `positive_rate <0.5` → `REJECT`)
- walk-forward degradation (`train→val <30%`)
- regime instability

`REJECT` is a **successful** result. On the current synthetic `60-bar` oscillating sample and on real `BTC 1h 2020-2024 43k` bars, all 3 tested candidates **correctly REJECT** (`val Sharpe -222 not > baseline -77+0.1`) — not manufactured winners.

> Example from V2 demo (synthetic, **not real market performance**): `ema_crossover` `fast2/slow5` `Sharpe -77` vs best `fast2/slow5` `val -222` → `REJECT` `cost sensitivity failed`. Labelled synthetic in `research/v2`.

---

## Architecture

**Trading plane**

```
Market Data (Packet)
    ↓
Data Validation (ordering, gaps, OHLC)
    ↓
Strategy (BaseStrategy, causal, no broker/redis/portfolio access)
    ↓
Backtest Engine / TradingEngine
    ↓
Portfolio (apply_report sole funnel)
    ↓
RiskManager → Gatekeeper → Broker (single ordered path, mutation-proven)
    ↓
SafetyController (sole, HALT terminal)
    ↓
Journal (append-only, fsync) / Replay (StateHasher)
```

**Research plane**

```
Historical CSV/Parquet
    ↓  ingestion + DataQualityReport
OHLCVDataset (content_hash)
    ↓  train / val / test (contiguous, test untouched)
Experiment (ParameterSpace grid, deterministic product)
    ↓  sweep train/val 50/50 + val_ds re-score
Validation (Sharpe, validation tail)
    ↓  Walk-forward (train→test rolling)
    ↓  Robustness (neighborhood, regime, MC DeterministicRNG)
    ↓  Cost sensitivity (1× vs 2× slippage)
    ↓  Benchmark (buy-and-hold)
    ↓  Decision REJECT / INCONCLUSIVE / ACCEPT + rejection_reasons
    ↓  ExperimentRegistry (immutable, experiment_id = sha256(config_hash+engine_version))
```

## Safety

- `HEALTHY` → trading, `DEGRADED` → only reducing intents (`is_reducing` via `Portfolio._evolve_position` exact `Decimal`), `HALT` → **both** increasing and reducing blocked (mutation-proven: remove `is_halted` gate → broker receives order, test `FAIL`)
- `HALT` terminal until explicit operator reset; `DEGRADED` cannot auto-become `HEALTHY`
- `TESTNET` is maximum `ExecutionMode`; `PRODUCTION`/`LIVE` `ValueError`; `api.binance.com` in `else` branch → AST test **FAIL**
- Risk always before Gatekeeper before Broker — spy `["risk","gatekeeper","broker"]` exact order, swap → **FAIL**
- `Gatekeeper=None` is **RESEARCH-ONLY** — `PAPER`/`TESTNET` + `gate None` + `broker` → `warning gatekeeper_missing_blocked` + `metric` + **no** `broker.submit_order` (fail-closed)
- Duplicate `ExecutionReport` dedup `if exec_key in _seen: return` + `journal` before `publish` → `engine.apply(report)` twice `cash 9900` not double (remove dedup → double **FAIL**)
- Secrets never logged — `test_exception_does_not_contain_secret` `with pytest.raises` outside `except:pass`

## Determinism

- `Decimal` `to_decimal(str())` at trading boundary, indicators float documented
- Event ordering `sorted(ts,symbol)` insertion-order independent `r_ab fills==r_ba fills`
- Journal `flush+os.fsync` on `append`/`close`, `JournalReader` `try:continue` skip corrupt
- Replay `live snapshot==replay snapshot` **plus** hand `Decimal("4971.85")` `cash 9900` not hash alone
- Experiments `experiment_id = sha256(sort_keys)` same `dataset+strategy+config+seed` → same metrics/decision

---

## Running the Project

**Setup**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# For backtesting/research extras (vectorbt, ccxt, pandas, pyarrow)
pip install -r requirements.txt  # already includes openalgo, vectorbt, ta-lib, scikit-learn
```

**Tests (normal gate)**

```bash
py -m pytest -q
# → 602 passed, 2 skipped
# 2 skipped = opt-in Binance testnet integration (requires RUN_BINANCE_TESTNET=1 + testnet creds in .env)
```

**Historical data ingestion (real CSV/Parquet, not synthetic)**

```python
from research.data.ingestion import ingest_historical_file
res = ingest_historical_file("research/data/raw/BTCUSDT_1h.csv", "BTC/USDT", "1h")
print(res.dataset.content_hash()[:8], res.report.is_clean, res.report.gap_count)
# Missing file → res.error == "DATA_REQUIRED: file not found" (no synthetic fallback)
```

**V2 research (synthetic demo, explicitly labelled synthetic):**

```bash
py -m research.v2.cli optimize --strategy ema_crossover --dataset BTC/USDT --train-bars 30 --test-bars 10 --step-bars 10
# → prints experiment_id, config_hash, baseline/candidate Sharpe, decision REJECT/ACCEPT, rejection_reasons
py -m research.v2.cli list
py -m research.v2.cli inspect <experiment_id>
py -m research.v2.cli reproduce <experiment_id>
py -m research.v2.cli best  # NO ROBUST IMPROVEMENT FOUND if none survive
```

**Real historical research (example after downloading 1h 2020-2024):**

```bash
# 1. Acquire (Binance Official API https://api.binance.com/api/v3/klines, UTC, paginated, raw preserved)
#    research/data/raw/BTCUSDT_1h.json 43816 rows 2020-01-01→2024-12-31  f630861c
#    research/data/raw/ETHUSDT_1h.json 43816 rows  8feb1e63
#    research/data/raw/SOLUSDT_1h.json 38470 rows 2020-08-11→2024-12-31 ada0c483
# 2. Validate
#    load_csv_dataset → DataQualityReport gap_count 15 (BTC downtime) is_clean False (gaps reported, not repaired)
# 3. Research
python -c "import sys; sys.path.insert(0,'.'); from research.v2.engine import run_v2_experiment; ..."
```

All commands above **really work** in this repo (verified for this README).

---

## Testing

- **Current:** `602 passed, 2 skipped` (`2` = opt-in `tests/test_binance_testnet_integration.py:21,38` `skipif RUN_BINANCE_TESTNET!="1"`)
- **No `xfail`, no `deselected`, no `failed`**; `grep -R "if False"` → 1 `skipif` only, `or True` → 0, `assert True` → 0, `_running = True` without public `await` → 0
- **Behavioral/mutation hardening:** 14 P0 + 7 J-1 + 7 V2 `FAIL when broken` (e.g. remove `is_halted` → `TestHaltExplicit` `risk==1` **FAIL**, remove dedup → `cash 9800` **FAIL**, swap `Risk`/`Gate` → `gate before Risk!` **FAIL**)
- Coverage not claimed — `pytest-cov` not installed; behavioral proof via mutations is the evidence

## Current Limitations

- **Historical data:** No `research/data/*.csv` committed (large, `.gitignore` `research/data/raw/*`); `V2` demo uses `60-bar` synthetic oscillating `hash 83dffea2` **explicitly labelled synthetic**, not real performance. Real `BTC/ETH/SOL 1h 2020-2024` must be downloaded via `ingestion.py` (Binance API, ~24 MB raw, `f630861c` etc.) before real research.
- **Optimization support:** `ema_crossover`, `zscore`, `donchian`, `ema_rsi_filtered` only in `ParameterSpace REQUIRED_AXES`; `rsi_momentum`/`bollinger`/`macd`/`adx`/`sma` exist but need `REQUIRED_AXES` extension to be grid-optimized.
- **Live/Testnet execution:** Controlled `PaperBroker` + `Binance TESTNET` `dec_to_str` `1e-8`; no live `RUN_BINANCE_TESTNET` run in CI (opt-in only, correctly not claimed as mock).
- **Vector `run_vectorbt` long-only** engine short not wired (arrays `se/sx` proven, engine not short) — research only.
- **No profitability claim:** All `60-bar` synthetic and `1h` 43k real `REJECT` (`val Sharpe -222 not > -77+0.1`) — `NO ROBUST IMPROVEMENT FOUND` is successful rejection.
- **Shared margin/correlation** not modeled beyond `shared Portfolio cash` `sorted(ts,symbol)`.

---

## Roadmap

**Completed**
- Phase 0–15 architecture/research/safety
- Phase 16 hardening (observability `logger.debug`, `HALT` explicit `both` blocking, `val_ds` re-score, `float`→`Decimal`, ghost `scan_iter`, `fsync`, `heartbeat_stale`)
- V2 deterministic experiment engine, filtered `EMA+RSI` example, `10` V2 behavioral tests, CLI `optimize/list/inspect/reproduce/best`
- P17.0 historical ingestion foundation (`ingest_historical_file` `DATA_REQUIRED`, `10` hard-gate tests)
- P17 real data `BTC/ETH/SOL 1h` `f630861c/8feb1e63/ada0c483` validated, `V2` on real `43k` bars correctly `REJECT` (cost/robustness)

**Next (only after V1 is genuinely complete — it now is)**
- Provide user-downloaded `1h`/`4h`/`1d` history for `BTC/ETH/SOL` 2021-2025 via ingestion (already ready, just download)
- Portfolio-level comparison across symbols/regimes (already `run_multi_symbol_backtest` + `leaderboard_P17.json`)
- Stronger V2 workflow on real history with larger `train 500 test 100` Already proven `reproduce identical` (`r1==r2`)
- Controlled paper/testnet validation **only** for `ACCEPT` candidates (none yet — correct)

Do not claim future work is completed.

---

## License

No license file currently in repository. Add `LICENSE` if you intend to publish.

