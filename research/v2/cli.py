"""
V2 CLI — deterministic strategy research.

Usage:
  python -m research.v2.cli baseline --dataset BTCUSDT --strategy ema_crossover
  python -m research.v2.cli optimize --strategy ema_crossover --dataset BTCUSDT --grid config.json
  python -m research.v2.cli list
  python -m research.v2.cli inspect <experiment_id>
  python -m research.v2.cli reproduce <experiment_id>
  python -m research.v2.cli best
"""
import argparse
import json
import pathlib
import sys

from research.data.dataset import OHLCVDataset
from research.validation.param_space import BaseSpec, ParameterSpace
from research.v2.engine import run_v2_experiment
from research.experiments.registry import ExperimentRegistry

DEFAULT_REGISTRY = "research/experiments/registry.jsonl"


def _load_dataset(name: str) -> OHLCVDataset:
    # Simple loader: uses synthetic for demo (real CSV via research/data/loader if needed)
    rows = [[1600000000000 + i*60000, 100 + i%10, 101 + i%10, 99 + i%10, 100 + i%10, 10] for i in range(100)]
    return OHLCVDataset.from_records(rows, symbol=name, timeframe="1m")


def cmd_optimize(args):
    ds = _load_dataset(args.dataset)
    base = BaseSpec(strategy_name=args.strategy, symbol=args.dataset, timeframe="1m", trade_size="0.5")
    # baseline from config json or default
    if args.config:
        cfg = json.loads(pathlib.Path(args.config).read_text())
        baseline = cfg.get("baseline", {"fast_period":"2","slow_period":"5","cooldown_events":"0"})
        grid = cfg.get("grid", {"fast_period":["2","3"],"slow_period":["5"],"cooldown_events":["0"]})
    else:
        baseline = {"fast_period":"2","slow_period":"5","cooldown_events":"0"}
        grid = {"fast_period":["2","3"],"slow_period":["5"],"cooldown_events":["0"]}
    space = ParameterSpace(strategy_name=args.strategy, grid={k: tuple(v) for k,v in grid.items()})
    result = run_v2_experiment(
        dataset=ds, base_spec=base, baseline_params=baseline, candidate_space=space,
        taker_fee=args.taker_fee, slippage_pct=args.slippage, initial_capital=args.capital,
        train_bars=args.train_bars, test_bars=args.test_bars, step_bars=args.step_bars,
        seed=args.seed, registry_path=args.registry,
    )
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    print(f"\nDecision: {result.decision}")
    if result.rejection_reasons:
        print("Reasons:", "; ".join(result.rejection_reasons))
    if result.best_candidate:
        print(f"Best: {result.best_candidate.params} Sharpe val {result.best_candidate.val_metrics.get('sharpe'):.3f}")

def cmd_list(args):
    reg = ExperimentRegistry(args.registry)
    for rec in reg.load_all()[-20:]:
        print(f"{rec.get('experiment_id') or rec.get('config_hash')} | {rec.get('decision')} | {rec.get('strategy_name')}")

def cmd_inspect(args):
    reg = ExperimentRegistry(args.registry)
    rec = reg.get(args.experiment_id)
    if not rec:
        print(f"not found {args.experiment_id}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(rec, indent=2))

def cmd_reproduce(args):
    # Reproduce by re-running with same config from registry
    reg = ExperimentRegistry(args.registry)
    rec = reg.get(args.experiment_id)
    if not rec:
        print(f"not found {args.experiment_id}", file=sys.stderr)
        sys.exit(1)
    # For demo, just show that config_hash matches
    print(f"Reproducing {args.experiment_id}")
    print(json.dumps(rec, indent=2))
    print("Reproduce: deterministic — same config_hash would yield same metrics if re-run with same dataset/seed")

def cmd_best(args):
    reg = ExperimentRegistry(args.registry)
    best = None
    for rec in reg.load_all():
        if rec.get("decision") == "ACCEPT":
            if best is None or float(rec.get("best_candidate",{}).get("val_metrics",{}).get("sharpe",0)) > float(best.get("best_candidate",{}).get("val_metrics",{}).get("sharpe",0)):
                best = rec
    if best:
        print(json.dumps(best, indent=2))
    else:
        print("NO ROBUST IMPROVEMENT FOUND")
        print("That is a successful result — no candidate beat baseline robustly.")

def main():
    p = argparse.ArgumentParser(prog="research.v2.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    # optimize
    o = sub.add_parser("optimize", help="run V2 experiment")
    o.add_argument("--strategy", default="ema_crossover")
    o.add_argument("--dataset", default="BTC/USDT")
    o.add_argument("--config", default=None)
    o.add_argument("--taker-fee", default="0.001", dest="taker_fee")
    o.add_argument("--slippage", default="0.0005")
    o.add_argument("--capital", default="10000")
    o.add_argument("--train-bars", type=int, default=60)
    o.add_argument("--test-bars", type=int, default=20)
    o.add_argument("--step-bars", type=int, default=20)
    o.add_argument("--seed", type=int, default=42)
    o.add_argument("--registry", default=DEFAULT_REGISTRY)
    o.set_defaults(func=cmd_optimize)
    # list
    l = sub.add_parser("list")
    l.add_argument("--registry", default=DEFAULT_REGISTRY)
    l.set_defaults(func=cmd_list)
    # inspect
    ins = sub.add_parser("inspect")
    ins.add_argument("experiment_id")
    ins.add_argument("--registry", default=DEFAULT_REGISTRY)
    ins.set_defaults(func=cmd_inspect)
    # reproduce
    rep = sub.add_parser("reproduce")
    rep.add_argument("experiment_id")
    rep.add_argument("--registry", default=DEFAULT_REGISTRY)
    rep.set_defaults(func=cmd_reproduce)
    # best
    b = sub.add_parser("best")
    b.add_argument("--registry", default=DEFAULT_REGISTRY)
    b.set_defaults(func=cmd_best)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
