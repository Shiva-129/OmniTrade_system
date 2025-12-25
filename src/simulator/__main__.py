"""
OmniTrade Simulator: Main Entry Point

Command-line interface for running deterministic replay.
Runs headless and unattended.
"""
import sys
import json
import argparse
from .context import SimulatorConfig, init_decimal_context
from .replay_engine import ReplayEngine
from .verdict import VerdictStatus

def main():
    parser = argparse.ArgumentParser(
        description="OmniTrade Deterministic Simulator"
    )
    parser.add_argument(
        "--journal", 
        required=True,
        help="Path to raw event journal file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for deterministic replay"
    )
    parser.add_argument(
        "--reference-hashes",
        help="Path to reference hash log for verification"
    )
    parser.add_argument(
        "--output-hashes",
        help="Path to save computed hash log"
    )
    parser.add_argument(
        "--config-hash",
        default="auto",
        help="Config hash (auto-computed if 'auto')"
    )

    args = parser.parse_args()

    # Initialize decimal context FIRST
    init_decimal_context()

    # Build config
    config_hash = args.config_hash
    if config_hash == "auto":
        # Auto-compute
        import hashlib
        data = f"{args.seed}:{args.journal}"
        config_hash = hashlib.sha256(data.encode()).hexdigest()[:16]

    config = SimulatorConfig(
        config_hash=config_hash,
        rng_seed=args.seed,
        journal_path=args.journal
    )

    print(f"=== OmniTrade Deterministic Simulator ===")
    print(f"Journal: {args.journal}")
    print(f"RNG Seed: {args.seed}")
    print(f"Config Hash: {config_hash}")
    print()

    # Create engine
    engine = ReplayEngine(config)

    # Load reference hashes if provided
    if args.reference_hashes:
        print(f"Loading reference hashes from: {args.reference_hashes}")
        engine.load_reference_hashes(args.reference_hashes)

    # Run replay
    print("Starting replay...")
    verdict = engine.run()

    # Output results
    print()
    print("=== REPLAY VERDICT ===")
    print(f"Status: {verdict.status.value}")
    print(f"Events: {verdict.events_processed}/{verdict.events_total}")
    print(verdict.summary())

    if verdict.divergence:
        print()
        print("=== DIVERGENCE DETAILS ===")
        print(f"Event Index: {verdict.divergence.event_index}")
        print(f"Expected Hash: {verdict.divergence.expected_hash}")
        print(f"Actual Hash: {verdict.divergence.actual_hash}")
        print(f"Causal Chain: {verdict.divergence.causal_chain}")
        print(f"Event Data: {json.dumps(verdict.divergence.event_data, indent=2)}")

    # Save hash log if requested
    if args.output_hashes:
        engine.save_hash_log(args.output_hashes)
        print(f"Hash log saved to: {args.output_hashes}")

    # Exit with appropriate code
    if verdict.status == VerdictStatus.PASS:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
