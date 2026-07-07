"""Quick baseline test — run each strategy and time each step."""
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


def main():
    print("=" * 60, flush=True)
    print("BASELINE TEST: Swing Strategies", flush=True)
    print("=" * 60, flush=True)

    from pathlib import Path
    from backtests.swing.auto.harness import SwingAutoHarness
    from backtests.swing.auto.scoring import composite_score, extract_metrics

    data_dir = Path("backtests/swing/data/raw")
    output_dir = Path("backtests/swing/auto/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    harness = SwingAutoHarness(
        data_dir=data_dir,
        output_dir=output_dir,
        initial_equity=100_000.0,
    )

    # 1. Load all data
    print("\n[1/3] Loading data...", flush=True)
    t0 = time.time()
    harness._load_data("all")
    print(f"  Data loaded in {time.time() - t0:.1f}s", flush=True)

    # 2. Run ATRSS baseline
    _run_strategy_baseline(harness, "atrss")

    # 3. Run Helix baseline
    _run_strategy_baseline(harness, "helix")

    print(f"\n{'=' * 60}", flush=True)
    print("BASELINE SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)

    for strategy, score in sorted(harness._baselines.items()):
        trades = harness._baseline_trades.get(strategy, [])
        if score.rejected:
            print(f"  {strategy:10s}: REJECTED ({score.reject_reason})", flush=True)
        else:
            print(f"  {strategy:10s}: score={score.total:.4f} "
                  f"calmar={score.calmar_component:.3f} "
                  f"pf={score.pf_component:.3f} "
                  f"inv_dd={score.inv_dd_component:.3f} "
                  f"net_pnl={score.net_profit_component:.3f} "
                  f"trades={len(trades)}", flush=True)

    print(f"\nTotal time: {time.time() - t0:.1f}s", flush=True)


def _run_strategy_baseline(harness, strategy: str) -> None:
    """Run a single strategy baseline with timing."""
    step = {"atrss": 2, "helix": 3}
    n = step.get(strategy, 0)
    print(f"\n[{n}/3] Running {strategy.upper()} baseline...", flush=True)
    t = time.time()

    success = harness._run_baseline(strategy)

    elapsed = time.time() - t
    status = "OK" if success else "FAILED"
    print(f"  {strategy.upper()} baseline done in {elapsed:.1f}s [{status}]", flush=True)


if __name__ == "__main__":
    main()
