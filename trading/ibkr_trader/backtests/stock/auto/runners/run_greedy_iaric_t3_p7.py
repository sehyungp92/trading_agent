"""Run IARIC T3 Phase 7 greedy optimization.

Phase 7: Mean-Reversion Pullback-Buy. A completely new entry signal (Tier 3)
that scans the full S&P universe for short-term oversold pullbacks in uptrends.
Tests 42 candidates across 14 themes: RSI tuning, alternative triggers,
stop/hold duration, profit targets, position limits, regime gates, carry, etc.

Usage:
    cd trading/ibkr_trader
    PYTHONUNBUFFERED=1 python -u -m backtests.stock.auto.runners.run_greedy_iaric_t3_p7
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

BOT_ROOT = Path(__file__).resolve().parents[4]
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtests.stock.auto.greedy_optimize import (
    IARIC_T3_P7_BASE_MUTATIONS,
    IARIC_T3_P7_CANDIDATES,
    run_greedy,
    save_result,
)
from backtests.stock.engine.research_replay import ResearchReplayEngine

DATA_DIR = BOT_ROOT / "backtests" / "stock" / "data" / "raw"
OUTPUT_DIR = BOT_ROOT / "backtests" / "stock" / "auto" / "output"


def main():
    print("=" * 60)
    print("IARIC T3 Phase 7 -- Mean-Reversion Pullback-Buy")
    print("=" * 60)
    print(f"Base: {len(IARIC_T3_P7_BASE_MUTATIONS)} mutations")
    print(f"Candidates: {len(IARIC_T3_P7_CANDIDATES)} across 14 themes")
    print("=" * 60, flush=True)

    # Load data
    print("\n[1/2] Loading data...", flush=True)
    t0 = time.time()
    replay = ResearchReplayEngine(data_dir=DATA_DIR)
    replay.load_all_data()
    print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)

    # Run greedy
    print("\n[2/2] Running greedy optimization...", flush=True)
    checkpoint = OUTPUT_DIR / "greedy_checkpoint_iaric_t3_p7.json"
    result = run_greedy(
        replay,
        strategy="iaric",
        tier=3,
        base_mutations=IARIC_T3_P7_BASE_MUTATIONS,
        candidates=IARIC_T3_P7_CANDIDATES,
        max_workers=3,
        checkpoint_path=checkpoint,
    )

    # Save
    out_path = OUTPUT_DIR / "greedy_optimal_iaric_t3_p7.json"
    save_result(result, out_path)
    print(f"\n  Saved: {out_path}")
    print(f"  Final score: {result.final_score:.6f}")
    print(f"  Kept features: {result.kept_features}")
    total = time.time() - t0
    print(f"  Total time: {total:.0f}s ({total / 60:.1f}min)")


if __name__ == "__main__":
    main()
