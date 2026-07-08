from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtests.shared.parity.baseline_regeneration import verify_manifest_regeneration


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate frozen backtest baselines in a sandbox and verify hash/metric parity."
    )
    parser.add_argument(
        "--artifact-id",
        action="append",
        default=[],
        help="Optional baseline artifact id(s) from tests/fixtures/backtest_baselines/manifest.json.",
    )
    parser.add_argument(
        "--sandbox-root",
        type=Path,
        default=None,
        help="Optional persistent sandbox root. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=3600,
        help="Per-artifact subprocess timeout.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a text summary.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output when --json is used.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    results = verify_manifest_regeneration(
        sandbox_root=args.sandbox_root,
        artifact_ids=args.artifact_id or None,
        timeout_seconds=max(1, args.timeout_seconds),
    )

    if args.json:
        payload = [asdict(result) for result in results]
        print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))
        return

    for result in results:
        print(
            f"[OK] {result.entry_id}: {result.sha256} "
            f"({result.artifact_path})"
        )


if __name__ == "__main__":
    main()
