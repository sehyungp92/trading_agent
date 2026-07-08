"""Replay local instrumentation JSONL files into PostgreSQL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crypto_trader.instrumentation.postgres_backfill import replay_jsonl_to_postgres


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--postgres-dsn", required=True)
    args = parser.parse_args()

    result = replay_jsonl_to_postgres(args.state_dir, args.postgres_dsn)
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
