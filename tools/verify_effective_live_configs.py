"""Verify generated materialized effective live config artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for src in (
    ROOT / "packages" / "trading_config" / "src",
    ROOT / "packages" / "trading_contracts" / "src",
):
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

from trading_config.verifier import verify_effective_configs  # noqa: E402


def main() -> int:
    result = verify_effective_configs(ROOT)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
