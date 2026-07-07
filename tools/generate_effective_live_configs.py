"""Generate canonical promotion manifests and materialized effective live configs."""

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

from trading_config.generator import generate_effective_configs  # noqa: E402


def main() -> int:
    result = generate_effective_configs(ROOT)
    print(json.dumps({"valid": True, **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
