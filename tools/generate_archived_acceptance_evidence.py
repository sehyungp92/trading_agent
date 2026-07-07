from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Historical-only archived evidence stub; not valid for final acceptance."
    )
    parser.add_argument(
        "--allow-historical-only",
        action="store_true",
        help="Acknowledge that no A10/A13/A14 final-acceptance evidence will be generated.",
    )
    args = parser.parse_args()
    if not args.allow_historical_only:
        parser.error(
            "archived/frozen-output-only evidence is not valid for final acceptance; "
            "use the runtime parity, optimizer equivalence, and backtest integrity gates instead"
        )
    print(json.dumps({"valid_for_final_acceptance": False, "generated_evidence": []}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
