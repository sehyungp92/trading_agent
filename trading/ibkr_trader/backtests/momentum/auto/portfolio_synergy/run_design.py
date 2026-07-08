from __future__ import annotations

import argparse
from pathlib import Path

from .round_design import write_round_design


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="momentum-portfolio-synergy-design",
        description="Write the next momentum four-strategy portfolio synergy phase-auto design.",
    )
    parser.add_argument(
        "--output-dir",
        default="backtests/output/momentum/portfolio_synergy/round_1",
        help="Directory for run_spec.json and related design artifacts.",
    )
    args = parser.parse_args()

    paths = write_round_design(Path(args.output_dir))
    print("Momentum portfolio synergy design written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

