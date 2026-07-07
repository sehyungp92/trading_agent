"""Explain or compute which bot images are affected by changed paths."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALL_IMAGES = {"ibkr-trading", "k-stock-trader", "crypto-trader"}
IMAGE_RULES = [
    ("packages/trading_contracts/", ALL_IMAGES),
    ("packages/trading_config/", ALL_IMAGES),
    ("packages/trading_deployment/", ALL_IMAGES),
    ("packages/trading_instrumentation/", ALL_IMAGES),
    ("contracts/", ALL_IMAGES),
    ("bots/ibkr_trading/", {"ibkr-trading"}),
    ("deployments/ibkr/", {"ibkr-trading"}),
    ("bots/k_stock_trader/", {"k-stock-trader"}),
    ("deployments/k_stock/", {"k-stock-trader"}),
    ("bots/crypto_trader/", {"crypto-trader"}),
    ("deployments/crypto/", {"crypto-trader"}),
]
ASSISTANT_ONLY_PREFIXES = (
    "packages/trading_assistant/",
    "packages/trading_assistant_data/",
    "packages/trading_assistant_backtest/",
)


def changed_files_from_git(base: str | None) -> list[str]:
    command = ["git", "diff", "--name-only"]
    if base:
        command.append(base)
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def affected_images(paths: list[str]) -> set[str]:
    affected: set[str] = set()
    for raw_path in paths:
        path = raw_path.replace("\\", "/")
        if path.startswith(ASSISTANT_ONLY_PREFIXES):
            continue
        if path.split("/", 1)[0].startswith("_"):
            continue
        for prefix, images in IMAGE_RULES:
            if path.startswith(prefix):
                affected.update(images)
    return affected


def print_rules() -> None:
    print("Affected-image rules:")
    for prefix, images in IMAGE_RULES:
        print(f"- {prefix} -> {', '.join(sorted(images))}")
    print("- packages/trading_assistant* -> assistant gates only unless shared artifacts changed")
    print("- root underscore-prefixed archive dirs -> no image; source has been ported")
    print("- docs/ and tools/ -> no image by default; CI gates still run")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explain-only", action="store_true", help="Print mapping rules and exit.")
    parser.add_argument("--base", help="Optional git ref to diff against.")
    parser.add_argument("--changed-file", action="append", default=[], help="Changed file path.")
    args = parser.parse_args(argv)

    if args.explain_only:
        print_rules()
        return 0

    paths = args.changed_file or changed_files_from_git(args.base)
    images = affected_images(paths)
    print("Changed files:")
    for path in paths:
        print(f"- {path}")
    print("Affected images:")
    if images:
        for image in sorted(images):
            print(f"- {image}")
    else:
        print("- none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
