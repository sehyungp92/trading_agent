"""CLI for generated live config promotion evidence."""

from __future__ import annotations

import argparse
import json
import sys

from trading_config.generator import generate_effective_configs
from trading_config.verifier import verify_effective_configs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trading-config")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-effective")
    generate.add_argument("--repo-root", default=".")
    generate.set_defaults(func=_cmd_generate)

    verify = subparsers.add_parser("verify-effective")
    verify.add_argument("--repo-root", default=".")
    verify.set_defaults(func=_cmd_verify)
    return parser


def _cmd_generate(args: argparse.Namespace) -> int:
    result = generate_effective_configs(args.repo_root)
    print(json.dumps({"valid": True, **result}, indent=2, sort_keys=True))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    result = verify_effective_configs(args.repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
