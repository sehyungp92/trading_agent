"""Command line validators for shared trading contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ValidationError

from trading_contracts.canonical import canonical_json_sha256, file_sha256
from trading_contracts.legacy import (
    validate_deployment_manifest,
    validate_plugin_contract,
    validate_promotion_manifest,
    validate_rounds_manifest,
)
from trading_contracts.known_artifacts import validate_known_artifacts
from trading_contracts.schemas import generate_schemas


Validator = Callable[[str | Path], BaseModel]


def _result(path: Path, model: BaseModel) -> dict:
    return {
        "valid": True,
        "path": str(path),
        "model": type(model).__name__,
        "file_sha256": file_sha256(path),
        "canonical_json_sha256": canonical_json_sha256(model),
    }


def _run_validator(path: str, validator: Validator) -> int:
    target = Path(path)
    try:
        model = validator(target)
    except (OSError, ValueError, ValidationError) as exc:
        print(json.dumps({"valid": False, "path": str(target), "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(_result(target, model), indent=2, sort_keys=True))
    return 0


def _cmd_generate_schemas(args: argparse.Namespace) -> int:
    paths = generate_schemas(args.output)
    print(json.dumps({"valid": True, "schemas": [str(path) for path in paths]}, indent=2))
    return 0


def _cmd_validate_known(args: argparse.Namespace) -> int:
    if not args.all_known_reference_artifacts:
        print(json.dumps({"valid": False, "error": "no validation target selected"}, indent=2))
        return 2
    result = validate_known_artifacts(args.repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trading-contracts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    commands: dict[str, Validator] = {
        "validate-rounds-manifest": validate_rounds_manifest,
        "validate-plugin-contract": validate_plugin_contract,
        "validate-promotion": validate_promotion_manifest,
        "validate-deployment-metadata": validate_deployment_manifest,
    }
    for name, validator in commands.items():
        sub = subparsers.add_parser(name)
        sub.add_argument("path")
        sub.set_defaults(func=lambda args, validator=validator: _run_validator(args.path, validator))

    schema_parser = subparsers.add_parser("generate-schemas")
    schema_parser.add_argument("--output", default="contracts/schemas")
    schema_parser.set_defaults(func=_cmd_generate_schemas)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--all-known-reference-artifacts", action="store_true")
    validate_parser.add_argument("--repo-root", default=".")
    validate_parser.set_defaults(func=_cmd_validate_known)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
