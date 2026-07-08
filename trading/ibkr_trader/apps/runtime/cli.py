"""CLI entrypoint for the unified runtime scaffold."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from libs.config.loader import load_strategy_registry
from libs.config.registry import write_registry_artifact

from .runtime import RuntimeShell


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Validate runtime config and registry")
    preflight.add_argument("--config-dir", default="config")
    preflight.add_argument("--json", action="store_true", dest="as_json")
    preflight.add_argument(
        "--write-registry-artifact",
        nargs="?",
        const="data/strategy-registry.json",
        default=None,
    )
    metadata = subparsers.add_parser(
        "emit-deployment-metadata",
        help="Emit no-secret runtime deployment metadata after startup preflight inputs are loaded",
    )
    metadata.add_argument("--contract", required=True)
    metadata.add_argument("--effective-config", required=True)
    metadata.add_argument("--output", required=True)
    metadata.add_argument("--repo-root", default=".")
    metadata.add_argument("--runtime-started-at-utc", required=True)
    metadata.add_argument("--runtime-instance-id", required=True)

    run = subparsers.add_parser("run", help="Start the unified runtime")
    run.add_argument("--config-dir", default="config")
    run.add_argument(
        "--effective-config",
        default=None,
        help="Generated effective config artifact mounted into the runtime container",
    )
    run.add_argument("--shadow", action="store_true")
    run.add_argument("--connect-ib", action="store_true")
    run.add_argument("--once", action="store_true")
    run.add_argument(
        "--family",
        default=None,
        choices=["swing", "momentum", "stock"],
        help="Run only strategies for the specified family",
    )
    run.add_argument(
        "--allow-no-db",
        action="store_true",
        help="Start without DB (no portfolio rules). Use for debugging only.",
    )
    run.add_argument(
        "--allow-partial-families",
        action="store_true",
        help=(
            "Permit paper/live startup to continue when one or more enabled "
            "family coordinators fail to construct or start. Default = strict "
            "(any failure aborts startup). Use only for development workflows "
            "that intentionally run a subset of families."
        ),
    )
    run.add_argument(
        "--allow-no-instrumentation",
        action="store_true",
        help=(
            "Permit paper/live startup to continue without sidecar forwarding. "
            "Use only for debugging; default paper/live behavior requires evidence."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    log = logging.getLogger(__name__)

    if args.command == "preflight":
        try:
            shell = RuntimeShell(args.config_dir)
            checks = shell.run_preflight()
        except Exception as exc:
            log.error("Preflight failed: %s", exc, exc_info=True)
            return 1
        if args.write_registry_artifact:
            registry = load_strategy_registry(args.config_dir)
            path = write_registry_artifact(registry, Path(args.write_registry_artifact))
            log.info("Wrote registry artifact to %s", path)

        payload = [
            {"name": check.name, "ok": check.ok, "detail": check.detail}
            for check in checks
        ]
        if args.as_json:
            print(json.dumps(payload, indent=2))
        else:
            for check in checks:
                status = "OK" if check.ok else "FAIL"
                print(f"[{status}] {check.name}: {check.detail}")
        return 0 if all(check.ok for check in checks) else 1

    if args.command == "emit-deployment-metadata":
        return _emit_deployment_metadata(args)

    if args.command == "run":
        try:
            shell = RuntimeShell(args.config_dir)
            asyncio.run(shell.run(
                shadow=args.shadow,
                connect_ib=args.connect_ib,
                once=args.once,
                family_filter=args.family,
                effective_config_path=args.effective_config,
                allow_no_db=args.allow_no_db,
                allow_partial_families=args.allow_partial_families,
                allow_no_instrumentation=args.allow_no_instrumentation,
            ))
            return 0
        except KeyboardInterrupt:
            return 0
        except Exception as exc:
            log.error("Runtime failed: %s", exc, exc_info=True)
            return 1

    parser.error(f"Unknown command {args.command!r}")
    return 2


def _emit_deployment_metadata(args: argparse.Namespace) -> int:
    from libs.instrumentation.deployment_metadata import build_deployment_metadata

    repo_root = Path(args.repo_root).resolve()
    contract_path = Path(args.contract)
    if not contract_path.is_absolute():
        contract_path = repo_root / contract_path
    effective = json.loads(Path(args.effective_config).read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    bridge_id = contract_path.parent.name
    lineage = {
        "family_id": "ibkr",
        "portfolio_id": "ibkr",
        "strategy_version": contract.get("decision_api_version", ""),
        "config_version": effective.get("effective_config_hash", ""),
        "deployment_id": f"ibkr-{bridge_id}",
        "code_sha": _git(repo_root, "rev-parse", "HEAD"),
    }
    metadata = build_deployment_metadata(
        lineage,
        bridge_id=bridge_id,
        repo_root=repo_root,
        effective_config=effective,
        strategy_plugin_contract_path=contract_path,
        runtime_entrypoint="apps.runtime.cli:main",
        runtime_started_at_utc=args.runtime_started_at_utc,
        runtime_instance_id=args.runtime_instance_id,
        dry_run=False,
        env=dict(os.environ),
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"deployment_metadata_path": str(output)}, sort_keys=True))
    return 0


def _git(repo_root: Path, *args: str) -> str:
    import subprocess

    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
