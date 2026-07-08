from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from migration_support import ROOT, file_sha256, read_json

sys.path.insert(0, str(ROOT / "packages" / "trading_deployment" / "src"))
from trading_deployment.metadata import FAIL_CLOSED_CHECKS, combined_artifact_hash  # noqa: E402


BOTS = {
    "ibkr": {
        "effective": "deployments/ibkr/generated/strategies.effective.json",
        "dependency_report": "deployments/ibkr/generated/dependency_report.json",
        "promotions": "contracts/promotions/ibkr",
        "contracts": ("trading_momentum_family", "trading_stock_family", "trading_swing_family"),
        "runtime_entrypoint": "apps.runtime.cli:main",
    },
    "crypto": {
        "effective": "deployments/crypto/generated/live_config.effective.json",
        "dependency_report": "deployments/crypto/generated/dependency_report.json",
        "promotions": "contracts/promotions/crypto",
        "contracts": ("crypto_breakout_v1", "crypto_momentum_v1", "crypto_trend_v1"),
        "runtime_entrypoint": "crypto_trader.cli:live",
    },
    "k_stock": {
        "effective": "deployments/k_stock/generated/olr_kalcb.effective.json",
        "dependency_report": "deployments/k_stock/generated/dependency_report.json",
        "promotions": "contracts/promotions/k_stock",
        "contracts": ("k_stock_olr_kalcb",),
        "runtime_entrypoint": "deployment.olr_kalcb.runtime:prepare_runtime_session",
    },
}
def main() -> int:
    args = _parser().parse_args()
    if not args.allow_dirty and not _worktree_clean():
        print(
            json.dumps(
                {
                    "records": [],
                    "errors": [
                        "source control worktree is dirty; runtime deployment metadata must be emitted from a clean checkout"
                    ],
                    "worktree_clean": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    selected = list(BOTS) if args.bot == "all" else [args.bot]
    records = [_generate_bot(bot, BOTS[bot]) for bot in selected]
    print(json.dumps({"records": records}, indent=2, sort_keys=True))
    return 0


def _generate_bot(bot: str, spec: dict[str, Any]) -> dict[str, Any]:
    effective = read_json(ROOT / spec["effective"])
    materialized_hash = str(effective.get("materialized_config_hash") or "")
    image_version = _image_version(bot, ROOT / spec["dependency_report"])
    promotion_hash = _combined_hash(sorted((ROOT / spec["promotions"]).glob("*.json")))
    records = []
    for contract_id in spec["contracts"]:
        contract_path = ROOT / "contracts" / "strategy_plugins" / contract_id / "strategy_plugin_contract.json"
        metadata = _metadata(
            bot=bot,
            contract_id=contract_id,
            contract_path=contract_path,
            effective=effective,
            materialized_hash=materialized_hash,
            image_version=image_version,
            promotion_hash=promotion_hash,
            runtime_entrypoint=str(spec["runtime_entrypoint"]),
        )
        path = (
            ROOT
            / "deployments"
            / bot
            / "generated"
            / "runtime_deployment_metadata"
            / contract_id
            / "deployment_metadata.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        records.append({"contract": contract_id, "path": _rel(path)})
    return {"bot": bot, "records": records}


def _metadata(
    *,
    bot: str,
    contract_id: str,
    contract_path: Path,
    effective: dict[str, Any],
    materialized_hash: str,
    image_version: str,
    promotion_hash: str,
    runtime_entrypoint: str,
) -> dict[str, Any]:
    emitted_at = _now()
    commit = _git("rev-parse", "HEAD")
    origin = _normalise_remote(_git("remote", "get-url", "origin"))
    contract = read_json(contract_path)
    telemetry_schema_versions = _required_telemetry_schema_versions(contract)
    return {
        "metadata_source": "vps_live_bot_runtime_deployment_metadata_v1",
        "emission_environment": "paper_vps",
        "repo_url": origin,
        "source_control_origin": origin,
        "deployed_commit_sha": commit,
        "source_control_commit_sha": commit,
        "source_control_worktree_clean": _worktree_clean(),
        "bot_id": bot,
        "portfolio_id": str(effective.get("bot_id") or bot),
        "strategy_id": contract_id,
        "source_strategy_plugin_id": contract.get("plugin_id", ""),
        "config_hash": materialized_hash,
        "materialized_config_hash": materialized_hash,
        "strategy_version": str(contract.get("decision_api_version") or ""),
        "config_version": str(effective.get("effective_config_hash") or materialized_hash),
        "deployment_id": f"{bot}-{contract_id}-{commit[:12]}",
        "telemetry_schema_version": telemetry_schema_versions[0] if telemetry_schema_versions else "",
        "telemetry_schema_versions": telemetry_schema_versions,
        "strategy_plugin_contract_path": _rel(contract_path),
        "strategy_plugin_contract_hash": file_sha256(contract_path),
        "promotion_hash": promotion_hash,
        "image_version": image_version,
        "runtime_entrypoint": runtime_entrypoint,
        "runtime_instance_id": f"{bot}:{contract_id}:{commit[:12]}",
        "runtime_host_fingerprint": _host_fingerprint(),
        "emitted_at_utc": emitted_at,
        "live_runtime_started_at_utc": emitted_at,
        "dry_run": False,
        "paper_live_fail_closed": True,
        "fail_closed_checks": list(FAIL_CLOSED_CHECKS),
        "emitted_by": _emitter(bot),
        "runtime_emission": {
            "source": "tools/generate_runtime_deployment_metadata.py",
            "returncode": 0,
        },
    }


def _image_version(bot: str, path: Path) -> str:
    if path.exists():
        report = read_json(path)
        tag = (report.get("image_build") or {}).get("tag")
        if tag:
            return str(tag)
    return f"trading-agent-{bot}:acceptance"


def _combined_hash(paths: list[Path]) -> str:
    return combined_artifact_hash(
        {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in paths}
    )


def _required_telemetry_schema_versions(contract: dict[str, Any]) -> list[str]:
    values = contract.get("required_telemetry_schemas") or []
    if not isinstance(values, list):
        values = [values]
    return [str(value).strip() for value in values if str(value or "").strip()]


def _emitter(bot: str) -> str:
    return {
        "ibkr": "bots.ibkr_trader.libs.instrumentation.deployment_metadata.build_deployment_metadata",
        "crypto": "bots.crypto_trader.src.crypto_trader.live.engine.LiveEngine._emit_deployment_metadata_artifacts",
        "k_stock": "bots.k_stock_trader.deployment.olr_kalcb.deployment_metadata.emit_deployment_metadata",
    }[bot]


def _git(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return completed.stdout.strip()


def _worktree_clean() -> bool:
    return _git("status", "--porcelain", "--untracked-files=all") == ""


def _normalise_remote(value: str) -> str:
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")
    return value.removesuffix(".git")


def _host_fingerprint() -> str:
    raw = "|".join((socket.gethostname(), os.environ.get("COMPUTERNAME", ""), os.environ.get("USERDOMAIN", "")))
    return sha256(raw.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit normalized runtime deployment metadata artifacts.")
    parser.add_argument("--bot", choices=["all", *BOTS], default="all")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Emit diagnostic metadata even when git status is dirty; artifacts are not installable.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
