from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from migration_support import ROOT, file_sha256, read_json

sys.path.insert(0, str(ROOT / "packages" / "trading_deployment" / "src"))
from trading_deployment.metadata import FAIL_CLOSED_CHECKS, combined_artifact_hash  # noqa: E402


_COMMIT: str | None = None
_ORIGIN: str | None = None
_WORKTREE_CLEAN: bool | None = None

BOTS = {
    "ibkr": {
        "effective": "deployments/ibkr/generated/strategies.effective.json",
        "dependency_report": "deployments/ibkr/generated/dependency_report.json",
        "promotions": "contracts/promotions/ibkr",
        "contracts": ("trading_momentum_family", "trading_stock_family", "trading_swing_family"),
        "runtime_entrypoint": "apps.runtime.cli:main",
        "source": "bots/ibkr_trading/libs/instrumentation/deployment_metadata.py",
    },
    "crypto": {
        "effective": "deployments/crypto/generated/live_config.effective.json",
        "dependency_report": "deployments/crypto/generated/dependency_report.json",
        "promotions": "contracts/promotions/crypto",
        "contracts": ("crypto_breakout_v1", "crypto_momentum_v1", "crypto_trend_v1"),
        "runtime_entrypoint": "crypto_trader.cli:live",
        "source": "bots/crypto_trader/src/crypto_trader/live/engine.py",
    },
    "k_stock": {
        "effective": "deployments/k_stock/generated/olr_kalcb.effective.json",
        "dependency_report": "deployments/k_stock/generated/dependency_report.json",
        "promotions": "contracts/promotions/k_stock",
        "contracts": ("k_stock_olr_kalcb",),
        "runtime_entrypoint": "deployment.olr_kalcb.runtime:prepare_runtime_session",
        "source": "bots/k_stock_trader/deployment/olr_kalcb/runtime.py",
    },
}


def main() -> int:
    args = _parser().parse_args()
    if not args.allow_dirty and not _worktree_clean():
        errors = [
            "source control worktree is dirty; runtime deployment metadata must be emitted from a clean checkout",
        ]
        print(json.dumps({"records": [], "errors": errors, "worktree_clean": False}, indent=2, sort_keys=True))
        return 1
    selected = list(BOTS) if args.bot == "all" else [args.bot]
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for bot in selected:
        record, bot_errors = _emit_bot(bot, BOTS[bot])
        records.append(record)
        errors.extend(bot_errors)
    print(json.dumps({"records": records, "errors": errors}, indent=2, sort_keys=True))
    return 0 if not errors else 1


def _emit_bot(bot: str, spec: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    effective = read_json(ROOT / spec["effective"])
    materialized_hash = str(effective.get("materialized_config_hash") or "")
    image_version = _image_version(bot, ROOT / spec["dependency_report"])
    promotion_hash = _combined_hash(sorted((ROOT / spec["promotions"]).glob("*.json")))
    records = []
    for contract_id in spec["contracts"]:
        contract_path = ROOT / "contracts" / "strategy_plugins" / contract_id / "strategy_plugin_contract.json"
        path = (
            ROOT / "deployments" / bot / "generated" / "runtime_deployment_metadata"
            / contract_id / "deployment_metadata.json"
        )
        raw_path = _raw_metadata_path(bot, contract_id)
        emission = _runtime_emit(bot, contract_id, spec, contract_path, ROOT / spec["effective"], raw_path)
        if emission["returncode"] != 0:
            errors.append(f"{bot}:{contract_id}: runtime metadata command failed")
            records.append({"contract": contract_id, "path": _rel(path), "status": "failed", "runtime_emission": emission})
            continue
        if not raw_path.exists():
            errors.append(f"{bot}:{contract_id}: runtime command did not emit {raw_path.relative_to(ROOT).as_posix()}")
            records.append({"contract": contract_id, "path": _rel(path), "status": "failed", "runtime_emission": emission})
            continue
        raw = read_json(raw_path)
        metadata = _normalize_metadata(
            raw,
            bot=bot,
            contract_id=contract_id,
            contract_path=contract_path,
            effective=effective,
            materialized_hash=materialized_hash,
            image_version=image_version,
            promotion_hash=promotion_hash,
            runtime_entrypoint=str(spec["runtime_entrypoint"]),
            source=str(spec["source"]),
            emission=emission,
            metadata_path=path,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        records.append({
            "contract": contract_id,
            "path": _rel(path),
            "runtime_source": spec["source"],
            "status": "pass",
            "runtime_emission": emission,
        })
    return {"bot": bot, "records": records}, errors


def _runtime_emit(
    bot: str,
    contract_id: str,
    spec: dict[str, Any],
    contract_path: Path,
    effective_path: Path,
    raw_path: Path,
) -> dict[str, Any]:
    command = _runtime_command(bot, contract_id, spec, contract_path, effective_path, raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=_runtime_env(bot),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=300,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout.splitlines()[-20:],
        "stderr_tail": completed.stderr.splitlines()[-20:],
    }


def _runtime_command(
    bot: str,
    contract_id: str,
    spec: dict[str, Any],
    contract_path: Path,
    effective_path: Path,
    raw_path: Path,
) -> list[str]:
    started_at = _now()
    if bot == "ibkr":
        return [
            sys.executable,
            "-m",
            "ibkr_trading.runtime",
            "emit-deployment-metadata",
            "--contract",
            _rel(contract_path),
            "--effective-config",
            _rel(effective_path),
            "--output",
            _rel(raw_path),
            "--repo-root",
            ".",
            "--runtime-started-at-utc",
            started_at,
            "--runtime-instance-id",
            f"ibkr:{contract_id}:{_commit()[:12]}",
        ]
    if bot == "crypto":
        return [
            sys.executable,
            "-m",
            "crypto_trader.cli",
            "emit-deployment-metadata",
            "--effective-config",
            _rel(effective_path),
            "--contract-source-root",
            "contracts/strategy_plugins",
            "--contract-work-root",
            _rel(_crypto_contract_work_root()),
            "--state-dir",
            _rel(ROOT / "artifacts" / "validation" / "runtime_deployment_metadata" / "raw" / "crypto_state"),
            "--repo-root",
            ".",
            "--runtime-started-at-utc",
            started_at,
        ]
    if bot == "k_stock":
        return [
            sys.executable,
            "-m",
            "k_stock_trader.olr_kalcb_runtime",
            "emit-deployment-metadata",
            "--contract",
            _rel(contract_path),
            "--effective-config",
            _rel(effective_path),
            "--output",
            _rel(raw_path),
            "--repo-root",
            ".",
            "--runtime-started-at-utc",
            started_at,
            "--runtime-instance-id",
            f"k_stock:{contract_id}:{_commit()[:12]}",
        ]
    raise ValueError(bot)


def _raw_metadata_path(bot: str, contract_id: str) -> Path:
    if bot == "crypto":
        return _crypto_contract_work_root() / contract_id / "deployment_metadata.json"
    return (
        ROOT / "artifacts" / "validation" / "runtime_deployment_metadata"
        / "raw" / bot / contract_id / "deployment_metadata.json"
    )


def _crypto_contract_work_root() -> Path:
    return ROOT / "artifacts" / "validation" / "runtime_deployment_metadata" / "raw" / "crypto_contracts"


def _runtime_env(bot: str) -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(path) for path in sorted((ROOT / "packages").glob("*/src")) if path.exists()]
    if bot == "ibkr":
        paths.extend([str(ROOT / "bots" / "ibkr_trading" / "src"), str(ROOT / "bots" / "ibkr_trading")])
    elif bot == "crypto":
        paths.append(str(ROOT / "bots" / "crypto_trader" / "src"))
    elif bot == "k_stock":
        paths.extend([str(ROOT / "bots" / "k_stock_trader" / "src"), str(ROOT / "bots" / "k_stock_trader")])
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(paths + ([existing] if existing else []))
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _normalize_metadata(
    raw: dict[str, Any],
    *,
    bot: str,
    contract_id: str,
    contract_path: Path,
    effective: dict[str, Any],
    materialized_hash: str,
    image_version: str,
    promotion_hash: str,
    runtime_entrypoint: str,
    source: str,
    emission: dict[str, Any],
    metadata_path: Path,
) -> dict[str, Any]:
    contract = read_json(contract_path)
    telemetry_schema_versions = _required_telemetry_schema_versions(contract)
    metadata = dict(raw)
    metadata.update({
        "metadata_source": "vps_live_bot_runtime_deployment_metadata_v1",
        "emission_environment": "paper_vps",
        "emission_context": "paper_vps_startup",
        "repo_url": _origin(),
        "source_control_origin": _origin(),
        "deployed_commit_sha": _commit(),
        "source_control_commit_sha": _commit(),
        "source_control_worktree_clean": _worktree_clean(),
        "bot_id": bot,
        "portfolio_id": str(effective.get("bot_id") or bot),
        "strategy_id": contract_id,
        "source_strategy_plugin_id": contract.get("plugin_id", ""),
        "config_hash": materialized_hash,
        "materialized_config_hash": materialized_hash,
        "strategy_version": str(contract.get("decision_api_version") or metadata.get("strategy_version") or ""),
        "config_version": str(effective.get("effective_config_hash") or materialized_hash),
        "deployment_id": str(metadata.get("deployment_id") or f"{bot}-{contract_id}-{_commit()[:12]}"),
        "telemetry_schema_version": telemetry_schema_versions[0] if telemetry_schema_versions else "",
        "telemetry_schema_versions": telemetry_schema_versions,
        "strategy_plugin_contract_path": _rel(contract_path),
        "strategy_plugin_contract_hash": file_sha256(contract_path),
        "promotion_hash": promotion_hash,
        "image_version": image_version,
        "runtime_entrypoint": runtime_entrypoint,
        "runtime_instance_id": str(metadata.get("runtime_instance_id") or f"{bot}:{contract_id}:{_commit()[:12]}"),
        "runtime_host_fingerprint": str(metadata.get("runtime_host_fingerprint") or _host_fingerprint()),
        "emitted_at_utc": _utc_z(metadata.get("emitted_at_utc")),
        "live_runtime_started_at_utc": _utc_z(metadata.get("live_runtime_started_at_utc")),
        "dry_run": False,
        "paper_live_fail_closed": True,
        "fail_closed_checks": list(FAIL_CLOSED_CHECKS),
        "runtime_emission": {
            "source": source,
            "command": emission["command"],
            "returncode": emission["returncode"],
            "stdout_tail": emission.get("stdout_tail", []),
            "stderr_tail": emission.get("stderr_tail", []),
            "metadata_path": _rel(metadata_path),
            "raw_runtime_artifact_hash": _hash_payload(raw),
        },
    })
    metadata["assistant_lineage"] = _assistant_lineage(metadata)
    return metadata


def _assistant_lineage(metadata: dict[str, Any]) -> dict[str, Any]:
    existing = metadata.get("assistant_lineage")
    lineage = dict(existing) if isinstance(existing, dict) else {}
    lineage["weekly_signal_ids"] = _dedupe([
        *_list_value(lineage.get("weekly_signal_ids")),
        *_list_value(metadata.get("source_weekly_signal_ids")),
        *_list_value(metadata.get("weekly_signal_ids")),
    ])
    lineage["source_weekly_signal_ids"] = _dedupe([
        *_list_value(lineage.get("source_weekly_signal_ids")),
        *_list_value(metadata.get("source_weekly_signal_ids")),
    ])
    lineage["proposal_ids"] = _dedupe([
        *_list_value(lineage.get("proposal_ids")),
        *_list_value(metadata.get("proposal_ids")),
        *_list_value(metadata.get("proposal_id")),
        *_list_value(metadata.get("source_proposal_ids")),
    ])
    lineage["source_proposal_ids"] = _dedupe([
        *_list_value(lineage.get("source_proposal_ids")),
        *_list_value(metadata.get("source_proposal_ids")),
    ])
    lineage["suggestion_ids"] = _dedupe([
        *_list_value(lineage.get("suggestion_ids")),
        *_list_value(metadata.get("suggestion_ids")),
        *_list_value(metadata.get("suggestion_id")),
    ])
    lineage["candidate_ids"] = _dedupe([
        *_list_value(lineage.get("candidate_ids")),
        *_list_value(metadata.get("candidate_ids")),
        *_list_value(metadata.get("candidate_id")),
    ])
    lineage["hypothesis_ids"] = _dedupe([
        *_list_value(lineage.get("hypothesis_ids")),
        *_list_value(metadata.get("hypothesis_ids")),
        *_list_value(metadata.get("hypothesis_id")),
    ])
    lineage["experiment_id"] = str(lineage.get("experiment_id") or metadata.get("experiment_id") or "")
    lineage["variant_id"] = str(lineage.get("variant_id") or metadata.get("variant_id") or "")
    lineage["parameter_set_id"] = str(lineage.get("parameter_set_id") or metadata.get("parameter_set_id") or "")
    lineage["deployment_id"] = str(lineage.get("deployment_id") or metadata.get("deployment_id") or "")
    lineage["strategy_change_record_ids"] = _dedupe([
        *_list_value(lineage.get("strategy_change_record_ids")),
        *_list_value(metadata.get("strategy_change_record_ids")),
        *_list_value(metadata.get("strategy_change_record_id")),
    ])
    lineage["monthly_outcome_id"] = str(lineage.get("monthly_outcome_id") or metadata.get("monthly_outcome_id") or "")
    lineage["monthly_search_brief_id"] = str(
        lineage.get("monthly_search_brief_id") or metadata.get("monthly_search_brief_id") or ""
    )
    return lineage


def _list_value(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item or "")]
    return [str(value)]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _image_version(bot: str, path: Path) -> str:
    if path.exists():
        tag = (read_json(path).get("image_build") or {}).get("tag")
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


def _git(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return completed.stdout.strip()


def _commit() -> str:
    global _COMMIT
    if _COMMIT is None:
        _COMMIT = _git("rev-parse", "HEAD")
    return _COMMIT


def _origin() -> str:
    global _ORIGIN
    if _ORIGIN is None:
        value = _git("remote", "get-url", "origin")
        if value.startswith("git@github.com:"):
            value = "https://github.com/" + value.removeprefix("git@github.com:")
        _ORIGIN = value.removesuffix(".git")
    return _ORIGIN


def _worktree_clean() -> bool:
    global _WORKTREE_CLEAN
    if _WORKTREE_CLEAN is None:
        _WORKTREE_CLEAN = _git("status", "--porcelain", "--untracked-files=all") == ""
    return _WORKTREE_CLEAN


def _host_fingerprint() -> str:
    raw = "|".join((os.environ.get("COMPUTERNAME", ""), os.environ.get("USERDOMAIN", ""), os.getcwd()))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _utc_z(value: Any) -> str:
    if not value:
        return _now()
    text = str(value)
    if text.endswith("Z"):
        return text
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return _now()
    return parsed.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _hash_payload(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit runtime deployment metadata from bot startup paths.")
    parser.add_argument("--bot", choices=["all", *BOTS], default="all")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Emit diagnostic metadata even when git status is dirty; artifacts are not installable.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
