"""Runtime deployment metadata artifact helpers."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .lineage import LineageContext, lineage_to_payload, redact_config, stable_hash


_ALLOWED_METADATA_SOURCES = {
    "live_bot_runtime_deployment_metadata_v1",
    "vps_live_bot_runtime_deployment_metadata_v1",
}
_ALLOWED_ENVIRONMENTS = {"live_bot", "vps", "paper_vps", "production_vps"}
_TEXT_ARTIFACT_SUFFIXES = frozenset(
    {
        ".json",
        ".md",
        ".py",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "contracts" / "strategy_plugins").is_dir():
            return parent
    return Path(__file__).resolve().parents[4]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _clean_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    cleaned = value.strip().lower()
    if cleaned in {"1", "true", "yes", "y", "clean"}:
        return True
    if cleaned in {"0", "false", "no", "n", "dirty"}:
        return False
    return None


def _lineage_list(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item or "")]
    return [str(value)]


def _assistant_lineage(lineage_payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "deployment_id",
        "monthly_search_brief_id",
        "weekly_signal_ids",
        "source_weekly_signal_ids",
        "proposal_id",
        "proposal_ids",
        "source_proposal_ids",
        "suggestion_id",
        "suggestion_ids",
        "candidate_id",
        "candidate_ids",
        "hypothesis_id",
        "hypothesis_ids",
        "experiment_id",
        "strategy_change_record_id",
        "strategy_change_record_ids",
        "monthly_outcome_id",
    )
    lineage: dict[str, Any] = {}
    for field in fields:
        value = lineage_payload.get(field)
        if field.endswith("_ids") or field in {"proposal_id", "suggestion_id", "candidate_id", "hypothesis_id", "strategy_change_record_id"}:
            values = _lineage_list(value)
            if values:
                target = f"{field}s" if field.endswith("_id") else field
                lineage.setdefault(target, values)
        elif value not in (None, "", [], {}):
            lineage[field] = str(value)
    if lineage.get("source_weekly_signal_ids") and not lineage.get("weekly_signal_ids"):
        lineage["weekly_signal_ids"] = list(lineage["source_weekly_signal_ids"])
    if lineage.get("source_proposal_ids") and not lineage.get("proposal_ids"):
        lineage["proposal_ids"] = list(lineage["source_proposal_ids"])
    return lineage


def _assistant_driven(lineage: Mapping[str, Any]) -> bool:
    return any(
        _lineage_list(lineage.get(field))
        for field in (
            "weekly_signal_ids",
            "source_weekly_signal_ids",
            "proposal_ids",
            "source_proposal_ids",
            "suggestion_ids",
            "candidate_ids",
            "hypothesis_ids",
            "strategy_change_record_ids",
        )
    ) or any(str(lineage.get(field) or "") for field in ("monthly_search_brief_id", "monthly_outcome_id"))


def _worktree_clean(repo_root: Path, env: Mapping[str, str]) -> bool:
    override = _clean_bool(env.get("SOURCE_CONTROL_WORKTREE_CLEAN"))
    if override is not None:
        return override
    status = _run_git(repo_root, "status", "--porcelain")
    return status == ""


def _normalise_remote(remote: str) -> str:
    value = remote.strip()
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")
    if value.endswith(".git"):
        value = value[:-4]
    return value


def _repo_url(repo_root: Path, env: Mapping[str, str]) -> str:
    explicit = env.get("SOURCE_CONTROL_ORIGIN") or env.get("REPO_URL") or env.get("GITHUB_REPOSITORY_URL")
    remote = explicit or _run_git(repo_root, "config", "--get", "remote.origin.url")
    if not remote:
        return ""
    return _normalise_remote(remote)


def _file_sha256(path: Path) -> str:
    try:
        if not path.exists() or not path.is_file():
            return ""
        data = path.read_bytes()
        if _is_text_artifact(path, data):
            data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        return hashlib.sha256(data).hexdigest()
    except Exception:
        return ""


def _is_text_artifact(path: Path, data: bytes) -> bool:
    if path.suffix.lower() in _TEXT_ARTIFACT_SUFFIXES:
        return True
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _required_telemetry_schema_versions(contract: Mapping[str, Any]) -> list[str]:
    values = contract.get("required_telemetry_schemas") or []
    if not isinstance(values, list):
        values = [values]
    return [str(value).strip() for value in values if str(value or "").strip()]


def _host_fingerprint(env: Mapping[str, str]) -> str:
    explicit = env.get("RUNTIME_HOST_FINGERPRINT")
    if explicit:
        return explicit
    return stable_hash(
        "host_",
        {
            "computer": env.get("COMPUTERNAME", ""),
            "user_domain": env.get("USERDOMAIN", ""),
            "runner": env.get("RUNNER_NAME", ""),
        },
    )


def _bridge_id(lineage: Mapping[str, Any], explicit: str = "") -> str:
    if explicit:
        return explicit
    family_id = str(lineage.get("family_id") or "").strip()
    if family_id:
        return f"trading_{family_id}_family"
    strategy_id = str(lineage.get("strategy_id") or "").strip()
    return strategy_id or "trading_default_bridge"


def _metadata_source(emission_environment: str, env: Mapping[str, str]) -> str:
    explicit = env.get("DEPLOYMENT_METADATA_SOURCE", "")
    if explicit in _ALLOWED_METADATA_SOURCES:
        return explicit
    if emission_environment in {"vps", "paper_vps", "production_vps"}:
        return "vps_live_bot_runtime_deployment_metadata_v1"
    return "live_bot_runtime_deployment_metadata_v1"


def _emission_environment(env: Mapping[str, str]) -> str:
    explicit = env.get("EMISSION_ENVIRONMENT") or env.get("DEPLOYMENT_EMISSION_ENVIRONMENT")
    if explicit in _ALLOWED_ENVIRONMENTS:
        return explicit
    mode = (env.get("TRADING_MODE") or env.get("TRADING_ENV") or "").strip().lower()
    if mode == "paper":
        return "paper_vps"
    if mode == "live":
        return "production_vps"
    return "live_bot"


def build_deployment_metadata(
    lineage: LineageContext | Mapping[str, Any] | None,
    *,
    bridge_id: str = "",
    repo_root: str | Path | None = None,
    effective_config: Mapping[str, Any] | None = None,
    strategy_plugin_contract_path: str | Path | None = None,
    runtime_entrypoint: str = "",
    runtime_started_at_utc: str = "",
    runtime_instance_id: str = "",
    dry_run: bool | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the approval-grade runtime deployment metadata payload.

    The artifact is evidence, not a gate.  A dirty local checkout therefore
    produces ``source_control_worktree_clean = false`` instead of pretending it
    is approval-ready.
    """
    env = env or os.environ
    root = Path(repo_root) if repo_root is not None else _repo_root()
    lineage_payload = lineage_to_payload(lineage)
    assistant_lineage = _assistant_lineage(lineage_payload)
    bridge = _bridge_id(lineage_payload, bridge_id)
    emitted_at = _utc_now()
    emission_environment = _emission_environment(env)
    repo_url = _repo_url(root, env)
    lineage_sha = str(lineage_payload.get("code_sha") or "")
    deployed_sha = (
        env.get("DEPLOYED_COMMIT_SHA")
        or env.get("SOURCE_CONTROL_COMMIT_SHA")
        or (lineage_sha if lineage_sha != "unknown" else "")
        or _run_git(root, "rev-parse", "HEAD")
        or "unknown"
    )
    source_sha = env.get("SOURCE_CONTROL_COMMIT_SHA") or deployed_sha
    worktree_clean = _worktree_clean(root, env)

    contract_path = Path(
        strategy_plugin_contract_path
        or env.get("STRATEGY_PLUGIN_CONTRACT_PATH")
        or f"contracts/strategy_plugins/{bridge}/strategy_plugin_contract.json"
    )
    contract_file = contract_path if contract_path.is_absolute() else root / contract_path
    contract = _read_json(contract_file)
    telemetry_schema_versions = (
        _required_telemetry_schema_versions(contract)
        or ["trading_live_shadow_contract_v1"]
    )
    contract_hash = env.get("STRATEGY_PLUGIN_CONTRACT_HASH") or _file_sha256(contract_file)
    config_hash = stable_hash(
        "cfg_",
        {
            "config_version": lineage_payload.get("config_version", ""),
            "effective_config": redact_config(dict(effective_config or {})),
        },
        length=32,
    )
    is_dry_run = dry_run if dry_run is not None else _clean_bool(env.get("DRY_RUN"))
    if is_dry_run is None:
        is_dry_run = False

    instance_id = runtime_instance_id or env.get("RUNTIME_INSTANCE_ID") or stable_hash(
        "runtime_",
        {
            "bridge_id": bridge,
            "deployment_id": lineage_payload.get("deployment_id", ""),
            "code_sha": deployed_sha,
        },
    )
    started_at = (
        runtime_started_at_utc
        or env.get("LIVE_RUNTIME_STARTED_AT_UTC")
        or env.get("RUNTIME_STARTED_AT_UTC")
        or emitted_at
    )

    return {
        "metadata_source": _metadata_source(emission_environment, env),
        "emission_environment": emission_environment,
        "repo_url": repo_url,
        "source_control_origin": repo_url,
        "deployed_commit_sha": deployed_sha,
        "source_control_commit_sha": source_sha,
        "source_control_worktree_clean": worktree_clean,
        "bot_id": "trading",
        "portfolio_id": str(lineage_payload.get("family_id") or lineage_payload.get("portfolio_id") or ""),
        "strategy_id": bridge,
        "config_hash": config_hash,
        "strategy_version": str(lineage_payload.get("strategy_version") or ""),
        "config_version": str(lineage_payload.get("config_version") or ""),
        "deployment_id": str(lineage_payload.get("deployment_id") or ""),
        "assistant_lineage_source": "runtime_deployment_metadata.lineage_context",
        "assistant_lineage": assistant_lineage,
        "assistant_driven": _assistant_driven(assistant_lineage),
        "telemetry_schema_version": telemetry_schema_versions[0],
        "telemetry_schema_versions": telemetry_schema_versions,
        "strategy_plugin_contract_path": str(contract_path).replace("\\", "/"),
        "strategy_plugin_contract_hash": contract_hash,
        "emitted_at_utc": emitted_at,
        "live_runtime_started_at_utc": started_at,
        "runtime_entrypoint": runtime_entrypoint or env.get("RUNTIME_ENTRYPOINT", ""),
        "runtime_instance_id": instance_id,
        "runtime_host_fingerprint": _host_fingerprint(env),
        "dry_run": bool(is_dry_run),
        "approval_ready": bool(
            repo_url
            and repo_url.startswith("https://github.com/")
            and source_sha == deployed_sha
            and deployed_sha not in {"", "unknown"}
            and worktree_clean
            and contract_hash
        ),
    }


def write_deployment_metadata(
    data_dir: str | Path,
    lineage: LineageContext | Mapping[str, Any] | None,
    *,
    bridge_id: str = "",
    repo_root: str | Path | None = None,
    effective_config: Mapping[str, Any] | None = None,
    strategy_plugin_contract_path: str | Path | None = None,
    runtime_entrypoint: str = "",
    runtime_started_at_utc: str = "",
    runtime_instance_id: str = "",
    dry_run: bool | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    metadata = build_deployment_metadata(
        lineage,
        bridge_id=bridge_id,
        repo_root=repo_root,
        effective_config=effective_config,
        strategy_plugin_contract_path=strategy_plugin_contract_path,
        runtime_entrypoint=runtime_entrypoint,
        runtime_started_at_utc=runtime_started_at_utc,
        runtime_instance_id=runtime_instance_id,
        dry_run=dry_run,
        env=env,
    )
    bridge = str(metadata["strategy_id"])
    out_dir = Path(data_dir) / "deployments" / bridge
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "deployment_metadata.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)
    return path
