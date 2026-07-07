from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from migration_support import (
    BASELINE_INDEX,
    BASELINES,
    INVENTORY_DOC,
    PROMOTION_DRAFTS,
    PORTED_SOURCE_ROOTS,
    ROOT,
    STRATEGY_CONTRACTS,
    artifact_roots_inventory,
    canonical_json_hash,
    canonical_json_hash_path,
    choose_latest_round,
    copy_with_record,
    crypto_strategy_round_manifests,
    direct_ibkr_round_manifests,
    ensure_no_path_escape,
    file_sha256,
    k_stock_round_manifests,
    key_metrics,
    parse_ibkr_enabled_strategies,
    parse_kalcb_alignment,
    read_json,
    rel,
    selected_files,
    source_roots_inventory,
    utc_now,
    write_json,
    IBKR_PROMOTION_MAP,
    K_STOCK_PROMOTION_MAP,
)


PHASE0_ALLOWED_BLOCKERS = {
    "k_stock_missing_backtest_output",
    "kalcb_frontier_size_alignment",
}

K_STOCK_CORE_EVIDENCE_NAMES = {
    "context_status.json",
    "diagnostics_summary.json",
    "evaluations.jsonl",
    "full_diagnostics_index.json",
    "holdout_evaluation.json",
    "live_backtest_parity_alignment.json",
    "live_backtest_parity_alignment.md",
    "live_parity_audit.json",
    "optimized_results.json",
    "paper_live_parity_contract.json",
    "phase_activity_log.jsonl",
    "phase_state.json",
    "progress.json",
    "progress.jsonl",
    "promotion_summary.json",
    "round_evaluation.txt",
    "round_final_diagnostics.txt",
    "round_final_diagnostics_status.json",
    "round_final_full_diagnostics.json",
    "round_final_full_diagnostics.md",
    "run_spec.json",
    "run_summary.json",
}


def main() -> int:
    args = build_parser().parse_args()
    if args.inventory_only:
        inventory = build_inventory()
        write_inventory_doc(inventory)
        print(f"Wrote {rel(INVENTORY_DOC)}")
        return 0
    if args.check:
        return check_phase0(strict=args.strict)
    index = freeze_phase0()
    write_json(BASELINE_INDEX, index)
    write_inventory_doc(index)
    write_strategy_plugin_contracts(index)
    write_promotion_drafts(index)
    print(f"Wrote {rel(BASELINE_INDEX)}")
    print(f"Wrote {rel(INVENTORY_DOC)}")
    print(f"Wrote strategy plugin contract copies under {rel(STRATEGY_CONTRACTS)}")
    print(f"Wrote draft promotion manifests under {rel(PROMOTION_DRAFTS)}")
    blocker_count = sum(1 for gate in index["gates"].values() if gate["status"] not in {"pass", "decision_recorded"})
    if blocker_count:
        print(f"Recorded {blocker_count} blocking gate finding(s); run --check for details.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze Phase 0 optimization baselines and source inventory."
    )
    parser.add_argument("--inventory-only", action="store_true", help="Only write docs/migration_inventory.md.")
    parser.add_argument("--check", action="store_true", help="Verify the frozen Phase 0 artifacts.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat documented scoped blockers as failures. Future completion gates should use this.",
    )
    return parser


def freeze_phase0() -> dict[str, Any]:
    index: dict[str, Any] = {
        "schema_version": "phase0_baseline_index.v1",
        "generated_at": utc_now(),
        "phase": "0",
        "duplicate_round_policy": (
            "For each direct rounds_manifest.json, prefer the manifest's explicit latest_round "
            "when present; otherwise choose the highest non-archived/non-invalidated round and "
            "break same-round ties by latest timestamp. Archive references are recorded and "
            "missing latest artifacts are blockers."
        ),
        "archive_resolution_policy": (
            "Resolve explicit archive_path/archive_replaced_round3_path only when the target exists "
            "inside this checkout. Otherwise fail closed and record the missing artifact."
        ),
        "inventory": build_inventory(),
        "baselines": [],
        "live_configs": {},
        "strategy_plugin_contracts": [],
        "portfolio_bundle": build_crypto_portfolio_bundle(),
        "k_stock_decision": {},
        "gates": {},
    }
    index["baselines"].extend(freeze_ibkr_baselines())
    index["baselines"].extend(freeze_crypto_baselines())
    k_stock_baselines = freeze_k_stock_baselines()
    index["baselines"].extend(k_stock_baselines)
    index["k_stock_decision"] = build_k_stock_decision(k_stock_baselines)
    index["live_configs"] = capture_live_configs()
    index["strategy_plugin_contracts"] = discover_strategy_plugin_contracts()
    index["gates"] = evaluate_phase0_gates(index)
    return index


def build_inventory() -> dict[str, Any]:
    files = selected_files()
    return {
        "schema_version": "migration_inventory.v1",
        "generated_at": utc_now(),
        "source_roots": source_roots_inventory(),
        "selected_files": [record.__dict__ for record in files],
        "artifact_roots": artifact_roots_inventory(),
    }


def write_inventory_doc(inventory_or_index: dict[str, Any]) -> None:
    inventory = inventory_or_index.get("inventory", inventory_or_index)
    gates = inventory_or_index.get("gates", {})
    k_stock_decision = inventory_or_index.get("k_stock_decision", {})
    portfolio_bundle = inventory_or_index.get("portfolio_bundle", {})
    lines = [
        "# Migration Inventory",
        "",
        f"Generated: {inventory.get('generated_at', utc_now())}",
        "",
        "This inventory is generated by `python tools/freeze_optimization_baselines.py --inventory-only`.",
        "",
        "## Source Roots",
        "",
        "| Path | Files | Nested Git | Commit |",
        "| --- | ---: | --- | --- |",
    ]
    for root in inventory.get("source_roots", []):
        lines.append(
            f"| `{root['path']}` | {root['file_count']} | {root['nested_git']} | "
            f"`{root.get('git_head') or 'n/a'}` |"
        )
    lines.extend(
        [
            "",
            "## Artifact Roots",
            "",
            "| Role | Path | Exists | Files |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for root in inventory.get("artifact_roots", []):
        lines.append(f"| {root['role']} | `{root['path']}` | {root['exists']} | {root['file_count']} |")
    lines.extend(
        [
            "",
            "## Selected Files",
            "",
            "| Role | Path | Size | SHA-256 |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for record in inventory.get("selected_files", []):
        lines.append(
            f"| {record['role']} | `{record['path']}` | {record['size_bytes']} | "
            f"`{record['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Phase 0 Findings",
            "",
        ]
    )
    if not gates:
        lines.append("- Run the full Phase 0 freezer to record gate findings.")
    else:
        for gate_id in ("A0", "A2", "A3", "A4", "A5_decision", "KALCB_frontier_size"):
            gate = gates.get(gate_id)
            if gate:
                lines.append(f"- {gate_id}: {gate['status']} - {gate['summary']}")
    if k_stock_decision:
        fingerprint = (k_stock_decision.get("output_fingerprint") or {}).get("canonical_tree_sha256")
        if fingerprint:
            lines.append(f"- K-stock output tree fingerprint: `{fingerprint}`.")
    if portfolio_bundle:
        superseded = (portfolio_bundle.get("superseded_rounds_manifest") or {}).get("path")
        if superseded:
            lines.append(f"- Crypto portfolio rounds manifest supersession: `{superseded}`.")
    INVENTORY_DOC.parent.mkdir(parents=True, exist_ok=True)
    INVENTORY_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def freeze_ibkr_baselines() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for manifest_path in direct_ibkr_round_manifests():
        data = read_json(manifest_path)
        latest = choose_latest_round(data.get("rounds", []), data.get("latest_round"))
        family = manifest_path.parent.parent.name
        strategy = manifest_path.parent.name
        target_dir = BASELINES / "ibkr" / family / strategy
        manifest_copy = copy_with_record(manifest_path, target_dir / "rounds_manifest.json", "rounds_manifest")
        record = {
            "bot": "ibkr",
            "baseline_id": f"ibkr:{family}:{strategy}",
            "family": family,
            "strategy": strategy,
            "source_root": rel(PORTED_SOURCE_ROOTS["ibkr_trading"]),
            "status": "frozen",
            "latest_round": latest.get("round") if latest else None,
            "latest_timestamp": latest.get("timestamp") if latest else None,
            "round_record_canonical_sha256": canonical_json_hash(latest or {}),
            "key_metrics": key_metrics(latest or {}),
            "files": [manifest_copy],
            "notes": [],
        }
        if latest is None:
            record["status"] = "blocked_no_non_archived_round"
            record["notes"].append("No non-archived, non-invalidated round entry exists.")
            records.append(record)
            continue
        round_dir = manifest_path.parent / f"round_{latest.get('round')}"
        optimized = round_dir / "optimized_config.json"
        if optimized.exists():
            record["files"].append(
                copy_with_record(optimized, target_dir / f"round_{latest.get('round')}" / "optimized_config.json", "optimized_config")
            )
        else:
            resolved = resolve_archive_artifact(manifest_path.parent, latest)
            if resolved is not None:
                record["files"].append(
                    copy_with_record(
                        resolved,
                        target_dir / f"round_{latest.get('round')}" / resolved.name,
                        "optimized_config_archive_resolved",
                    )
                )
                record["artifact_resolution"] = {
                    "method": "ported_baseline_resolution",
                    "resolved_source_path": rel(resolved),
                }
            else:
                record["status"] = "blocked_missing_latest_optimized_config"
                record["missing_artifacts"] = [rel(optimized)]
                record["notes"].append(
                    "Latest non-archived manifest entry has no adjacent optimized_config.json and no existing archive replacement path."
                )
        records.append(record)
    return records


def resolve_archive_artifact(strategy_root: Path, latest: dict[str, Any]) -> Path | None:
    raw_candidates = [
        latest.get("archive_path"),
        latest.get("archive_replaced_round3_path"),
        latest.get("cleanup_note"),
    ]
    for raw in raw_candidates:
        if not isinstance(raw, str) or not raw:
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = strategy_root / candidate
        if candidate.is_dir():
            optimized = candidate / "optimized_config.json"
            if optimized.exists():
                return optimized
        if candidate.is_file() and candidate.name == "optimized_config.json":
            return candidate
    return None


def freeze_crypto_baselines() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for manifest_path in crypto_strategy_round_manifests():
        data = read_json(manifest_path)
        latest = choose_latest_round(data.get("rounds", []), data.get("latest_round"))
        strategy = manifest_path.parent.name
        target_dir = BASELINES / "crypto" / strategy
        manifest_copy = copy_with_record(manifest_path, target_dir / "rounds_manifest.json", "rounds_manifest")
        record = {
            "bot": "crypto",
            "baseline_id": f"crypto:{strategy}",
            "strategy": strategy,
            "source_root": rel(PORTED_SOURCE_ROOTS["crypto_trader"]),
            "status": "frozen",
            "latest_round": latest.get("round") if latest else None,
            "latest_timestamp": latest.get("timestamp") if latest else None,
            "round_record_canonical_sha256": canonical_json_hash(latest or {}),
            "contract_hash": latest.get("contract_hash") if latest else None,
            "profile_hash": latest.get("profile_hash") if latest else None,
            "strategy_config_hash": latest.get("strategy_config_hash") if latest else None,
            "key_metrics": key_metrics(latest or {}),
            "files": [manifest_copy],
            "notes": [],
        }
        if latest is None:
            record["status"] = "blocked_no_non_archived_round"
            record["notes"].append("No non-archived, non-invalidated round entry exists.")
            records.append(record)
            continue
        round_dir = manifest_path.parent / f"round_{latest.get('round')}"
        for name, role in [
            ("optimized_config.json", "optimized_config"),
            ("parity_alignment.json", "parity_alignment"),
        ]:
            source = round_dir / name
            if source.exists():
                record["files"].append(copy_with_record(source, target_dir / f"round_{latest.get('round')}" / name, role))
            else:
                record.setdefault("missing_artifacts", []).append(rel(source))
                record["status"] = "blocked_missing_latest_artifact"
        records.append(record)
    return records


def freeze_k_stock_baselines() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for manifest_path in k_stock_round_manifests():
        data = read_json(manifest_path)
        latest = choose_latest_round(data.get("rounds", []), data.get("latest_round"))
        strategy = manifest_path.parent.name
        target_dir = BASELINES / "k_stock" / strategy
        manifest_copy = copy_with_record(manifest_path, target_dir / "rounds_manifest.json", "rounds_manifest")
        record = {
            "bot": "k_stock",
            "baseline_id": f"k_stock:{strategy}",
            "strategy": strategy,
            "source_root": rel(PORTED_SOURCE_ROOTS["k_stock_trader"]),
            "status": "frozen",
            "latest_round": latest.get("round") if latest else None,
            "latest_timestamp": latest.get("timestamp") if latest else None,
            "round_record_canonical_sha256": canonical_json_hash(latest or {}),
            "key_metrics": key_metrics(latest or {}),
            "source_fingerprint": latest.get("source_fingerprint") if latest else None,
            "feature_manifest_hash": latest.get("feature_manifest_hash") if latest else None,
            "candidate_snapshot_hash": latest.get("candidate_snapshot_hash") if latest else None,
            "audit_status": latest.get("audit_status") if latest else None,
            "promotion_status": latest.get("promotion_status") if latest else None,
            "files": [manifest_copy],
            "notes": [],
        }
        if latest is None:
            record["status"] = "blocked_no_non_archived_round"
            record["notes"].append("No non-archived, non-invalidated round entry exists.")
            records.append(record)
            continue
        round_dir = manifest_path.parent / f"round_{latest.get('round')}"
        optimized = round_dir / "optimized_config.json"
        if optimized.exists():
            record["files"].append(
                copy_with_record(
                    optimized,
                    target_dir / f"round_{latest.get('round')}" / "optimized_config.json",
                    "optimized_config",
                )
            )
        else:
            record["status"] = "blocked_missing_latest_optimized_config"
            record["missing_artifacts"] = [rel(optimized)]
            record["notes"].append("Latest K-stock round has no adjacent optimized_config.json.")
            records.append(record)
            continue

        for source in sorted(round_dir.iterdir()):
            if not source.is_file() or source.name == "optimized_config.json":
                continue
            if source.name not in K_STOCK_CORE_EVIDENCE_NAMES:
                continue
            role = source.stem.replace(".", "_")
            record["files"].append(
                copy_with_record(
                    source,
                    target_dir / f"round_{latest.get('round')}" / source.name,
                    role,
                )
            )
        records.append(record)
    return records


def build_crypto_portfolio_bundle() -> dict[str, Any]:
    portfolio_root = BASELINES / "crypto" / "portfolio"
    round3 = portfolio_root / "round_3"
    deployment_manifest = round3 / "deployment_manifest.json"
    target_dir = BASELINES / "crypto" / "portfolio"
    bundle: dict[str, Any] = {
        "status": "superseded_missing_source_rounds_manifest",
        "source_root": rel(portfolio_root),
        "required_source_rounds_manifest": rel(portfolio_root / "rounds_manifest.json"),
        "required_source_rounds_manifest_exists": (portfolio_root / "rounds_manifest.json").exists(),
        "supersession_note": (
            "The ported deployment manifest points at output/portfolio/rounds_manifest.json, "
            "which is absent. Phase 0 creates a small superseding evidence manifest from the "
            "available round_3 deployment, parity, recommended configs, and portfolio config."
        ),
        "files": [],
        "missing_artifacts": [],
    }
    if deployment_manifest.exists():
        deployment = read_json(deployment_manifest)
        bundle["deployment_manifest"] = copy_with_record(
            deployment_manifest, target_dir / "round_3" / "deployment_manifest.json", "deployment_manifest"
        )
        bundle["files"].append(bundle["deployment_manifest"])
        required_paths = [
            deployment.get("portfolio_config_path"),
            deployment.get("parity_alignment_path"),
            *list((deployment.get("strategy_configs") or {}).values()),
        ]
        for raw in required_paths:
            if not raw:
                continue
            source = _ported_crypto_portfolio_path(str(raw))
            if source.exists():
                destination = target_dir / Path(raw).relative_to("output/portfolio")
                bundle["files"].append(copy_with_record(source, destination, "portfolio_bundle_artifact"))
            else:
                bundle["missing_artifacts"].append(rel(source))
    else:
        bundle["status"] = "blocked_missing_deployment_manifest"
        bundle["missing_artifacts"].append(rel(deployment_manifest))
    superseded = build_superseded_crypto_portfolio_manifest(bundle)
    superseded_path = target_dir / "rounds_manifest.superseded.json"
    write_json(superseded_path, superseded)
    bundle["superseded_rounds_manifest"] = {
        "path": rel(superseded_path),
        "canonical_json_sha256": canonical_json_hash(superseded),
        "sha256": file_sha256(superseded_path),
    }
    if bundle["missing_artifacts"]:
        bundle["status"] = "blocked_missing_portfolio_artifact"
    return bundle


def build_superseded_crypto_portfolio_manifest(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "crypto_portfolio_rounds_manifest_supersession.v1",
        "generated_at": utc_now(),
        "supersedes_missing_path": bundle["required_source_rounds_manifest"],
        "status": "explicit_phase0_supersession",
        "available_rounds": [
            {
                "round": 3,
                "deployment_manifest": bundle.get("deployment_manifest", {}).get("source_path"),
                "evidence_files": [
                    item["source_path"]
                    for item in bundle.get("files", [])
                    if item.get("role") == "portfolio_bundle_artifact"
                ],
            }
        ],
        "approval_state": "draft_pending_human_approval_before_crypto_cutover",
    }


def _ported_crypto_portfolio_path(raw_path: str) -> Path:
    normalized = raw_path.replace("\\", "/")
    prefix = "output/portfolio/"
    if normalized.startswith(prefix):
        return BASELINES / "crypto" / "portfolio" / normalized.removeprefix(prefix)
    return PORTED_SOURCE_ROOTS["crypto_trader"] / normalized


def build_k_stock_decision(k_stock_baselines: list[dict[str, Any]]) -> dict[str, Any]:
    output = BASELINES / "k_stock"
    alignment = parse_kalcb_alignment()
    frozen = [record for record in k_stock_baselines if record.get("status") == "frozen"]
    blockers = [record for record in k_stock_baselines if record.get("status") != "frozen"]
    output_fingerprint = directory_fingerprint(output) if output.exists() else None
    if not output.exists():
        status = "blocked_missing_backtest_output"
        decision = (
            "Authoritative K-stock latest accepted artifacts are not present in this checkout. "
            "Do not extract K-stock optimizer/live promotion code until the artifacts are restored "
            "or regenerated in a frozen environment with source SHAs, data fingerprints, commands, "
            "and approval notes."
        )
    elif blockers or not frozen:
        status = "restored_pending_validation"
        decision = (
            "K-stock output root exists, but one or more latest-round artifacts could not be "
            "frozen. Do not extract K-stock optimizer/live promotion code until the blockers are "
            "resolved."
        )
    else:
        status = "restored_frozen"
        decision = (
            "K-stock output root exists and latest accepted KALCB, OLR, and portfolio-synergy "
            "artifacts are frozen with source hashes and an output-tree fingerprint."
        )
    return {
        "status": status,
        "required_output_root": output.relative_to(ROOT).as_posix(),
        "required_output_root_exists": output.exists(),
        "decision": decision,
        "baseline_ids": [record["baseline_id"] for record in frozen],
        "blockers": [record["baseline_id"] for record in blockers],
        "restoration_provenance": (
            "ported_baseline_tree_present_before_phase0_freeze"
            if output.exists()
            else "missing"
        ),
        "output_fingerprint": output_fingerprint,
        "kalcb_alignment": alignment,
    }


def directory_fingerprint(path: Path) -> dict[str, Any]:
    files = [
        {
            "path": rel(child),
            "sha256": file_sha256(child),
            "size_bytes": child.stat().st_size,
        }
        for child in sorted(path.rglob("*"))
        if child.is_file()
    ]
    return {
        "file_count": len(files),
        "canonical_tree_sha256": canonical_json_hash(files),
    }


def capture_live_configs() -> dict[str, Any]:
    paths = {
        "ibkr_strategies": PORTED_SOURCE_ROOTS["ibkr_trading"] / "config" / "strategies.yaml",
        "crypto_live_example": PORTED_SOURCE_ROOTS["crypto_trader"] / "config" / "live_config.example.json",
        "k_stock_kalcb": PORTED_SOURCE_ROOTS["k_stock_trader"] / "config" / "kalcb.yaml",
        "k_stock_olr_universe": PORTED_SOURCE_ROOTS["k_stock_trader"] / "config" / "olr_kalcb" / "olr_deployment_universe_103.yaml",
    }
    captured: dict[str, Any] = {}
    for key, path in paths.items():
        if path.exists():
            captured[key] = {
                "path": rel(path),
                "sha256": file_sha256(path),
            }
    strategies_dir = PORTED_SOURCE_ROOTS["crypto_trader"] / "config" / "strategies"
    if strategies_dir.exists():
        captured["crypto_strategy_configs"] = [
            {"path": rel(path), "sha256": file_sha256(path), "canonical_json_sha256": canonical_json_hash_path(path)}
            for path in sorted(strategies_dir.glob("*.json"))
        ]
    return captured


def discover_strategy_plugin_contracts() -> list[dict[str, Any]]:
    paths = sorted(STRATEGY_CONTRACTS.rglob("strategy_plugin_contract.json"))
    return [
        {
            "source_path": rel(path),
            "sha256": file_sha256(path),
            "canonical_json_sha256": canonical_json_hash_path(path),
        }
        for path in paths
    ]


def write_strategy_plugin_contracts(index: dict[str, Any]) -> None:
    for contract in index.get("strategy_plugin_contracts", []):
        source = ROOT / contract["source_path"]
        parent_name = source.parent.name
        if parent_name == "contracts":
            parent_name = source.parent.parent.name
        destination = STRATEGY_CONTRACTS / parent_name / "strategy_plugin_contract.json"
        ensure_no_path_escape(destination)
        payload = canonicalize_strategy_plugin_contract(read_json(source), source)
        write_json(destination, payload)
        contract["monorepo_path"] = rel(destination)
        contract["monorepo_canonical_json_sha256"] = canonical_json_hash_path(destination)


def canonicalize_strategy_plugin_contract(payload: dict[str, Any], source: Path) -> dict[str, Any]:
    contract = dict(payload)
    source_text = rel(source)
    live_path = str(contract.get("live_repo_path") or "")
    plugin_id = str(contract.get("plugin_id") or "")
    if "crypto" in plugin_id or "crypto_trader" in live_path or "crypto_trader" in source_text:
        live_path = "bots/crypto_trader"
    elif "k-stock" in plugin_id or "k_stock" in plugin_id or "k_stock_trader" in live_path or "k_stock_trader" in source_text:
        live_path = "bots/k_stock_trader"
    elif plugin_id.startswith("trading-") or "trading_" in source_text:
        live_path = "bots/ibkr_trading"
    contract["live_repo_path"] = _monorepo_contract_path(live_path)

    adapter_path = str(contract.get("backtest_adapter_path") or "")
    if adapter_path.startswith("src/"):
        contract["backtest_adapter_path"] = f"packages/trading_assistant_backtest/{adapter_path}"

    fixtures = contract.get("parity_fixture_set")
    if isinstance(fixtures, list):
        contract["parity_fixture_set"] = [
            _monorepo_contract_path(str(path))
            for path in fixtures
            if str(path).strip()
        ]
    return contract


def _monorepo_contract_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    return normalized


def write_promotion_drafts(index: dict[str, Any]) -> None:
    write_ibkr_promotion_drafts(index)
    write_crypto_promotion_drafts(index)
    write_k_stock_decision_drafts(index)


def write_ibkr_promotion_drafts(index: dict[str, Any]) -> None:
    records = {(r.get("family"), r.get("strategy")): r for r in index["baselines"] if r["bot"] == "ibkr"}
    for strategy_id, metadata in parse_ibkr_enabled_strategies().items():
        family_strategy = IBKR_PROMOTION_MAP.get(str(strategy_id))
        disabled = metadata.get("enabled") is False
        promoted_state = "disabled" if disabled else "draft"
        record = records.get(family_strategy) if family_strategy else None
        phase3_status = _phase3_approval_status(record, disabled=disabled)
        payload = {
            "schema_version": "strategy_promotion_manifest.v1.draft",
            "strategy_id": strategy_id,
            "bot_id": "ibkr_trading",
            "venue": "ibkr",
            "promotion_state": promoted_state,
            "generated_at": utc_now(),
            "source_live_config": index["live_configs"].get("ibkr_strategies", {}),
            "baseline_id": record.get("baseline_id") if record else None,
            "baseline_status": record.get("status") if record else "not_mapped",
            "optimizer_round": {
                "round_id": record.get("latest_round") if record else None,
                "rounds_manifest_path": _file_by_role(record, "rounds_manifest") if record else None,
                "optimized_config_path": _file_by_role(record, "optimized_config") if record else None,
            },
            "approval": {
                "status": phase3_status,
                "notes": record.get("notes", []) if record else ["No baseline mapping found."],
            },
        }
        write_json(PROMOTION_DRAFTS / "ibkr" / f"{strategy_id}.json", _monorepo_evidence_paths(payload))


def write_crypto_promotion_drafts(index: dict[str, Any]) -> None:
    for record in [r for r in index["baselines"] if r["bot"] == "crypto"]:
        payload = {
            "schema_version": "strategy_promotion_manifest.v1.draft",
            "strategy_id": record["strategy"],
            "bot_id": "crypto_trader",
            "venue": "hyperliquid",
            "promotion_state": "draft",
            "generated_at": utc_now(),
            "baseline_id": record["baseline_id"],
            "baseline_status": record["status"],
            "optimizer_round": {
                "round_id": record["latest_round"],
                "rounds_manifest_path": _file_by_role(record, "rounds_manifest"),
                "optimized_config_path": _file_by_role(record, "optimized_config"),
                "parity_alignment_path": _file_by_role(record, "parity_alignment"),
            },
            "approval": {
                "status": _phase3_approval_status(record),
                "notes": record.get("notes", []),
            },
        }
        write_json(PROMOTION_DRAFTS / "crypto" / f"{record['strategy']}.json", _monorepo_evidence_paths(payload))
    bundle = index["portfolio_bundle"]
    portfolio_payload = {
        "schema_version": "strategy_promotion_manifest.v1.draft",
        "strategy_id": "portfolio_round_3",
        "bot_id": "crypto_trader",
        "venue": "hyperliquid",
        "promotion_state": "draft_portfolio_bundle_supersession",
        "generated_at": utc_now(),
        "baseline_status": bundle["status"],
        "portfolio_round": {
            "round_id": 3,
            "deployment_manifest_path": bundle.get("deployment_manifest", {}).get("baseline_path"),
            "superseded_rounds_manifest_path": bundle.get("superseded_rounds_manifest", {}).get("path"),
        },
        "approval": {
            "status": (
                "phase3_supersession_recorded"
                if bundle["status"] == "superseded_missing_source_rounds_manifest"
                else "blocked_missing_portfolio_artifact"
            ),
            "notes": [bundle["supersession_note"]],
        },
    }
    write_json(PROMOTION_DRAFTS / "crypto" / "portfolio_round_3.json", _monorepo_evidence_paths(portfolio_payload))


def write_k_stock_decision_drafts(index: dict[str, Any]) -> None:
    decision = index["k_stock_decision"]
    records = {r.get("strategy"): r for r in index["baselines"] if r["bot"] == "k_stock"}
    for strategy_id in ("kalcb", "olr", "olr_kalcb_portfolio"):
        record = records.get(K_STOCK_PROMOTION_MAP[strategy_id])
        frozen = record is not None and record.get("status") == "frozen"
        payload = {
            "schema_version": "strategy_promotion_manifest.v1.draft",
            "strategy_id": strategy_id,
            "bot_id": "k_stock_trader",
            "venue": "kis_krx",
            "promotion_state": "draft" if frozen else "blocked_pending_k_stock_baseline_restore",
            "generated_at": utc_now(),
            "baseline_id": record.get("baseline_id") if record else None,
            "baseline_status": record.get("status") if record else decision["status"],
            "optimizer_round": {
                "round_id": record.get("latest_round") if record else None,
                "rounds_manifest_path": _file_by_role(record, "rounds_manifest") if record else None,
                "optimized_config_path": _file_by_role(record, "optimized_config") if record else None,
            },
            "k_stock_decision": decision,
            "kalcb_alignment": decision["kalcb_alignment"],
            "approval": {
                "status": _phase3_approval_status(record) if frozen else "blocked",
                "notes": [decision["decision"], decision["kalcb_alignment"]["decision"]],
            },
        }
        write_json(PROMOTION_DRAFTS / "k_stock" / f"{strategy_id}.json", _monorepo_evidence_paths(payload))


def _phase3_approval_status(record: dict[str, Any] | None, *, disabled: bool = False) -> str:
    if disabled:
        return "not_required_disabled_strategy"
    if record and record.get("status") == "frozen":
        return "phase3_alignment_passed"
    return "blocked_missing_latest_round"


def _monorepo_evidence_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _monorepo_evidence_paths(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_monorepo_evidence_paths(child) for child in value]
    if isinstance(value, str):
        return _monorepo_contract_path(value)
    return value


def _file_by_role(record: dict[str, Any] | None, role: str) -> str | None:
    if not record:
        return None
    for item in record.get("files", []):
        if item.get("role") == role:
            return item.get("baseline_path")
    return None


def evaluate_phase0_gates(index: dict[str, Any]) -> dict[str, Any]:
    ibkr_blockers = [
        r for r in index["baselines"] if r["bot"] == "ibkr" and r["status"] != "frozen"
    ]
    crypto_blockers = [
        r for r in index["baselines"] if r["bot"] == "crypto" and r["status"] != "frozen"
    ]
    k_stock_records = [r for r in index["baselines"] if r["bot"] == "k_stock"]
    k_stock_blockers = [r for r in k_stock_records if r["status"] != "frozen"]
    portfolio_status = index["portfolio_bundle"]["status"]
    k_stock_status = index["k_stock_decision"]["status"]
    kalcb_status = index["k_stock_decision"]["kalcb_alignment"]["status"]
    return {
        "A0": {
            "status": "pass",
            "summary": "Source inventory generated with selected file hashes and artifact roots.",
            "evidence": rel(INVENTORY_DOC),
        },
        "A2": {
            "status": "pass" if not ibkr_blockers else "blocked",
            "summary": (
                "All direct IBKR latest rounds froze with optimized configs."
                if not ibkr_blockers
                else "Some IBKR latest non-archived rounds are missing optimized config artifacts."
            ),
            "blockers": [r["baseline_id"] for r in ibkr_blockers],
        },
        "A3": {
            "status": "pass" if not crypto_blockers else "blocked",
            "summary": (
                "Crypto momentum/trend/breakout latest round 3 artifacts froze."
                if not crypto_blockers
                else "Some crypto strategy latest artifacts are missing."
            ),
            "blockers": [r["baseline_id"] for r in crypto_blockers],
        },
        "A4": {
            "status": "pass" if portfolio_status == "superseded_missing_source_rounds_manifest" else "blocked",
            "summary": "Crypto portfolio round 3 bundle has explicit supersession evidence for missing rounds_manifest.json.",
            "evidence": index["portfolio_bundle"].get("superseded_rounds_manifest", {}).get("path"),
            "source_status": portfolio_status,
        },
        "A5_decision": {
            "status": (
                "decision_recorded"
                if k_stock_status == "blocked_missing_backtest_output"
                else "pass"
                if k_stock_status == "restored_frozen" and not k_stock_blockers
                else "blocked"
            ),
            "summary": index["k_stock_decision"]["decision"],
            "required_output_root_exists": index["k_stock_decision"]["required_output_root_exists"],
            "blockers": [r["baseline_id"] for r in k_stock_blockers],
            "baseline_ids": [r["baseline_id"] for r in k_stock_records],
        },
        "KALCB_frontier_size": {
            "status": "blocked" if kalcb_status != "aligned" else "pass",
            "summary": index["k_stock_decision"]["kalcb_alignment"]["decision"],
        },
    }


def check_phase0(*, strict: bool) -> int:
    if not BASELINE_INDEX.exists():
        print(f"FAIL A0 - missing {rel(BASELINE_INDEX)}")
        return 1
    index = read_json(BASELINE_INDEX)
    failures: list[str] = []
    if not INVENTORY_DOC.exists():
        failures.append("A0 inventory document is missing")
    for record in index.get("baselines", []):
        for item in record.get("files", []):
            source = ROOT / item["source_path"]
            baseline = ROOT / item["baseline_path"]
            if not source.exists():
                failures.append(f"{record['baseline_id']} source missing: {item['source_path']}")
                continue
            if not baseline.exists():
                failures.append(f"{record['baseline_id']} baseline copy missing: {item['baseline_path']}")
                continue
            if file_sha256(source) != item["sha256"]:
                failures.append(f"{record['baseline_id']} source hash drift: {item['source_path']}")
            if file_sha256(baseline) != item["sha256"]:
                failures.append(f"{record['baseline_id']} baseline copy hash drift: {item['baseline_path']}")
            if item.get("canonical_json_sha256") and canonical_json_hash_path(source) != item["canonical_json_sha256"]:
                failures.append(f"{record['baseline_id']} canonical JSON drift: {item['source_path']}")
        if record.get("status") != "frozen":
            failures.append(f"{record['baseline_id']} status is {record.get('status')}")
    gates = evaluate_phase0_gates(index)
    index["gates"] = gates
    write_json(BASELINE_INDEX, index)
    for gate_id, gate in gates.items():
        status = gate["status"]
        ok = status == "pass" or (status == "decision_recorded" and not strict)
        if gate_id == "KALCB_frontier_size" and not strict:
            ok = True
        prefix = "PASS" if ok else "FAIL"
        print(f"{prefix} {gate_id} - {gate['summary']}")
        if not ok:
            failures.append(f"{gate_id}: {gate['summary']}")
    if failures:
        print("\nPhase 0 check found blockers:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Phase 0 check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
