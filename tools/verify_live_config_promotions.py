from __future__ import annotations

import argparse
import sys

from migration_support import (
    IBKR_PROMOTION_MAP,
    K_STOCK_PROMOTION_MAP,
    PROMOTION_DRAFTS,
    ROOT,
    file_sha256,
    load_baseline_index,
    parse_ibkr_enabled_strategies,
    parse_kalcb_alignment,
    print_result,
    read_json,
)


PROMOTIONS = ROOT / "contracts" / "promotions"
PHASE3_EVIDENCE_STATUSES = {
    "phase3_alignment_passed",
    "phase3_supersession_recorded",
}


def main() -> int:
    args = build_parser().parse_args()
    index = load_baseline_index()
    failures: list[str] = []
    require_latest_round = args.require_latest_round or args.strict
    require_effective_configs = args.require_effective_configs or args.strict
    require_portfolio_bundle = args.require_portfolio_bundle or args.strict
    bots = ["ibkr", "crypto", "k_stock"] if args.bot == "all" else [args.bot]
    for bot in bots:
        if bot == "ibkr":
            failures.extend(check_ibkr(index, require_latest_round=require_latest_round))
        elif bot == "crypto":
            failures.extend(
                check_crypto(
                    index,
                    require_latest_round=require_latest_round,
                    require_portfolio_bundle=require_portfolio_bundle,
                )
            )
        elif bot == "k_stock":
            failures.extend(check_k_stock(index, require_latest_round=require_latest_round))
            if args.strict:
                failures.extend(check_active_kalcb_alignment())
    if require_effective_configs:
        failures.extend(check_effective_configs())
    if failures:
        print("\nLive-config promotion check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Live-config promotion check passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify live config promotion evidence.")
    parser.add_argument("--bot", choices=["all", "ibkr", "crypto", "k_stock"], default="all")
    parser.add_argument("--require-latest-round", action="store_true")
    parser.add_argument("--require-portfolio-bundle", action="store_true")
    parser.add_argument("--require-effective-configs", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Require latest rounds, effective configs, and active KALCB source alignment.")
    return parser


def check_ibkr(index: dict, *, require_latest_round: bool) -> list[str]:
    failures: list[str] = []
    records = {(r.get("family"), r.get("strategy")): r for r in index.get("baselines", []) if r.get("bot") == "ibkr"}
    for strategy_id, metadata in parse_ibkr_enabled_strategies().items():
        path = promotion_path("ibkr", f"{strategy_id}.json")
        enabled = metadata.get("enabled") is True
        if not path.exists():
            if enabled:
                failures.append(f"ibkr:{strategy_id} missing promotion manifest {path.relative_to(ROOT).as_posix()}")
                print_result(False, f"ibkr:{strategy_id}", "missing promotion manifest")
            continue
        promotion = read_json(path)
        family_strategy = IBKR_PROMOTION_MAP.get(strategy_id)
        record = records.get(family_strategy) if family_strategy else None
        if not enabled:
            if str(promotion.get("promotion_state") or "") not in {"disabled", "not_promoted", "research"}:
                failures.append(f"ibkr:{strategy_id} disabled strategy promotion state is not explicit")
                print_result(False, f"ibkr:{strategy_id}", "disabled strategy state is not explicit")
                continue
            print_result(True, f"ibkr:{strategy_id}", "disabled strategy tagged in promotion manifest")
            continue
        if require_latest_round and (not record or record.get("status") != "frozen"):
            reason = record.get("status") if record else "not_mapped"
            failures.append(f"ibkr:{strategy_id} latest round is not frozen ({reason})")
            print_result(False, f"ibkr:{strategy_id}", f"latest round is not frozen ({reason})")
            continue
        if require_latest_round and not check_latest_round_binding(
            failures,
            f"ibkr:{strategy_id}",
            promotion,
            record,
            ("rounds_manifest", "optimized_config"),
        ):
            continue
        if require_latest_round and not check_phase3_evidence_status(failures, f"ibkr:{strategy_id}", promotion):
            continue
        print_result(True, f"ibkr:{strategy_id}", f"promotion manifest {path.relative_to(ROOT).as_posix()}")
    return failures


def check_crypto(
    index: dict,
    *,
    require_latest_round: bool,
    require_portfolio_bundle: bool,
) -> list[str]:
    failures: list[str] = []
    records = {r["strategy"]: r for r in index.get("baselines", []) if r.get("bot") == "crypto"}
    for strategy_id in ("momentum", "trend", "breakout"):
        path = promotion_path("crypto", f"{strategy_id}.json")
        record = records.get(strategy_id)
        if not path.exists():
            failures.append(f"crypto:{strategy_id} missing promotion manifest")
            print_result(False, f"crypto:{strategy_id}", "missing promotion manifest")
            continue
        promotion = read_json(path)
        if require_latest_round and (not record or record.get("status") != "frozen"):
            reason = record.get("status") if record else "not_mapped"
            failures.append(f"crypto:{strategy_id} latest round is not frozen ({reason})")
            print_result(False, f"crypto:{strategy_id}", f"latest round is not frozen ({reason})")
            continue
        if require_latest_round and not check_latest_round_binding(
            failures,
            f"crypto:{strategy_id}",
            promotion,
            record,
            ("rounds_manifest", "optimized_config", "parity_alignment"),
        ):
            continue
        if require_latest_round and not check_phase3_evidence_status(failures, f"crypto:{strategy_id}", promotion):
            continue
        print_result(True, f"crypto:{strategy_id}", f"promotion manifest {path.relative_to(ROOT).as_posix()}")
    if require_portfolio_bundle or require_latest_round:
        failures.extend(check_crypto_portfolio(index))
    return failures


def check_crypto_portfolio(index: dict) -> list[str]:
    failures: list[str] = []
    path = promotion_path("crypto", "portfolio_round_3.json")
    bundle = index.get("portfolio_bundle", {})
    if not path.exists():
        failures.append("crypto:portfolio missing portfolio_round_3 promotion manifest")
        print_result(False, "crypto:portfolio", "missing promotion manifest")
        return failures
    promotion = read_json(path)
    portfolio_round = promotion.get("portfolio_round", {})
    deployment_path = portfolio_round.get("deployment_manifest_path")
    superseded_path = portfolio_round.get("superseded_rounds_manifest_path")
    expected_deployment = (bundle.get("deployment_manifest") or {}).get("baseline_path")
    expected_superseded = (bundle.get("superseded_rounds_manifest") or {}).get("path")
    if not deployment_path or not (ROOT / deployment_path).exists():
        failures.append("crypto:portfolio deployment manifest evidence missing")
        print_result(False, "crypto:portfolio", "deployment manifest evidence missing")
    elif not superseded_path or not (ROOT / superseded_path).exists():
        failures.append("crypto:portfolio superseded rounds manifest evidence missing")
        print_result(False, "crypto:portfolio", "superseded rounds manifest evidence missing")
    elif promotion.get("baseline_status") != bundle.get("status"):
        failures.append("crypto:portfolio promotion baseline status does not match bundle status")
        print_result(False, "crypto:portfolio", "baseline status mismatch")
    elif deployment_path != expected_deployment:
        failures.append("crypto:portfolio deployment manifest path does not match frozen bundle")
        print_result(False, "crypto:portfolio", "deployment manifest path mismatch")
    elif superseded_path != expected_superseded:
        failures.append("crypto:portfolio superseded manifest path does not match frozen bundle")
        print_result(False, "crypto:portfolio", "superseded manifest path mismatch")
    elif str(portfolio_round.get("round_id")) != "3":
        failures.append("crypto:portfolio promotion round_id is not round 3")
        print_result(False, "crypto:portfolio", "round_id mismatch")
    elif bundle.get("missing_artifacts"):
        failures.append("crypto:portfolio has missing bundle artifacts: " + ", ".join(bundle["missing_artifacts"]))
        print_result(False, "crypto:portfolio", "bundle artifacts missing")
    elif not check_phase3_evidence_status(failures, "crypto:portfolio", promotion):
        return failures
    else:
        print_result(True, "crypto:portfolio", "round 3 bundle supersession evidence present")
    return failures


def check_k_stock(index: dict, *, require_latest_round: bool) -> list[str]:
    failures: list[str] = []
    decision = index.get("k_stock_decision", {})
    records = {r.get("strategy"): r for r in index.get("baselines", []) if r.get("bot") == "k_stock"}
    alignment = decision.get("kalcb_alignment", {})
    for strategy_id in ("kalcb", "olr", "olr_kalcb_portfolio"):
        path = promotion_path("k_stock", f"{strategy_id}.json")
        if not path.exists():
            failures.append(f"k_stock:{strategy_id} missing promotion decision manifest")
            print_result(False, f"k_stock:{strategy_id}", "missing promotion decision manifest")
            continue
        promotion = read_json(path)
        if require_latest_round:
            record = records.get(K_STOCK_PROMOTION_MAP[strategy_id])
            if decision.get("status") != "restored_frozen":
                status = decision.get("status")
                failures.append(f"k_stock:{strategy_id} latest baseline not frozen ({status})")
                print_result(False, f"k_stock:{strategy_id}", decision.get("decision", "baseline not frozen"))
                continue
            if alignment.get("status") != "aligned":
                failures.append(f"k_stock:{strategy_id} KALCB frontier alignment is not green")
                print_result(False, f"k_stock:{strategy_id}", alignment.get("decision", "alignment failed"))
                continue
            if not record or record.get("status") != "frozen":
                reason = record.get("status") if record else "not_mapped"
                failures.append(f"k_stock:{strategy_id} latest round is not frozen ({reason})")
                print_result(False, f"k_stock:{strategy_id}", f"latest round is not frozen ({reason})")
                continue
            if not check_latest_round_binding(
                failures,
                f"k_stock:{strategy_id}",
                promotion,
                record,
                ("rounds_manifest", "optimized_config"),
            ):
                continue
            if not check_phase3_evidence_status(failures, f"k_stock:{strategy_id}", promotion):
                continue
            print_result(True, f"k_stock:{strategy_id}", f"promotion manifest {path.relative_to(ROOT).as_posix()}")
        else:
            print_result(True, f"k_stock:{strategy_id}", "blocked promotion decision is recorded")
    return failures


def check_latest_round_binding(
    failures: list[str],
    label: str,
    promotion: dict,
    record: dict,
    required_roles: tuple[str, ...],
) -> bool:
    errors: list[str] = []
    optimizer_round = promotion.get("optimizer_round") or {}
    if promotion.get("baseline_id") != record.get("baseline_id"):
        errors.append("baseline_id does not match frozen baseline record")
    if promotion.get("baseline_status") != "frozen":
        errors.append("baseline_status is not frozen")
    if str(optimizer_round.get("round_id")) != str(record.get("latest_round")):
        errors.append("round_id does not match latest frozen round")
    for role in required_roles:
        key = {
            "optimized_config": "optimized_config_path",
            "parity_alignment": "parity_alignment_path",
            "rounds_manifest": "rounds_manifest_path",
        }[role]
        expected = baseline_path_for_role(record, role)
        actual = optimizer_round.get(key)
        if not expected:
            errors.append(f"frozen baseline lacks {role}")
        elif actual != expected:
            errors.append(f"{key} does not match frozen baseline")
        elif not (ROOT / actual).exists():
            errors.append(f"{key} target is missing")
    source_live_config = promotion.get("source_live_config") or {}
    if source_live_config:
        path = source_live_config.get("path")
        expected_hash = source_live_config.get("sha256")
        if not path or not expected_hash:
            errors.append("source_live_config is incomplete")
        elif not (ROOT / path).exists():
            errors.append("source_live_config path is missing")
        elif file_sha256(ROOT / path) != expected_hash:
            errors.append("source_live_config hash drift")
    for error in errors:
        failures.append(f"{label} {error}")
        print_result(False, label, error)
    return not errors


def baseline_path_for_role(record: dict, role: str) -> str:
    for item in record.get("files", []):
        if item.get("role") == role:
            return str(item.get("baseline_path") or "")
    return ""


def check_phase3_evidence_status(failures: list[str], label: str, promotion: dict) -> bool:
    schema_version = str(promotion.get("schema_version") or "")
    promotion_state = str(promotion.get("promotion_state") or "")
    if "draft" in schema_version or promotion_state.startswith("draft"):
        failures.append(f"{label} promotion manifest is still draft ({promotion_state or 'missing'})")
        print_result(False, label, f"promotion state is {promotion_state or 'missing'}")
        return False
    if promotion_state not in {"approved", "disabled", "not_promoted", "research"}:
        failures.append(f"{label} promotion state is not accepted ({promotion_state or 'missing'})")
        print_result(False, label, f"promotion state is {promotion_state or 'missing'}")
        return False
    status = str((promotion.get("approval") or {}).get("status") or "")
    if status in PHASE3_EVIDENCE_STATUSES:
        return True
    failures.append(f"{label} phase3 evidence status is not accepted ({status or 'missing'})")
    print_result(False, label, f"phase3 evidence status is {status or 'missing'}")
    return False


def check_active_kalcb_alignment() -> list[str]:
    alignment = parse_kalcb_alignment(ROOT / "trading" / "k_stock_trader")
    if alignment.get("status") == "aligned":
        print_result(True, "k_stock:kalcb-frontier", "active source frontier.size values are 103")
        return []
    detail = alignment.get("decision", "frontier alignment failed")
    print_result(False, "k_stock:kalcb-frontier", detail)
    return [f"k_stock:kalcb-frontier {detail}"]


def promotion_path(bot: str, filename: str):
    canonical = PROMOTIONS / bot / filename
    if canonical.exists():
        return canonical
    return PROMOTION_DRAFTS / bot / filename


def check_effective_configs() -> list[str]:
    for src in (
        ROOT / "packages" / "trading_config" / "src",
        ROOT / "packages" / "trading_contracts" / "src",
    ):
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
    from trading_config.verifier import verify_effective_configs

    result = verify_effective_configs(ROOT)
    if result["valid"]:
        for record in result["records"]:
            print_result(
                True,
                f"effective:{record['bot_id']}",
                f"artifact {record['path']} {record['effective_config_hash']}",
            )
        return []
    failures: list[str] = []
    for error in result["errors"]:
        failures.append(
            f"effective:{error.get('bot_id', 'unknown')} {error.get('path', '')}: {error.get('error', '')}"
        )
        print_result(False, f"effective:{error.get('bot_id', 'unknown')}", error.get("error", "failed"))
    return failures


if __name__ == "__main__":
    sys.exit(main())
