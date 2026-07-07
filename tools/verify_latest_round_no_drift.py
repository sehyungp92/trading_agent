from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from migration_support import (
    BASELINE_INDEX,
    ROOT,
    canonical_json_hash_path,
    file_sha256,
    iter_baseline_records,
    print_result,
    read_json,
)


def main() -> int:
    args = build_parser().parse_args()
    baseline_path = ROOT / args.baseline if not Path(args.baseline).is_absolute() else Path(args.baseline)
    if not baseline_path.exists():
        print(f"FAIL baseline - missing {baseline_path}")
        return 1
    index = read_json(baseline_path)
    requested_bot = args.bot
    failures: list[str] = []
    if args.strict:
        failures.extend(check_data_portability(requested_bot))
    checked = 0
    for record in iter_baseline_records(index, requested_bot):
        checked += 1
        baseline_id = record["baseline_id"]
        if record.get("status") != "frozen":
            message = f"status is {record.get('status')}; latest artifact evidence is incomplete"
            print_result(False, baseline_id, message)
            failures.append(f"{baseline_id}: {message}")
            continue
        for item in record.get("files", []):
            failures.extend(check_file_record(baseline_id, item, strict=args.strict))
    if requested_bot in {"all", "crypto"}:
        failures.extend(check_crypto_portfolio_bundle(index, strict=args.strict))
    if requested_bot in {"all", "k_stock"}:
        decision = index.get("k_stock_decision", {})
        status = decision.get("status")
        if status != "restored_frozen":
            message = decision.get("decision", "K-stock baseline evidence is not restored.")
            print_result(False, "k_stock:A5", message)
            failures.append(f"k_stock:A5: {message}")
        elif not any(record.get("bot") == "k_stock" for record in index.get("baselines", [])):
            message = "K-stock decision is restored_frozen but no baseline records are present"
            print_result(False, "k_stock:A5", message)
            failures.append(f"k_stock:A5: {message}")
        else:
            print_result(True, "k_stock:A5", "restored K-stock baselines are frozen")
    if checked == 0 and requested_bot not in {"all", "k_stock"}:
        failures.append(f"No baseline records found for bot {requested_bot!r}")
    if failures:
        print("\nLatest-round no-drift check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"Latest-round no-drift check passed using {baseline_path.relative_to(ROOT).as_posix()}.")
    return 0


def check_data_portability(bot: str) -> list[str]:
    command = [sys.executable, "tools/verify_backtest_data_portability.py", "--bot", bot]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    label = f"data-portability:{bot}"
    if completed.returncode == 0:
        print_result(True, label, "historical data manifest verified")
        return []
    details = "\n".join((completed.stdout or completed.stderr).splitlines()[-20:])
    message = "historical data portability prerequisite failed"
    print_result(False, label, message)
    return [f"{label}: {message}\n{details}"]


def check_file_record(label: str, item: dict, *, strict: bool) -> list[str]:
    failures: list[str] = []
    source = ROOT / item["source_path"]
    baseline = ROOT / item["baseline_path"]
    archived = ROOT / item["archived_source_path"] if item.get("archived_source_path") else None
    if not source.exists():
        message = f"source missing: {item['source_path']}"
        print_result(False, label, message)
        return [f"{label}: {message}"]
    if not baseline.exists():
        message = f"baseline copy missing: {item['baseline_path']}"
        print_result(False, label, message)
        return [f"{label}: {message}"]
    source_hash = file_sha256(source)
    baseline_hash = file_sha256(baseline)
    if source_hash != item["sha256"]:
        message = f"source hash drift in {item['source_path']}"
        print_result(False, label, message)
        failures.append(f"{label}: {message}")
    elif baseline_hash != item["sha256"]:
        message = f"baseline hash drift in {item['baseline_path']}"
        print_result(False, label, message)
        failures.append(f"{label}: {message}")
    elif item.get("canonical_json_sha256") and canonical_json_hash_path(source) != item["canonical_json_sha256"]:
        message = f"canonical JSON drift in {item['source_path']}"
        print_result(False, label, message)
        failures.append(f"{label}: {message}")
    else:
        print_result(True, f"{label}:{item['role']}", item["sha256"])
    if strict and archived is not None:
        archived_label = f"{label}:{item['role']}:archived-source"
        if not archived.exists():
            message = f"archived source missing: {item['archived_source_path']}"
            print_result(False, archived_label, message)
            failures.append(f"{label}: {message}")
        elif file_sha256(archived) != item["sha256"]:
            message = f"archived source hash drift in {item['archived_source_path']}"
            print_result(False, archived_label, message)
            failures.append(f"{label}: {message}")
        elif item.get("canonical_json_sha256") and canonical_json_hash_path(archived) != item["canonical_json_sha256"]:
            message = f"archived source canonical JSON drift in {item['archived_source_path']}"
            print_result(False, archived_label, message)
            failures.append(f"{label}: {message}")
        else:
            print_result(True, archived_label, item["sha256"])
    return failures


def check_crypto_portfolio_bundle(index: dict, *, strict: bool) -> list[str]:
    bundle = index.get("portfolio_bundle") or {}
    if not bundle:
        print_result(False, "crypto:portfolio_round_3", "missing portfolio bundle evidence")
        return ["crypto:portfolio_round_3: missing portfolio bundle evidence"]
    failures: list[str] = []
    if bundle.get("missing_artifacts"):
        message = "missing bundle artifacts: " + ", ".join(bundle["missing_artifacts"])
        print_result(False, "crypto:portfolio_round_3", message)
        failures.append(f"crypto:portfolio_round_3: {message}")
    for item in bundle.get("files", []):
        failures.extend(check_file_record("crypto:portfolio_round_3", item, strict=strict))
    superseded = bundle.get("superseded_rounds_manifest") or {}
    if not superseded:
        message = "missing superseded rounds manifest evidence"
        print_result(False, "crypto:portfolio_round_3", message)
        failures.append(f"crypto:portfolio_round_3: {message}")
    else:
        failures.extend(check_file_record(
            "crypto:portfolio_round_3",
            {
                "role": "superseded_rounds_manifest",
                "source_path": superseded["path"],
                "baseline_path": superseded["path"],
                "sha256": superseded["sha256"],
                "canonical_json_sha256": superseded.get("canonical_json_sha256", ""),
            },
            strict=False,
        ))
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify frozen latest optimization baselines have no drift.")
    parser.add_argument("--bot", choices=["all", "ibkr", "crypto", "k_stock"], default="all")
    parser.add_argument("--baseline", default=str(BASELINE_INDEX), help="Baseline index path.")
    parser.add_argument("--strict", action="store_true", help="Also verify archived reference source files against frozen hashes.")
    return parser


if __name__ == "__main__":
    sys.exit(main())
