"""Generate replay-backed validation evidence for scopes with authoritative bundles."""

from __future__ import annotations

import argparse
import json
from calendar import monthrange
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from trading_assistant_backtest.auto.types import Candidate
from trading_assistant_backtest.contract_models import (
    DataBundleManifest,
    DataBundleStatus,
    MonthlyRunManifest,
    MonthlyRunMode,
)
from trading_assistant_backtest.replay.types import WindowSpec
from trading_assistant_backtest.strategies.contracts import load_strategy_plugin_contract
from trading_assistant_backtest.strategies.crypto.replay_evaluator import (
    CryptoReplayPlugin,
)
from trading_assistant_backtest.strategies.crypto.trend import PLUGIN_ID as CRYPTO_TREND_PLUGIN_ID
from trading_assistant_backtest.strategies.krx.olr_kalcb import PLUGIN_ID as K_STOCK_PLUGIN_ID
from trading_assistant_backtest.paths import monorepo_root, resolve_workspace_path, workspace_root
from trading_assistant_backtest.strategies.krx.replay_evaluator import KStockReplayPlugin
from trading_assistant_backtest.strategies.trading.equity_replay_evaluator import (
    TradingStockReplayPlugin,
    TradingSwingReplayPlugin,
)
from trading_assistant_backtest.strategies.trading.momentum import (
    PLUGIN_ID as TRADING_MOMENTUM_PLUGIN_ID,
)
from trading_assistant_backtest.strategies.trading.momentum_replay_evaluator import (
    TradingMomentumReplayPlugin,
)
from trading_assistant_backtest.strategies.trading.stock import (
    PLUGIN_ID as TRADING_STOCK_PLUGIN_ID,
)
from trading_assistant_backtest.strategies.trading.swing import (
    PLUGIN_ID as TRADING_SWING_PLUGIN_ID,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate replay-backed validation evidence.")
    parser.add_argument("--agent-root", type=Path, default=_default_agent_root())
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--bundle-manifest", action="append", type=Path, default=[])
    parser.add_argument(
        "--scope",
        choices=(
            "crypto_trader_portfolio",
            "trading_momentum_family",
            "trading_stock_family",
            "trading_swing_family",
            "k_stock_olr_kalcb",
            "all",
        ),
        default="crypto_trader_portfolio",
    )
    args = parser.parse_args(argv)
    artifact_root_arg = Path(args.artifact_root).resolve() if args.artifact_root else None

    def artifact_root_for(scope_id: str) -> Path:
        if artifact_root_arg is None:
            return _default_scope_artifact_root(args.agent_root, scope_id)
        if args.scope == "all":
            return artifact_root_arg / scope_id
        return artifact_root_arg

    reports: list[dict[str, Any]] = []
    if args.scope in {"crypto_trader_portfolio", "all"}:
        reports.append(
            generate_crypto_replay_evidence(
                agent_root=args.agent_root,
                artifact_root=artifact_root_for("crypto_trader_portfolio"),
                bundle_manifest_paths=args.bundle_manifest or None
                if args.scope == "crypto_trader_portfolio"
                else None,
            )
        )
    if args.scope in {"trading_momentum_family", "all"}:
        reports.append(
            generate_trading_momentum_replay_evidence(
                agent_root=args.agent_root,
                artifact_root=artifact_root_for("trading_momentum_family"),
                bundle_manifest_paths=args.bundle_manifest or None
                if args.scope == "trading_momentum_family"
                else None,
            )
        )
    if args.scope in {"trading_stock_family", "all"}:
        reports.append(
            generate_bar_scope_replay_evidence(
                agent_root=args.agent_root,
                artifact_root=artifact_root_for("trading_stock_family"),
                scope_id="trading_stock_family",
                bundle_manifest_paths=args.bundle_manifest or None
                if args.scope == "trading_stock_family"
                else None,
                default_bundle_paths_factory=_default_trading_stock_bundle_paths,
                contract_relative_path=Path(
                    "contracts/strategy_plugins/trading_stock_family/"
                    "strategy_plugin_contract.json"
                ),
                run_id_prefix="replay-evidence-stock",
                bot_id="trading",
                strategy_id="trading_stock_family",
                plugin_id=TRADING_STOCK_PLUGIN_ID,
                plugin_factory=TradingStockReplayPlugin,
                candidate_family="risk_size_repair",
                starting_baseline_config={
                    "strategy_plugin_id": TRADING_STOCK_PLUGIN_ID,
                    "symbol": "MSFT",
                    "timeframe": "5m",
                    "adoption_enabled": False,
                },
            )
        )
    if args.scope in {"trading_swing_family", "all"}:
        reports.append(
            generate_bar_scope_replay_evidence(
                agent_root=args.agent_root,
                artifact_root=artifact_root_for("trading_swing_family"),
                scope_id="trading_swing_family",
                bundle_manifest_paths=args.bundle_manifest or None
                if args.scope == "trading_swing_family"
                else None,
                default_bundle_paths_factory=_default_trading_swing_bundle_paths,
                contract_relative_path=Path(
                    "contracts/strategy_plugins/trading_swing_family/"
                    "strategy_plugin_contract.json"
                ),
                run_id_prefix="replay-evidence-swing",
                bot_id="trading",
                strategy_id="trading_swing_family",
                plugin_id=TRADING_SWING_PLUGIN_ID,
                plugin_factory=TradingSwingReplayPlugin,
                candidate_family="exit_repair",
                starting_baseline_config={
                    "strategy_plugin_id": TRADING_SWING_PLUGIN_ID,
                    "symbol": "QQQ",
                    "timeframe": "1h",
                    "adoption_enabled": False,
                },
            )
        )
    if args.scope in {"k_stock_olr_kalcb", "all"}:
        reports.append(
            generate_bar_scope_replay_evidence(
                agent_root=args.agent_root,
                artifact_root=artifact_root_for("k_stock_olr_kalcb"),
                scope_id="k_stock_olr_kalcb",
                bundle_manifest_paths=args.bundle_manifest or None
                if args.scope == "k_stock_olr_kalcb"
                else None,
                default_bundle_paths_factory=_default_k_stock_bundle_paths,
                contract_relative_path=Path(
                    "contracts/strategy_plugins/k_stock_olr_kalcb/"
                    "strategy_plugin_contract.json"
                ),
                run_id_prefix="replay-evidence-k-stock",
                bot_id="k_stock",
                strategy_id="k_stock_olr_kalcb",
                plugin_id=K_STOCK_PLUGIN_ID,
                plugin_factory=KStockReplayPlugin,
                candidate_family="session_repair",
                starting_baseline_config={
                    "strategy_plugin_id": K_STOCK_PLUGIN_ID,
                    "symbol": "005930",
                    "timeframe": "5m",
                    "adoption_enabled": False,
                },
            )
        )
    report = reports[0] if len(reports) == 1 else {"status": "pass", "reports": reports}
    if len(reports) > 1 and any(item["status"] not in {"pass", "partial_pass"} for item in reports):
        report["status"] = "partial_pass"
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] in {"pass", "partial_pass"} else 1


def generate_crypto_replay_evidence(
    *,
    agent_root: Path,
    artifact_root: Path,
    bundle_manifest_paths: list[Path] | None = None,
) -> dict[str, Any]:
    agent_root = Path(agent_root).resolve()
    artifact_root = Path(artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    paths = bundle_manifest_paths or _default_crypto_bundle_paths(agent_root)
    runs = [_run_bundle_replay(path, artifact_root=artifact_root) for path in paths]
    incumbent_pass = bool(runs) and all(run["incumbent"]["trade_count"] > 0 for run in runs)
    round_pass = bool(runs) and all(run["round_reproduction"]["ok"] for run in runs)
    leakage_checks = _walk_forward_leakage_checks(runs)
    leakage_pass = _walk_forward_leakage_pass(leakage_checks)
    leakage_checks["status"] = "pass" if leakage_pass else "blocked"
    historical_pass = len(runs) >= 3 and incumbent_pass and round_pass and leakage_pass
    walk_forward_report = {
        "schema_version": "historical_walk_forward_report_v1",
        "scope_id": "crypto_trader_portfolio",
        "status": "pass" if historical_pass else "blocked",
        "window_count": len(runs),
        "minimum_window_count": 3,
        "windows": runs,
        "leakage_checks": leakage_checks,
        "reason": ""
        if historical_pass
        else _walk_forward_leakage_reason(leakage_checks)
        or "at least three authoritative monthly bundle replays are required",
    }
    walk_forward_path = artifact_root / "historical_walk_forward_report.json"
    walk_forward_path.write_text(
        json.dumps(walk_forward_report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    round_path = artifact_root / "round_reproduction_report.json"
    round_path.write_text(
        json.dumps(
            {
                "schema_version": "round_reproduction_report_v1",
                "scope_id": "crypto_trader_portfolio",
                "status": "pass" if round_pass else "blocked",
                "runs": [run["round_reproduction"] for run in runs],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    baseline_path = artifact_root / "frozen_baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema_version": "frozen_replay_baseline_v1",
                "scope_id": "crypto_trader_portfolio",
                "status": "pass" if incumbent_pass else "blocked",
                "latest_run": runs[-1] if runs else {},
                "run_count": len(runs),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "replay_evidence_report_v1",
        "scope_id": "crypto_trader_portfolio",
        "status": "pass" if historical_pass else "partial_pass",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "tests": {
            "incumbent_replay": {
                "ok": incumbent_pass,
                "status": "pass" if incumbent_pass else "blocked",
                "artifact_paths": [str(baseline_path)],
            },
            "round_reproduction": {
                "ok": round_pass,
                "status": "pass" if round_pass else "blocked",
                "artifact_paths": [str(round_path)],
            },
            "historical_walk_forward": {
                "ok": historical_pass,
                "status": "pass" if historical_pass else "blocked",
                "artifact_paths": [str(walk_forward_path)],
            },
        },
        "artifact_paths": [str(baseline_path), str(round_path), str(walk_forward_path)],
    }
    report_path = artifact_root / "replay_evidence_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def generate_trading_momentum_replay_evidence(
    *,
    agent_root: Path,
    artifact_root: Path,
    bundle_manifest_paths: list[Path] | None = None,
) -> dict[str, Any]:
    agent_root = Path(agent_root).resolve()
    artifact_root = Path(artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    paths = bundle_manifest_paths or _default_trading_momentum_bundle_paths(agent_root)
    contract = _load_contract(
        workspace_root(agent_root, "trading_assistant_backtest")
        / "contracts"
        / "trading_momentum_family"
        / "strategy_plugin_contract.json"
    )
    runs = [
        _run_bundle_replay_with_plugin(
            path,
            artifact_root=artifact_root,
            scope_id="trading_momentum_family",
            run_id_prefix="replay-evidence-momentum",
            bot_id="trading",
            strategy_id="trading_momentum_family",
            plugin_id=TRADING_MOMENTUM_PLUGIN_ID,
            plugin_factory=TradingMomentumReplayPlugin,
            candidate_family="filter_repair",
            trading_repo_commit_sha=str(getattr(contract, "live_repo_commit_sha", "") or ""),
            backtest_repo_commit_sha=str(
                getattr(contract, "backtest_adapter_commit_sha", "") or ""
            ),
        )
        for path in paths
    ]
    incumbent_pass = bool(runs) and all(run["incumbent"]["trade_count"] > 0 for run in runs)
    round_pass = bool(runs) and all(run["round_reproduction"]["ok"] for run in runs)
    leakage_checks = _walk_forward_leakage_checks(runs)
    leakage_pass = _walk_forward_leakage_pass(leakage_checks)
    leakage_checks["status"] = "pass" if leakage_pass else "blocked"
    historical_pass = len(runs) >= 3 and incumbent_pass and round_pass and leakage_pass
    objective_scores = [float(run["incumbent"]["objective_score"]) for run in runs]
    walk_forward_report = {
        "schema_version": "historical_walk_forward_report_v1",
        "scope_id": "trading_momentum_family",
        "status": "pass" if historical_pass else "blocked",
        "window_count": len(runs),
        "minimum_window_count": 3,
        "windows": runs,
        "leakage_checks": leakage_checks,
        "objective_stability": {
            "min_score": min(objective_scores) if objective_scores else 0.0,
            "max_score": max(objective_scores) if objective_scores else 0.0,
            "mean_score": sum(objective_scores) / len(objective_scores)
            if objective_scores
            else 0.0,
        },
        "fee_model_version": _single_bundle_field(runs, "fee_model_version"),
        "slippage_model_version": _single_bundle_field(runs, "slippage_model_version"),
        "reason": ""
        if historical_pass
        else _walk_forward_leakage_reason(leakage_checks)
        or "at least three authoritative monthly bundle replays are required",
    }
    walk_forward_path = artifact_root / "historical_walk_forward_report.json"
    walk_forward_path.write_text(
        json.dumps(walk_forward_report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    round_path = artifact_root / "round_reproduction_report.json"
    round_path.write_text(
        json.dumps(
            {
                "schema_version": "round_reproduction_report_v1",
                "scope_id": "trading_momentum_family",
                "status": "pass" if round_pass else "blocked",
                "objective_version": "objective_weights_v1",
                "runs": [run["round_reproduction"] for run in runs],
                "deterministic_rerun_assertions": [
                    {
                        "run_month": run["run_month"],
                        "status": "pass" if run["round_reproduction"]["ok"] else "fail",
                        "trade_hash": run["round_reproduction"]["trade_hash"],
                    }
                    for run in runs
                ],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    baseline_path = artifact_root / "frozen_baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema_version": "frozen_replay_baseline_v1",
                "scope_id": "trading_momentum_family",
                "status": "pass" if incumbent_pass else "blocked",
                "objective_version": "objective_weights_v1",
                "starting_baseline_config": {
                    "strategy_plugin_id": TRADING_MOMENTUM_PLUGIN_ID,
                    "symbol": "NQ",
                    "timeframe": "5m",
                    "adoption_enabled": False,
                },
                "latest_run": runs[-1] if runs else {},
                "run_count": len(runs),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "replay_evidence_report_v1",
        "scope_id": "trading_momentum_family",
        "status": "pass" if historical_pass else "partial_pass",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "tests": {
            "incumbent_replay": {
                "ok": incumbent_pass,
                "status": "pass" if incumbent_pass else "blocked",
                "artifact_paths": [str(baseline_path)],
            },
            "round_reproduction": {
                "ok": round_pass,
                "status": "pass" if round_pass else "blocked",
                "artifact_paths": [str(round_path)],
            },
            "historical_walk_forward": {
                "ok": historical_pass,
                "status": "pass" if historical_pass else "blocked",
                "artifact_paths": [str(walk_forward_path)],
            },
        },
        "artifact_paths": [str(baseline_path), str(round_path), str(walk_forward_path)],
    }
    report_path = artifact_root / "replay_evidence_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def generate_bar_scope_replay_evidence(
    *,
    agent_root: Path,
    artifact_root: Path,
    scope_id: str,
    bundle_manifest_paths: list[Path] | None,
    default_bundle_paths_factory: Any,
    contract_relative_path: Path,
    run_id_prefix: str,
    bot_id: str,
    strategy_id: str,
    plugin_id: str,
    plugin_factory: Any,
    candidate_family: str,
    starting_baseline_config: dict[str, Any],
) -> dict[str, Any]:
    agent_root = Path(agent_root).resolve()
    artifact_root = Path(artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    candidate_paths = bundle_manifest_paths or default_bundle_paths_factory(agent_root)
    paths, skipped_bundles = _authoritative_bundle_paths(candidate_paths)
    contract = _load_contract(resolve_workspace_path(agent_root, contract_relative_path))
    runs = [
        _run_bundle_replay_with_plugin(
            path,
            artifact_root=artifact_root,
            scope_id=scope_id,
            run_id_prefix=run_id_prefix,
            bot_id=bot_id,
            strategy_id=strategy_id,
            plugin_id=plugin_id,
            plugin_factory=plugin_factory,
            candidate_family=candidate_family,
            trading_repo_commit_sha=str(getattr(contract, "live_repo_commit_sha", "") or ""),
            backtest_repo_commit_sha=str(
                getattr(contract, "backtest_adapter_commit_sha", "") or ""
            ),
        )
        for path in paths
    ]
    incumbent_pass = bool(runs) and all(run["incumbent"]["trade_count"] > 0 for run in runs)
    round_pass = bool(runs) and all(run["round_reproduction"]["ok"] for run in runs)
    leakage_checks = _walk_forward_leakage_checks(runs, require_data_repo_source_sha=True)
    leakage_pass = _walk_forward_leakage_pass(leakage_checks)
    leakage_checks["status"] = "pass" if leakage_pass else "blocked"
    historical_pass = len(runs) >= 3 and incumbent_pass and round_pass and leakage_pass
    objective_scores = [float(run["incumbent"]["objective_score"]) for run in runs]
    walk_forward_report = {
        "schema_version": "historical_walk_forward_report_v1",
        "scope_id": scope_id,
        "status": "pass" if historical_pass else "blocked",
        "window_count": len(runs),
        "minimum_window_count": 3,
        "skipped_bundle_count": len(skipped_bundles),
        "skipped_bundles": skipped_bundles,
        "windows": runs,
        "leakage_checks": leakage_checks,
        "objective_stability": {
            "min_score": min(objective_scores) if objective_scores else 0.0,
            "max_score": max(objective_scores) if objective_scores else 0.0,
            "mean_score": sum(objective_scores) / len(objective_scores)
            if objective_scores
            else 0.0,
        },
        "fee_model_version": _single_bundle_field(runs, "fee_model_version"),
        "slippage_model_version": _single_bundle_field(runs, "slippage_model_version"),
        "reason": ""
        if historical_pass
        else _walk_forward_leakage_reason(leakage_checks)
        or _walk_forward_block_reason(runs, skipped_bundles),
    }
    walk_forward_path = artifact_root / "historical_walk_forward_report.json"
    walk_forward_path.write_text(
        json.dumps(walk_forward_report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    round_path = artifact_root / "round_reproduction_report.json"
    round_path.write_text(
        json.dumps(
            {
                "schema_version": "round_reproduction_report_v1",
                "scope_id": scope_id,
                "status": "pass" if round_pass else "blocked",
                "objective_version": "objective_weights_v1",
                "runs": [run["round_reproduction"] for run in runs],
                "deterministic_rerun_assertions": [
                    {
                        "run_month": run["run_month"],
                        "status": "pass" if run["round_reproduction"]["ok"] else "fail",
                        "trade_hash": run["round_reproduction"]["trade_hash"],
                    }
                    for run in runs
                ],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    baseline_path = artifact_root / "frozen_baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema_version": "frozen_replay_baseline_v1",
                "scope_id": scope_id,
                "status": "pass" if incumbent_pass else "blocked",
                "objective_version": "objective_weights_v1",
                "starting_baseline_config": starting_baseline_config,
                "latest_run": runs[-1] if runs else {},
                "run_count": len(runs),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "replay_evidence_report_v1",
        "scope_id": scope_id,
        "status": "pass" if historical_pass else "partial_pass",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "tests": {
            "incumbent_replay": {
                "ok": incumbent_pass,
                "status": "pass" if incumbent_pass else "blocked",
                "artifact_paths": [str(baseline_path)],
            },
            "round_reproduction": {
                "ok": round_pass,
                "status": "pass" if round_pass else "blocked",
                "artifact_paths": [str(round_path)],
            },
            "historical_walk_forward": {
                "ok": historical_pass,
                "status": "pass" if historical_pass else "blocked",
                "artifact_paths": [str(walk_forward_path)],
            },
        },
        "artifact_paths": [str(baseline_path), str(round_path), str(walk_forward_path)],
        "skipped_bundle_count": len(skipped_bundles),
        "skipped_bundles": skipped_bundles,
    }
    report_path = artifact_root / "replay_evidence_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def _authoritative_bundle_paths(paths: list[Path]) -> tuple[list[Path], list[dict[str, Any]]]:
    authoritative_paths: list[Path] = []
    skipped: list[dict[str, Any]] = []
    for path in sorted(Path(item).resolve() for item in paths):
        try:
            bundle = DataBundleManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            skipped.append(
                {
                    "bundle_manifest_path": str(path),
                    "status": "unreadable",
                    "reason": str(exc),
                }
            )
            continue
        if bundle.status == DataBundleStatus.AUTHORITATIVE:
            authoritative_paths.append(path)
            continue
        skipped.append(
            {
                "bundle_manifest_path": str(path),
                "status": bundle.status.value,
                "reason": bundle.diagnostics_only_reason or bundle.status.value,
            }
        )
    return authoritative_paths, skipped


def _walk_forward_block_reason(
    runs: list[dict[str, Any]],
    skipped_bundles: list[dict[str, Any]],
) -> str:
    reason = "at least three authoritative monthly bundle replays are required"
    if skipped_bundles:
        return (
            f"{reason}; skipped {len(skipped_bundles)} non-authoritative "
            "or unreadable bundle(s)"
        )
    return reason


def _walk_forward_leakage_checks(
    runs: list[dict[str, Any]],
    *,
    require_data_repo_source_sha: bool = False,
) -> dict[str, Any]:
    checks = {
        "window_order_strictly_increasing": _windows_strictly_increasing(runs),
        "bundle_checksums_unique": len({run["bundle_checksum"] for run in runs}) == len(runs),
    }
    if require_data_repo_source_sha:
        checks["data_repo_source_sha_present"] = all(
            bool(run.get("data_repo_commit_sha")) for run in runs
        )
    return checks


def _walk_forward_leakage_pass(checks: dict[str, Any]) -> bool:
    bool_checks = [
        value
        for key, value in checks.items()
        if key != "status" and isinstance(value, bool)
    ]
    return bool(bool_checks) and all(bool_checks)


def _walk_forward_leakage_reason(checks: dict[str, Any]) -> str:
    failed = [
        key
        for key, value in checks.items()
        if key != "status" and isinstance(value, bool) and not value
    ]
    if not failed:
        return ""
    return "walk-forward leakage checks failed: " + ", ".join(failed)


def _run_bundle_replay(bundle_path: Path, *, artifact_root: Path) -> dict[str, Any]:
    bundle_path = Path(bundle_path).resolve()
    bundle = DataBundleManifest.model_validate(json.loads(bundle_path.read_text(encoding="utf-8")))
    start, end = _walk_forward_window(bundle_path, bundle)
    run_month = f"{start.year:04d}-{start.month:02d}"
    manifest = MonthlyRunManifest(
        run_id=f"replay-evidence-crypto-{run_month}",
        run_month=run_month,
        mode=MonthlyRunMode.PHASED_AUTO,
        bot_id="crypto_portfolio",
        strategy_id="walk_forward",
        latest_month_start=start,
        latest_month_end=end,
        selection_oos_start=start,
        selection_oos_end=end,
        market_data_manifest_path=str(bundle_path),
        data_bundle_manifest_path=str(bundle_path),
        data_bundle_checksum=bundle.bundle_checksum,
        data_manifest_checksum=bundle.bundle_checksum,
        telemetry_manifest_path=str(artifact_root / f"{run_month}-telemetry.json"),
        artifact_root=str(artifact_root / run_month),
        strategy_plugin_id=CRYPTO_TREND_PLUGIN_ID,
        round_id=f"{run_month}-round-0",
        prior_round_id=f"{run_month}-round-minus-1",
        next_round_id=f"{run_month}-round-1",
    )
    plugin = CryptoReplayPlugin(plugin_id=CRYPTO_TREND_PLUGIN_ID, strategy_id="walk_forward")
    baseline = plugin.load_baseline(manifest, bundle)
    window = WindowSpec("walk_forward", start, end)
    incumbent = plugin.run_incumbent(window, baseline)
    candidate = Candidate(
        candidate_id=f"{run_month}-filter-repair-1",
        family="filter_repair",
        payload={"phase_id": "historical_walk_forward"},
    )
    first = plugin.evaluate_candidate(candidate, window)
    second = plugin.evaluate_candidate(candidate, window)
    round_ok = (
        first.objective_score == second.objective_score
        and first.passed == second.passed
        and first.candidate.payload.get("replay_result", {}).get("trade_hash")
        == second.candidate.payload.get("replay_result", {}).get("trade_hash")
    )
    return {
        "run_month": run_month,
        "bundle_manifest_path": str(bundle_path),
        "bundle_id": bundle.bundle_id,
        "bundle_checksum": bundle.bundle_checksum,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "incumbent": {
            "trade_count": incumbent.trade_count,
            "objective_score": incumbent.objective_score,
            "net_return": incumbent.net_return,
            "max_drawdown": incumbent.max_drawdown,
            "trade_hash": incumbent.diagnostics.get("trade_hash", ""),
        },
        "round_reproduction": {
            "ok": round_ok,
            "run_month": run_month,
            "candidate_id": candidate.candidate_id,
            "objective_score": first.objective_score,
            "passed": first.passed,
            "trade_hash": first.candidate.payload.get("replay_result", {}).get("trade_hash", ""),
        },
    }


def _run_bundle_replay_with_plugin(
    bundle_path: Path,
    *,
    artifact_root: Path,
    scope_id: str,
    run_id_prefix: str,
    bot_id: str,
    strategy_id: str,
    plugin_id: str,
    plugin_factory: Any,
    candidate_family: str,
    trading_repo_commit_sha: str = "",
    backtest_repo_commit_sha: str = "",
) -> dict[str, Any]:
    bundle_path = Path(bundle_path).resolve()
    bundle = DataBundleManifest.model_validate(json.loads(bundle_path.read_text(encoding="utf-8")))
    start, end = _walk_forward_window(bundle_path, bundle)
    run_month = f"{start.year:04d}-{start.month:02d}"
    manifest = MonthlyRunManifest(
        run_id=f"{run_id_prefix}-{run_month}",
        run_month=run_month,
        mode=MonthlyRunMode.PHASED_AUTO,
        bot_id=bot_id,
        strategy_id=strategy_id,
        latest_month_start=start,
        latest_month_end=end,
        selection_oos_start=start,
        selection_oos_end=end,
        market_data_manifest_path=str(bundle_path),
        data_bundle_manifest_path=str(bundle_path),
        data_bundle_checksum=bundle.bundle_checksum,
        data_manifest_checksum=bundle.bundle_checksum,
        telemetry_manifest_path=str(artifact_root / f"{run_month}-telemetry.json"),
        artifact_root=str(artifact_root / run_month),
        strategy_plugin_id=plugin_id,
        trading_repo_commit_sha=trading_repo_commit_sha,
        backtest_repo_commit_sha=backtest_repo_commit_sha,
        round_id=f"{run_month}-round-0",
        prior_round_id=f"{run_month}-round-minus-1",
        next_round_id=f"{run_month}-round-1",
    )
    plugin = plugin_factory(plugin_id=plugin_id, strategy_id=strategy_id)
    baseline = plugin.load_baseline(manifest, bundle)
    window = WindowSpec("walk_forward", start, end)
    incumbent = plugin.run_incumbent(window, baseline)
    candidate = Candidate(
        candidate_id=f"{run_month}-{candidate_family}-1",
        family=candidate_family,
        payload={"phase_id": "historical_walk_forward"},
    )
    first = plugin.evaluate_candidate(candidate, window)
    second = plugin.evaluate_candidate(candidate, window)
    first_hash = first.candidate.payload.get("replay_result", {}).get("trade_hash", "")
    second_hash = second.candidate.payload.get("replay_result", {}).get("trade_hash", "")
    round_ok = (
        first.objective_score == second.objective_score
        and first.passed == second.passed
        and first_hash == second_hash
    )
    return {
        "run_month": run_month,
        "scope_id": scope_id,
        "bundle_manifest_path": str(bundle_path),
        "bundle_id": bundle.bundle_id,
        "bundle_checksum": bundle.bundle_checksum,
        "data_repo_commit_sha": bundle.data_repo_commit_sha,
        "fee_model_version": bundle.fee_model_version,
        "slippage_model_version": bundle.slippage_model_version,
        "adjustment_policy": bundle.adjustment_policy,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "incumbent": {
            "trade_count": incumbent.trade_count,
            "objective_score": incumbent.objective_score,
            "net_return": incumbent.net_return,
            "max_drawdown": incumbent.max_drawdown,
            "trade_hash": incumbent.diagnostics.get("trade_hash", ""),
            "diagnostics": plugin.run_diagnostics(incumbent),
        },
        "round_reproduction": {
            "ok": round_ok,
            "run_month": run_month,
            "round_id": manifest.round_id,
            "prior_round_id": manifest.prior_round_id,
            "candidate_id": candidate.candidate_id,
            "candidate_list": [candidate.candidate_id],
            "selected_rows": [],
            "rejected_rows": [
                {
                    "candidate_id": candidate.candidate_id,
                    "objective_score": first.objective_score,
                    "passed": first.passed,
                    "reason": "; ".join(first.reasons),
                }
            ],
            "objective_version": manifest.objective_version,
            "data_bundle_checksum": bundle.bundle_checksum,
            "source_shas": {
                "data_repo_commit_sha": bundle.data_repo_commit_sha,
                "backtest_repo_commit_sha": manifest.backtest_repo_commit_sha,
                "trading_repo_commit_sha": manifest.trading_repo_commit_sha,
            },
            "objective_score": first.objective_score,
            "passed": first.passed,
            "trade_hash": first_hash,
        },
    }


def _bundle_window(bundle: DataBundleManifest) -> tuple[date, date]:
    starts = [_as_date(item.start_ts) for item in bundle.slice_manifests if item.start_ts]
    ends = [_as_date(item.end_ts) for item in bundle.slice_manifests if item.end_ts]
    if not starts or not ends:
        raise ValueError("bundle slices must have start_ts and end_ts for replay evidence")
    return min(starts), max(ends)


def _walk_forward_window(
    bundle_path: Path,
    bundle: DataBundleManifest,
) -> tuple[date, date]:
    run_month = _run_month_from_bundle_path(bundle_path)
    if not run_month:
        return _bundle_window(bundle)
    year, month = (int(part) for part in run_month.split("-"))
    return date(year, month, 1), date(year, month, monthrange(year, month)[1])


def _run_month_from_bundle_path(bundle_path: Path) -> str:
    parts = Path(bundle_path).parts
    for index, part in enumerate(parts):
        if part == "monthly" and index + 1 < len(parts):
            candidate = parts[index + 1]
            if _is_run_month(candidate):
                return candidate
    for part in reversed(parts):
        if _is_run_month(part):
            return part
    return ""


def _is_run_month(value: str) -> bool:
    parts = value.split("-")
    if len(parts) != 2:
        return False
    try:
        year = int(parts[0])
        month = int(parts[1])
    except ValueError:
        return False
    return 1900 <= year <= 2100 and 1 <= month <= 12


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()


def _default_crypto_bundle_paths(agent_root: Path) -> list[Path]:
    root = workspace_root(agent_root, "trading_assistant_data") / "data" / "bundles" / "monthly"
    preferred = sorted(
        root.glob("*/crypto_portfolio/btc_1d_walk_forward/data_bundle_manifest.json")
    )
    if preferred:
        return preferred
    return sorted(root.glob("*/crypto_portfolio/*/data_bundle_manifest.json"))


def _default_trading_momentum_bundle_paths(agent_root: Path) -> list[Path]:
    root = workspace_root(agent_root, "trading_assistant_data") / "data" / "bundles" / "monthly"
    return _preferred_replay_paths(
        root.glob("*/trading_momentum_family/cme_nq_5m/data_bundle_manifest.json"),
        fallback=root.glob("*/trading_momentum_family/portfolio/data_bundle_manifest.json"),
    )


def _default_trading_stock_bundle_paths(agent_root: Path) -> list[Path]:
    root = workspace_root(agent_root, "trading_assistant_data") / "data" / "bundles" / "monthly"
    return _preferred_replay_paths(
        root.glob("*/trading_stock_family/us_msft_5m/data_bundle_manifest.json"),
        fallback=root.glob("*/trading_stock_family/portfolio/data_bundle_manifest.json"),
    )


def _default_trading_swing_bundle_paths(agent_root: Path) -> list[Path]:
    root = workspace_root(agent_root, "trading_assistant_data") / "data" / "bundles" / "monthly"
    return _preferred_replay_paths(
        root.glob("*/trading_swing_family/us_qqq_1h/data_bundle_manifest.json"),
        fallback=root.glob("*/trading_swing_family/portfolio/data_bundle_manifest.json"),
    )


def _default_k_stock_bundle_paths(agent_root: Path) -> list[Path]:
    root = workspace_root(agent_root, "trading_assistant_data") / "data" / "bundles" / "monthly"
    return _preferred_replay_paths(
        root.glob("*/k_stock_olr_kalcb/krx_kis_005930_5m/data_bundle_manifest.json"),
        fallback=root.glob("*/k_stock_olr_kalcb/portfolio/data_bundle_manifest.json"),
    )


def _preferred_replay_paths(
    preferred: Any,
    *,
    fallback: Any,
) -> list[Path]:
    paths = sorted(Path(path) for path in preferred)
    if len(paths) >= 3:
        return paths
    return sorted(Path(path) for path in fallback)


def _default_scope_artifact_root(agent_root: Path, scope_id: str) -> Path:
    return (
        workspace_root(agent_root, "trading_assistant_backtest")
        / "artifacts"
        / "validation"
        / "replay_evidence"
        / scope_id
    )


def _load_contract(path: Path) -> Any | None:
    contract, errors = load_strategy_plugin_contract(path)
    return None if errors else contract


def _windows_strictly_increasing(runs: list[dict[str, Any]]) -> bool:
    windows = [run.get("window", {}) for run in runs]
    starts = [str(window.get("start") or "") for window in windows]
    return starts == sorted(starts) and len(starts) == len(set(starts))


def _single_bundle_field(runs: list[dict[str, Any]], field_name: str) -> str:
    values = sorted({str(run.get(field_name) or "") for run in runs if run.get(field_name)})
    return values[0] if len(values) == 1 else ",".join(values)


def _default_agent_root() -> Path:
    return monorepo_root()


if __name__ == "__main__":
    raise SystemExit(main())
