from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from trading_assistant_backtest.contract_models import (
    DataBundleManifest,
    DataBundleSlice,
    DECISION_PARITY_DIMENSIONS,
    DecisionParityReport,
    DecisionParityStatus,
    MonthlyRunManifest,
    StrategyPluginMaturity,
)
from trading_assistant_backtest.file_hashes import sha256_file
from trading_assistant_backtest.replay.decision_trace import DecisionTraceEvent
from trading_assistant_backtest.replay.parity import decision_parity_report_from_traces
from trading_assistant_backtest.strategies.contracts import (
    load_strategy_plugin_contract,
    strategy_plugin_errors,
)
from trading_assistant_backtest.strategies.crypto.trend import (
    build_crypto_trend_decision_parity_report,
    decision_trace_from_fixture,
)
from trading_assistant_backtest.strategies.deployment import (
    deployment_metadata_errors,
    load_deployment_metadata,
)
from trading_assistant_backtest.strategies.live_clone import validate_clean_checkout
from trading_assistant_backtest.validation.decision_parity_run import (
    run_crypto_trend_decision_parity_validation,
)
from trading_assistant_backtest.validation.week1_decision_parity_run import (
    run_week1_decision_parity_validations,
)
from tests.paths import MONOREPO_ROOT, package_workspace

AGENT_ROOT = MONOREPO_ROOT
CRYPTO_TRADER_REPO = AGENT_ROOT / "bots" / "crypto_trader"
K_STOCK_TRADER_REPO = AGENT_ROOT / "bots" / "k_stock_trader"
TRADING_REPO = AGENT_ROOT / "bots" / "ibkr_trading"
PERSISTED_CRYPTO_CONTRACT = (
    package_workspace("trading_assistant_backtest")
    / "contracts"
    / "crypto_trend_v1"
    / "strategy_plugin_contract.json"
)
PERSISTED_CRYPTO_DEPLOYMENT = PERSISTED_CRYPTO_CONTRACT.parent / "deployment_metadata.json"
PERSISTED_K_STOCK_CONTRACT = (
    package_workspace("trading_assistant_backtest")
    / "contracts"
    / "k_stock_olr_kalcb"
    / "strategy_plugin_contract.json"
)
PERSISTED_TRADING_STOCK_CONTRACT = (
    package_workspace("trading_assistant_backtest")
    / "contracts"
    / "trading_stock_family"
    / "strategy_plugin_contract.json"
)


def _manifest(
    tmp_path: Path,
    *,
    deployment_metadata_path: str = "",
    config_hash: str = "",
) -> MonthlyRunManifest:
    return MonthlyRunManifest(
        run_id="monthly-bot1-strat1-2026-04",
        run_month="2026-04",
        bot_id="bot1",
        strategy_id="strat1",
        latest_month_start=date(2026, 4, 1),
        latest_month_end=date(2026, 4, 30),
        market_data_manifest_path=str(tmp_path / "data_bundle.json"),
        telemetry_manifest_path=str(tmp_path / "telemetry.json"),
        artifact_root=str(tmp_path / "artifacts"),
        strategy_plugin_id="strat1-plugin",
        trading_repo_commit_sha="live-sha",
        backtest_repo_commit_sha="backtest-sha",
        deployment_metadata_path=deployment_metadata_path,
        config_hash=config_hash,
    )


def test_passing_decision_parity_requires_lineage_and_evidence() -> None:
    with pytest.raises(ValueError, match="decision parity pass missing required fields"):
        DecisionParityReport(
            run_id="run1",
            candidate_id="cand1",
            status=DecisionParityStatus.PASS,
            checks=[
                {
                    "dimension": dimension,
                    "status": DecisionParityStatus.PASS,
                    "match_rate": 1.0,
                    "mismatch_count": 0,
                    "evidence_paths": ["fixture.json"],
                }
                for dimension in sorted(DECISION_PARITY_DIMENSIONS)
            ],
        )


def test_decision_trace_report_passes_when_all_dimensions_match(tmp_path: Path) -> None:
    evidence = tmp_path / "decision_trace.json"
    evidence.write_text("{}", encoding="utf-8")
    events = [
        DecisionTraceEvent(
            ts=datetime(2026, 4, 1, 9, 30, tzinfo=UTC),
            dimension=dimension,
            key="AAPL:1m",
            payload={"decision": "hold", "score": 1},
        )
        for dimension in sorted(DECISION_PARITY_DIMENSIONS)
    ]

    report = decision_parity_report_from_traces(
        _manifest(tmp_path),
        candidate_id="candidate-1",
        live_events=events,
        adapter_events=list(reversed(events)),
        evidence_paths=[str(evidence)],
    )

    assert report.status == DecisionParityStatus.PASS
    assert report.eligible_for_structural_approval is True
    assert {check.dimension for check in report.checks} == DECISION_PARITY_DIMENSIONS


def test_decision_trace_report_blocks_missing_dimension(tmp_path: Path) -> None:
    evidence = tmp_path / "decision_trace.json"
    evidence.write_text("{}", encoding="utf-8")
    events = [
        DecisionTraceEvent(None, dimension, "AAPL:1m", {"decision": "hold"})
        for dimension in sorted(DECISION_PARITY_DIMENSIONS - {"stops"})
    ]

    report = decision_parity_report_from_traces(
        _manifest(tmp_path),
        candidate_id="candidate-1",
        live_events=events,
        adapter_events=events,
        evidence_paths=[str(evidence)],
    )

    stops = next(check for check in report.checks if check.dimension == "stops")
    assert report.status == DecisionParityStatus.INSUFFICIENT_DATA
    assert stops.status == DecisionParityStatus.INSUFFICIENT_DATA


def test_deployment_metadata_mismatch_blocks_bridge(tmp_path: Path) -> None:
    metadata_path = tmp_path / "deployment_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "bot_id": "bot1",
                "strategy_id": "strat1",
                "repo_url": "https://github.com/example/live.git",
                "deployed_commit_sha": "other-live-sha",
                "config_hash": "config-sha",
                "strategy_version": "strategy_v1",
                "config_version": "config_v1",
                "telemetry_schema_version": "trade_event_v1",
                "strategy_plugin_contract_path": "strategy_plugin_contract.json",
                "strategy_plugin_contract_hash": "contract-sha",
            }
        ),
        encoding="utf-8",
    )

    errors = deployment_metadata_errors(
        _manifest(tmp_path, deployment_metadata_path=str(metadata_path))
    )

    assert "deployment metadata deployed_commit_sha does not match run manifest" in errors


def test_deployment_metadata_config_hash_mismatch_blocks_bridge(tmp_path: Path) -> None:
    metadata_path = tmp_path / "deployment_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "bot_id": "bot1",
                "strategy_id": "strat1",
                "repo_url": "https://github.com/example/live.git",
                "deployed_commit_sha": "live-sha",
                "config_hash": "other-config-sha",
                "strategy_version": "strategy_v1",
                "config_version": "config_v1",
                "telemetry_schema_version": "trade_event_v1",
                "strategy_plugin_contract_path": "strategy_plugin_contract.json",
                "strategy_plugin_contract_hash": "contract-sha",
            }
        ),
        encoding="utf-8",
    )

    errors = deployment_metadata_errors(
        _manifest(
            tmp_path,
            deployment_metadata_path=str(metadata_path),
            config_hash="config-sha",
        )
    )

    assert "deployment metadata config_hash does not match run manifest" in errors


def test_deployment_metadata_is_checked_without_plugin_contract(tmp_path: Path) -> None:
    metadata_path = tmp_path / "deployment_metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")

    errors = strategy_plugin_errors(
        _manifest(tmp_path, deployment_metadata_path=str(metadata_path)),
        bundle=None,
    )

    assert errors and errors[0].startswith("deployment metadata is invalid")


def test_mature_strategy_plugin_requires_deployment_metadata_path(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    repo = tmp_path / "live"
    repo.mkdir()
    _git(["init"], repo)
    _git(["config", "user.email", "fixture@example.com"], repo)
    _git(["config", "user.name", "Fixture"], repo)
    (repo / "strategy.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(["add", "strategy.py"], repo)
    _git(["commit", "-m", "fixture"], repo)
    head = _git(["rev-parse", "HEAD"], repo).strip()
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text("{}", encoding="utf-8")
    contract_path = tmp_path / "strategy_plugin_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "plugin_id": "strat1-plugin",
                "live_repo_path": str(repo),
                "live_repo_commit_sha": head,
                "backtest_adapter_path": "adapters/strat1.py",
                "backtest_adapter_commit_sha": "backtest-sha",
                "config_schema_version": "config_v1",
                "decision_api_version": "decision_api_v1",
                "required_telemetry_schemas": ["trade_event_v1"],
                "supported_symbols": ["AAPL"],
                "supported_timeframes": ["1m"],
                "parity_fixture_set": [str(fixture_path)],
                "maturity": "shadow_validated",
            }
        ),
        encoding="utf-8",
    )
    manifest = _manifest(tmp_path)
    manifest.strategy_plugin_contract_path = str(contract_path)
    manifest.trading_repo_commit_sha = head
    manifest.backtest_repo_commit_sha = "backtest-sha"

    errors = strategy_plugin_errors(manifest, bundle=None)

    assert "deployment metadata path is required for mature strategy plugin contract" in errors


def test_family_contract_does_not_reject_sibling_bundle_slices(tmp_path: Path) -> None:
    bundle = DataBundleManifest(
        data_repo_path=".",
        data_repo_commit_sha="a" * 40,
        data_repo_branch="main",
        slice_manifests=[
            DataBundleSlice(
                manifest_path="msft.json",
                manifest_id="msft-5m",
                source="ibkr",
                market="us_equity",
                symbol="MSFT",
                timeframe="5m",
                start_ts="2026-05-01T13:30:00Z",
                end_ts="2026-05-29T20:00:00Z",
                checksum="b" * 64,
                calendar="us_equity_rth",
                authoritative=True,
            ),
            DataBundleSlice(
                manifest_path="aapl.json",
                manifest_id="aapl-30m",
                source="ibkr",
                market="us_equity",
                symbol="AAPL",
                timeframe="30m",
                start_ts="2026-05-01T13:30:00Z",
                end_ts="2026-05-29T20:00:00Z",
                checksum="c" * 64,
                calendar="us_equity_eth",
                authoritative=True,
            ),
        ],
        calendars=["us_equity_eth", "us_equity_rth"],
        fee_model_version="fee_v1",
        slippage_model_version="slippage_v1",
        adjustment_policy="raw",
        status="authoritative",
    )
    manifest = MonthlyRunManifest(
        run_id="monthly-trading-stock-2026-05",
        run_month="2026-05",
        bot_id="trading",
        strategy_id="trading_stock_family",
        strategy_version="trading_stock_shadow_v1",
        config_version="trading_stock_family_shadow_2026-05-31",
        config_hash="4bfd5d145d581942bf785fe1bea580c4fea286cdab94ea3e72b32d059f94411c",
        deployment_id="trading-stock-family-shadow-5a901bc9",
        latest_month_start=date(2026, 5, 1),
        latest_month_end=date(2026, 5, 31),
        market_data_manifest_path=str(tmp_path / "data_bundle.json"),
        telemetry_manifest_path=str(tmp_path / "telemetry.json"),
        artifact_root=str(tmp_path / "artifacts"),
        strategy_plugin_id="trading-stock-family",
        strategy_plugin_contract_path=str(PERSISTED_TRADING_STOCK_CONTRACT),
        trading_repo_commit_sha="5a901bc9547cf90fd2beaa8b820994feb384e6d6",
        backtest_repo_commit_sha="212e5d7115af878cb3032c0ec6adac2b2e12b073a292e615b00a062b7164eded",
        deployment_metadata_path=str(PERSISTED_TRADING_STOCK_CONTRACT.parent / "deployment_metadata.json"),
    )

    errors = strategy_plugin_errors(manifest, bundle=bundle)

    assert not [error for error in errors if error.startswith("strategy plugin does not support")]


def test_clean_checkout_validator_rejects_dirty_or_wrong_head(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    repo = tmp_path / "live"
    repo.mkdir()
    _git(["init"], repo)
    _git(["config", "user.email", "fixture@example.com"], repo)
    _git(["config", "user.name", "Fixture"], repo)
    (repo / "strategy.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(["add", "strategy.py"], repo)
    _git(["commit", "-m", "fixture"], repo)
    head = _git(["rev-parse", "HEAD"], repo).strip()

    assert validate_clean_checkout(repo, head) == []
    assert validate_clean_checkout(repo, "wrong-sha")[0].startswith("live repo HEAD")
    (repo / "strategy.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert "uncommitted changes" in validate_clean_checkout(repo, head)[0]


def test_crypto_trend_shared_kernel_parity_uses_pinned_live_repo(tmp_path: Path) -> None:
    if not CRYPTO_TRADER_REPO.exists():
        pytest.skip("crypto_trader reference repo is not available")
    live_sha = _git(["rev-parse", "HEAD"], CRYPTO_TRADER_REPO).strip()
    fixture_path = tmp_path / "crypto_trend_entry_fixture.json"
    fixture_path.write_text(json.dumps(_crypto_trend_fixture()), encoding="utf-8")
    manifest = _manifest(tmp_path)
    manifest.trading_repo_commit_sha = live_sha

    report = build_crypto_trend_decision_parity_report(
        manifest,
        candidate_id="strategy-plugin-contract",
        fixture_paths=[fixture_path],
        live_repo_path=CRYPTO_TRADER_REPO,
        live_repo_commit_sha=live_sha,
        backtest_adapter_commit_sha="backtest-sha",
    )

    assert report.status == DecisionParityStatus.PASS
    assert report.strategy_plugin_id == "strat1-plugin"
    assert report.live_repo_commit_sha == live_sha
    assert {check.dimension for check in report.checks} == DECISION_PARITY_DIMENSIONS
    assert all(check.evidence_paths == [str(fixture_path)] for check in report.checks)


def test_persisted_crypto_trend_shadow_contract_has_broad_parity_fixtures(tmp_path: Path) -> None:
    if not PERSISTED_CRYPTO_CONTRACT.exists():
        pytest.skip("persisted crypto trend strategy contract is not available")
    contract, errors = load_strategy_plugin_contract(PERSISTED_CRYPTO_CONTRACT)
    assert errors == []
    assert contract is not None
    assert contract.maturity == StrategyPluginMaturity.SHADOW_VALIDATED
    assert contract.eligible_for_optimizer is True
    assert contract.eligible_for_approval is False
    assert Path(contract.live_repo_path).resolve() == CRYPTO_TRADER_REPO.resolve()
    reference_config = CRYPTO_TRADER_REPO / "config" / "strategies" / "trend.json"
    deployment = load_deployment_metadata(PERSISTED_CRYPTO_DEPLOYMENT)
    deployment_payload = json.loads(PERSISTED_CRYPTO_DEPLOYMENT.read_text(encoding="utf-8"))
    assert deployment.deployed_commit_sha == contract.live_repo_commit_sha
    assert (
        PERSISTED_CRYPTO_DEPLOYMENT.parent
        / deployment_payload["strategy_plugin_contract_path"]
    ).resolve() == PERSISTED_CRYPTO_CONTRACT.resolve()
    assert deployment_payload["strategy_plugin_contract_hash"] == _sha256_file(
        PERSISTED_CRYPTO_CONTRACT
    )
    assert contract.backtest_adapter_commit_sha == _sha256_file(
        package_workspace("trading_assistant_backtest")
        / "src"
        / "trading_assistant_backtest"
        / "strategies"
        / "crypto"
        / "trend.py"
    )
    assert len(contract.parity_fixture_set) >= 4
    assert all(Path(path).exists() for path in contract.parity_fixture_set)
    if not reference_config.exists():
        pytest.skip("crypto_trader reference repo is local-only and not available in CI")
    assert deployment.config_hash == _stable_strategy_config_hash(reference_config)

    manifest = _manifest(tmp_path)
    manifest.strategy_plugin_id = contract.plugin_id
    manifest.trading_repo_commit_sha = contract.live_repo_commit_sha
    manifest.backtest_repo_commit_sha = contract.backtest_adapter_commit_sha

    report = build_crypto_trend_decision_parity_report(
        manifest,
        candidate_id="strategy-plugin-contract",
        fixture_paths=contract.parity_fixture_set,
        live_repo_path=contract.live_repo_path,
        live_repo_commit_sha=contract.live_repo_commit_sha,
        backtest_adapter_commit_sha=contract.backtest_adapter_commit_sha,
    )

    assert report.status == DecisionParityStatus.PASS
    assert report.strategy_plugin_id == contract.plugin_id
    assert {check.dimension for check in report.checks} == DECISION_PARITY_DIMENSIONS
    assert len({path for check in report.checks for path in check.evidence_paths}) >= 4


def test_formal_crypto_trend_decision_parity_validation_emits_artifact(tmp_path: Path) -> None:
    if not PERSISTED_CRYPTO_CONTRACT.exists():
        pytest.skip("persisted crypto trend strategy contract is not available")
    if not (CRYPTO_TRADER_REPO / "config" / "strategies" / "trend.json").exists():
        pytest.skip("crypto_trader reference repo is local-only and not available in CI")

    result = run_crypto_trend_decision_parity_validation(
        contract_path=PERSISTED_CRYPTO_CONTRACT,
        deployment_metadata_path=PERSISTED_CRYPTO_DEPLOYMENT,
        artifact_root=tmp_path / "decision_parity",
    )

    report_path = Path(result["decision_parity_report_path"])
    summary_path = Path(result["summary_path"])
    report = DecisionParityReport.model_validate(
        json.loads(report_path.read_text(encoding="utf-8"))
    )

    assert result["ok"] is True
    assert report_path.exists()
    assert summary_path.exists()
    assert result["contract_maturity"] == "shadow_validated"
    assert result["eligible_for_optimizer"] is True
    assert result["eligible_for_approval"] is False
    assert result["decision_parity_status"] == "pass"
    assert report.status == DecisionParityStatus.PASS
    assert {check.dimension for check in report.checks} == DECISION_PARITY_DIMENSIONS
    assert all(check["passed"] for check in result["checks"])


def test_formal_week1_decision_parity_validation_emits_artifacts(tmp_path: Path) -> None:
    if not PERSISTED_K_STOCK_CONTRACT.exists() or not PERSISTED_TRADING_STOCK_CONTRACT.exists():
        pytest.skip("persisted week-1 strategy contracts are not available")
    if not K_STOCK_TRADER_REPO.exists() or not TRADING_REPO.exists():
        pytest.skip("week-1 reference repos are local-only and not available in CI")

    result = run_week1_decision_parity_validations(
        contract_paths=[PERSISTED_K_STOCK_CONTRACT, PERSISTED_TRADING_STOCK_CONTRACT],
        artifact_root=tmp_path / "week1",
    )

    assert result["ok"] is True
    assert [item["strategy_plugin_id"] for item in result["results"]] == [
        "k-stock-olr-kalcb",
        "trading-stock-family",
    ]
    for item in result["results"]:
        report_path = Path(item["decision_parity_report_path"])
        report = DecisionParityReport.model_validate(
            json.loads(report_path.read_text(encoding="utf-8"))
        )
        assert report_path.exists()
        assert Path(item["summary_path"]).exists()
        assert item["contract_maturity"] == "shadow_validated"
        assert item["eligible_for_optimizer"] is True
        assert item["eligible_for_approval"] is False
        assert item["decision_parity_status"] == "pass"
        assert item["fixture_count"] >= 2
        assert report.status == DecisionParityStatus.PASS
        assert {check.dimension for check in report.checks} == DECISION_PARITY_DIMENSIONS
        assert all(check["passed"] for check in item["checks"])


def test_crypto_trend_parity_rejects_wrong_live_repo_sha(tmp_path: Path) -> None:
    if not CRYPTO_TRADER_REPO.exists():
        pytest.skip("crypto_trader reference repo is not available")
    fixture_path = tmp_path / "crypto_trend_entry_fixture.json"
    fixture_path.write_text(json.dumps(_crypto_trend_fixture()), encoding="utf-8")

    with pytest.raises(ValueError, match="live repo HEAD"):
        build_crypto_trend_decision_parity_report(
            _manifest(tmp_path),
            candidate_id="strategy-plugin-contract",
            fixture_paths=[fixture_path],
            live_repo_path=CRYPTO_TRADER_REPO,
            live_repo_commit_sha="wrong-live-sha",
            backtest_adapter_commit_sha="backtest-sha",
        )


def test_crypto_trend_fixture_can_cover_blocked_trade_without_position(tmp_path: Path) -> None:
    if not CRYPTO_TRADER_REPO.exists():
        pytest.skip("crypto_trader reference repo is not available")
    fixture = _crypto_trend_fixture()
    fixture["fixture_id"] = "trend_blocked_no_trigger"
    fixture["trigger"] = None
    fixture["position_open"] = False
    fixture["setup_passed"] = False
    fixture["config"] = {"entry": {"mode": "break"}}
    fixture["sizing"]["was_reduced"] = True
    fixture["sizing"]["reduction_reason"] = "risk_cap"

    events = decision_trace_from_fixture(fixture, live_repo_path=CRYPTO_TRADER_REPO)
    by_dimension = {event.dimension: event for event in events}

    assert set(by_dimension) == DECISION_PARITY_DIMENSIONS
    assert by_dimension["entries"].payload["action"] == "no_order"
    assert by_dimension["exits"].payload == {"orders": []}
    assert by_dimension["order_intent"].payload["action"] == "no_order"
    assert by_dimension["risk_caps"].payload["was_reduced"] is True


def test_mature_strategy_plugin_requires_clean_live_checkout(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    repo = tmp_path / "live"
    repo.mkdir()
    _git(["init"], repo)
    _git(["config", "user.email", "fixture@example.com"], repo)
    _git(["config", "user.name", "Fixture"], repo)
    (repo / "strategy.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(["add", "strategy.py"], repo)
    _git(["commit", "-m", "fixture"], repo)
    head = _git(["rev-parse", "HEAD"], repo).strip()
    (repo / "strategy.py").write_text("VALUE = 2\n", encoding="utf-8")
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps({
        "plugin_id": "strat1-plugin",
        "live_repo_path": str(repo),
        "live_repo_commit_sha": head,
        "backtest_adapter_path": "adapters/strat1.py",
        "backtest_adapter_commit_sha": "backtest-sha",
        "config_schema_version": "config_v1",
        "decision_api_version": "decision_api_v1",
        "required_telemetry_schemas": ["trade_event_v1"],
        "supported_symbols": ["AAPL"],
        "supported_timeframes": ["1m"],
        "parity_fixture_set": [str(tmp_path / "fixture.json")],
        "maturity": "approval_ready",
    }), encoding="utf-8")
    manifest = _manifest(tmp_path)
    manifest.strategy_plugin_contract_path = str(contract_path)

    errors = strategy_plugin_errors(manifest, bundle=None)

    assert "live repo checkout has uncommitted changes" in errors


def _crypto_trend_fixture() -> dict:
    return {
        "fixture_id": "trend_entry_market",
        "symbol": "BTC",
        "timeframe": "1h",
        "timestamp": "2026-03-15T10:00:00+00:00",
        "direction": "LONG",
        "entry_price": 50050.0,
        "atr": 250.0,
        "bar": {
            "open": 50000.0,
            "high": 50100.0,
            "low": 49900.0,
            "close": 50050.0,
            "volume": 100.0,
        },
        "setup": {
            "grade": "B",
            "direction": "LONG",
            "impulse_start": 49000.0,
            "impulse_end": 50500.0,
            "impulse_atr_move": 2.0,
            "pullback_depth": 0.3,
            "confluences": ["h1_ema_zone", "rsi_pullback"],
            "zone_price": 50050.0,
            "room_r": 2.5,
            "stop_level": 49500.0,
            "setup_score": 2.0,
        },
        "trigger": {
            "pattern": "engulfing",
            "trigger_price": 50050.0,
            "bar_offset": 0,
            "valid": True,
        },
        "sizing": {
            "qty": 0.1,
            "leverage": 5.0,
            "liquidation_price": 40000.0,
            "risk_pct_actual": 0.005,
            "notional": 5005.0,
            "was_reduced": False,
            "reduction_reason": None,
        },
        "exit_state": {
            "entry_price": 50050.0,
            "stop_distance": 550.0,
            "qty": 0.1,
            "bars_since_entry": 20,
            "mfe_r": 0.05,
            "peak_r": 0.05,
            "ema_fast": 50000.0,
        },
    }


def _git(args: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout


def _sha256_file(path: Path) -> str:
    return sha256_file(path)


def _stable_strategy_config_hash(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    strategy = payload.get("strategy") if isinstance(payload, dict) else payload
    raw = json.dumps(strategy or payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

