from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading_contracts.cli import main
from trading_contracts.legacy import (
    LegacyRoundsManifest,
    StrategyPromotionManifest,
    load_rounds_manifest,
    validate_plugin_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_loads_legacy_ibkr_rounds_manifest() -> None:
    manifest = load_rounds_manifest(
        REPO_ROOT / "backtests/baselines/ibkr/momentum/nq_regime/rounds_manifest.json"
    )

    assert isinstance(manifest, LegacyRoundsManifest)
    assert manifest.rounds


def test_validates_strategy_plugin_contract() -> None:
    contract = validate_plugin_contract(
        REPO_ROOT / "contracts/strategy_plugins/k_stock_olr_kalcb/strategy_plugin_contract.json"
    )

    assert contract.plugin_id == "k-stock-olr-kalcb"
    assert contract.eligible_for_optimizer


def test_validates_promotion_manifest() -> None:
    payload = json.loads(
        (REPO_ROOT / "contracts/promotions/draft/ibkr/ALCB_v1.json").read_text(encoding="utf-8")
    )
    manifest = StrategyPromotionManifest.model_validate(payload)

    assert manifest.strategy_id == "ALCB_v1"
    assert manifest.promotion_state == "draft"


def test_rejects_legacy_reference_paths_in_promotion_manifest() -> None:
    payload = {
        "schema_version": "strategy_promotion_manifest.v1.draft",
        "bot_id": "ibkr_trading",
        "strategy_id": "ALCB_v1",
        "promotion_state": "draft",
        "source_live_config": {
            "path": "_ref" "erences/trading/config/strategies.yaml",
            "sha256": "0" * 64,
        },
    }

    with pytest.raises(ValueError, match="legacy reference paths"):
        StrategyPromotionManifest.model_validate(payload)


def test_cli_validation_and_schema_generation(tmp_path: Path, capsys) -> None:
    assert main([
        "validate-rounds-manifest",
        str(REPO_ROOT / "backtests/baselines/crypto/momentum/rounds_manifest.json"),
    ]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["valid"] is True

    assert main(["generate-schemas", "--output", str(tmp_path)]) == 0
    assert (tmp_path / "rounds_manifest.schema.json").exists()
