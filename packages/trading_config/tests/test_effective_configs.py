from __future__ import annotations

import json
import shutil
from pathlib import Path

from trading_config.generator import BOT_SPECS, generate_effective_configs
from trading_config.verifier import verify_effective_configs


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_generate_and_verify_effective_configs(tmp_path: Path) -> None:
    for relative in (
        "bots/ibkr_trading/config/strategies.yaml",
        "bots/crypto_trader/config/live_config.example.json",
        "bots/crypto_trader/config/strategies",
        "bots/k_stock_trader/config/kalcb.yaml",
        "bots/k_stock_trader/config/optimization",
        "bots/k_stock_trader/config/olr_kalcb/olr_deployment_universe_103.yaml",
        "bots/k_stock_trader/strategy_kalcb/config.py",
        "bots/k_stock_trader/backtests/strategies/kalcb/phase_candidates.py",
        "contracts/promotions",
    ):
        source = REPO_ROOT / relative
        target = tmp_path / relative
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    result = generate_effective_configs(tmp_path)
    assert len(result["generated"]) == len(BOT_SPECS)
    verification = verify_effective_configs(tmp_path)
    assert verification["valid"] is True

    ibkr_output = tmp_path / "deployments/ibkr/generated/strategies.effective.json"
    first_snapshot = json.loads(ibkr_output.read_text(encoding="utf-8"))
    assert first_snapshot["materialized_config"]["strategies"]
    assert first_snapshot["materialized_config_hash"]
    draft_promotion = tmp_path / "contracts/promotions/draft/ibkr/stale.json"
    draft_promotion.write_text('{"strategy_id": "stale"}\n', encoding="utf-8")

    generate_effective_configs(tmp_path)

    second_snapshot = json.loads(ibkr_output.read_text(encoding="utf-8"))
    assert draft_promotion.exists()
    assert second_snapshot["effective_config_hash"] == first_snapshot["effective_config_hash"]
    assert second_snapshot["generated_at"] == first_snapshot["generated_at"]
