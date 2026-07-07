"""Drift tests between data/strategy_profiles.yaml and the trading repo.

Catches:
- ``unit_risk_dollars`` drift between strategy_profiles.yaml and the live
  trading config at bots/ibkr_trading/config/strategies.yaml.
- ``bot_id`` drift outside the canonical bot identifiers emitted by the
  trading runtime (k_stock_trader, stock_trader, swing_multi_01,
  momentum_nq_01, crypto_trader).
- Strategies whose ``bot_id`` belongs to a bot in the trading monorepo
  (swing_multi_01 / momentum_nq_01 / stock_trader) but which are missing
  from the trading repo's ``strategies.yaml`` ??almost always means the
  strategy was retired upstream and the assistant profile is stale.

Strategies for ``k_stock_trader`` and ``crypto_trader`` are exempt from
the trading-repo presence check because those bots live in their own
repos, not in the ``trading`` monorepo.
"""
from __future__ import annotations


import pytest
import yaml

from trading_assistant.orchestrator.strategy_registry_loader import load_strategy_registry
from trading_assistant.paths import monorepo_root, package_root

PACKAGE_ROOT = package_root()
MONOREPO_ROOT = monorepo_root()
TRADING_STRATEGIES_YAML = MONOREPO_ROOT / "bots" / "ibkr_trading" / "config" / "strategies.yaml"

CANONICAL_BOT_IDS = {
    "k_stock_trader",
    "stock_trader",
    "swing_multi_01",
    "momentum_nq_01",
    "crypto_trader",
}

# Bots whose strategies live in `bots/ibkr_trading/config/strategies.yaml`.
# Anything else (k_stock_trader, crypto_trader) is exempt from the
# trading-repo presence check.
MONOREPO_BOT_IDS = {"swing_multi_01", "momentum_nq_01", "stock_trader"}


@pytest.fixture(scope="module")
def assistant_registry():
    return load_strategy_registry()


@pytest.fixture(scope="module")
def trading_strategies() -> dict[str, dict]:
    if not TRADING_STRATEGIES_YAML.exists():
        pytest.skip(
            f"Trading reference config not present at {TRADING_STRATEGIES_YAML}; "
            "drift comparison cannot run in this environment."
        )
    raw = yaml.safe_load(TRADING_STRATEGIES_YAML.read_text(encoding="utf-8"))
    return raw.get("strategies") or {}


def test_unit_risk_dollars_match_trading(assistant_registry, trading_strategies):
    """Every strategy present in both files must declare the same unit_risk_dollars."""
    mismatches: list[str] = []
    overlap = 0
    for sid, profile in assistant_registry.strategies.items():
        if sid not in trading_strategies:
            continue
        overlap += 1
        live = trading_strategies[sid].get("risk", {}).get("unit_risk_dollars")
        ours = profile.risk.unit_risk_dollars
        if live is None:
            mismatches.append(
                f"{sid}: trading strategies.yaml is missing risk.unit_risk_dollars"
            )
            continue
        if float(live) != float(ours):
            mismatches.append(
                f"{sid}: strategy_profiles.yaml={ours} != strategies.yaml={live}"
            )

    assert overlap > 0, "no overlapping strategies found ??file paths likely broken"
    assert not mismatches, "URD drift detected:\n  " + "\n  ".join(mismatches)


def test_monorepo_strategies_exist_in_trading(assistant_registry, trading_strategies):
    """Strategies attributed to a trading-monorepo bot must exist upstream.

    If a profile claims a strategy belongs to swing_multi_01 / momentum_nq_01 /
    stock_trader but the trading repo doesn't define it, the strategy was
    almost certainly retired upstream and the profile is stale.
    """
    missing: list[str] = []
    for sid, profile in assistant_registry.strategies.items():
        if profile.bot_id not in MONOREPO_BOT_IDS:
            continue
        if sid not in trading_strategies:
            missing.append(f"{sid} (bot_id={profile.bot_id})")

    assert not missing, (
        "strategies in strategy_profiles.yaml are attributed to a trading-"
        "monorepo bot but absent from bots/ibkr_trading/config/strategies.yaml "
        "??retire from profiles or restore upstream:\n  " + "\n  ".join(sorted(missing))
    )


def test_bot_ids_are_canonical(assistant_registry):
    """Every strategy's bot_id must match what the trading runtime actually emits."""
    bad: list[str] = []
    for sid, profile in assistant_registry.strategies.items():
        if profile.bot_id not in CANONICAL_BOT_IDS:
            bad.append(f"{sid}: bot_id={profile.bot_id!r}")

    assert not bad, (
        "bot_id values must be one of "
        f"{sorted(CANONICAL_BOT_IDS)}.\nNon-canonical:\n  " + "\n  ".join(bad)
    )


def test_bot_config_strategies_match_trading(trading_strategies):
    """Strategies listed in data/bot_configs/<bot>.yaml must exist upstream.

    Catches the same retired-strategy drift as
    ``test_monorepo_strategies_exist_in_trading`` but on the bot-config side
    (parameter registry), where stale strategy entries cause downstream
    PR-generation pipelines to target dead code paths.
    """
    bot_configs_dir = PACKAGE_ROOT / "data" / "bot_configs"

    stale: list[str] = []
    for cfg_path in sorted(bot_configs_dir.glob("*.yaml")):
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        bot_id = cfg.get("bot_id", cfg_path.stem)
        if bot_id not in MONOREPO_BOT_IDS:
            continue
        for sid in cfg.get("strategies") or []:
            if sid not in trading_strategies:
                stale.append(f"{cfg_path.name}: lists {sid!r} (no longer in trading repo)")

    assert not stale, (
        "bot_configs reference strategies that no longer exist upstream ??"
        "remove from the strategies list and prune any matching parameter "
        "blocks:\n  " + "\n  ".join(stale)
    )


def test_bot_config_param_paths_resolve():
    """Every parameter ``file_path`` in a monorepo bot config must point at a
    file that actually exists under ``bots/ibkr_trading/``.

    Catches the class of bug where a strategy directory was renamed or
    refactored upstream (e.g. ``nq_regime/`` ??``nqdtc/``) but the param
    blocks in the bot config still point at the old path. Without this guard,
    PR-generation pipelines silently target nonexistent files.
    """
    bot_configs_dir = PACKAGE_ROOT / "data" / "bot_configs"
    trading_root = MONOREPO_ROOT / "bots" / "ibkr_trading"
    if not trading_root.exists():
        pytest.skip(f"Trading reference not present at {trading_root}")

    broken: list[str] = []
    for cfg_path in sorted(bot_configs_dir.glob("*.yaml")):
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        bot_id = cfg.get("bot_id", cfg_path.stem)
        if bot_id not in MONOREPO_BOT_IDS:
            continue
        for param in cfg.get("parameters") or []:
            target = trading_root / param["file_path"]
            if not target.exists():
                broken.append(
                    f"{cfg_path.name}: {param['strategy_id']}.{param['param_name']} "
                    f"-> {param['file_path']} (not found)"
                )

    assert not broken, (
        "bot_config parameter file_paths do not resolve in the trading "
        "monorepo ??fix the path or remove the param block:\n  "
        + "\n  ".join(broken)
    )
