from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_CONTRACT_IMPORTS = ("trading_assistant_backtest",)
FORBIDDEN_LIVE_DEPS = (
    "trading-assistant",
    "trading-assistant-backtest",
    "trading_assistant",
    "trading_assistant_backtest",
)
LIVE_BOT_PYPROJECTS = (
    ROOT / "bots/ibkr_trading/pyproject.toml",
    ROOT / "bots/k_stock_trader/pyproject.toml",
    ROOT / "bots/crypto_trader/pyproject.toml",
)


def main() -> int:
    failures: list[str] = []
    failures.extend(_contract_failures())
    failures.extend(_assistant_backtest_contract_failures())
    failures.extend(_live_dependency_failures())
    if failures:
        print("Dependency boundary check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Dependency boundary check passed.")
    return 0


def _contract_failures() -> list[str]:
    failures: list[str] = []
    pyproject = ROOT / "packages/trading_contracts/pyproject.toml"
    for forbidden in FORBIDDEN_CONTRACT_IMPORTS:
        if forbidden.replace("_", "-") in pyproject.read_text(encoding="utf-8"):
            failures.append(f"{pyproject.relative_to(ROOT)} declares {forbidden}")
    for path in (ROOT / "packages/trading_contracts/src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_CONTRACT_IMPORTS:
            if f"import {forbidden}" in text or f"from {forbidden}" in text:
                failures.append(f"{path.relative_to(ROOT)} imports {forbidden}")
    return failures


def _assistant_backtest_contract_failures() -> list[str]:
    failures: list[str] = []
    pyproject = ROOT / "packages/trading_assistant_backtest/pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    if "trading-contracts" not in text:
        failures.append(f"{pyproject.relative_to(ROOT)} does not depend on trading-contracts")
    models = ROOT / "packages/trading_assistant_backtest/src/trading_assistant_backtest/contract_models.py"
    source = models.read_text(encoding="utf-8")
    for class_name in ("MonthlyRunManifest", "StrategyPluginContract", "DataBundleManifest"):
        if f"class {class_name}" in source:
            failures.append(f"{models.relative_to(ROOT)} defines duplicate {class_name}")
    return failures


def _live_dependency_failures() -> list[str]:
    failures: list[str] = []
    for path in LIVE_BOT_PYPROJECTS:
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_LIVE_DEPS:
            if f'"{forbidden}' in text or f"'{forbidden}" in text:
                failures.append(f"{path.relative_to(ROOT)} declares {forbidden}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
