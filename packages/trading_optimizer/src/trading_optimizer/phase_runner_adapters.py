from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trading_optimizer.archived_smoke import dimension_payloads


@dataclass(frozen=True)
class PhaseRunnerSpec:
    bot: str
    module_path: str
    class_name: str
    source_path: str


LEGACY_PHASE_RUNNERS = {
    "ibkr": PhaseRunnerSpec(
        bot="ibkr",
        module_path="backtests.shared.auto.phase_runner",
        class_name="PhaseRunner",
        source_path="bots/ibkr_trading/backtests/shared/auto/phase_runner.py",
    ),
    "crypto": PhaseRunnerSpec(
        bot="crypto",
        module_path="crypto_trader.optimize.phase_runner",
        class_name="PhaseRunner",
        source_path="bots/crypto_trader/src/crypto_trader/optimize/phase_runner.py",
    ),
    "k_stock": PhaseRunnerSpec(
        bot="k_stock",
        module_path="backtests.auto.shared.phase_runner",
        class_name="PhaseRunner",
        source_path="bots/k_stock_trader/backtests/auto/shared/phase_runner.py",
    ),
}


class ArchivedPhaseRunnerAdapter:
    """Preflight adapter for frozen legacy outputs; not A13 runner equivalence."""

    def __init__(self, spec: PhaseRunnerSpec) -> None:
        self.spec = spec

    def canonical_outputs(self, records: list[dict[str, Any]], root: Path) -> dict[str, Any]:
        return dimension_payloads(records, root)


class LegacyPhaseRunnerAdapter:
    """Shared adapter over a bot's existing PhaseRunner implementation."""

    def __init__(self, spec: PhaseRunnerSpec) -> None:
        self.spec = spec

    def run(self, plugin: Any, output_dir: Path, **kwargs: Any) -> Any:
        module = importlib.import_module(self.spec.module_path)
        runner_cls = getattr(module, self.spec.class_name)
        runner = runner_cls(plugin, output_dir, **kwargs)
        if self.spec.bot == "crypto":
            return runner.run_all_phases(runner.load_state())
        return runner.run_all_phases()


def runner_specs_for_records(records: list[dict[str, Any]]) -> tuple[PhaseRunnerSpec, ...]:
    bots = sorted({str(record.get("bot") or "") for record in records})
    return tuple(LEGACY_PHASE_RUNNERS[bot] for bot in bots if bot in LEGACY_PHASE_RUNNERS)
