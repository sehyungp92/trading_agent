"""Drift checker — detect mismatches between data/strategy_profiles.yaml
and the live strategy directories under trading/ibkr_trader/strategies/.

Surfaces three categories of drift:
  - registered_but_missing: in registry but the reference dir is empty/absent
  - present_but_unregistered: live in references but no registry entry
  - empty_shell: dir exists but has no strategy code (deleted strategies leave
    empty parent dirs, e.g. breakout/, keltner/, helix_v40/)

Records the finding to memory/findings/strategy_registry_drift.jsonl.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from trading_assistant.paths import monorepo_root
from trading_assistant.schemas.strategy_profile import StrategyRegistry

logger = logging.getLogger(__name__)


_STRATEGY_ID_RE = re.compile(
    r"""STRATEGY_ID\s*[:=]\s*(?:Final\[[^\]]+\]\s*=\s*)?["']([A-Za-z0-9_.]+)["']""",
    re.MULTILINE,
)
_INFRA_NAMES = frozenset({
    "_shared", "instrumentation", "coordinator.py", "__init__.py",
    "__pycache__", "core", "tests", "models.py", "events.py",
    "actions.py", "plugin_runtime.py", "serialization.py", "contracts.py",
    "artifact_generator.py", "live_universe.py", "readiness.py",
})


@dataclass
class StrategyDirInfo:
    """One live strategy dir under a bot strategies/<family>/ root."""
    family: str
    dir_name: str
    is_empty: bool
    has_engine: bool
    extracted_strategy_id: str = ""


@dataclass
class RegistryDrift:
    """Result of comparing the registry to the reference dirs."""
    checked_at: str
    reference_root: str
    registered_but_missing: list[str] = field(default_factory=list)
    present_but_unregistered: list[str] = field(default_factory=list)
    empty_shells: list[str] = field(default_factory=list)
    live_dirs: list[StrategyDirInfo] = field(default_factory=list)

    @property
    def has_drift(self) -> bool:
        """Actionable drift: registry/code mismatch. Empty shells are informational only."""
        return bool(self.registered_but_missing or self.present_but_unregistered)

    def summary(self) -> str:
        if not self.has_drift:
            return "No drift detected."
        parts = []
        if self.registered_but_missing:
            parts.append(
                f"In registry but missing from references: {', '.join(self.registered_but_missing)}"
            )
        if self.present_but_unregistered:
            parts.append(
                f"In references but unregistered: {', '.join(self.present_but_unregistered)}"
            )
        if self.empty_shells:
            parts.append(
                f"Empty/deleted dirs (likely retired): {', '.join(self.empty_shells)}"
            )
        return " | ".join(parts)


def _extract_strategy_id(config_path: Path) -> str:
    """Pull STRATEGY_ID from a strategy's config.py via regex. Returns '' if absent."""
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    match = _STRATEGY_ID_RE.search(text)
    return match.group(1) if match else ""


def _scan_family(family_dir: Path, family: str) -> list[StrategyDirInfo]:
    """Scan one family dir (swing/momentum/stock) for strategy subdirs."""
    if not family_dir.is_dir():
        return []
    infos: list[StrategyDirInfo] = []
    for entry in sorted(family_dir.iterdir()):
        if entry.name in _INFRA_NAMES or not entry.is_dir():
            continue
        # Filter substantive content: anything other than empty dir or only "tests/"
        children = [c for c in entry.iterdir() if c.name != "tests"]
        is_empty = not children
        has_engine = (entry / "engine.py").exists() or (entry / "plugin.py").exists()
        sid = ""
        cfg = entry / "config.py"
        if cfg.exists():
            sid = _extract_strategy_id(cfg)
        infos.append(StrategyDirInfo(
            family=family,
            dir_name=entry.name,
            is_empty=is_empty,
            has_engine=has_engine,
            extracted_strategy_id=sid,
        ))
    return infos


def check_drift(
    registry: StrategyRegistry,
    reference_roots: list[Path] | None = None,
    families: tuple[str, ...] = ("swing", "momentum", "stock"),
) -> RegistryDrift:
    """Compare a loaded StrategyRegistry against live reference dirs.

    `reference_roots` lets callers point at multiple bot repos
    or archived evidence roots. Defaults to the extracted IBKR bot.
    """
    if reference_roots is None:
        reference_roots = [monorepo_root() / "trading" / "ibkr_trader"]

    drift = RegistryDrift(
        checked_at=datetime.now(timezone.utc).isoformat(),
        reference_root=";".join(str(r) for r in reference_roots),
    )

    seen_ids: set[str] = set()
    empty_seen: set[str] = set()

    for root in reference_roots:
        strategies_root = root / "strategies"
        for family in families:
            for info in _scan_family(strategies_root / family, family):
                drift.live_dirs.append(info)
                if info.is_empty or not info.has_engine:
                    empty_seen.add(info.dir_name)
                    drift.empty_shells.append(f"{family}/{info.dir_name}")
                    continue
                if info.extracted_strategy_id:
                    seen_ids.add(info.extracted_strategy_id)

    registered_ids = set(registry.strategies)

    # Registry → references: registered ids whose code we can't find as a live dir.
    # Restrict to bot_ids that the trading monorepo actually covers; other bots
    # (k_stock, crypto) live in separate repos and are not in scope here.
    monorepo_bots = {"swing_multi_01", "momentum_nq_01", "stock_trader"}
    for sid, profile in registry.strategies.items():
        if profile.bot_id not in monorepo_bots:
            continue
        if sid not in seen_ids:
            drift.registered_but_missing.append(sid)

    # References → registry: live dirs whose STRATEGY_ID isn't in the registry.
    for sid in sorted(seen_ids):
        if sid not in registered_ids:
            drift.present_but_unregistered.append(sid)

    return drift


def record_drift(drift: RegistryDrift, findings_dir: Path) -> Path:
    """Append the drift finding to memory/findings/strategy_registry_drift.jsonl."""
    findings_dir.mkdir(parents=True, exist_ok=True)
    path = findings_dir / "strategy_registry_drift.jsonl"
    payload = {
        "checked_at": drift.checked_at,
        "reference_root": drift.reference_root,
        "has_drift": drift.has_drift,
        "registered_but_missing": drift.registered_but_missing,
        "present_but_unregistered": drift.present_but_unregistered,
        "empty_shells": drift.empty_shells,
        "summary": drift.summary(),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    return path


def _cli() -> int:
    from trading_assistant.orchestrator.strategy_registry_loader import load_strategy_registry

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    registry = load_strategy_registry()
    drift = check_drift(registry)
    print(drift.summary())
    if drift.live_dirs:
        print("\nLive dirs scanned:")
        for info in drift.live_dirs:
            tag = "EMPTY" if info.is_empty else ("OK" if info.has_engine else "NO_ENGINE")
            print(f"  [{tag}] {info.family}/{info.dir_name} → {info.extracted_strategy_id or '?'}")
    return 0 if not drift.has_drift else 1


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
