"""Hyperliquid asset metadata: tick sizes, lot sizes, margin tiers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import structlog

log = structlog.get_logger()


@dataclass
class MarginTier:
    """A single margin tier for a perpetual contract."""
    max_notional: float
    maintenance_margin: float


@dataclass
class AssetMeta:
    """Metadata for all listed Hyperliquid perpetuals."""
    asset_index: dict[str, int] = field(default_factory=dict)
    tick_sizes: dict[str, float] = field(default_factory=dict)
    lot_sizes: dict[str, float] = field(default_factory=dict)
    margin_tiers: dict[str, list[MarginTier]] = field(default_factory=dict)

    @classmethod
    def from_exchange(cls) -> AssetMeta:
        """Fetch metadata from the Hyperliquid exchange."""
        from hyperliquid.info import Info

        info = Info(skip_ws=True)
        raw = info.meta()
        meta = cls()

        for i, asset in enumerate(raw["universe"]):
            name: str = asset["name"]
            meta.asset_index[name] = i
            sz_decimals: int = asset["szDecimals"]
            meta.lot_sizes[name] = 10 ** (-sz_decimals)
            # Tick size: Hyperliquid uses 5 significant figures for price
            # but we derive from szDecimals as a sensible default
            # Actual tick sizes vary by price level; this is a conservative estimate
            meta.tick_sizes[name] = 10 ** (-sz_decimals)

        # Parse margin tiers if available
        if "marginTiers" in raw:
            for name, tiers in raw["marginTiers"].items():
                meta.margin_tiers[name] = [
                    MarginTier(
                        max_notional=t["maxNotional"],
                        maintenance_margin=t["maintenanceMargin"],
                    )
                    for t in tiers
                ]

        log.info("asset_meta.loaded", symbols=len(meta.asset_index))
        return meta

    @classmethod
    def from_cache(cls, path: Path) -> AssetMeta:
        """Load metadata from a JSON cache file."""
        data = json.loads(path.read_text())
        meta = cls()
        meta.asset_index = data["asset_index"]
        meta.tick_sizes = {k: float(v) for k, v in data["tick_sizes"].items()}
        meta.lot_sizes = {k: float(v) for k, v in data["lot_sizes"].items()}
        meta.margin_tiers = {
            name: [MarginTier(**t) for t in tiers]
            for name, tiers in data.get("margin_tiers", {}).items()
        }
        return meta

    def save_cache(self, path: Path) -> None:
        """Save metadata to a JSON cache file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "asset_index": self.asset_index,
            "tick_sizes": self.tick_sizes,
            "lot_sizes": self.lot_sizes,
            "margin_tiers": {
                name: [{"max_notional": t.max_notional, "maintenance_margin": t.maintenance_margin} for t in tiers]
                for name, tiers in self.margin_tiers.items()
            },
        }
        path.write_text(json.dumps(data, indent=2))
        log.info("asset_meta.cached", path=str(path))
