"""Slice authority adapter views."""

from __future__ import annotations

from dataclasses import dataclass, field

from trading_assistant_data.manifests import MarketDataManifest


@dataclass(frozen=True)
class SliceAuthorityStatus:
    status: str
    source_contract_id: str = ""
    blocking_reasons: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.blocking_reasons and self.status not in {"", "blocked"}


def authority_status_for_manifest(manifest: MarketDataManifest) -> SliceAuthorityStatus:
    lineage = dict(manifest.lineage or {})
    return SliceAuthorityStatus(
        status=str(lineage.get("authority_status") or manifest.usability.value),
        source_contract_id=str(lineage.get("authority_contract_id") or ""),
        blocking_reasons=list(manifest.blocking_reasons or []),
    )


def is_authoritative_slice_manifest(manifest: MarketDataManifest) -> bool:
    return (
        bool(manifest.usable_for_authoritative_validation)
        and authority_status_for_manifest(manifest).usable
    )
