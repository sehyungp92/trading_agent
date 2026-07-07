# skills/leakage_detector.py
"""Leakage detector — audits features and labels for temporal correctness.

Two core checks:
1. No lookahead in features: every feature at time t uses only data at or before t.
2. No forward-fill labels: trade outcomes (TP/SL hit) computed from post-entry data only.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trading_assistant.schemas.validation_results import LeakageAuditEntry


@dataclass
class FeatureRecord:
    """Metadata for a computed feature — tracks its data dependency."""

    feature_name: str
    computed_at: str  # ISO timestamp
    latest_data_used: str  # ISO timestamp


@dataclass
class LabelRecord:
    """Metadata for a trade label — tracks when the label was computed."""

    trade_id: str
    entry_time: str  # ISO timestamp
    label_computed_from: str  # ISO timestamp of earliest data used to compute label


class LeakageDetector:
    """Audits feature and label timestamps for lookahead or forward-fill leakage."""

    def audit_features(self, features: list[FeatureRecord]) -> list[LeakageAuditEntry]:
        """Check that every feature value uses only data at or before its computation time."""
        entries: list[LeakageAuditEntry] = []
        for f in features:
            computed = _parse(f.computed_at)
            latest = _parse(f.latest_data_used)
            if latest > computed:
                entries.append(LeakageAuditEntry(
                    feature_name=f.feature_name,
                    computed_at=f.computed_at,
                    latest_data_used=f.latest_data_used,
                    passed=False,
                    violation=f"Used future data: latest_data_used ({f.latest_data_used}) > computed_at ({f.computed_at})",
                ))
            else:
                entries.append(LeakageAuditEntry(
                    feature_name=f.feature_name,
                    computed_at=f.computed_at,
                    latest_data_used=f.latest_data_used,
                    passed=True,
                ))
        return entries

    def audit_labels(self, labels: list[LabelRecord]) -> list[LeakageAuditEntry]:
        """Check that trade labels are computed from data after entry only."""
        entries: list[LeakageAuditEntry] = []
        for label in labels:
            entry = _parse(label.entry_time)
            computed_from = _parse(label.label_computed_from)
            if computed_from < entry:
                entries.append(LeakageAuditEntry(
                    feature_name=f"label:{label.trade_id}",
                    computed_at=label.entry_time,
                    latest_data_used=label.label_computed_from,
                    passed=False,
                    violation=f"Label computed from before entry: {label.label_computed_from} < {label.entry_time}",
                ))
            else:
                entries.append(LeakageAuditEntry(
                    feature_name=f"label:{label.trade_id}",
                    computed_at=label.entry_time,
                    latest_data_used=label.label_computed_from,
                    passed=True,
                ))
        return entries

    def full_audit(
        self, features: list[FeatureRecord], labels: list[LabelRecord]
    ) -> list[LeakageAuditEntry]:
        """Run all leakage checks, return combined audit log."""
        return self.audit_features(features) + self.audit_labels(labels)


def _parse(iso_str: str) -> datetime:
    return datetime.fromisoformat(iso_str)
