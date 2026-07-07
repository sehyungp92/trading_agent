# tests/test_leakage_detector.py
"""Tests for temporal leakage detection."""
from trading_assistant.skills.leakage_detector import LeakageDetector, FeatureRecord, LabelRecord


class TestFeatureRecord:
    def test_creates_record(self):
        r = FeatureRecord(
            feature_name="rsi_14",
            computed_at="2026-01-15T12:00:00",
            latest_data_used="2026-01-15T11:00:00",
        )
        assert r.feature_name == "rsi_14"


class TestLabelRecord:
    def test_creates_record(self):
        r = LabelRecord(
            trade_id="t1",
            entry_time="2026-01-15T12:00:00",
            label_computed_from="2026-01-15T13:00:00",
        )
        assert r.trade_id == "t1"


class TestNoLookaheadInFeatures:
    def test_passes_when_all_features_valid(self):
        detector = LeakageDetector()
        features = [
            FeatureRecord(feature_name="rsi", computed_at="2026-01-15T12:00:00", latest_data_used="2026-01-15T11:00:00"),
            FeatureRecord(feature_name="ema", computed_at="2026-01-15T12:00:00", latest_data_used="2026-01-15T12:00:00"),
        ]
        entries = detector.audit_features(features)
        assert all(e.passed for e in entries)
        assert len(entries) == 2

    def test_fails_when_feature_uses_future_data(self):
        detector = LeakageDetector()
        features = [
            FeatureRecord(feature_name="rsi", computed_at="2026-01-15T12:00:00", latest_data_used="2026-01-15T11:00:00"),
            FeatureRecord(feature_name="regime", computed_at="2026-01-15T12:00:00", latest_data_used="2026-01-16T00:00:00"),
        ]
        entries = detector.audit_features(features)
        assert entries[0].passed is True
        assert entries[1].passed is False
        assert "future data" in entries[1].violation.lower()

    def test_equal_timestamps_is_valid(self):
        detector = LeakageDetector()
        features = [
            FeatureRecord(feature_name="vol", computed_at="2026-01-15T12:00:00", latest_data_used="2026-01-15T12:00:00"),
        ]
        entries = detector.audit_features(features)
        assert entries[0].passed is True


class TestNoForwardFillLabels:
    def test_passes_when_labels_from_after_entry(self):
        detector = LeakageDetector()
        labels = [
            LabelRecord(trade_id="t1", entry_time="2026-01-15T12:00:00", label_computed_from="2026-01-15T13:00:00"),
            LabelRecord(trade_id="t2", entry_time="2026-01-15T14:00:00", label_computed_from="2026-01-15T14:00:00"),
        ]
        entries = detector.audit_labels(labels)
        assert all(e.passed for e in entries)

    def test_fails_when_label_from_before_entry(self):
        detector = LeakageDetector()
        labels = [
            LabelRecord(trade_id="t1", entry_time="2026-01-15T12:00:00", label_computed_from="2026-01-15T11:00:00"),
        ]
        entries = detector.audit_labels(labels)
        assert entries[0].passed is False
        assert "before entry" in entries[0].violation.lower()


class TestFullAudit:
    def test_combined_audit(self):
        detector = LeakageDetector()
        features = [
            FeatureRecord(feature_name="rsi", computed_at="2026-01-15T12:00:00", latest_data_used="2026-01-15T11:00:00"),
        ]
        labels = [
            LabelRecord(trade_id="t1", entry_time="2026-01-15T12:00:00", label_computed_from="2026-01-15T13:00:00"),
        ]
        entries = detector.full_audit(features, labels)
        assert len(entries) == 2
        assert all(e.passed for e in entries)

    def test_empty_inputs(self):
        detector = LeakageDetector()
        entries = detector.full_audit([], [])
        assert len(entries) == 0
