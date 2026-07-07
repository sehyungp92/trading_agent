"""Tests for conversation chain tracking with loop protection (M5)."""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from trading_assistant.orchestrator.conversation_tracker import ConversationTracker

class TestConversationTracker:
    def test_begin_chain_creates_chain(self):
        tracker = ConversationTracker()
        chain = tracker.begin_chain("evt-1")
        assert chain.chain_id.startswith("chain-")
        assert chain.depth == 0
        assert "evt-1" in chain.event_ids

    def test_extend_chain_increments_depth(self):
        tracker = ConversationTracker()
        chain = tracker.begin_chain("evt-1")
        ok = tracker.extend_chain(chain.chain_id, "evt-2")
        assert ok is True
        assert chain.depth == 1
        assert "evt-2" in chain.event_ids

    def test_extend_chain_rejects_at_max_depth(self):
        tracker = ConversationTracker(max_depth=3)
        chain = tracker.begin_chain()
        for i in range(3):
            assert tracker.extend_chain(chain.chain_id, f"evt-{i}") is True
        # 4th extension should fail (depth would be 4 > max_depth 3)
        assert tracker.extend_chain(chain.chain_id, "evt-overflow") is False

    def test_extend_chain_rejects_after_timeout(self):
        tracker = ConversationTracker(timeout_minutes=30)
        chain = tracker.begin_chain()
        # Manually set started_at to 31 minutes ago
        chain.started_at = datetime.now(timezone.utc) - timedelta(minutes=31)
        assert tracker.extend_chain(chain.chain_id, "evt-late") is False

    def test_extend_nonexistent_chain_returns_false(self):
        tracker = ConversationTracker()
        assert tracker.extend_chain("nonexistent", "evt-1") is False

    def test_get_chain(self):
        tracker = ConversationTracker()
        chain = tracker.begin_chain("evt-1")
        found = tracker.get_chain(chain.chain_id)
        assert found is chain

    def test_get_active_chains(self):
        tracker = ConversationTracker(timeout_minutes=30)
        c1 = tracker.begin_chain()
        c2 = tracker.begin_chain()
        # Expire c1
        c1.started_at = datetime.now(timezone.utc) - timedelta(minutes=31)
        active = tracker.get_active_chains()
        assert len(active) == 1
        assert active[0].chain_id == c2.chain_id

    def test_cleanup_expired(self):
        tracker = ConversationTracker(timeout_minutes=30)
        c1 = tracker.begin_chain()
        c2 = tracker.begin_chain()
        c1.started_at = datetime.now(timezone.utc) - timedelta(minutes=31)
        removed = tracker.cleanup_expired()
        assert removed == 1
        assert tracker.get_chain(c1.chain_id) is None
        assert tracker.get_chain(c2.chain_id) is not None

    def test_begin_chain_triggers_cleanup(self):
        """begin_chain() should lazily clean up expired chains."""
        tracker = ConversationTracker(timeout_minutes=30)
        c1 = tracker.begin_chain()
        c1.started_at = datetime.now(timezone.utc) - timedelta(minutes=31)
        # begin_chain should trigger cleanup of c1
        c2 = tracker.begin_chain()
        assert tracker.get_chain(c1.chain_id) is None
        assert tracker.get_chain(c2.chain_id) is not None
