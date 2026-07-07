from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

@dataclass
class ConversationChain:
    """Tracks a chain of causally-linked events."""
    chain_id: str
    depth: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_ids: list[str] = field(default_factory=list)

class ConversationTracker:
    """Tracks event processing chains and prevents infinite loops."""

    def __init__(self, max_depth: int = 10, timeout_minutes: int = 30) -> None:
        self._max_depth = max_depth
        self._timeout = timedelta(minutes=timeout_minutes)
        self._chains: dict[str, ConversationChain] = {}

    @property
    def max_depth(self) -> int:
        return self._max_depth

    def begin_chain(self, event_id: str = "") -> ConversationChain:
        """Start a new conversation chain."""
        self.cleanup_expired()
        chain_id = f"chain-{uuid.uuid4().hex[:8]}"
        chain = ConversationChain(chain_id=chain_id)
        if event_id:
            chain.event_ids.append(event_id)
        self._chains[chain_id] = chain
        return chain

    def extend_chain(self, chain_id: str, event_id: str) -> bool:
        """Extend an existing chain. Returns False if loop detected (depth exceeded or timed out)."""
        chain = self._chains.get(chain_id)
        if chain is None:
            logger.warning("Chain %s not found", chain_id)
            return False

        # Check timeout
        elapsed = datetime.now(timezone.utc) - chain.started_at
        if elapsed > self._timeout:
            logger.warning("Chain %s timed out after %s", chain_id, elapsed)
            return False

        # Check depth
        chain.depth += 1
        if chain.depth > self._max_depth:
            logger.warning(
                "Chain %s exceeded max depth %d (loop detected?)",
                chain_id, self._max_depth,
            )
            return False

        chain.event_ids.append(event_id)
        return True

    def get_chain(self, chain_id: str) -> ConversationChain | None:
        return self._chains.get(chain_id)

    def get_active_chains(self) -> list[ConversationChain]:
        """Get chains that haven't timed out."""
        now = datetime.now(timezone.utc)
        return [
            c for c in self._chains.values()
            if (now - c.started_at) <= self._timeout
        ]

    def cleanup_expired(self) -> int:
        """Remove expired chains. Returns count removed."""
        now = datetime.now(timezone.utc)
        expired = [
            cid for cid, chain in self._chains.items()
            if (now - chain.started_at) > self._timeout
        ]
        for cid in expired:
            del self._chains[cid]
        return len(expired)
