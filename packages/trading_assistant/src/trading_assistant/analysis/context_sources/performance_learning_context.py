"""Performance-learning prompt evidence source."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PerformanceLearningContextSource:
    def __init__(self, memory_dir: Path) -> None:
        self._path = Path(memory_dir) / "findings" / "performance_learning_ledger.jsonl"

    def load(
        self,
        *,
        bot_id: str = "",
        strategy_id: str = "",
        portfolio_id: str = "",
        limit: int = 10,
    ) -> list[dict]:
        try:
            from trading_assistant.skills.performance_learning_ledger import (
                PerformanceLearningLedgerStore,
            )

            return PerformanceLearningLedgerStore(self._path).recent_summaries(
                bot_id=bot_id,
                strategy_id=strategy_id,
                portfolio_id=portfolio_id,
                limit=limit,
            )
        except Exception:
            logger.debug("Performance-learning context loading failed; skipping")
            return []
