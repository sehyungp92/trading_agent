"""Run recall prompt evidence source."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class RunRecallSource:
    def __init__(self, memory_dir: Path, *, run_index: object | None = None) -> None:
        self._memory_dir = Path(memory_dir)
        self._run_index = run_index

    def load_similar_runs(
        self,
        *,
        agent_type: str = "",
        bot_id: str = "",
        query: str = "",
        formatter,
        limit: int = 5,
        days: int = 60,
    ) -> list[dict]:
        if self._run_index is None:
            return []
        try:
            runs = []
            if query and hasattr(self._run_index, "search"):
                min_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
                runs = self._run_index.search(
                    query=query,
                    limit=limit,
                    agent_type=agent_type,
                    bot_id=bot_id,
                    min_date=min_date,
                )
            if not runs:
                runs = self._run_index.get_recent_runs(
                    agent_type=agent_type,
                    bot_id=bot_id,
                    limit=limit,
                    days=days,
                )
            return formatter(runs)
        except Exception:
            logger.debug("Similar runs loading failed; skipping")
            return []

    def load_focused_recall(
        self,
        *,
        agent_type: str = "",
        bot_id: str = "",
        strategy_id: str = "",
        tags: list[str] | None = None,
        limit: int = 5,
        days: int = 90,
    ) -> list[dict]:
        if not agent_type:
            return []
        try:
            from trading_assistant.skills.run_recall_summarizer import RunRecallSummarizer

            cards = RunRecallSummarizer(
                self._memory_dir,
                run_index=self._run_index,
            ).summarize(
                workflow=agent_type,
                bot_id=bot_id,
                strategy_id=strategy_id,
                tags=tags or [],
                limit=limit,
                days=days,
            )
            return [card.to_prompt_dict() for card in cards]
        except Exception:
            logger.debug("Focused recall loading failed; skipping")
            return []
