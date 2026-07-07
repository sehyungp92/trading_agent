"""Session and run-recall prompt context loaders."""

from __future__ import annotations




class SessionRecallContextMixin:
    def load_session_history(self, session_store, agent_type: str, days: int = 7) -> str:
        """Load recent session summaries as formatted text.

        Args:
            session_store: SessionStore instance.
            agent_type: Type of agent to load history for.
            days: Number of days to look back.

        Returns:
            Formatted string summarizing recent sessions, or empty string.
        """
        try:
            sessions = session_store.get_recent_sessions(agent_type, days=days)
            if not isinstance(sessions, list):
                return ""
        except Exception:
            return ""
        if not sessions:
            return ""

        formatted_lines = [f"Recent {agent_type} sessions (last {days} days):"]
        for s in sessions[:20]:  # cap to avoid context bloat
            details: list[str] = []
            provider = s.get("provider")
            effective_model = s.get("effective_model")
            if provider and effective_model:
                details.append(f"{provider}/{effective_model}")
            elif provider:
                details.append(str(provider))

            duration = s.get("duration_ms", 0)
            details.append(f"{duration}ms")

            first_output_ms = s.get("first_output_ms")
            if isinstance(first_output_ms, int) and first_output_ms > 0:
                details.append(f"first {first_output_ms}ms")

            tool_call_count = s.get("tool_call_count")
            if isinstance(tool_call_count, int) and tool_call_count > 0:
                details.append(f"tools {tool_call_count}")

            stream_event_count = s.get("stream_event_count")
            if isinstance(stream_event_count, int) and stream_event_count > 0:
                details.append(f"stream {stream_event_count}")

            auth_mode = s.get("auth_mode")
            if auth_mode:
                details.append(str(auth_mode))

            summary = s.get("response_summary", "")[:100]
            formatted_lines.append(
                f"- {s.get('date', '?')}: {', '.join(details)} -- {summary}"
            )
        return "\n".join(formatted_lines)

    def load_similar_runs(
        self,
        agent_type: str = "",
        bot_id: str = "",
        limit: int = 5,
        days: int = 60,
        retrieval_profile: dict | None = None,
    ) -> list[dict]:
        """Load recent similar runs from RunIndex for prompt context."""
        profile = retrieval_profile or self.build_retrieval_profile(
            agent_type=agent_type,
            bot_id=bot_id,
        )
        return self._evidence_memory().run_recall.load_similar_runs(
            agent_type=agent_type,
            bot_id=bot_id,
            query=self._run_search_query(profile),
            formatter=self._format_similar_runs,
            limit=limit,
            days=days,
        )

    def load_focused_recall(
        self,
        agent_type: str = "",
        bot_id: str = "",
        strategy_id: str = "",
        tags: list[str] | None = None,
        limit: int = 5,
        days: int = 90,
    ) -> list[dict]:
        """Load provenance-rich recall cards for prompt context."""
        return self._evidence_memory().run_recall.load_focused_recall(
            agent_type=agent_type,
            bot_id=bot_id,
            strategy_id=strategy_id,
            tags=tags or [],
            limit=limit,
            days=days,
        )

    def build_retrieval_profile(self, agent_type: str = "", bot_id: str = "") -> dict:
        """Build structured retrieval tags and query terms from current context."""
        tags: list[str] = []
        query_terms: list[str] = []

        def _add_tag(prefix: str, value: str) -> None:
            normalized = self._retrieval_tag(prefix, value)
            if normalized and normalized not in tags:
                tags.append(normalized)

        def _add_query(value: str) -> None:
            normalized = str(value or "").strip()
            if normalized and normalized not in query_terms:
                query_terms.append(normalized)

        if agent_type:
            _add_tag("workflow", agent_type)
            _add_query(agent_type.replace("_", " "))
        if bot_id:
            _add_tag("bot", bot_id)
            _add_query(bot_id)

        macro_regime = self.load_macro_regime_context().get("macro_regime", "")
        if macro_regime:
            _add_tag("regime", macro_regime)
            _add_query(macro_regime)

        validation_patterns = self.load_validation_patterns(bot_id=bot_id)
        top_blocked = sorted(
            validation_patterns.items(),
            key=lambda item: item[1].get("blocked_count", 0),
            reverse=True,
        )[:3]
        for category, info in top_blocked:
            _add_tag("category", category)
            _add_query(category)
            for reason in (info.get("common_reasons") or [])[:2]:
                _add_tag("reason", reason)
                _add_query(reason)

        weakest = [
            score for score in self.load_category_scorecard().get("scores", [])
            if score.get("sample_size", 0) >= 3
            and (not bot_id or score.get("bot_id") in ("", bot_id))
        ]
        weakest.sort(key=lambda score: (score.get("win_rate", 1.0), -score.get("sample_size", 0)))
        for score in weakest[:3]:
            category = score.get("category", "")
            _add_tag("category", category)
            _add_query(category)

        return {
            "tags": tags,
            "query_terms": query_terms,
            "macro_regime": macro_regime,
            "agent_type": agent_type,
            "bot_id": bot_id,
        }

    @staticmethod
    def _retrieval_tag(prefix: str, value: str) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        chars: list[str] = []
        prev_sep = False
        for char in text:
            if char.isalnum():
                chars.append(char)
                prev_sep = False
            elif not prev_sep:
                chars.append("_")
                prev_sep = True
        slug = "".join(chars).strip("_")
        return f"{prefix}:{slug}" if slug else ""

    @staticmethod
    def _run_search_query(profile: dict) -> str:
        terms: list[str] = []
        for term in profile.get("query_terms", []):
            cleaned = "".join(char if char.isalnum() or char.isspace() else " " for char in str(term))
            cleaned = " ".join(cleaned.split())
            if not cleaned:
                continue
            if " " in cleaned:
                terms.append(f"\"{cleaned}\"")
            else:
                terms.append(cleaned)
        return " OR ".join(terms[:8])

    @staticmethod
    def _format_similar_runs(runs: list[dict]) -> list[dict]:
        return [
            {
                "run_id": r.get("run_id", ""),
                "date": r.get("date", ""),
                "agent_type": r.get("agent_type", ""),
                "provider": r.get("provider", ""),
                "snippet": (r.get("snippet") or r.get("response_preview", "") or "")[:200],
            }
            for r in runs
        ]
