# analysis/outcome_reasoning_prompt.py
"""Outcome reasoning prompt — asks Claude WHY a suggestion worked or didn't.

Invoked after outcome measurement to build causal understanding that enables
transfer learning and avoids repeating mistakes.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from trading_assistant.analysis.context_builder import ContextBuilder
from trading_assistant.analysis.evidence_memory import EvidenceMemory
from trading_assistant.schemas.prompt_package import PromptPackage

logger = logging.getLogger(__name__)

_OUTCOME_REASONING_INSTRUCTIONS = """\
You are analyzing the outcomes of previously implemented suggestions. For each
outcome below, determine WHETHER the suggestion genuinely caused the observed
effect, and WHY.

## OUTCOMES TO ANALYZE
{outcomes_text}

## YOUR ANALYTICAL TASKS
For each outcome:
1. **Genuine effect?** — Did the suggestion actually cause the measured improvement/degradation,
   or was it coincidental (market regime change, other concurrent changes, etc.)?
2. **Mechanism** — If genuine, WHAT mechanism explains the effect? Be specific about
   the causal chain from parameter change to outcome.
3. **Transferable?** — Can this insight be applied to other bots? If so, which ones and why?
4. **Lessons learned** — What should the system remember for future decisions?
5. **Revised confidence** — Given what you now know, what confidence (0.0-1.0) would you
   assign to this TYPE of suggestion in the future?

## CONSTRAINTS
- Consider regime_matched and volatility_ratio when assessing genuine effect
- Check concurrent_changes — if other suggestions were deployed simultaneously,
  the effect cannot be cleanly attributed
- measurement_quality LOW or INSUFFICIENT means you should be skeptical
- Be honest about uncertainty — "inconclusive" is a valid answer

## STRUCTURED OUTPUT (REQUIRED)
<!-- STRUCTURED_OUTPUT
{{
  "reasonings": [
    {{
      "suggestion_id": "...",
      "genuine_effect": true,
      "mechanism": "...",
      "transferable": true,
      "lessons_learned": "...",
      "revised_confidence": 0.7,
      "market_context": "...",
      "confounders": ["..."]
    }}
  ]
}}
-->"""


class OutcomeReasoningAssembler:
    """Assembles context for outcome reasoning — focused analysis of measured outcomes."""

    def __init__(
        self,
        memory_dir: Path,
        curated_dir: Path | None = None,
        bot_configs: dict | None = None,
    ) -> None:
        self.memory_dir = memory_dir
        self.curated_dir = curated_dir
        self.bot_configs = bot_configs
        self._evidence = EvidenceMemory(memory_dir)
        self._ctx = ContextBuilder(
            memory_dir,
            curated_dir=curated_dir,
            evidence_memory=self._evidence,
        )

    def assemble(self, outcomes: list[dict], session_store=None) -> PromptPackage:
        """Build prompt package for reasoning about measured outcomes.

        Args:
            outcomes: List of OutcomeMeasurement dicts to reason about.
            session_store: Optional SessionStore for loading session history.
        """
        pkg = self._ctx.base_package(
            session_store=session_store,
            agent_type="outcome_reasoning",
            bot_configs=self.bot_configs,
        )
        pkg.task_prompt = self._build_task_prompt(outcomes)
        pkg.instructions = self._build_instructions(outcomes)
        pkg.data["outcomes_for_reasoning"] = outcomes

        # Load suggestion details for context
        suggestions = self._load_suggestion_details(outcomes)
        if suggestions:
            pkg.data["suggestion_details"] = suggestions

        pkg.metadata["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        bot_ids = sorted({o.get("bot_id", "") for o in outcomes} - {""})
        if bot_ids:
            pkg.metadata["bot_ids"] = bot_ids
        return pkg

    def _build_task_prompt(self, outcomes: list[dict]) -> str:
        count = len(outcomes)
        return (
            f"Analyze {count} measured suggestion outcome(s). "
            f"Determine whether each suggestion genuinely caused the observed effect, "
            f"explain the mechanism, and assess transferability."
        )

    def _build_instructions(self, outcomes: list[dict]) -> str:
        """Format outcomes into the instruction template."""
        outcome_lines = []
        for i, o in enumerate(outcomes, 1):
            sid = o.get("suggestion_id", "?")
            verdict = o.get("verdict", "unknown")
            quality = o.get("measurement_quality", "unknown")
            pnl_before = o.get("pnl_before", 0)
            pnl_after = o.get("pnl_after", 0)
            regime_matched = o.get("regime_matched")
            concurrent = o.get("concurrent_changes", [])
            sig = o.get("significance_score")

            sig_str = f"{sig:.2f}" if sig is not None else "n/a"
            regime_str = str(regime_matched) if regime_matched is not None else "unknown"
            concurrent_str = ", ".join(concurrent) if concurrent else "none"

            outcome_lines.append(
                f"{i}. **Suggestion #{sid}** — Verdict: {verdict} (quality: {quality})\n"
                f"   PnL: {pnl_before:.2f} → {pnl_after:.2f} (delta: {pnl_after - pnl_before:+.2f})\n"
                f"   Regime matched: {regime_str}, Significance: {sig_str}\n"
                f"   Concurrent changes: {concurrent_str}"
            )

        outcomes_text = "\n".join(outcome_lines) if outcome_lines else "(No outcomes to analyze)"

        return _OUTCOME_REASONING_INSTRUCTIONS.format(outcomes_text=outcomes_text)

    def _load_suggestion_details(self, outcomes: list[dict]) -> list[dict]:
        """Load the original suggestion records for context."""
        sids = {o.get("suggestion_id", "") for o in outcomes}
        path = self.memory_dir / "findings" / "suggestions.jsonl"
        if not path.exists():
            return []
        details = []
        try:
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("suggestion_id") in sids:
                    details.append(rec)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load suggestion details: %s", exc)
        return details
