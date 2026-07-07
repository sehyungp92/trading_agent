"""Structural experiment recording support."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class StructuralExperimentActions:
    """Structural experiment recording support."""

    def _record_structural_experiments(self, proposals, run_id: str) -> None:
        """Convert structural proposals with acceptance_criteria into experiment records."""
        if not proposals:
            return

        import hashlib

        from trading_assistant.schemas.structural_experiment import AcceptanceCriteria, ExperimentRecord

        for proposal in proposals:
            # Validate criteria have at least a metric field
            valid_criteria: list[AcceptanceCriteria] = []
            for raw_c in proposal.acceptance_criteria:
                if isinstance(raw_c, dict) and raw_c.get("metric"):
                    try:
                        valid_criteria.append(AcceptanceCriteria(**raw_c))
                    except Exception:
                        pass
            exp_id = ""
            if valid_criteria:
                exp_id = "exp_" + hashlib.sha256(
                    f"{run_id}:{proposal.bot_id}:{proposal.title}".encode()
                ).hexdigest()[:12]

            self._ledger_write_candidate(
                source="structural",
                kind_hint="structural_change",
                bot_id=proposal.bot_id or "",
                title=proposal.title or "",
                description=proposal.description or "",
                suggestion_id=proposal.linked_suggestion_id or "",
                experiment_id=exp_id,
                hypothesis_id=proposal.hypothesis_id or "",
                stable_link_key=exp_id or proposal.linked_suggestion_id or proposal.hypothesis_id or "",
                run_id=run_id,
                evaluation_method="experiment" if valid_criteria else "approval",
                acceptance_criteria=[
                    f"{c.metric}:{c.direction}:{c.minimum_change}"
                    for c in valid_criteria
                ],
            )

            if not self._structural_experiment_tracker or not valid_criteria:
                continue

            experiment = ExperimentRecord(
                experiment_id=exp_id,
                bot_id=proposal.bot_id,
                title=proposal.title,
                description=proposal.description,
                hypothesis_id=proposal.hypothesis_id,
                suggestion_id=proposal.linked_suggestion_id,
                proposal_run_id=run_id,
                acceptance_criteria=valid_criteria,
            )
            recorded = self._structural_experiment_tracker.record_experiment(experiment)
            if recorded:
                logger.info("Recorded structural experiment %s: %s", exp_id, proposal.title)
