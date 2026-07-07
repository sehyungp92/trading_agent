"""Experiment config generator — creates A/B experiment configurations from suggestions.

When a suggestion is approved but the user opts for "experiment first" instead of
direct deployment, this generator creates a proper A/B config.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from trading_assistant.schemas.repo_changes import FileChange, PRRequest
from trading_assistant.schemas.experiments import (
    ExperimentConfig,
    ExperimentType,
    ExperimentVariant,
)


class ExperimentConfigGenerator:
    """Generates experiment configurations from approved suggestions."""

    def __init__(self, config_registry=None) -> None:
        self._registry = config_registry

    def generate_from_suggestion(
        self,
        suggestion_id: str,
        bot_id: str,
        param_name: str,
        current_value: Any,
        proposed_value: Any,
        title: str = "",
        duration_days: int = 14,
    ) -> ExperimentConfig:
        """Create an ExperimentConfig from a parameter suggestion."""
        experiment_id = self._generate_id(suggestion_id, param_name)

        return ExperimentConfig(
            experiment_id=experiment_id,
            bot_id=bot_id,
            experiment_type=ExperimentType.PARAMETER_AB,
            title=title or f"A/B test: {param_name} on {bot_id}",
            description=f"Testing {param_name} change from {current_value} to {proposed_value}",
            variants=[
                ExperimentVariant(
                    name="control",
                    params={param_name: current_value},
                    allocation_pct=50.0,
                ),
                ExperimentVariant(
                    name="treatment",
                    params={param_name: proposed_value},
                    allocation_pct=50.0,
                ),
            ],
            max_duration_days=duration_days,
            source_suggestion_id=suggestion_id,
        )

    def generate_bot_yaml(self, config: ExperimentConfig) -> str:
        """Generate YAML snippet for bot experiment config."""
        import yaml

        experiment_data = {
            "experiment_id": config.experiment_id,
            "title": config.title,
            "status": "active",
            "variants": {},
            "allocation_method": "hash",
            "success_metric": config.success_metric,
            "max_duration_days": config.max_duration_days,
        }

        for variant in config.variants:
            experiment_data["variants"][variant.name] = {
                "params": variant.params,
                "allocation_pct": variant.allocation_pct,
            }

        return yaml.dump(
            {"experiments": [experiment_data]},
            default_flow_style=False,
            sort_keys=False,
        )

    def generate_experiment_pr(
        self,
        config: ExperimentConfig,
        repo_dir: str,
    ) -> PRRequest:
        """Build a PRRequest that adds the experiment config to the bot repo."""
        yaml_content = self.generate_bot_yaml(config)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        return PRRequest(
            approval_request_id=config.experiment_id,
            suggestion_id=config.source_suggestion_id or config.experiment_id,
            bot_id=config.bot_id,
            repo_dir=repo_dir,
            branch_name=f"codex/experiment-{config.experiment_id[:8]}-{date_str}",
            title=f"[trading-assistant] Experiment: {config.title}",
            body=(
                f"## A/B Experiment\n\n"
                f"**{config.title}**\n\n"
                f"{config.description}\n\n"
                f"### Variants\n"
                + "\n".join(
                    f"- **{v.name}** ({v.allocation_pct}%): {v.params}"
                    for v in config.variants
                )
                + f"\n\n### Settings\n"
                f"- Success metric: {config.success_metric}\n"
                f"- Max duration: {config.max_duration_days} days\n"
                f"- Min trades per variant: {config.min_trades_per_variant}\n"
                f"- Significance level: {config.significance_level}\n"
            ),
            file_changes=[
                FileChange(
                    file_path=f"config/experiments/{config.experiment_id}.yaml",
                    original_content="",
                    new_content=yaml_content,
                    diff_preview=f"Add experiment config: {config.experiment_id}",
                )
            ],
        )

    def _generate_id(self, suggestion_id: str, param_name: str) -> str:
        """Generate deterministic experiment ID from suggestion + param."""
        raw = f"{suggestion_id}|{param_name}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
