# skills/config_registry.py
"""Config registry — loads bot parameter profiles from YAML config files.

Each bot has a YAML file defining its tunable parameters, file paths,
valid ranges, and safety flags. The registry resolves suggestion text
to matching parameter definitions.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from trading_assistant.schemas.bot_profile import BotConfigProfile
from trading_assistant.schemas.parameter_definition import ParameterDefinition
from trading_assistant.skills.file_change_generator import FileChangeGenerator

logger = logging.getLogger(__name__)


class ConfigRegistry:
    """Loads and queries bot configuration profiles."""

    def __init__(self, config_dir: Path) -> None:
        self._profiles: dict[str, BotConfigProfile] = {}
        self._load_errors: list[str] = []
        self._missing_repo_dirs_logged: set[Path] = set()
        self._file_change_generator = FileChangeGenerator()
        self._load_profiles(config_dir)

    def _load_profiles(self, config_dir: Path) -> None:
        config_dir = Path(config_dir)
        if not config_dir.exists():
            logger.warning("Config directory %s does not exist", config_dir)
            return
        for yaml_file in sorted(config_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
                if not data or not isinstance(data, dict):
                    continue
                bot_id = data.get("bot_id", yaml_file.stem)
                repo_dir_raw = data.get("repo_dir", "")
                repo_dir = Path(repo_dir_raw) if repo_dir_raw else None
                params = []
                for p in data.get("parameters", []):
                    raw_param = dict(p)
                    raw_param["bot_id"] = bot_id
                    try:
                        param = ParameterDefinition(**raw_param)
                    except Exception as exc:
                        self._record_load_error(yaml_file, bot_id, raw_param, str(exc))
                        continue
                    ok, reason = self._validate_parameter_definition(param, repo_dir)
                    if not ok:
                        self._record_load_error(yaml_file, bot_id, raw_param, reason or "invalid")
                        continue
                    params.append(param)
                profile = BotConfigProfile(
                    bot_id=bot_id,
                    repo_url=data.get("repo_url", ""),
                    repo_dir=data.get("repo_dir", ""),
                    default_branch=data.get("default_branch", "main"),
                    strategy_version=data.get("strategy_version"),
                    config_version=data.get("config_version"),
                    code_sha=data.get("code_sha"),
                    allowed_edit_paths=data.get("allowed_edit_paths", []) or [],
                    structural_context_paths=data.get("structural_context_paths", []) or [],
                    verification_commands=data.get("verification_commands", []) or [],
                    parameters=params,
                    strategies=data.get("strategies", []),
                )
                self._profiles[bot_id] = profile
                logger.info("Loaded config profile for %s (%d params)", bot_id, len(params))
            except Exception:
                logger.exception("Failed to load config from %s", yaml_file)

    def _record_load_error(
        self,
        yaml_file: Path,
        bot_id: str,
        raw_param: dict,
        reason: str,
    ) -> None:
        param_name = raw_param.get("param_name", "<unknown>")
        msg = f"{yaml_file.name}:{bot_id}:{param_name}: {reason}"
        self._load_errors.append(msg)
        logger.warning("Skipping invalid config parameter %s", msg)

    def _validate_parameter_definition(
        self,
        param: ParameterDefinition,
        repo_dir: Path | None,
    ) -> tuple[bool, str | None]:
        if not param.file_path:
            return False, "file_path is required"

        if repo_dir is None:
            return True, None
        if not repo_dir.exists():
            if repo_dir not in self._missing_repo_dirs_logged:
                logger.info(
                    "Repo dir %s is not present; deferring file-shape validation",
                    repo_dir,
                )
                self._missing_repo_dirs_logged.add(repo_dir)
            return True, None

        target = repo_dir / param.file_path
        if not target.exists():
            return False, f"target file does not exist: {param.file_path}"

        try:
            self._file_change_generator.generate_change(
                param,
                param.current_value,
                repo_dir,
            )
        except Exception as exc:
            return False, f"unsupported parameter target: {exc}"
        return True, None

    def get_profile(self, bot_id: str) -> BotConfigProfile | None:
        return self._profiles.get(bot_id)

    @property
    def load_errors(self) -> list[str]:
        return list(self._load_errors)

    def get_parameter(self, bot_id: str, param_name: str) -> ParameterDefinition | None:
        profile = self._profiles.get(bot_id)
        if profile is None:
            return None
        return profile.get_parameter(param_name)

    def find_parameters_by_category(self, bot_id: str, category: str) -> list[ParameterDefinition]:
        profile = self._profiles.get(bot_id)
        if profile is None:
            return []
        return profile.get_parameters_by_category(category)

    def resolve_suggestion_to_params(self, suggestion) -> list[ParameterDefinition]:
        """Match a suggestion to parameter definitions by category and keyword.

        Args:
            suggestion: SuggestionRecord (dict or model with bot_id, category, title, description)
        """
        bot_id = suggestion.get("bot_id") if isinstance(suggestion, dict) else getattr(suggestion, "bot_id", "")
        category = suggestion.get("category") if isinstance(suggestion, dict) else getattr(suggestion, "category", "")
        title = suggestion.get("title") if isinstance(suggestion, dict) else getattr(suggestion, "title", "")
        description = suggestion.get("description") if isinstance(suggestion, dict) else getattr(suggestion, "description", "")

        profile = self._profiles.get(bot_id)
        if profile is None:
            return []

        # Match by category first
        category_matches = profile.get_parameters_by_category(category)

        # If we have category matches, filter by keyword match in title/description
        text = f"{title} {description}".lower()
        if category_matches:
            keyword_matches = [
                p for p in category_matches
                if self._keyword_match(p.param_name, text)
            ]
            return keyword_matches if keyword_matches else category_matches

        # Fallback: keyword match across all params
        return [
            p for p in profile.parameters
            if self._keyword_match(p.param_name, text)
        ]

    @staticmethod
    def _keyword_match(param_name: str, text: str) -> bool:
        """Check if a parameter name appears in the suggestion text."""
        # Convert param_name to searchable variants
        variants = {param_name.lower()}
        # Add space-separated version (quality_min_threshold -> quality min threshold)
        variants.add(param_name.lower().replace("_", " "))
        # Add individual words
        words = param_name.lower().split("_")
        return any(v in text for v in variants) or all(w in text for w in words if len(w) > 2)

    def validate_value(self, param: ParameterDefinition, value: Any) -> tuple[bool, str | None]:
        """Validate a proposed value against parameter constraints."""
        if param.valid_range is not None:
            try:
                fval = float(value)
            except (TypeError, ValueError):
                return False, f"Cannot convert {value!r} to float for range check"
            if fval < param.valid_range[0] or fval > param.valid_range[1]:
                return False, f"Value {fval} outside range [{param.valid_range[0]}, {param.valid_range[1]}]"
        if param.valid_values is not None:
            if value not in param.valid_values:
                return False, f"Value {value!r} not in allowed values {param.valid_values}"
        return True, None

    def list_bot_ids(self) -> list[str]:
        return list(self._profiles.keys())
