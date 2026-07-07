# tests/test_config_registry.py
"""Tests for the ConfigRegistry."""
from __future__ import annotations

import pytest
import yaml
from pathlib import Path

from trading_assistant.schemas.parameter_definition import ParameterType
from trading_assistant.skills.config_registry import ConfigRegistry


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    d = tmp_path / "bot_configs"
    d.mkdir()
    (d / "test_bot.yaml").write_text(yaml.dump({
        "bot_id": "test_bot",
        "repo_url": "git@github.com:user/test_bot.git",
        "repo_dir": "/repos/test_bot",
        "strategies": ["alpha"],
        "parameters": [
            {
                "param_name": "quality_min_threshold",
                "strategy_id": "alpha",
                "param_type": "YAML_FIELD",
                "file_path": "config.yaml",
                "yaml_key": "alpha.quality_min_threshold",
                "current_value": 0.6,
                "valid_range": [0.0, 1.0],
                "value_type": "float",
                "category": "entry_signal",
                "is_safety_critical": False,
            },
            {
                "param_name": "base_risk_pct",
                "param_type": "PYTHON_CONSTANT",
                "file_path": "config/risk.py",
                "python_path": "BASE_RISK_PCT",
                "current_value": 0.02,
                "valid_range": [0.005, 0.05],
                "value_type": "float",
                "category": "risk_management",
                "is_safety_critical": True,
            },
            {
                "param_name": "max_positions",
                "param_type": "YAML_FIELD",
                "file_path": "config.yaml",
                "yaml_key": "alpha.max_positions",
                "current_value": 5,
                "valid_range": [1, 20],
                "value_type": "int",
                "category": "risk_management",
                "is_safety_critical": True,
            },
            {
                "param_name": "mode",
                "param_type": "YAML_FIELD",
                "file_path": "config.yaml",
                "yaml_key": "alpha.mode",
                "current_value": "conservative",
                "valid_values": ["conservative", "aggressive", "balanced"],
                "value_type": "str",
                "category": "risk_management",
                "is_safety_critical": False,
            },
        ],
    }), encoding="utf-8")
    return d


@pytest.fixture
def registry(config_dir: Path) -> ConfigRegistry:
    return ConfigRegistry(config_dir)


class TestConfigRegistry:
    def test_load_profiles(self, registry: ConfigRegistry):
        assert "test_bot" in registry.list_bot_ids()
        profile = registry.get_profile("test_bot")
        assert profile is not None
        assert len(profile.parameters) == 4

    def test_get_parameter(self, registry: ConfigRegistry):
        p = registry.get_parameter("test_bot", "quality_min_threshold")
        assert p is not None
        assert p.current_value == 0.6
        assert p.param_type == ParameterType.YAML_FIELD

    def test_find_parameters_by_category(self, registry: ConfigRegistry):
        risk_params = registry.find_parameters_by_category("test_bot", "risk_management")
        assert len(risk_params) == 3
        entry_params = registry.find_parameters_by_category("test_bot", "entry_signal")
        assert len(entry_params) == 1

    def test_resolve_suggestion_category_match(self, registry: ConfigRegistry):
        suggestion = {
            "bot_id": "test_bot",
            "category": "entry_signal",
            "title": "Increase quality min threshold to 0.7",
            "description": "Quality filter too strict",
        }
        params = registry.resolve_suggestion_to_params(suggestion)
        assert len(params) >= 1
        assert params[0].param_name == "quality_min_threshold"

    def test_resolve_suggestion_no_match(self, registry: ConfigRegistry):
        suggestion = {
            "bot_id": "test_bot",
            "category": "unknown_category",
            "title": "something completely unrelated",
            "description": "no keywords match",
        }
        params = registry.resolve_suggestion_to_params(suggestion)
        assert len(params) == 0

    def test_validate_value_in_range(self, registry: ConfigRegistry):
        p = registry.get_parameter("test_bot", "quality_min_threshold")
        ok, msg = registry.validate_value(p, 0.7)
        assert ok is True
        assert msg is None

    def test_validate_value_out_of_range(self, registry: ConfigRegistry):
        p = registry.get_parameter("test_bot", "quality_min_threshold")
        ok, msg = registry.validate_value(p, 1.5)
        assert ok is False
        assert "outside range" in msg

    def test_validate_value_enum_check(self, registry: ConfigRegistry):
        p = registry.get_parameter("test_bot", "mode")
        ok, msg = registry.validate_value(p, "conservative")
        assert ok is True
        ok, msg = registry.validate_value(p, "turbo")
        assert ok is False
        assert "not in allowed values" in msg

    def test_safety_critical_flags(self, registry: ConfigRegistry):
        risk = registry.get_parameter("test_bot", "base_risk_pct")
        assert risk.is_safety_critical is True
        entry = registry.get_parameter("test_bot", "quality_min_threshold")
        assert entry.is_safety_critical is False

    def test_unknown_bot_returns_none(self, registry: ConfigRegistry):
        assert registry.get_profile("nonexistent") is None
        assert registry.get_parameter("nonexistent", "x") is None

    def test_excludes_unsupported_parameter_when_repo_file_available(self, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "config.py").write_text("SUPPORTED = 1\n", encoding="utf-8")
        cfg = tmp_path / "bot_configs"
        cfg.mkdir()
        (cfg / "bot.yaml").write_text(yaml.dump({
            "bot_id": "bot",
            "repo_dir": str(repo),
            "parameters": [
                {
                    "param_name": "supported",
                    "param_type": "PYTHON_CONSTANT",
                    "file_path": "config.py",
                    "python_path": "SUPPORTED",
                    "current_value": 1,
                    "value_type": "int",
                },
                {
                    "param_name": "missing",
                    "param_type": "PYTHON_CONSTANT",
                    "file_path": "config.py",
                    "python_path": "MISSING.value",
                    "current_value": 1,
                    "value_type": "int",
                },
            ],
        }), encoding="utf-8")

        registry = ConfigRegistry(cfg)

        assert registry.get_parameter("bot", "supported") is not None
        assert registry.get_parameter("bot", "missing") is None
        assert any("missing" in err for err in registry.load_errors)

    def test_loads_json_field_parameter_when_path_resolves(self, tmp_path: Path):
        repo = tmp_path / "repo"
        (repo / "config" / "strategies").mkdir(parents=True)
        (repo / "config" / "strategies" / "momentum.json").write_text(
            '{"strategy": {"indicators": {"ema_fast": 20}}}\n',
            encoding="utf-8",
        )
        cfg = tmp_path / "bot_configs"
        cfg.mkdir()
        (cfg / "crypto.yaml").write_text(yaml.dump({
            "bot_id": "crypto",
            "repo_dir": str(repo),
            "parameters": [{
                "param_name": "ema_fast",
                "param_type": "JSON_FIELD",
                "file_path": "config/strategies/momentum.json",
                "python_path": "strategy.indicators.ema_fast",
                "current_value": 20,
                "value_type": "int",
            }],
        }), encoding="utf-8")

        registry = ConfigRegistry(cfg)

        param = registry.get_parameter("crypto", "ema_fast")
        assert param is not None
        assert param.param_type == ParameterType.JSON_FIELD
        assert registry.load_errors == []
