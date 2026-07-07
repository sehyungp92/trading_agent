# tests/test_file_change_generator.py
"""Tests for FileChangeGenerator."""
from __future__ import annotations

from pathlib import Path

import pytest

from trading_assistant.schemas.parameter_definition import ParameterDefinition, ParameterType
from trading_assistant.schemas.repo_changes import FileChangeMode
from trading_assistant.skills.file_change_generator import FileChangeGenerator


@pytest.fixture
def gen() -> FileChangeGenerator:
    return FileChangeGenerator()


def _yaml_param(yaml_key: str = "kmp.quality_min") -> ParameterDefinition:
    return ParameterDefinition(
        param_name="quality_min",
        bot_id="bot1",
        param_type=ParameterType.YAML_FIELD,
        file_path="config.yaml",
        yaml_key=yaml_key,
        current_value=0.6,
        value_type="float",
    )


def _python_param(
    python_path: str = "BASE_RISK_PCT",
    file_path: str = "risk.py",
) -> ParameterDefinition:
    return ParameterDefinition(
        param_name="base_risk_pct",
        bot_id="bot1",
        param_type=ParameterType.PYTHON_CONSTANT,
        file_path=file_path,
        python_path=python_path,
        current_value=0.02,
        value_type="float",
    )


def _toml_param(toml_path: str, file_path: str = "config.toml") -> ParameterDefinition:
    return ParameterDefinition(
        param_name=toml_path.split(".")[-1],
        bot_id="bot1",
        param_type=ParameterType.TOML_FIELD,
        file_path=file_path,
        python_path=toml_path,  # TOML reuses python_path field for the dotted path
        current_value=20,
        value_type="int",
    )


def _json_param(json_path: str, file_path: str = "config.json") -> ParameterDefinition:
    return ParameterDefinition(
        param_name=json_path.split(".")[-1],
        bot_id="bot1",
        param_type=ParameterType.JSON_FIELD,
        file_path=file_path,
        python_path=json_path,
        current_value=20,
        value_type="int",
    )


class TestFileChangeGenerator:
    def test_yaml_simple_key_change(self, gen: FileChangeGenerator, tmp_path: Path):
        (tmp_path / "config.yaml").write_text(
            "kmp:\n  quality_min: 0.6\n  other: 10\n"
        )
        change = gen.generate_change(_yaml_param(), 0.7, tmp_path)
        assert "quality_min: 0.7" in change.new_content
        assert "quality_min: 0.6" not in change.new_content
        assert "other: 10" in change.new_content  # untouched

    def test_yaml_nested_dotted_path(self, gen: FileChangeGenerator, tmp_path: Path):
        (tmp_path / "config.yaml").write_text(
            "kmp:\n  quality_min: 0.6\n"
        )
        change = gen.generate_change(_yaml_param("kmp.quality_min"), 0.8, tmp_path)
        assert "quality_min: 0.8" in change.new_content

    def test_yaml_preserves_comments(self, gen: FileChangeGenerator, tmp_path: Path):
        content = "# Top comment\nkmp:\n  quality_min: 0.6  # inline\n  other: 5\n"
        (tmp_path / "config.yaml").write_text(content)
        change = gen.generate_change(_yaml_param(), 0.7, tmp_path)
        assert "# Top comment" in change.new_content
        assert "# inline" in change.new_content
        assert "quality_min: 0.7" in change.new_content

    def test_yaml_missing_key_raises(self, gen: FileChangeGenerator, tmp_path: Path):
        (tmp_path / "config.yaml").write_text("kmp:\n  other: 5\n")
        with pytest.raises(ValueError, match="YAML key not found"):
            gen.generate_change(_yaml_param(), 0.7, tmp_path)

    def test_unsupported_param_type_raises(self, gen: FileChangeGenerator, tmp_path: Path):
        """Non-YAML param types are not supported."""
        (tmp_path / "config.yaml").write_text("x: 1\n")
        param = ParameterDefinition(
            param_name="x", bot_id="bot1",
            param_type=ParameterType.YAML_FIELD,
            file_path="config.yaml",
            yaml_key="x",
            current_value=1,
            value_type="int",
        )
        # This should work (YAML_FIELD)
        change = gen.generate_change(param, 2, tmp_path)
        assert "x: 2" in change.new_content

    def test_file_not_found_raises(self, gen: FileChangeGenerator, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            gen.generate_change(_yaml_param(), 0.7, tmp_path)

    def test_diff_preview_contains_values(self, gen: FileChangeGenerator, tmp_path: Path):
        (tmp_path / "config.yaml").write_text(
            "kmp:\n  quality_min: 0.6\n"
        )
        change = gen.generate_change(_yaml_param(), 0.7, tmp_path)
        assert "0.6" in change.diff_preview
        assert "0.7" in change.diff_preview

    def test_same_value_no_diff(self, gen: FileChangeGenerator, tmp_path: Path):
        (tmp_path / "config.yaml").write_text(
            "kmp:\n  quality_min: 0.6\n"
        )
        change = gen.generate_change(_yaml_param(), 0.6, tmp_path)
        assert change.original_content == change.new_content
        assert change.diff_preview == ""

    def test_python_constant_change(self, gen: FileChangeGenerator, tmp_path: Path):
        (tmp_path / "risk.py").write_text("BASE_RISK_PCT = 0.02\nOTHER = 5\n")
        change = gen.generate_change(_python_param(), 0.03, tmp_path)
        assert change.change_mode == FileChangeMode.PYTHON_CONSTANT
        assert "BASE_RISK_PCT = 0.03" in change.new_content
        assert "OTHER = 5" in change.new_content

    def test_unified_diff_change(self, gen: FileChangeGenerator, tmp_path: Path):
        (tmp_path / "algo.py").write_text("VALUE = 1\nOTHER = 2\n")
        patch = "\n".join([
            "--- a/algo.py",
            "+++ b/algo.py",
            "@@ -1,2 +1,2 @@",
            "-VALUE = 1",
            "+VALUE = 2",
            " OTHER = 2",
        ])
        change = gen.generate_patch_change("algo.py", patch, tmp_path)
        assert change.change_mode == FileChangeMode.UNIFIED_DIFF
        assert "VALUE = 2" in change.new_content
        assert "VALUE = 1" not in change.new_content

    def test_unified_diff_rejects_bad_context(self, gen: FileChangeGenerator, tmp_path: Path):
        (tmp_path / "algo.py").write_text("VALUE = 1\nOTHER = 2\n")
        patch = "\n".join([
            "--- a/algo.py",
            "+++ b/algo.py",
            "@@ -1,2 +1,2 @@",
            "-VALUE = 9",
            "+VALUE = 2",
            " OTHER = 2",
        ])
        with pytest.raises(ValueError):
            gen.generate_patch_change("algo.py", patch, tmp_path)

    # ----- P1-7: class attribute editing -----

    def test_python_class_attribute_change(self, gen: FileChangeGenerator, tmp_path: Path):
        """python_path like `StrategySettings.tier_a_min` should patch the
        `tier_a_min = ...` line inside `class StrategySettings:`."""
        (tmp_path / "config.py").write_text(
            "class StrategySettings:\n"
            "    tier_a_min = 0.65\n"
            "    tier_b_min = 0.40\n"
        )
        param = _python_param("StrategySettings.tier_a_min", file_path="config.py")
        change = gen.generate_change(param, 0.70, tmp_path)
        assert change.change_mode == FileChangeMode.PYTHON_CONSTANT
        assert "tier_a_min = 0.7" in change.new_content
        assert "tier_b_min = 0.40" in change.new_content  # untouched

    def test_python_class_attribute_with_annotation(
        self, gen: FileChangeGenerator, tmp_path: Path,
    ):
        """Type-annotated class attributes should also patch correctly."""
        (tmp_path / "config.py").write_text(
            "class StrategySettings:\n"
            "    tier_a_min: float = 0.65\n"
        )
        param = _python_param("StrategySettings.tier_a_min", file_path="config.py")
        change = gen.generate_change(param, 0.55, tmp_path)
        assert "tier_a_min: float = 0.55" in change.new_content

    def test_python_class_attribute_preserves_trailing_comment(
        self, gen: FileChangeGenerator, tmp_path: Path,
    ):
        (tmp_path / "config.py").write_text(
            "class StrategySettings:\n"
            "    tier_a_min = 0.65  # gate threshold\n"
        )
        param = _python_param("StrategySettings.tier_a_min", file_path="config.py")
        change = gen.generate_change(param, 0.70, tmp_path)
        assert "tier_a_min = 0.7" in change.new_content
        assert "# gate threshold" in change.new_content

    def test_python_class_attribute_missing_raises(
        self, gen: FileChangeGenerator, tmp_path: Path,
    ):
        (tmp_path / "config.py").write_text("class StrategySettings:\n    pass\n")
        param = _python_param("StrategySettings.tier_a_min", file_path="config.py")
        with pytest.raises(ValueError, match="not found"):
            gen.generate_change(param, 0.70, tmp_path)

    def test_python_subscript_assignment_change(
        self, gen: FileChangeGenerator, tmp_path: Path,
    ):
        (tmp_path / "config.py").write_text(
            'R7C_FLAGS["reversal_engine"] = True  # enabled\n'
        )
        param = _python_param("R7C_FLAGS.reversal_engine", file_path="config.py")
        change = gen.generate_change(param, False, tmp_path)
        assert 'R7C_FLAGS["reversal_engine"] = False' in change.new_content
        assert "# enabled" in change.new_content

    def test_python_dict_literal_change(
        self, gen: FileChangeGenerator, tmp_path: Path,
    ):
        (tmp_path / "config.py").write_text(
            "R7C_FLAGS = {\n"
            "    'reversal_engine': True,\n"
            "    'breakdown_engine': True,\n"
            "}\n"
        )
        param = _python_param("R7C_FLAGS.reversal_engine", file_path="config.py")
        change = gen.generate_change(param, False, tmp_path)
        assert "'reversal_engine': False" in change.new_content
        assert "'breakdown_engine': True" in change.new_content

    def test_python_multiline_dict_literal_value_fails_closed(
        self, gen: FileChangeGenerator, tmp_path: Path,
    ):
        (tmp_path / "config.py").write_text(
            "R7C_FLAGS = {\n"
            "    'reversal_engine': [\n"
            "        True\n"
            "    ],\n"
            "}\n"
        )
        param = _python_param("R7C_FLAGS.reversal_engine", file_path="config.py")

        with pytest.raises(ValueError, match="not found"):
            gen.generate_change(param, False, tmp_path)

    # ----- P1-7: TOML editing -----

    def test_toml_field_change(self, gen: FileChangeGenerator, tmp_path: Path):
        """TOML_FIELD param should update a dotted path while preserving formatting."""
        (tmp_path / "config.toml").write_text(
            "# entry signal config\n"
            "[entry]\n"
            "ema_fast = 20\n"
            "ema_slow = 200\n"
        )
        param = _toml_param("entry.ema_fast")
        change = gen.generate_change(param, 12, tmp_path)
        assert change.change_mode == FileChangeMode.TOML_FIELD
        assert "ema_fast = 12" in change.new_content
        assert "ema_slow = 200" in change.new_content
        assert "# entry signal config" in change.new_content  # comment preserved

    def test_toml_field_missing_section_raises(
        self, gen: FileChangeGenerator, tmp_path: Path,
    ):
        (tmp_path / "config.toml").write_text("[risk]\nrisk_pct = 0.01\n")
        param = _toml_param("entry.ema_fast")
        with pytest.raises(ValueError, match="TOML section not found"):
            gen.generate_change(param, 12, tmp_path)

    def test_toml_field_missing_leaf_raises(
        self, gen: FileChangeGenerator, tmp_path: Path,
    ):
        (tmp_path / "config.toml").write_text("[entry]\nema_slow = 200\n")
        param = _toml_param("entry.ema_fast")
        with pytest.raises(ValueError, match="TOML key not found"):
            gen.generate_change(param, 12, tmp_path)

    # ----- JSON editing -----

    @pytest.mark.parametrize(
        ("path", "new_value", "expected"),
        [
            ("strategy.indicators.ema_fast", 12, '"ema_fast": 12'),
            ("strategy.exits.tp1_r", 1.4, '"tp1_r": 1.4'),
            ("strategy.entry.entry_on_close", False, '"entry_on_close": false'),
            ("strategy.entry.mode", "confirm", '"mode": "confirm"'),
        ],
    )
    def test_json_field_change_round_trip(
        self,
        gen: FileChangeGenerator,
        tmp_path: Path,
        path: str,
        new_value,
        expected: str,
    ):
        (tmp_path / "config.json").write_text(
            "{\n"
            "  \"strategy\": {\n"
            "    \"indicators\": {\"ema_fast\": 20},\n"
            "    \"exits\": {\"tp1_r\": 1.2},\n"
            "    \"entry\": {\"entry_on_close\": true, \"mode\": \"legacy\"}\n"
            "  }\n"
            "}\n"
        )
        change = gen.generate_change(_json_param(path), new_value, tmp_path)
        assert change.change_mode == FileChangeMode.JSON_FIELD
        assert expected in change.new_content
        assert change.new_content.endswith("\n")

    def test_json_field_missing_path_raises(self, gen: FileChangeGenerator, tmp_path: Path):
        (tmp_path / "config.json").write_text("{\"strategy\": {\"risk\": {}}}\n")
        with pytest.raises(ValueError, match="JSON section not found|JSON key not found"):
            gen.generate_change(_json_param("strategy.risk.max_leverage_major"), 5.0, tmp_path)
