"""Unit tests for libs.config.loader — _load_yaml_file and related helpers."""
from __future__ import annotations

from pathlib import Path

import pytest

from libs.config.loader import _load_yaml_file


# ---------------------------------------------------------------------------
# Tests — YAML error wrapping
# ---------------------------------------------------------------------------

class TestYamlErrorWrapping:
    """Malformed YAML should raise ValueError with the filename in the message."""

    def test_malformed_yaml_raises_value_error(self, tmp_path: Path) -> None:
        bad_yaml = tmp_path / "bad.yaml"
        # Produce genuinely invalid YAML: a tab character in an indentation context
        bad_yaml.write_text("key:\n\t- mixed tabs and spaces\n  - item", encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            _load_yaml_file(bad_yaml)
        assert "bad.yaml" in str(exc_info.value)

    def test_malformed_yaml_message_contains_filename(self, tmp_path: Path) -> None:
        bad_yaml = tmp_path / "broken.yaml"
        bad_yaml.write_text(":\n  :\n    - {bad", encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            _load_yaml_file(bad_yaml)
        assert "broken.yaml" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests — missing file
# ---------------------------------------------------------------------------

class TestMissingFile:
    """A non-existent path should raise ValueError mentioning the file path."""

    def test_missing_file_raises_value_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.yaml"
        assert not missing.exists()

        with pytest.raises(ValueError, match="does_not_exist.yaml"):
            _load_yaml_file(missing)

    def test_missing_file_mentions_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "subdir" / "gone.yaml"
        with pytest.raises(ValueError) as exc_info:
            _load_yaml_file(missing)
        assert "gone.yaml" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests — non-dict YAML
# ---------------------------------------------------------------------------

class TestNonDictYaml:
    """YAML that is not a mapping at the top level should raise ValueError."""

    def test_list_raises_value_error(self, tmp_path: Path) -> None:
        list_yaml = tmp_path / "list.yaml"
        list_yaml.write_text("[1, 2, 3]", encoding="utf-8")

        with pytest.raises(ValueError, match="YAML mapping"):
            _load_yaml_file(list_yaml)

    def test_scalar_string_raises_value_error(self, tmp_path: Path) -> None:
        scalar_yaml = tmp_path / "scalar.yaml"
        scalar_yaml.write_text('"just a string"', encoding="utf-8")

        with pytest.raises(ValueError, match="YAML mapping"):
            _load_yaml_file(scalar_yaml)

    def test_valid_dict_succeeds(self, tmp_path: Path) -> None:
        good_yaml = tmp_path / "good.yaml"
        good_yaml.write_text("key: value\nother: 42", encoding="utf-8")

        result = _load_yaml_file(good_yaml)
        assert result == {"key": "value", "other": 42}

    def test_empty_file_returns_empty_dict(self, tmp_path: Path) -> None:
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("", encoding="utf-8")

        result = _load_yaml_file(empty_yaml)
        assert result == {}
