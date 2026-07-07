"""Generate validated file changes for bot repository updates."""
from __future__ import annotations

import ast
import difflib
import json
import re
from pathlib import Path
from typing import Any

import tomlkit

from trading_assistant.schemas.parameter_definition import ParameterDefinition, ParameterType
from trading_assistant.schemas.repo_changes import FileChange, FileChangeMode


class FileChangeGenerator:
    """Generates repo file mutations for parameters and agent-produced patches."""

    def generate_change(
        self,
        param: ParameterDefinition,
        new_value: Any,
        repo_dir: Path,
    ) -> FileChange:
        """Read a file, apply a parameter change, and return a FileChange."""
        repo_dir = Path(repo_dir)
        full_path = repo_dir / param.file_path
        if not full_path.exists():
            raise FileNotFoundError(f"Config file not found: {full_path}")

        original = full_path.read_text(encoding="utf-8")
        if param.param_type == ParameterType.YAML_FIELD:
            modified = self._modify_yaml(original, param.yaml_key or "", new_value)
            change_mode = FileChangeMode.YAML_FIELD
            metadata = {"yaml_key": param.yaml_key or ""}
        elif param.param_type == ParameterType.PYTHON_CONSTANT:
            modified = self._modify_python_constant(
                original, param.python_path or "", new_value,
            )
            change_mode = FileChangeMode.PYTHON_CONSTANT
            metadata = {"python_path": param.python_path or ""}
        elif param.param_type == ParameterType.TOML_FIELD:
            modified = self._modify_toml_field(
                original, param.python_path or "", new_value,
            )
            change_mode = FileChangeMode.TOML_FIELD
            metadata = {"toml_path": param.python_path or ""}
        elif param.param_type == ParameterType.JSON_FIELD:
            modified = self._modify_json_field(
                original, param.python_path or "", new_value,
            )
            change_mode = FileChangeMode.JSON_FIELD
            metadata = {"json_path": param.python_path or ""}
        else:
            raise ValueError(f"Unsupported param_type: {param.param_type}")

        return self._build_change(
            file_path=param.file_path,
            original=original,
            modified=modified,
            change_mode=change_mode,
            metadata=metadata,
        )

    def generate_patch_change(
        self,
        file_path: str,
        patch: str,
        repo_dir: Path,
    ) -> FileChange:
        """Apply a validated unified diff and return the resulting FileChange."""
        repo_dir = Path(repo_dir)
        full_path = repo_dir / file_path
        if not full_path.exists():
            raise FileNotFoundError(f"Patch target not found: {full_path}")

        original = full_path.read_text(encoding="utf-8")
        modified = self._apply_unified_diff(
            original=original,
            patch=patch,
            expected_file_path=file_path,
        )
        return self._build_change(
            file_path=file_path,
            original=original,
            modified=modified,
            change_mode=FileChangeMode.UNIFIED_DIFF,
            metadata={},
            patch=patch,
        )

    def _build_change(
        self,
        file_path: str,
        original: str,
        modified: str,
        change_mode: FileChangeMode,
        metadata: dict[str, Any],
        patch: str = "",
    ) -> FileChange:
        diff = "\n".join(difflib.unified_diff(
            original.splitlines(),
            modified.splitlines(),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        ))
        return FileChange(
            file_path=file_path,
            original_content=original,
            new_content=modified,
            change_mode=change_mode,
            metadata=metadata,
            patch=patch,
            diff_preview=diff,
        )

    def _modify_yaml(self, content: str, yaml_key: str, new_value: Any) -> str:
        keys = yaml_key.split(".")
        lines = content.split("\n")
        result, changed = self._set_yaml_value(lines, keys, 0, 0, new_value)
        if not changed:
            raise ValueError(f"YAML key not found: {yaml_key}")
        return "\n".join(result)

    def _set_yaml_value(
        self,
        lines: list[str],
        keys: list[str],
        key_idx: int,
        min_indent: int,
        new_value: Any,
    ) -> tuple[list[str], bool]:
        target_key = keys[key_idx]
        is_last = key_idx == len(keys) - 1
        result = list(lines)

        for idx, line in enumerate(result):
            stripped = line.lstrip()
            if stripped.startswith("#") or not stripped:
                continue
            indent = len(line) - len(stripped)
            if indent < min_indent:
                continue

            match = re.match(r"^(\s*)([\w\-]+)\s*:\s*(.*)", line)
            if not match or match.group(2) != target_key or indent < min_indent:
                continue

            if is_last:
                prefix = match.group(1) + match.group(2) + ": "
                comment = ""
                rest = match.group(3)
                comment_match = re.search(r"\s+#\s+.*$", rest)
                if comment_match:
                    comment = comment_match.group()
                result[idx] = prefix + self._format_yaml_value(new_value) + comment
                return result, True

            nested_result, nested_changed = self._set_yaml_value(
                result[idx + 1:],
                keys,
                key_idx + 1,
                indent + 2,
                new_value,
            )
            if nested_changed:
                return result[:idx + 1] + list(nested_result), True
            return result, False
        return result, False

    def _modify_python_constant(
        self,
        content: str,
        python_path: str,
        new_value: Any,
    ) -> str:
        # Fast-path: module-level constant or rebound `Class.attr = X` line.
        # Matches both `tier_a_min = 0.65` and `StrategySettings.tier_a_min = 0.65`.
        pattern = re.compile(
            rf"^(?P<indent>\s*){re.escape(python_path)}\s*=\s*(?P<value>.+?)\s*$",
            re.MULTILINE,
        )

        def repl(match: re.Match[str]) -> str:
            indent = match.group("indent")
            return f"{indent}{python_path} = {self._format_python_value(new_value)}"

        modified, count = pattern.subn(repl, content, count=1)
        if count > 0:
            return modified

        if "." in python_path:
            subscript_modified = self._modify_python_subscript_assignment(
                content, python_path, new_value,
            )
            if subscript_modified is not None:
                return subscript_modified

            # P1-7: AST fallback for class attributes — `class Foo: bar = 0.65`
            # where the file line is just `    bar = 0.65` with no `Foo.` prefix.
            class_modified = self._modify_python_class_attribute(
                content, python_path, new_value,
            )
            if class_modified is not None:
                return class_modified

            dict_modified = self._modify_python_dict_literal(
                content, python_path, new_value,
            )
            if dict_modified is not None:
                return dict_modified

        raise ValueError(f"Python constant not found: {python_path}")

    def _modify_python_subscript_assignment(
        self,
        content: str,
        dotted_path: str,
        new_value: Any,
    ) -> str | None:
        """Patch `FOO["bar"] = value` for dotted path `FOO.bar`."""
        parts = dotted_path.split(".")
        if len(parts) != 2:
            return None
        root, key = parts
        pattern = re.compile(
            rf"^(?P<indent>\s*){re.escape(root)}\s*\[\s*"
            rf"(?P<quote>['\"]){re.escape(key)}(?P=quote)\s*\]\s*=\s*"
            rf"(?P<value>.*?)(?P<comment>\s+#.*)?(?P<newline>\r?\n?)$",
            re.MULTILINE,
        )

        def repl(match: re.Match[str]) -> str:
            quote = match.group("quote")
            comment = match.group("comment") or ""
            newline = match.group("newline") or ""
            return (
                f"{match.group('indent')}{root}[{quote}{key}{quote}] = "
                f"{self._format_python_value(new_value)}{comment}{newline}"
            )

        modified, count = pattern.subn(repl, content, count=1)
        return modified if count > 0 else None

    def _modify_python_class_attribute(
        self,
        content: str,
        dotted_path: str,
        new_value: Any,
    ) -> str | None:
        """Locate `class <Prefix>: <attr> = ...` and replace the assignment.

        Supports `ClassName.attr` (one-level) and `Outer.Inner.attr` (nested).
        Returns the modified content or None if no match.
        """
        parts = dotted_path.split(".")
        class_chain, attr_name = parts[:-1], parts[-1]
        if not class_chain:
            return None

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return None

        target = self._find_class_attribute_assignment(tree, class_chain, attr_name)
        if target is None:
            return None

        lines = content.splitlines(keepends=True)
        # ast line numbers are 1-indexed.
        line_idx = target.lineno - 1
        if line_idx < 0 or line_idx >= len(lines):
            return None
        original_line = lines[line_idx]

        # Preserve leading whitespace + optional trailing comment.
        match = re.match(
            rf"^(?P<indent>\s*){re.escape(attr_name)}\s*"
            rf"(?P<annotation>:\s*[^=]+?)?\s*=\s*"
            rf"(?P<value>.+?)\s*(?P<comment>#.*)?(?P<newline>\r?\n?)$",
            original_line,
        )
        if match is None:
            return None
        indent = match.group("indent")
        annotation = match.group("annotation") or ""
        comment = match.group("comment") or ""
        newline = match.group("newline") or ""
        rendered_value = self._format_python_value(new_value)
        if comment:
            new_line = f"{indent}{attr_name}{annotation} = {rendered_value}  {comment}{newline}"
        else:
            new_line = f"{indent}{attr_name}{annotation} = {rendered_value}{newline}"
        lines[line_idx] = new_line
        return "".join(lines)

    @staticmethod
    def _find_class_attribute_assignment(
        node: ast.AST,
        class_chain: list[str],
        attr_name: str,
    ) -> ast.AST | None:
        """Walk the AST to find `class <chain>: <attr> = ...`."""
        target_class_name = class_chain[0]
        remaining = class_chain[1:]
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef) and child.name == target_class_name:
                if remaining:
                    found = FileChangeGenerator._find_class_attribute_assignment(
                        child, remaining, attr_name,
                    )
                    if found is not None:
                        return found
                else:
                    for stmt in child.body:
                        if isinstance(stmt, ast.Assign):
                            for tgt in stmt.targets:
                                if isinstance(tgt, ast.Name) and tgt.id == attr_name:
                                    return stmt
                        elif isinstance(stmt, ast.AnnAssign):
                            if (
                                isinstance(stmt.target, ast.Name)
                                and stmt.target.id == attr_name
                            ):
                                return stmt
        return None

    def _modify_python_dict_literal(
        self,
        content: str,
        dotted_path: str,
        new_value: Any,
    ) -> str | None:
        """Patch `FOO = {"bar": value}` for dotted path `FOO.bar`."""
        parts = dotted_path.split(".")
        if len(parts) != 2:
            return None
        root, key = parts

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return None

        value_node = self._find_dict_literal_value(tree, root, key)
        if value_node is None:
            return None
        return self._replace_ast_node_source(
            content,
            value_node,
            self._format_python_value(new_value),
        )

    @staticmethod
    def _find_dict_literal_value(
        tree: ast.AST,
        root_name: str,
        key_name: str,
    ) -> ast.AST | None:
        for node in ast.walk(tree):
            value: ast.AST | None = None
            if isinstance(node, ast.Assign):
                if any(
                    isinstance(target, ast.Name) and target.id == root_name
                    for target in node.targets
                ):
                    value = node.value
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == root_name
            ):
                value = node.value

            if not isinstance(value, ast.Dict):
                continue
            for key_node, item_node in zip(value.keys, value.values, strict=False):
                if (
                    isinstance(key_node, ast.Constant)
                    and isinstance(key_node.value, str)
                    and key_node.value == key_name
                ):
                    return item_node
        return None

    @staticmethod
    def _replace_ast_node_source(content: str, node: ast.AST, replacement: str) -> str | None:
        if not all(
            hasattr(node, attr)
            for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset")
        ):
            return None
        lines = content.splitlines(keepends=True)
        start_line = node.lineno - 1
        end_line = node.end_lineno - 1
        if start_line < 0 or end_line >= len(lines):
            return None
        if start_line == end_line:
            line = lines[start_line]
            lines[start_line] = (
                line[:node.col_offset]
                + replacement
                + line[node.end_col_offset:]
            )
            return "".join(lines)

        return None

    def _modify_toml_field(
        self,
        content: str,
        toml_path: str,
        new_value: Any,
    ) -> str:
        """Update a dotted TOML path (e.g. `entry.ema_fast`) preserving formatting.

        Uses tomlkit so comments, blank lines, and key ordering survive.
        """
        if not toml_path:
            raise ValueError("toml_path is required")
        try:
            doc = tomlkit.parse(content)
        except Exception as exc:
            raise ValueError(f"Could not parse TOML: {exc}") from exc

        parts = toml_path.split(".")
        cursor: Any = doc
        for key in parts[:-1]:
            if key not in cursor:
                raise ValueError(f"TOML section not found: {key} (in {toml_path})")
            cursor = cursor[key]
        leaf = parts[-1]
        if leaf not in cursor:
            raise ValueError(f"TOML key not found: {toml_path}")
        cursor[leaf] = new_value
        return tomlkit.dumps(doc)

    def _modify_json_field(
        self,
        content: str,
        json_path: str,
        new_value: Any,
    ) -> str:
        """Update a dotted JSON path (e.g. `strategy.indicators.ema_fast`).

        Preserves the source file's indent and trailing newline. Bools, ints,
        floats, and strings are written as-is.
        """
        if not json_path:
            raise ValueError("json_path is required")
        try:
            doc = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse JSON: {exc}") from exc
        if not isinstance(doc, dict):
            raise ValueError("JSON root must be an object")

        parts = json_path.split(".")
        cursor: Any = doc
        for key in parts[:-1]:
            if not isinstance(cursor, dict) or key not in cursor:
                raise ValueError(f"JSON section not found: {key} (in {json_path})")
            cursor = cursor[key]
        leaf = parts[-1]
        if not isinstance(cursor, dict) or leaf not in cursor:
            raise ValueError(f"JSON key not found: {json_path}")
        cursor[leaf] = new_value

        indent = self._infer_json_indent(content)
        rendered = json.dumps(doc, indent=indent)
        if content.endswith("\n") and not rendered.endswith("\n"):
            rendered += "\n"
        return rendered

    @staticmethod
    def _infer_json_indent(content: str) -> int:
        """Infer indent width from the first indented line. Default to 2."""
        for line in content.splitlines():
            stripped = line.lstrip(" ")
            if stripped and stripped != line:
                return len(line) - len(stripped)
        return 2

    def _apply_unified_diff(
        self,
        original: str,
        patch: str,
        expected_file_path: str,
    ) -> str:
        lines = patch.splitlines()
        if len(lines) < 3:
            raise ValueError("Unified diff is too short")

        header_index = self._find_diff_header(lines)
        if header_index is None:
            raise ValueError("Unified diff headers not found")

        old_path = self._strip_prefix(lines[header_index], "--- ")
        new_path = self._strip_prefix(lines[header_index + 1], "+++ ")
        expected_variants = {
            expected_file_path,
            f"a/{expected_file_path}",
            f"b/{expected_file_path}",
        }
        if old_path not in expected_variants or new_path not in expected_variants:
            raise ValueError(
                f"Patch paths do not match expected file {expected_file_path}: "
                f"{old_path} -> {new_path}",
            )

        source_lines = original.splitlines()
        result: list[str] = []
        cursor = 0
        idx = header_index + 2

        while idx < len(lines):
            line = lines[idx]
            if not line:
                idx += 1
                continue
            if not line.startswith("@@"):
                raise ValueError(f"Expected hunk header, got: {line}")

            match = re.match(
                r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
                r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@",
                line,
            )
            if not match:
                raise ValueError(f"Malformed hunk header: {line}")

            old_start = int(match.group("old_start"))
            old_index = max(old_start - 1, 0)
            result.extend(source_lines[cursor:old_index])
            cursor = old_index
            idx += 1

            while idx < len(lines):
                hunk_line = lines[idx]
                if hunk_line.startswith("@@"):
                    break
                if hunk_line.startswith("\\"):
                    idx += 1
                    continue

                prefix = hunk_line[:1]
                payload = hunk_line[1:]
                if prefix == " ":
                    self._expect_source_line(source_lines, cursor, payload)
                    result.append(payload)
                    cursor += 1
                elif prefix == "-":
                    self._expect_source_line(source_lines, cursor, payload)
                    cursor += 1
                elif prefix == "+":
                    result.append(payload)
                else:
                    raise ValueError(f"Unexpected unified diff line: {hunk_line}")
                idx += 1

        result.extend(source_lines[cursor:])
        rebuilt = "\n".join(result)
        if original.endswith("\n"):
            rebuilt += "\n"
        return rebuilt

    @staticmethod
    def _find_diff_header(lines: list[str]) -> int | None:
        for idx in range(len(lines) - 1):
            if lines[idx].startswith("--- ") and lines[idx + 1].startswith("+++ "):
                return idx
        return None

    @staticmethod
    def _strip_prefix(line: str, prefix: str) -> str:
        if not line.startswith(prefix):
            raise ValueError(f"Expected line prefix {prefix!r}: {line}")
        return line[len(prefix):].strip()

    @staticmethod
    def _expect_source_line(source_lines: list[str], index: int, payload: str) -> None:
        if index >= len(source_lines):
            raise ValueError("Unified diff references lines past end of file")
        if source_lines[index] != payload:
            raise ValueError(
                f"Unified diff context mismatch at source line {index + 1}: "
                f"expected {source_lines[index]!r}, got {payload!r}",
            )

    @staticmethod
    def _format_yaml_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    @staticmethod
    def _format_python_value(value: Any) -> str:
        if isinstance(value, str):
            return repr(value)
        if isinstance(value, bool):
            return "True" if value else "False"
        return str(value)
