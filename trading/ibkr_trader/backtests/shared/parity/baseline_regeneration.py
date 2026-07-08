from __future__ import annotations

import copy
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .diagnostic_baselines import (
    collect_baseline_snapshot,
    load_manifest,
    normalized_sha256_file,
    repo_root,
    sha256_file,
)

_OUTPUT_FLAGS = {
    "--output",
    "--json-output",
    "--summary-json",
    "--output-dir",
    "--mutations-output",
}


@dataclass(slots=True, frozen=True)
class RegenerationVerification:
    entry_id: str
    command: tuple[str, ...]
    sandbox_root: str
    artifact_path: str
    sha256: str
    expected_sha256: str
    metrics: dict[str, float]
    expected_metrics: dict[str, float]
    returncode: int
    stdout: str
    stderr: str


def verify_manifest_regeneration(
    *,
    manifest: dict[str, Any] | None = None,
    root: Path | None = None,
    python_executable: str | None = None,
    timeout_seconds: int = 3600,
    sandbox_root: Path | None = None,
    artifact_ids: Iterable[str] | None = None,
) -> list[RegenerationVerification]:
    base_root = Path(root or repo_root())
    manifest_data = manifest or load_manifest()
    requested_ids = list(artifact_ids or [])
    allowed_ids = set(requested_ids)
    selected = [
        entry
        for entry in manifest_data.get("artifacts", [])
        if not allowed_ids or entry.get("id") in allowed_ids
    ]
    if allowed_ids:
        found_ids = {entry.get("id") for entry in selected}
        missing_ids = sorted(allowed_ids - found_ids)
        if missing_ids:
            raise ValueError(
                "Unknown baseline artifact id(s): " + ", ".join(missing_ids)
            )

    if sandbox_root is not None:
        sandbox_root = Path(sandbox_root)
        sandbox_root.mkdir(parents=True, exist_ok=True)
        return [
            regenerate_manifest_entry(
                entry,
                root=base_root,
                sandbox_root=sandbox_root / str(entry["id"]),
                python_executable=python_executable,
                timeout_seconds=timeout_seconds,
            )
            for entry in selected
        ]

    with tempfile.TemporaryDirectory(prefix="baseline-regeneration-") as temp_dir:
        temp_root = Path(temp_dir)
        return [
            regenerate_manifest_entry(
                entry,
                root=base_root,
                sandbox_root=temp_root / str(entry["id"]),
                python_executable=python_executable,
                timeout_seconds=timeout_seconds,
            )
            for entry in selected
        ]


def regenerate_manifest_entry(
    entry: dict[str, Any],
    *,
    root: Path | None = None,
    sandbox_root: Path,
    python_executable: str | None = None,
    timeout_seconds: int = 3600,
) -> RegenerationVerification:
    base_root = Path(root or repo_root())
    sandbox = Path(sandbox_root)
    sandbox.mkdir(parents=True, exist_ok=True)

    command = build_regeneration_command(
        entry,
        root=base_root,
        sandbox_root=sandbox,
        python_executable=python_executable,
    )
    completed = subprocess.run(
        command,
        cwd=str(base_root),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Regeneration command failed for {entry['id']} with exit code {completed.returncode}.\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    sandbox_entry = _sandbox_manifest_entry(entry, sandbox_root=sandbox, root=base_root)
    snapshot = collect_baseline_snapshot(sandbox_entry, root=sandbox)
    artifact_path = sandbox / Path(entry["artifact_path"])
    expected_artifact_path = base_root / Path(entry["artifact_path"])
    comparison_sha = snapshot["sha256"]
    expected_comparison_sha = entry["sha256"]
    if snapshot["sha256"] != entry["sha256"]:
        comparison_sha = normalized_sha256_file(artifact_path)
        expected_comparison_sha = normalized_sha256_file(expected_artifact_path)
        if comparison_sha != expected_comparison_sha:
            raise AssertionError(
                f"Baseline regeneration hash drift for {entry['id']}: "
                f"expected {entry['sha256']}, got {snapshot['sha256']} "
                f"(normalized expected {expected_comparison_sha}, got {comparison_sha})"
            )
    if snapshot["metrics"] != entry["expected_metrics"]:
        # Use explicit float comparison here to make drift visible in failures.
        for key, expected in entry["expected_metrics"].items():
            actual = snapshot["metrics"].get(key)
            if actual != expected:
                raise AssertionError(
                    f"Baseline regeneration metric drift for {entry['id']} metric {key}: "
                    f"expected {expected}, got {actual}"
                )
    return RegenerationVerification(
        entry_id=entry["id"],
        command=tuple(command),
        sandbox_root=str(sandbox),
        artifact_path=str(artifact_path),
        sha256=comparison_sha,
        expected_sha256=expected_comparison_sha,
        metrics=dict(snapshot["metrics"]),
        expected_metrics=dict(entry["expected_metrics"]),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def build_regeneration_command(
    entry: dict[str, Any],
    *,
    root: Path | None = None,
    sandbox_root: Path,
    python_executable: str | None = None,
) -> list[str]:
    base_root = Path(root or repo_root())
    regeneration = entry["regeneration"]
    executor = regeneration["executor"]
    entrypoint = regeneration["entrypoint"]
    arguments = remap_regeneration_arguments(
        regeneration.get("arguments", []),
        sandbox_root=sandbox_root,
        root=base_root,
    )
    python_cmd = python_executable or sys.executable

    if executor == "python_file":
        script_path = base_root / Path(entrypoint)
        return [python_cmd, str(script_path), *arguments]
    if executor == "python_module":
        return [python_cmd, "-m", entrypoint, *arguments]
    if executor == "manual":
        raise ValueError(f"Cannot execute manual regeneration entry {entry['id']}")
    raise ValueError(f"Unsupported regeneration executor: {executor}")


def remap_regeneration_arguments(
    arguments: Sequence[str],
    *,
    sandbox_root: Path,
    root: Path | None = None,
    output_flags: Iterable[str] = _OUTPUT_FLAGS,
) -> list[str]:
    base_root = Path(root or repo_root())
    sandbox = Path(sandbox_root)
    remapped: list[str] = []
    output_flag_set = set(output_flags)
    pending_output_flag: str | None = None

    for argument in arguments:
        if pending_output_flag is not None:
            remapped_path = _sandbox_path(argument, root=base_root, sandbox_root=sandbox)
            if pending_output_flag == "--output-dir":
                remapped_path.mkdir(parents=True, exist_ok=True)
            else:
                remapped_path.parent.mkdir(parents=True, exist_ok=True)
            remapped.append(str(remapped_path))
            pending_output_flag = None
            continue

        inline_output_flag = next(
            (flag for flag in output_flag_set if argument.startswith(f"{flag}=")),
            None,
        )
        if inline_output_flag is not None:
            _, raw_value = argument.split("=", 1)
            remapped_path = _sandbox_path(raw_value, root=base_root, sandbox_root=sandbox)
            if inline_output_flag == "--output-dir":
                remapped_path.mkdir(parents=True, exist_ok=True)
            else:
                remapped_path.parent.mkdir(parents=True, exist_ok=True)
            remapped.append(f"{inline_output_flag}={remapped_path}")
            continue

        remapped.append(argument)
        if argument in output_flag_set:
            pending_output_flag = argument

    if pending_output_flag is not None:
        raise ValueError(f"Output flag {pending_output_flag} is missing a value")
    return remapped


def _sandbox_manifest_entry(entry: dict[str, Any], *, sandbox_root: Path, root: Path) -> dict[str, Any]:
    sandbox_entry = copy.deepcopy(entry)
    artifact_path = _sandbox_path(entry["artifact_path"], root=root, sandbox_root=sandbox_root)
    sandbox_entry["artifact_path"] = str(artifact_path.relative_to(sandbox_root)).replace("\\", "/")

    summary_source = sandbox_entry.get("summary_source")
    if isinstance(summary_source, dict) and summary_source.get("path"):
        summary_path = _sandbox_path(summary_source["path"], root=root, sandbox_root=sandbox_root)
        summary_source["path"] = str(summary_path.relative_to(sandbox_root)).replace("\\", "/")

    regeneration = sandbox_entry.get("regeneration")
    if isinstance(regeneration, dict) and regeneration.get("expected_output"):
        expected_output = _sandbox_path(regeneration["expected_output"], root=root, sandbox_root=sandbox_root)
        regeneration["expected_output"] = str(expected_output.relative_to(sandbox_root)).replace("\\", "/")
    return sandbox_entry


def _sandbox_path(raw_path: str, *, root: Path, sandbox_root: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            relative = Path(candidate.name)
    else:
        relative = candidate
    return sandbox_root / relative


def copy_regenerated_outputs(
    verification_results: Sequence[RegenerationVerification],
    *,
    destination_root: Path,
) -> list[Path]:
    destination = Path(destination_root)
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for result in verification_results:
        source = Path(result.artifact_path)
        target = destination / Path(result.entry_id) / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target)
    return copied
