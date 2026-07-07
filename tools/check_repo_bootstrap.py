"""Validate the Phase -1 repository/tooling bootstrap gate."""

from __future__ import annotations

import argparse
from collections import Counter
import re
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_BRANCH_PROTECTION_CHECKS = (
    "repo-bootstrap",
    "workspace-lock",
    "workspace-imports",
    "contracts",
    "baselines",
    "live-configs",
    "decision-parity",
    "optimizer-compatibility",
    "backtest-integrity",
    "deployment-gate",
    "deployment-metadata",
    "affected-images",
    "docker",
    "strict-refactor-acceptance",
)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def ok(label: str, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    print(f"PASS {label}{suffix}")


def fail(label: str, detail: str) -> None:
    print(f"FAIL {label} - {detail}")


def warn(label: str, detail: str) -> None:
    print(f"WARN {label} - {detail}")


def load_pyproject() -> dict:
    pyproject_path = ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError("missing pyproject.toml")
    with pyproject_path.open("rb") as handle:
        return tomllib.load(handle)


def check_git(strict_remote: bool) -> int:
    failures = 0
    inside = run(["git", "rev-parse", "--is-inside-work-tree"])
    if inside.returncode != 0 or inside.stdout.strip().lower() != "true":
        fail("git.repository", "root is not a Git worktree")
        return failures + 1
    top_level = run(["git", "rev-parse", "--show-toplevel"])
    resolved_top = Path(top_level.stdout.strip()).resolve() if top_level.stdout.strip() else None
    if resolved_top != ROOT:
        fail("git.repository", f"expected root {ROOT}, got {resolved_top}")
        failures += 1
    else:
        ok("git.repository", str(ROOT))

    for key in ("user.name", "user.email"):
        value = run(["git", "config", "--get", key]).stdout.strip()
        if value:
            ok(f"git.{key}", value)
        else:
            fail(f"git.{key}", "missing Git identity")
            failures += 1

    branch = run(["git", "branch", "--show-current"]).stdout.strip()
    if branch == "main":
        ok("git.branch", branch)
    else:
        fail("git.branch", f"expected main, got {branch or '<detached/none>'}")
        failures += 1

    remote = run(["git", "remote", "get-url", "origin"])
    remote_url = remote.stdout.strip()
    if not remote_url:
        fail("git.remote.origin", "missing root origin remote")
        failures += 1
    else:
        ok("git.remote.origin", remote_url)
        probe = run(["git", "ls-remote", "--exit-code", "origin", "HEAD"])
        if probe.returncode == 0:
            ok("git.remote.reachable", "origin HEAD is reachable")
        elif strict_remote:
            detail = (probe.stderr or probe.stdout).strip().splitlines()[-1:]
            fail("git.remote.reachable", detail[0] if detail else "origin HEAD is not reachable")
            failures += 1
        else:
            warn("git.remote.reachable", "origin HEAD is not reachable in non-strict mode")

    return failures


def check_files() -> int:
    failures = 0
    required_files = [
        ".github/workflows/ci.yml",
        ".gitattributes",
        ".gitignore",
        "docs/repo-bootstrap-decisions.md",
        "pyproject.toml",
        "tools/check_repo_bootstrap.py",
        "tools/workspace_lock_check.py",
        "tools/workspace_import_smoke.py",
        "tools/detect_affected_images.py",
        "uv.lock",
    ]
    for rel_path in required_files:
        path = ROOT / rel_path
        if path.exists():
            ok(f"file.{rel_path}")
        else:
            fail(f"file.{rel_path}", "missing")
            failures += 1
    return failures


def check_pyproject() -> int:
    failures = 0
    try:
        pyproject = load_pyproject()
    except Exception as exc:  # pragma: no cover - diagnostic path
        fail("pyproject.parse", str(exc))
        return 1

    project = pyproject.get("project", {})
    if project.get("requires-python") == ">=3.12":
        ok("pyproject.python", "live-image target >=3.12")
    else:
        fail("pyproject.python", "project.requires-python must be >=3.12")
        failures += 1

    workspace = pyproject.get("tool", {}).get("uv", {}).get("workspace", {})
    members = workspace.get("members", [])
    if "packages/*" in members and "bots/*" in members:
        ok("pyproject.uv.workspace", ", ".join(members))
    else:
        fail("pyproject.uv.workspace", "members must include packages/* and bots/*")
        failures += 1

    bootstrap = pyproject.get("tool", {}).get("trading_agent", {}).get("bootstrap", {})
    if bootstrap.get("ci_provider") == "github-actions":
        ok("bootstrap.ci_provider", "github-actions")
    else:
        fail("bootstrap.ci_provider", "GitHub Actions must be selected")
        failures += 1

    checks = bootstrap.get("branch_protection_required_checks", [])
    declared_checks = [item.strip() for item in checks if isinstance(item, str)]
    malformed_checks = len(declared_checks) != len(checks) or any(not item for item in declared_checks)
    duplicate_checks = sorted(
        item for item, count in Counter(declared_checks).items() if count > 1
    )
    expected_checks = set(REQUIRED_BRANCH_PROTECTION_CHECKS)
    declared_check_set = set(declared_checks)
    missing_checks = sorted(expected_checks.difference(declared_check_set))
    if malformed_checks:
        fail("bootstrap.branch_protection", "required checks must be non-empty strings")
        failures += 1
    elif duplicate_checks:
        fail("bootstrap.branch_protection", f"duplicate required checks: {duplicate_checks}")
        failures += 1
    elif missing_checks:
        fail("bootstrap.branch_protection", f"missing required checks: {missing_checks}")
        failures += 1
    else:
        ok("bootstrap.branch_protection", "required checks selected")
    workflow_jobs = workflow_job_names(ROOT / ".github" / "workflows" / "ci.yml")
    checks_for_ci = declared_check_set or expected_checks
    missing_workflow_jobs = sorted(checks_for_ci.difference(workflow_jobs))
    if missing_workflow_jobs:
        fail("bootstrap.ci_required_jobs", f"missing workflow jobs: {missing_workflow_jobs}")
        failures += 1
    else:
        ok("bootstrap.ci_required_jobs", "required checks are present in CI")

    policy_path = bootstrap.get("artifact_policy")
    if policy_path and (ROOT / policy_path).exists():
        ok("bootstrap.artifact_policy", policy_path)
    else:
        fail("bootstrap.artifact_policy", "artifact policy path is missing")
        failures += 1
    return failures


def workflow_job_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8")
    return {
        match.group(1)
        for match in re.finditer(r"(?m)^  ([A-Za-z0-9_-]+):\s*$", text)
        if match.group(1) != "steps"
    }


def check_artifact_policy() -> int:
    failures = 0
    policy = ROOT.joinpath("docs/repo-bootstrap-decisions.md").read_text(encoding="utf-8")
    required_phrases = [
        "Source control",
        "Git LFS",
        "Object storage",
        "Generated local-only outputs",
    ]
    for phrase in required_phrases:
        if phrase in policy:
            ok(f"artifact_policy.{phrase}")
        else:
            fail(f"artifact_policy.{phrase}", "missing storage class")
            failures += 1
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-unreachable-remote",
        action="store_true",
        help="Warn instead of failing when origin HEAD cannot be reached.",
    )
    args = parser.parse_args(argv)

    failures = 0
    failures += check_git(strict_remote=not args.allow_unreachable_remote)
    failures += check_files()
    failures += check_pyproject()
    failures += check_artifact_policy()

    if failures:
        print(f"\nBootstrap gate failed with {failures} issue(s).")
        return 1
    print("\nBootstrap gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
