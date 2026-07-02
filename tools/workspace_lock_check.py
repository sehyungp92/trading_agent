"""Check that the root uv lockfile matches the Phase -1 workspace skeleton."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def main() -> int:
    if not shutil.which("uv"):
        print("FAIL uv - uv executable is not on PATH")
        return 1
    print(f"PASS uv - {run(['uv', '--version']).stdout.strip()}")

    pyproject_path = ROOT / "pyproject.toml"
    lock_path = ROOT / "uv.lock"
    if not pyproject_path.exists():
        print("FAIL pyproject.toml - missing")
        return 1
    with pyproject_path.open("rb") as handle:
        pyproject = tomllib.load(handle)

    requires_python = pyproject.get("project", {}).get("requires-python")
    if requires_python != ">=3.12":
        print(f"FAIL project.requires-python - expected >=3.12, got {requires_python!r}")
        return 1
    print("PASS project.requires-python - >=3.12")

    workspace = pyproject.get("tool", {}).get("uv", {}).get("workspace", {})
    members = workspace.get("members", [])
    if "packages/*" not in members or "bots/*" not in members:
        print("FAIL tool.uv.workspace - members must include packages/* and bots/*")
        return 1
    print(f"PASS tool.uv.workspace - {', '.join(members)}")

    if not lock_path.exists():
        print("FAIL uv.lock - missing; run `uv lock --python 3.12` after resolving metadata")
        return 1

    check = run(["uv", "lock", "--check", "--python", "3.12"])
    if check.returncode != 0:
        output = (check.stderr or check.stdout).strip()
        print("FAIL uv.lock - lockfile is missing or out of date")
        if output:
            print(output)
        return check.returncode
    print("PASS uv.lock - matches pyproject.toml for Python 3.12")
    return 0


if __name__ == "__main__":
    sys.exit(main())
