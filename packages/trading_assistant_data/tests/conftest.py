from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONOREPO_ROOT = PROJECT_ROOT.parent.parent if PROJECT_ROOT.parent.name == "packages" else PROJECT_ROOT.parent
CONTROL_WORKSPACE = (
    MONOREPO_ROOT / "packages" / "trading_assistant"
    if (MONOREPO_ROOT / "packages" / "trading_assistant").exists()
    else MONOREPO_ROOT / "trading_assistant"
)
CONTROL_IMPORT_ROOT = (
    CONTROL_WORKSPACE / "src"
    if (CONTROL_WORKSPACE / "src" / "trading_assistant").exists()
    else CONTROL_WORKSPACE
)

sys.path.insert(0, str(PROJECT_ROOT / "src"))
if CONTROL_IMPORT_ROOT.exists():
    sys.path.insert(0, str(CONTROL_IMPORT_ROOT))
