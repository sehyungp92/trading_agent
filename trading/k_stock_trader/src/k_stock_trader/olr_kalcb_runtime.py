"""Compatibility wrapper for the OLR/KALCB runtime-session CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["emit-deployment-metadata"]:
        return _emit_deployment_metadata(args[1:])
    from .runtime_session import main as runtime_session_main

    return int(runtime_session_main(argv))


def _emit_deployment_metadata(argv: list[str]) -> int:
    from deployment.olr_kalcb.deployment_metadata import emit_deployment_metadata

    parser = argparse.ArgumentParser(prog="k-stock-olr-kalcb-runtime emit-deployment-metadata")
    parser.add_argument("--contract", required=True)
    parser.add_argument("--effective-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--runtime-started-at-utc", required=True)
    parser.add_argument("--runtime-instance-id", required=True)
    ns = parser.parse_args(argv)
    repo_root = Path(ns.repo_root).resolve()
    effective = json.loads(Path(ns.effective_config).read_text(encoding="utf-8"))
    metadata = emit_deployment_metadata(
        ns.output,
        repo_root=repo_root,
        contract_path=ns.contract,
        mode="paper",
        strategy_ids=("OLR", "KALCB"),
        strategy_configs={"effective": effective},
        portfolio_policy_config=effective,
        strategy_artifacts={},
        deployment_id="k-stock-olr-kalcb",
        runtime_started_at_utc=ns.runtime_started_at_utc,
        runtime_entrypoint="k_stock_trader.olr_kalcb_runtime:main",
        runtime_instance_id=ns.runtime_instance_id,
        emission_environment="paper_vps",
        metadata_source="vps_live_bot_runtime_deployment_metadata_v1",
    )
    print(json.dumps({"deployment_metadata_path": ns.output, "strategy_id": metadata["strategy_id"]}, sort_keys=True))
    return 0


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
