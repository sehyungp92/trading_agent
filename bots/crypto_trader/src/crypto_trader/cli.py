"""CLI entry point for crypto_trader."""

from __future__ import annotations

import json
import hashlib
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import click
import structlog


def _configure_logging() -> None:
    """Configure structlog with JSON output."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
    )


_PUBLIC_LIVE_CONFIG_HASH_EXCLUDED_FIELDS = {
    "wallet_address",
    "private_key",
    "relay_url",
    "relay_secret",
    "postgres_dsn",
}


@click.group()
def cli() -> None:
    """Crypto Trader — Hyperliquid perpetual futures trading system."""
    _configure_logging()


@cli.command()
@click.option("--coin", required=True, help="Comma-separated coin symbols (e.g. BTC,ETH)")
@click.option("--interval", default="15m", help="Comma-separated intervals (e.g. 15m,1h,4h)")
@click.option("--days", default=90, type=int, help="Number of days of history to download")
@click.option("--data-dir", default="data", type=click.Path(), help="Base data directory")
@click.option("--include-funding/--no-funding", default=True, help="Download funding rates")
def download(coin: str, interval: str, days: int, data_dir: str, include_funding: bool) -> None:
    """Download historical candle and funding data from Hyperliquid."""
    from crypto_trader.data.downloader import HyperliquidDownloader
    from crypto_trader.data.store import ParquetStore

    log = structlog.get_logger()

    store = ParquetStore(base_dir=Path(data_dir))
    downloader = HyperliquidDownloader(store=store)

    coins = [c.strip().upper() for c in coin.split(",")]
    intervals = [i.strip() for i in interval.split(",")]

    log.info(
        "download.start",
        coins=coins,
        intervals=intervals,
        days=days,
        include_funding=include_funding,
    )

    for c in coins:
        for iv in intervals:
            log.info("download.candles", coin=c, interval=iv)
            try:
                downloader.download_and_store(c, iv, days=days)
            except Exception:
                log.exception("download.candles.failed", coin=c, interval=iv)

        if include_funding:
            log.info("download.funding", coin=c)
            try:
                downloader.download_and_store_funding(c, days=days)
            except Exception:
                log.exception("download.funding.failed", coin=c)

    log.info("download.complete", coins=len(coins), intervals=len(intervals))


def _build_strategy_config(strategy: str, config_path: str | None, raw: dict | None = None):
    """Build strategy config from type and optional raw dict."""
    if strategy == "trend":
        from crypto_trader.strategy.trend.config import TrendConfig
        if raw:
            return TrendConfig.from_dict(raw.get("strategy", {}))
        return TrendConfig()
    elif strategy == "breakout":
        from crypto_trader.strategy.breakout.config import BreakoutConfig
        if raw:
            return BreakoutConfig.from_dict(raw.get("strategy", {}))
        return BreakoutConfig()
    else:
        from crypto_trader.strategy.momentum.config import MomentumConfig
        if raw:
            return MomentumConfig.from_dict(raw.get("strategy", {}))
        return MomentumConfig()


@cli.command()
@click.option("--config", "config_path", default=None, type=click.Path(exists=True), help="YAML config file")
@click.option("--start-date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", required=True, help="End date (YYYY-MM-DD)")
@click.option("--symbols", default="BTC,ETH,SOL", help="Comma-separated symbols")
@click.option("--output-dir", default="output", type=click.Path(), help="Output directory")
@click.option("--data-dir", default="data", type=click.Path(), help="Data directory")
@click.option("--equity", default=10000.0, type=float, help="Initial equity")
@click.option("--walk-forward", is_flag=True, help="Run walk-forward analysis")
@click.option("--strategy", default="momentum", type=click.Choice(["momentum", "trend", "breakout"]),
              help="Strategy type")
@click.option("--warmup-days", default=None, type=int, help="Extra days before start for indicator warmup")
def backtest(
    config_path: str | None,
    start_date: str,
    end_date: str,
    symbols: str,
    output_dir: str,
    data_dir: str,
    equity: float,
    walk_forward: bool,
    strategy: str,
    warmup_days: int | None,
) -> None:
    """Run a backtest with the specified strategy."""
    from datetime import date

    import yaml

    from crypto_trader.backtest.analysis import (
        export_equity_curve,
        export_trade_journal,
        generate_report,
        print_summary,
    )
    from crypto_trader.backtest.profiles import (
        LIVE_PARITY_PROFILE,
        build_backtest_config_from_profile,
    )
    from crypto_trader.backtest.runner import run, run_walk_forward

    log = structlog.get_logger()

    # Build strategy config
    raw = None
    if config_path:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
    strategy_cfg = _build_strategy_config(strategy, config_path, raw)

    sym_list = [s.strip().upper() for s in symbols.split(",")]
    strategy_cfg.symbols = sym_list

    bt_cfg = build_backtest_config_from_profile(
        profile=LIVE_PARITY_PROFILE,
        symbols=sym_list,
        start_date=date.fromisoformat(start_date),
        end_date=date.fromisoformat(end_date),
        initial_equity=equity,
        warmup_days=warmup_days,
    )

    out = Path(output_dir)

    if walk_forward:
        log.info("backtest.walk_forward", train_pct=bt_cfg.train_pct)
        wf_result = run_walk_forward(strategy_cfg, bt_cfg, data_dir=Path(data_dir),
                                     strategy_type=strategy)
        print("\n--- TRAIN ---")
        print_summary(wf_result.train)
        generate_report(wf_result.train, out / "train")
        print("\n--- TEST ---")
        print_summary(wf_result.test)
        generate_report(wf_result.test, out / "test")
    else:
        result = run(strategy_cfg, bt_cfg, data_dir=Path(data_dir), strategy_type=strategy)
        print_summary(result)
        generate_report(result, out)
        export_equity_curve(result, out)
        export_trade_journal(result, out)
        log.info("backtest.complete", output_dir=str(out))


def _detect_next_round(base_dir: Path) -> int:
    """Scan for round_N/ dirs and return max+1 (or 1 if none)."""
    max_round = 0
    if base_dir.is_dir():
        for child in base_dir.iterdir():
            if child.is_dir():
                m = re.match(r"^round_(\d+)$", child.name)
                if m:
                    max_round = max(max_round, int(m.group(1)))
    return max_round + 1


def _update_rounds_manifest(
    base_dir: Path,
    round_num: int,
    mutations: dict,
    metrics: dict | None,
    *,
    contract: dict | None = None,
    phase_result: dict | None = None,
    gate_result: dict | None = None,
    reject_reason: str = "",
) -> None:
    """Append round entry to rounds_manifest.json."""
    from crypto_trader.optimize.phase_state import _atomic_write_json

    manifest_path = base_dir / "rounds_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"rounds": []}
    manifest.setdefault("rounds", [])
    try:
        manifest["schema_version"] = max(int(manifest.get("schema_version", 1)), 3)
    except (TypeError, ValueError):
        manifest["schema_version"] = 3

    # Replace entry for same round_num if re-running
    manifest["rounds"] = [
        r for r in manifest["rounds"] if r.get("round") != round_num
    ]

    if contract is None:
        metadata_path = base_dir / f"round_{round_num}" / "optimized_config.json"
        if metadata_path.exists():
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata = payload.get("metadata", {})
                contract = metadata.get("contract") or {
                    "contract_hash": metadata.get("contract_hash", ""),
                    "profile_hash": metadata.get("profile_hash", ""),
                    "strategy_config_hash": metadata.get("strategy_config_hash", ""),
                    "portfolio_config_hash": metadata.get("portfolio_config_hash", ""),
                    "data_window": metadata.get("data_window", {}),
                    "data_fingerprint": metadata.get("data_fingerprint", {}),
                    "symbols": metadata.get("symbols", []),
                    "required_timeframes": metadata.get("required_timeframes", []),
                }
            except (OSError, json.JSONDecodeError):
                contract = {}
        else:
            contract = {}
    phase_result = phase_result or {}
    gate_result = gate_result or {}
    if not phase_result or not gate_result:
        state_path = base_dir / f"round_{round_num}" / "phase_state.json"
        if state_path.exists():
            try:
                state_payload = json.loads(state_path.read_text(encoding="utf-8"))
                phase_results = state_payload.get("phase_results", {})
                gate_results = state_payload.get("phase_gate_results", {})
                phase_ids = sorted(int(key) for key in phase_results)
                if phase_ids:
                    last_phase = str(phase_ids[-1])
                    phase_result = phase_result or phase_results.get(last_phase, {})
                    gate_result = gate_result or gate_results.get(last_phase, {})
            except (OSError, json.JSONDecodeError, ValueError):
                pass
    failure_reasons = gate_result.get("failure_reasons") or []
    gate_passed = gate_result.get("passed")
    if reject_reason:
        manifest_reject_reason = reject_reason
    elif failure_reasons:
        manifest_reject_reason = "; ".join(str(reason) for reason in failure_reasons)
    else:
        manifest_reject_reason = ""

    entry: dict = {
        "round": round_num,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mutations_count": len(mutations),
        "mutations": mutations,
        "score": phase_result.get("final_score"),
        "gate_status": (
            "passed" if gate_passed is True
            else "failed" if gate_passed is False
            else "unknown"
        ),
        "gate_passed": gate_passed,
        "gate_failure_reasons": failure_reasons,
        "reject_reason": manifest_reject_reason,
        "contract_hash": contract.get("contract_hash", ""),
        "profile_hash": contract.get("profile_hash", ""),
        "strategy_config_hash": contract.get("strategy_config_hash", ""),
        "portfolio_config_hash": contract.get("portfolio_config_hash", ""),
        "data_window": contract.get("data_window", {}),
        "symbols": contract.get("symbols", []),
        "required_timeframes": contract.get("required_timeframes", []),
        "data_fingerprint": contract.get("data_fingerprint", {}),
        "economic_profile": contract.get("economic_profile", {}),
        "contract": contract,
        "metrics": metrics or {},
        "gate_result": gate_result,
        "final_validation": phase_result.get("final_validation", {}),
        "accepted_count": phase_result.get("accepted_count"),
        "new_mutations": phase_result.get("new_mutations", {}),
    }
    if metrics:
        for k in [
            "total_trades",
            "win_rate",
            "profit_factor",
            "max_drawdown_pct",
            "sharpe_ratio",
            "calmar_ratio",
            "net_return_pct",
            "expectancy_r",
            "exit_efficiency",
            "realized_pnl_net",
            "terminal_mark_pnl_net",
            "net_profit",
            "total_fees",
            "funding_cost_total",
            "terminal_mark_count",
        ]:
            if k in metrics:
                entry[k] = metrics[k]

    manifest["rounds"].append(entry)
    manifest["rounds"].sort(key=lambda r: r["round"])

    _atomic_write_json(manifest, manifest_path)


@cli.command()
@click.option("--config", "config_path", default=None, type=click.Path(exists=True), help="YAML config file")
@click.option("--start-date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", required=True, help="End date (YYYY-MM-DD)")
@click.option("--symbols", default="BTC,ETH,SOL", help="Comma-separated symbols")
@click.option("--output-dir", default=None, type=click.Path(), help="Output directory (default: output/{strategy})")
@click.option("--data-dir", default="data", type=click.Path(), help="Data directory")
@click.option("--equity", default=10000.0, type=float, help="Initial equity")
@click.option("--phase", "phase_num", default=None, type=int, help="Run only phase N (default: all)")
@click.option("--resume", is_flag=True, help="Resume from phase_state.json checkpoint")
@click.option("--workers", default=None, type=int, help="Parallel workers (default: cpu_count - 1)")
@click.option("--round", "round_num", default=None, type=int,
              help="Load round N's optimized config as baseline and start round N+1")
@click.option("--strategy", default="momentum", type=click.Choice(["momentum", "trend", "breakout"]),
              help="Strategy type")
@click.option("--warmup-days", default=None, type=int, help="Extra days before start for indicator warmup")
@click.option("--validation-mode", default="strict", type=click.Choice(["strict", "fast", "dev"]),
              help="Final validation mode; strict refuses fallback metrics")
def optimize(
    config_path: str | None,
    start_date: str,
    end_date: str,
    symbols: str,
    output_dir: str,
    data_dir: str,
    equity: float,
    phase_num: int | None,
    resume: bool,
    workers: int | None,
    round_num: int | None,
    strategy: str,
    warmup_days: int | None,
    validation_mode: str,
) -> None:
    """Run phased auto-optimization of the specified strategy."""
    from datetime import date

    import yaml

    from crypto_trader.backtest.profiles import (
        LIVE_PARITY_PROFILE,
        build_backtest_config_from_profile,
    )
    from crypto_trader.optimize.contracts import (
        build_optimization_contract,
        run_optimization_preflight,
    )
    from crypto_trader.optimize.phase_runner import PhaseRunner
    from crypto_trader.optimize.phase_state import PhaseState

    log = structlog.get_logger()

    # Default output dir is strategy-specific
    if output_dir is None:
        output_dir = f"output/{strategy}"
    base_out = Path(output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    # Build strategy config — --round overrides --config
    if round_num is not None and config_path:
        log.warning("optimize.config_ignored", reason="--round overrides --config")

    if config_path and round_num is None:
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        strategy_cfg = _build_strategy_config(strategy, config_path, raw)
    else:
        strategy_cfg = _build_strategy_config(strategy, None)

    # Load previous round's optimized config if --round specified
    if round_num is not None:
        prev_config_path = base_out / f"round_{round_num}" / "optimized_config.json"
        if not prev_config_path.exists():
            raise click.ClickException(
                f"No optimized config found at {prev_config_path}. "
                f"Round {round_num} must complete before starting round {round_num + 1}."
            )
        with open(prev_config_path, encoding="utf-8") as f:
            prev_raw = json.load(f)
        strategy_cfg = _build_strategy_config(strategy, None, prev_raw)
        next_round = round_num + 1
        log.info("optimize.loaded_round", source_round=round_num, next_round=next_round)
    else:
        next_round = _detect_next_round(base_out)

    # Round output directory
    round_dir = base_out / f"round_{next_round}"
    round_dir.mkdir(parents=True, exist_ok=True)

    sym_list = [s.strip().upper() for s in symbols.split(",")]
    strategy_cfg.symbols = sym_list

    bt_cfg = build_backtest_config_from_profile(
        profile=LIVE_PARITY_PROFILE,
        symbols=sym_list,
        start_date=date.fromisoformat(start_date),
        end_date=date.fromisoformat(end_date),
        initial_equity=equity,
        warmup_days=warmup_days,
    )

    # Create strategy-specific plugin
    if strategy == "trend":
        from crypto_trader.optimize.trend_plugin import TrendPlugin
        plugin = TrendPlugin(bt_cfg, strategy_cfg, data_dir=Path(data_dir), max_workers=workers)
    elif strategy == "breakout":
        from crypto_trader.optimize.breakout_plugin import BreakoutPlugin
        plugin = BreakoutPlugin(bt_cfg, strategy_cfg, data_dir=Path(data_dir), max_workers=workers)
    else:
        from crypto_trader.optimize.momentum_plugin import MomentumPlugin
        plugin = MomentumPlugin(bt_cfg, strategy_cfg, data_dir=Path(data_dir), max_workers=workers)

    contract = build_optimization_contract(
        strategy_type=strategy,
        strategy_config=strategy_cfg,
        backtest_config=bt_cfg,
        data_dir=Path(data_dir),
        profile=LIVE_PARITY_PROFILE,
        plugin=plugin,
    )
    run_optimization_preflight(
        contract=contract,
        backtest_config=bt_cfg,
        data_dir=Path(data_dir),
        output_dir=round_dir,
        profile=LIVE_PARITY_PROFILE,
        validation_mode=validation_mode,
    )
    runner = PhaseRunner(
        plugin,
        round_dir,
        contract=contract,
        validation_mode=validation_mode,
    )

    # Load or create state
    state_path = round_dir / "phase_state.json"
    if resume and state_path.exists():
        state = PhaseState.load(state_path)
        log.info("optimize.resumed", current_phase=state.current_phase, round=next_round)
    else:
        state = PhaseState(_path=state_path)
    if not state.completed_phases and not state.cumulative_mutations:
        state.ensure_contract(contract, strict=validation_mode == "strict")
        initial = plugin.initial_mutations
        if initial:
            state.cumulative_mutations.update(initial)
    else:
        state.ensure_contract(contract, strict=validation_mode == "strict")
    state.save(state_path)

    # Run
    log.info("optimize.start", round=next_round, output_dir=str(round_dir))
    if phase_num is not None:
        log.info("optimize.single_phase", phase=phase_num)
        runner.run_phase(phase_num, state)
    else:
        log.info("optimize.all_phases")
        runner.run_all_phases(state)

    # Update manifest
    final_metrics = None
    if state.phase_metrics:
        last_phase = max(state.phase_metrics.keys())
        final_metrics = state.phase_metrics[last_phase]
        phase_result = state.phase_results.get(last_phase, {})
        gate_result = state.phase_gate_results.get(last_phase, {})
    else:
        phase_result = {}
        gate_result = {}

    _update_rounds_manifest(
        base_out,
        next_round,
        state.cumulative_mutations,
        final_metrics,
        contract=contract,
        phase_result=phase_result,
        gate_result=gate_result,
    )

    # Summary
    print(f"\n=== Optimization Complete (Round {next_round}) ===")
    print(f"Output: {round_dir}")
    print(f"Completed phases: {state.completed_phases}")
    print(f"Total mutations: {len(state.cumulative_mutations)}")
    for key, val in state.cumulative_mutations.items():
        print(f"  {key} = {val}")

    if final_metrics:
        print(f"\nFinal metrics:")
        for k in ["total_trades", "win_rate", "profit_factor", "max_drawdown_pct",
                   "sharpe_ratio", "calmar_ratio"]:
            print(f"  {k}: {final_metrics.get(k, 0):.2f}")


@cli.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True),
              help="Live config JSON file")
def paper(config_path: str) -> None:
    """Run paper trading on Hyperliquid testnet."""
    import asyncio

    from crypto_trader.live.config import LiveConfig
    from crypto_trader.live.engine import LiveEngine
    from crypto_trader.live.execution_adapter import HyperliquidExecutionAdapter
    from crypto_trader.live.parity_warnings import (
        collect_live_parity_warnings,
        should_block_live_startup,
    )
    from crypto_trader.portfolio.config import PortfolioConfig

    log = structlog.get_logger()

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    config = LiveConfig.from_dict(data)

    errors = config.validate()
    if errors:
        for e in errors:
            log.error("config.validation", error=e)
        raise click.ClickException("Invalid configuration. See errors above.")

    path_errors = _live_config_path_errors(config)
    if path_errors:
        for e in path_errors:
            log.error("config.path_validation", error=e)
        raise click.ClickException("Live configuration preflight failed. See errors above.")

    portfolio_cfg = None
    if config.portfolio_config_path and config.portfolio_config_path.exists():
        with open(config.portfolio_config_path, "r", encoding="utf-8") as f:
            portfolio_cfg = PortfolioConfig.from_dict(json.load(f))

    strategy_cfgs = {}
    for strategy_id, path in config.strategy_configs.items():
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                strategy_cfgs[strategy_id] = _build_strategy_config(
                    strategy_id,
                    str(path),
                    json.load(f),
                )

    parity_warnings = collect_live_parity_warnings(
        config,
        portfolio_cfg,
        durable_oms_available=True,
        exchange_metadata_enforced=config.asset_meta_path is not None,
        strategy_configs=strategy_cfgs,
        capabilities=HyperliquidExecutionAdapter.capabilities,
    )
    for warning in parity_warnings:
        log.warning("config.live_parity_warning", **warning.to_dict())
    if should_block_live_startup(parity_warnings, config):
        raise click.ClickException("Live parity warnings must be resolved before this run.")

    engine = LiveEngine(config)

    log.info("paper.starting", testnet=config.is_testnet)
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        log.info("paper.interrupted")


@cli.command("deployment-preflight")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True),
              help="Mounted live config JSON file")
@click.option("--effective-config", "effective_config_path", required=True, type=click.Path(exists=True),
              help="Generated effective config artifact")
@click.option("--runtime-root", default=".", type=click.Path(),
              help="Runtime root used to resolve config paths")
def deployment_preflight(config_path: str, effective_config_path: str, runtime_root: str) -> None:
    """Validate the mounted VPS config before starting the crypto runtime."""
    with open(config_path, "r", encoding="utf-8") as f:
        config_payload = json.load(f)
    with open(effective_config_path, "r", encoding="utf-8") as f:
        effective_payload = json.load(f)

    errors = _deployment_preflight_errors(
        config_payload,
        effective_payload,
        runtime_root=Path(runtime_root),
    )
    result = {"valid": not errors, "errors": errors}
    click.echo(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise click.ClickException("Crypto deployment preflight failed")


@cli.command("emit-deployment-metadata")
@click.option("--effective-config", required=True, type=click.Path(exists=True))
@click.option("--contract-source-root", required=True, type=click.Path(exists=True))
@click.option("--contract-work-root", required=True, type=click.Path())
@click.option("--state-dir", required=True, type=click.Path())
@click.option("--repo-root", default=".", type=click.Path(exists=True))
@click.option("--runtime-started-at-utc", required=True)
def emit_deployment_metadata(
    effective_config: str,
    contract_source_root: str,
    contract_work_root: str,
    state_dir: str,
    repo_root: str,
    runtime_started_at_utc: str,
) -> None:
    """Emit approval metadata through the runtime engine without exchange credentials."""
    import shutil
    import subprocess

    from crypto_trader.instrumentation.lineage import stable_hash
    from crypto_trader.live.engine import LiveEngine
    from crypto_trader.portfolio.config import PortfolioConfig

    root = Path(repo_root).resolve()
    effective = json.loads(Path(effective_config).read_text(encoding="utf-8"))
    source_root = Path(contract_source_root)
    work_root = Path(contract_work_root)
    if work_root.exists():
        shutil.rmtree(work_root)
    for contract_path in source_root.glob("*/strategy_plugin_contract.json"):
        target = work_root / contract_path.parent.name
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(contract_path, target / "strategy_plugin_contract.json")

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()

    state_dir_path = Path(state_dir)

    class Config:
        bot_id = "crypto"
        is_testnet = True
        state_dir = state_dir_path

    class Lineage:
        code_sha = git("rev-parse", "HEAD")
        portfolio_id = "crypto_portfolio"
        strategy_config_versions = {
            "trend": str(effective.get("effective_config_hash") or ""),
            "momentum": str(effective.get("effective_config_hash") or ""),
            "breakout": str(effective.get("effective_config_hash") or ""),
        }
        deployment_id = f"crypto-{code_sha[:12]}"
        portfolio_config_version = str(effective.get("effective_config_hash") or "")
        risk_config_version = str(effective.get("effective_config_hash") or "")
        allocation_version = str(effective.get("effective_config_hash") or "")
        symbol_universe: list[str] = []
        deployment_manifest_version = str(effective.get("effective_config_hash") or "")

    class EngineStub:
        _config = Config()
        _lineage = Lineage()

        def _git_remote_url(self) -> str:
            return git("remote", "get-url", "origin")

        def _git_worktree_clean(self) -> bool:
            return git("status", "--porcelain", "--untracked-files=all") == ""

        def _package_version(self) -> str:
            return "acceptance"

        def _runtime_host_fingerprint(self) -> str:
            return stable_hash({"repo": str(root), "runtime": "crypto"}, length=32)

        def _file_sha256(self, path: Path) -> str:
            import hashlib

            return hashlib.sha256(path.read_bytes()).hexdigest()

        def _bridge_contract_root(self) -> Path:
            return work_root

        def _emission_environment(self) -> str:
            return "paper_vps"

        def _runtime_instance_id_value(self) -> str:
            return f"crypto:{self._lineage.code_sha[:12]}"

        def _deployment_metadata_blockers(self, **kwargs) -> list[str]:
            return [] if kwargs.get("contract_hash") else ["contract_hash_missing"]

    strategy_configs = {"trend": effective, "momentum": effective, "breakout": effective}
    artifacts = LiveEngine._emit_deployment_metadata_artifacts(  # noqa: SLF001
        EngineStub(),
        strategy_configs,
        PortfolioConfig(),
        started_at=runtime_started_at_utc,
    )
    print(json.dumps({"deployment_metadata_paths": artifacts}, sort_keys=True))


def _live_config_path_errors(
    config,
    *,
    runtime_root: Path | str | None = None,
    require_deployment_manifest: bool = True,
) -> list[str]:
    """Return runtime config path errors before LiveEngine can fall back to defaults."""
    root = Path.cwd() if runtime_root is None else Path(runtime_root)
    errors: list[str] = []
    manifest_path = getattr(config, "deployment_manifest_path", None)

    if config.portfolio_config_path is None:
        errors.append("portfolio_config_path is required")
    elif not _runtime_path(config.portfolio_config_path, root).exists():
        errors.append(f"portfolio_config_path does not exist: {config.portfolio_config_path}")

    if manifest_path is not None and not _runtime_path(manifest_path, root).exists():
        errors.append(f"deployment_manifest_path does not exist: {manifest_path}")

    if not config.strategy_configs:
        errors.append("at least one strategy config path is required")
    for strategy_id, path in config.strategy_configs.items():
        if not _runtime_path(path, root).exists():
            errors.append(f"strategy_configs.{strategy_id} does not exist: {path}")

    if config.asset_meta_path is not None and not _runtime_path(config.asset_meta_path, root).exists():
        errors.append(f"asset_meta_path does not exist: {config.asset_meta_path}")
    if config.asset_meta_path is not None:
        errors.extend(_asset_meta_path_errors(config, root))

    if require_deployment_manifest:
        if manifest_path is None:
            errors.append("deployment_manifest_path is required for deployment bundle preflight")
        elif _runtime_path(manifest_path, root).exists():
            errors.extend(_live_config_deployment_manifest_errors(config, root, _runtime_path(manifest_path, root)))

    return errors


def _deployment_preflight_errors(
    config_payload: dict,
    effective_payload: dict,
    *,
    runtime_root: Path,
) -> list[str]:
    from crypto_trader.live.config import LiveConfig

    errors: list[str] = []
    config = LiveConfig.from_dict(config_payload)
    errors.extend(config.validate())
    errors.extend(_live_config_path_errors(config, runtime_root=runtime_root))

    materialized = effective_payload.get("materialized_config")
    contract = materialized.get("runtime_config_contract") if isinstance(materialized, dict) else None
    if not isinstance(contract, dict):
        errors.append("effective config missing materialized_config.runtime_config_contract")
        return errors
    if contract.get("schema_version") != "crypto_runtime_config_contract.v1":
        errors.append("runtime_config_contract schema_version must be crypto_runtime_config_contract.v1")
    if contract.get("sidecar_forwarding_required") is not True:
        errors.append("runtime_config_contract.sidecar_forwarding_required must be true")

    for field in contract.get("required_non_empty_fields", []):
        value = config_payload.get(str(field))
        if not str(value or "").strip():
            errors.append(f"{field} is required by runtime_config_contract")

    expected_hash = str(contract.get("public_live_config_sha256") or "")
    actual_hash = _live_config_public_hash(config_payload)
    if expected_hash != actual_hash:
        errors.append("mounted live config public hash does not match generated effective config")

    expected_mount = "config/live_config.json"
    mounted_path = str(contract.get("mounted_config_path") or "")
    if mounted_path and Path(mounted_path) != Path(expected_mount):
        errors.append(f"runtime_config_contract.mounted_config_path must be {expected_mount}")
    return errors


def _live_config_public_hash(config_payload: dict) -> str:
    public_payload = {
        key: value
        for key, value in config_payload.items()
        if key not in _PUBLIC_LIVE_CONFIG_HASH_EXCLUDED_FIELDS
    }
    return _canonical_json_sha256(public_payload)


def _canonical_json_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    ).hexdigest()


def _runtime_path(path: Path, runtime_root: Path) -> Path:
    return path if path.is_absolute() else runtime_root / path


def _asset_meta_path_errors(config, runtime_root: Path) -> list[str]:
    path = _runtime_path(config.asset_meta_path, runtime_root)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"asset_meta_path could not be read: {exc}"]
    if not isinstance(payload, dict):
        return ["asset_meta_path must contain a JSON object"]

    errors: list[str] = []
    required_maps = ("asset_index", "tick_sizes", "lot_sizes")
    for key in required_maps:
        if not isinstance(payload.get(key), dict):
            errors.append(f"asset_meta_path.{key} must be an object")
    if errors:
        return errors

    symbols = [str(symbol) for symbol in getattr(config, "symbols", [])]
    for key in required_maps:
        values = payload[key]
        missing = sorted(symbol for symbol in symbols if symbol not in values)
        if missing:
            errors.append(f"asset_meta_path.{key} missing symbols: {', '.join(missing)}")
            continue
        if key == "asset_index":
            continue
        invalid = []
        for symbol in symbols:
            try:
                parsed = float(values[symbol])
            except (TypeError, ValueError):
                invalid.append(symbol)
                continue
            if not math.isfinite(parsed) or parsed <= 0.0:
                invalid.append(symbol)
        if invalid:
            errors.append(f"asset_meta_path.{key} has invalid positive numeric values for: {', '.join(invalid)}")
    return errors


def _live_config_deployment_manifest_errors(config, runtime_root: Path, manifest_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"deployment_manifest_path could not be read: {exc}"]
    if not isinstance(manifest, dict):
        return ["deployment_manifest_path must contain a JSON object"]

    strategy_refs = manifest.get("strategy_configs")
    if not isinstance(strategy_refs, dict) or not strategy_refs:
        errors.append("deployment_manifest.strategy_configs must be a non-empty object")
        strategy_refs = {}
    required_strategy_ids = manifest.get("required_strategy_ids", list(strategy_refs))
    if not isinstance(required_strategy_ids, list) or not all(isinstance(item, str) for item in required_strategy_ids):
        errors.append("deployment_manifest.required_strategy_ids must be a list of strategy ids")
        required_strategy_ids = []

    required = set(required_strategy_ids)
    configured = set(config.strategy_configs)
    missing = sorted(required - configured)
    unexpected = sorted(configured - required)
    if missing:
        errors.append(f"strategy_configs missing required deployment strategies: {', '.join(missing)}")
    if unexpected:
        errors.append(f"strategy_configs includes strategies not in deployment manifest: {', '.join(unexpected)}")

    if config.portfolio_config_path is not None:
        portfolio_ref = manifest.get("portfolio_config_path")
        if not isinstance(portfolio_ref, str) or not portfolio_ref:
            errors.append("deployment_manifest.portfolio_config_path is required")
        else:
            expected_portfolio = _runtime_path(Path(portfolio_ref), runtime_root)
            actual_portfolio = _runtime_path(config.portfolio_config_path, runtime_root)
            error = _json_identity_error(
                "portfolio_config_path",
                actual_portfolio,
                expected_portfolio,
            )
            if error:
                errors.append(error)

    for strategy_id in required_strategy_ids:
        ref = strategy_refs.get(strategy_id)
        if not isinstance(ref, str) or not ref:
            errors.append(f"deployment_manifest.strategy_configs.{strategy_id} is required")
            continue
        actual_config = config.strategy_configs.get(strategy_id)
        if actual_config is None:
            continue
        error = _json_identity_error(
            f"strategy_configs.{strategy_id}",
            _runtime_path(actual_config, runtime_root),
            _runtime_path(Path(ref), runtime_root),
        )
        if error:
            errors.append(error)

    errors.extend(_portfolio_manifest_errors(manifest, runtime_root))
    errors.extend(_parity_alignment_errors(manifest, runtime_root))
    return errors


def _portfolio_manifest_errors(manifest: dict, runtime_root: Path) -> list[str]:
    path_text = manifest.get("portfolio_rounds_manifest_path")
    required_rounds = manifest.get("required_portfolio_rounds")
    if path_text is None and required_rounds is None:
        return []
    if not isinstance(path_text, str) or not path_text:
        return ["deployment_manifest.portfolio_rounds_manifest_path is required when portfolio rounds are required"]
    path = _runtime_path(Path(path_text), runtime_root)
    if not path.exists():
        return [f"portfolio rounds manifest is missing: {path_text}"]
    if required_rounds is None:
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"portfolio rounds manifest could not be read: {exc}"]
    rounds = payload.get("rounds") if isinstance(payload, dict) else None
    actual_rounds = [item.get("round") for item in rounds if isinstance(item, dict)] if isinstance(rounds, list) else []
    if actual_rounds != required_rounds:
        return [f"portfolio rounds manifest rounds {actual_rounds} do not match required {required_rounds}"]
    return []


def _parity_alignment_errors(manifest: dict, runtime_root: Path) -> list[str]:
    path_text = manifest.get("parity_alignment_path")
    if path_text is None:
        return []
    if not isinstance(path_text, str) or not path_text:
        return ["deployment_manifest.parity_alignment_path must be a path string"]
    path = _runtime_path(Path(path_text), runtime_root)
    if not path.exists():
        return [f"portfolio parity evidence is missing: {path_text}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"portfolio parity evidence could not be read: {exc}"]
    replay = payload.get("portfolio_metric_replay", {}) if isinstance(payload, dict) else {}
    if not isinstance(replay, dict):
        return [f"portfolio parity evidence has invalid portfolio_metric_replay: {path_text}"]
    if replay.get("status") != "matched":
        return [f"portfolio parity evidence is not matched: {path_text}"]
    missing = [key for key in ("max_abs_delta", "tolerance") if key not in replay]
    if missing:
        return [f"portfolio parity evidence missing numeric fields: {', '.join(missing)}"]
    try:
        max_abs_delta = float(replay["max_abs_delta"])
        tolerance = float(replay["tolerance"])
    except (TypeError, ValueError) as exc:
        return [f"portfolio parity evidence has invalid numeric fields: {exc}"]
    if (
        not math.isfinite(max_abs_delta)
        or not math.isfinite(tolerance)
        or max_abs_delta < 0.0
        or tolerance < 0.0
    ):
        return ["portfolio parity evidence has non-finite or negative numeric fields"]
    if max_abs_delta > tolerance:
        return [f"portfolio parity max_abs_delta {max_abs_delta} exceeds tolerance {tolerance}"]
    return []


def _json_identity_error(label: str, actual: Path, expected: Path) -> str:
    if not expected.exists():
        return f"{label} deployment manifest reference is missing: {expected}"
    if not actual.exists():
        return ""
    try:
        actual_payload = json.loads(actual.read_text(encoding="utf-8"))
        expected_payload = json.loads(expected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"{label} could not be compared with deployment manifest reference: {exc}"
    if actual_payload != expected_payload:
        return f"{label} does not match deployment manifest reference: {actual}"
    return ""


@cli.group()
def admin() -> None:
    """Operator tools for durable live OMS state."""


@admin.command("resolve-discrepancy")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True),
              help="Live config JSON file")
@click.option("--id", "discrepancy_id", required=True, type=int,
              help="OMS reconciliation discrepancy id to resolve")
@click.option("--resolution", required=True,
              help="Human-readable resolution note")
@click.option("--resolved-by", default="admin", show_default=True,
              help="Operator, ticket, or automation id applying the correction")
@click.option("--action", default="resolve_discrepancy", show_default=True,
              help="Lifecycle action name to emit")
@click.option("--description", default="", help="Optional lifecycle event description")
@click.option("--metadata", default=None,
              help="Optional JSON object merged into the correction metadata")
def admin_resolve_discrepancy(
    config_path: str,
    discrepancy_id: int,
    resolution: str,
    resolved_by: str,
    action: str,
    description: str,
    metadata: str | None,
) -> None:
    """Resolve an OMS discrepancy and emit admin-correction evidence."""
    from crypto_trader.live.config import LiveConfig
    from crypto_trader.live.engine import LiveEngine

    log = structlog.get_logger()
    metadata_payload: dict = {}
    if metadata:
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"--metadata must be a JSON object: {exc}") from exc
        if not isinstance(parsed, dict):
            raise click.ClickException("--metadata must be a JSON object")
        metadata_payload = parsed

    with open(config_path, "r", encoding="utf-8") as f:
        config = LiveConfig.from_dict(json.load(f))

    engine = LiveEngine(config)
    try:
        try:
            engine.load_instrumentation_context_from_config()
        except Exception:
            log.exception("admin.instrumentation_context_load_failed")
        resolved = engine.record_admin_correction(
            discrepancy_id,
            resolution=resolution,
            resolved_by=resolved_by,
            action=action,
            description=description,
            metadata=metadata_payload,
        )
        if not resolved:
            raise click.ClickException(f"Discrepancy {discrepancy_id} was not found or could not be resolved")
        discrepancy = engine._oms.get_discrepancy(discrepancy_id)
        click.echo(json.dumps({
            "resolved": True,
            "discrepancy_id": discrepancy_id,
            "status": (discrepancy or {}).get("status"),
            "resolved_at": (discrepancy or {}).get("resolved_at"),
        }, indent=2, sort_keys=True))
    finally:
        engine._oms.close()


@cli.command()
@click.option("--state-dir", default="state", type=click.Path(), help="State directory with JSONL files")
def status(state_dir: str) -> None:
    """Show live system status from the latest health report and pipeline funnels."""
    from crypto_trader.live.health_report import HealthReport

    state = Path(state_dir)

    # --- Helper: read last line of a JSONL file ---
    def _read_last_jsonl(filename: str) -> dict | None:
        path = state / filename
        if not path.exists():
            return None
        last_line = ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        last_line = stripped
        except Exception:
            return None
        if not last_line:
            return None
        try:
            return json.loads(last_line)
        except json.JSONDecodeError:
            return None

    # --- Health report ---
    health_data = _read_last_jsonl("health_reports.jsonl")
    if health_data:
        # HealthReportSnapshot wraps the report in a "report" key
        report_dict = health_data.get("report", health_data)
        report = HealthReport(
            timestamp=report_dict.get("timestamp", "?"),
            uptime_sec=report_dict.get("uptime_sec", 0),
            data_flow=report_dict.get("data_flow", {}),
            signal_funnels=report_dict.get("signal_funnels", {}),
            gate_breakdown=report_dict.get("gate_breakdown", {}),
            positions=report_dict.get("positions", []),
            portfolio=report_dict.get("portfolio", {}),
            system=report_dict.get("system", {}),
            alerts=report_dict.get("alerts", []),
            assessment=report_dict.get("assessment", "unknown"),
        )
        print(report.to_text())
        print(f"\n  Last report: {report.timestamp}")
    else:
        print("=== System Health: NO DATA ===")
        print(f"  No health reports found in {state / 'health_reports.jsonl'}")

    # --- Pipeline funnels ---
    funnel_data = _read_last_jsonl("pipeline_funnels.jsonl")
    if funnel_data:
        print(f"\n--- Latest Pipeline Funnel ---")
        sid = funnel_data.get("strategy_id", "?")
        ts = funnel_data.get("timestamp", "?")
        assessment = funnel_data.get("assessment", "?")
        print(f"  Strategy: {sid} | Assessment: {assessment} | At: {ts}")
        funnel = funnel_data.get("funnel", {})
        if funnel:
            bars = sum(funnel.get("bars_received", {}).values()) if isinstance(funnel.get("bars_received"), dict) else funnel.get("bars_received", 0)
            ind = sum(funnel.get("indicators_ready", {}).values()) if isinstance(funnel.get("indicators_ready"), dict) else funnel.get("indicators_ready", 0)
            setups = sum(funnel.get("setups_detected", {}).values()) if isinstance(funnel.get("setups_detected"), dict) else funnel.get("setups_detected", 0)
            confirms = sum(funnel.get("confirmations", {}).values()) if isinstance(funnel.get("confirmations"), dict) else funnel.get("confirmations", 0)
            entries = sum(funnel.get("entries_attempted", {}).values()) if isinstance(funnel.get("entries_attempted"), dict) else funnel.get("entries_attempted", 0)
            fills = sum(funnel.get("fills", {}).values()) if isinstance(funnel.get("fills"), dict) else funnel.get("fills", 0)
            print(
                f"  bars={bars} -> indicators={ind} -> setups={setups} "
                f"-> confirms={confirms} -> entries={entries} -> fills={fills}"
            )
            # Gate rejections
            gates = funnel.get("gate_rejections", {})
            if gates:
                print("  Gate rejections:")
                for gate, count in gates.items():
                    total = sum(count.values()) if isinstance(count, dict) else count
                    print(f"    {gate}: {total}")
    else:
        print(f"\n--- Latest Pipeline Funnel ---")
        print(f"  No funnel data found in {state / 'pipeline_funnels.jsonl'}")

    print()


@cli.command("parity-report")
@click.option("--state-dir", default="data/live_state", type=click.Path(), help="Live state directory")
@click.option("--output", default=None, type=click.Path(), help="Optional JSON report output path")
def parity_report(state_dir: str, output: str | None) -> None:
    """Build a parity report from recorded canonical live/paper events."""
    from crypto_trader.parity.report import build_parity_report

    report = build_parity_report(Path(state_dir))
    payload = report.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    click.echo(text)


@cli.command("parity-gate")
@click.option("--report", "report_path", required=True, type=click.Path(exists=True), help="Parity report JSON file")
def parity_gate(report_path: str) -> None:
    """Evaluate promotion/deployment gates from a parity report."""
    from crypto_trader.parity.report import ParityReport, evaluate_promotion_gate

    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    report = ParityReport(
        stream_counts=payload.get("stream_counts", {}),
        decision_drift_count=payload.get("decision_drift_count", 0),
        order_intent_drift_count=payload.get("order_intent_drift_count", 0),
        unresolved_oms_discrepancies=payload.get("unresolved_oms_discrepancies", []),
        fill_watermark_age_sec=payload.get("fill_watermark_age_sec"),
        stale_fill_watermark=payload.get("stale_fill_watermark", False),
        unprotected_entry_fills=payload.get("unprotected_entry_fills", []),
        accounting_mismatch_count=payload.get("accounting_mismatch_count", 0),
        allocation_count=payload.get("allocation_count", 0),
        unallocated_exposure_count=payload.get("unallocated_exposure_count", 0),
        max_allocation_net_residual=payload.get("max_allocation_net_residual", 0.0),
        position_ownership_drift=payload.get("position_ownership_drift", False),
    )
    result = evaluate_promotion_gate(report)
    click.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    if not result.passed:
        raise click.ClickException("Parity promotion gate failed")


@cli.command("paper-status")
@click.option("--address", required=True, help="Wallet address")
@click.option("--testnet/--mainnet", default=True, help="Use testnet or mainnet")
def paper_status(address: str, testnet: bool) -> None:
    """Show current positions, orders, and equity from Hyperliquid."""
    from crypto_trader.live.broker import HyperliquidBroker

    log = structlog.get_logger()

    broker = HyperliquidBroker(
        wallet_address=address,
        private_key=None,  # read-only
        is_testnet=testnet,
    )

    print(f"\n{'='*50}")
    print(f"Hyperliquid {'Testnet' if testnet else 'Mainnet'} Status")
    print(f"Address: {address[:8]}...{address[-4:]}")
    print(f"{'='*50}")

    # Equity
    equity = broker.get_equity()
    print(f"\nEquity: ${equity:,.2f}")

    # Positions
    positions = broker.get_positions()
    if positions:
        print(f"\nOpen Positions ({len(positions)}):")
        for pos in positions:
            pnl_str = f"${pos.unrealized_pnl:+,.2f}" if pos.unrealized_pnl else "$0.00"
            print(f"  {pos.symbol:>5} {pos.direction.value:>5} qty={pos.qty:.4f} "
                  f"entry=${pos.avg_entry:,.2f} uPnL={pnl_str} "
                  f"lev={pos.leverage:.0f}x")
    else:
        print("\nNo open positions.")

    # Open orders
    orders = broker.get_open_orders()
    if orders:
        print(f"\nOpen Orders ({len(orders)}):")
        for o in orders:
            px = o.limit_price or o.stop_price or 0
            print(f"  {o.symbol:>5} {o.side.value:>5} {o.order_type.value:>8} "
                  f"qty={o.qty:.4f} px=${px:,.2f}")
    else:
        print("\nNo open orders.")

    print()


if __name__ == "__main__":
    cli()
