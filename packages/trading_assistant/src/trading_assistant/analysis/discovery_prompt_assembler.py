# analysis/discovery_prompt_assembler.py
"""Discovery prompt assembler — builds context for the discovery agent type.

Unlike daily/weekly analysis, the discovery agent gets access to raw trade
data (JSONL) and higher max_turns for iterative exploration. The goal is to
find patterns NOT covered by the 30 automated detectors.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from trading_assistant.analysis.context_builder import ContextBuilder
from trading_assistant.analysis.evidence_memory import EvidenceMemory
from trading_assistant.schemas.prompt_package import PromptPackage

_DISCOVERY_INSTRUCTIONS = """\
You are exploring raw trade data to discover patterns that the automated
detectors miss. You have access to raw JSONL trade files via Read/Grep/Glob.

## CONTEXT
The system has 30 automated detectors:
1. Alpha decay (signal correlation degradation)
2. Signal decay (win rate decline)
3. Component signal decay (per-signal-component decay)
4. Factor decay (factor-level signal degradation)
5. Exit timing issues (premature exits)
6. Correlation breakdown (cross-bot)
7. Time-of-day patterns
8. Drawdown concentration (clustered drawdowns)
9. Position sizing issues
10. Tight stop (stop-loss too tight)
11. Wide stop (stop-loss too wide)
12. Filter cost (filter threshold cost exceeds benefit)
13. Filter interactions (inter-filter dependency effects)
14. Regime loss (regime gate mismatch)
15. Regime config effectiveness (regime configuration quality)
16. Regime transition cost (cost of regime switches)
17. Stress entry pattern (entries during stress periods)
18. Microstructure (market microstructure anomalies)
19. Execution bottleneck (pipeline latency → slippage correlation)
20. Sizing methodology (risk efficiency and model divergence)
21. Portfolio crowding (correlated-position drag on win rate)
22. Funding impact (crypto funding rate cost erosion)
23. Grade selectivity (setup grade A/B performance differential)
24. Confluence quality (confluence factor lift analysis)
25. Leverage utilization (realized leverage versus cap)
26. Multi-timeframe alignment drift (trade direction versus higher-timeframe bias)
27. Liquidation proximity (MAE times leverage guardrail)
28. Symbol concentration (BTC/ETH/SOL loss concentration)
29. 24/7 session patterns (Asia/EU/US UTC window quality)
30. Funding trend (per-symbol weekly funding cost drift)

Your job is to find patterns OUTSIDE this coverage. The detectors are
grouped by category: stop_loss, filter_threshold, regime_gate, signal,
exit_timing, position_sizing.

## DATA ACCESS
Raw trade data is available in the curated directories listed in context_files.
Use Read/Grep/Glob to explore the JSONL files directly. Each line is a JSON
trade record with fields like: pnl, signal_strength, regime, entry_time,
exit_time, mae_pct, mfe_pct, position_size_pct, root_cause, etc.

## METHODOLOGY
1. Start with a broad scan: distribution shapes, outlier clusters, temporal patterns
2. Form a hypothesis about a pattern you observe
3. Test it: filter for confirming AND refuting data
4. If the pattern holds with >= 5 supporting data points, record it as a discovery

## ANTI-PATTERNS (do NOT do these)
- Do NOT restate curated summary statistics — those are already computed
- Do NOT report patterns with fewer than 5 supporting trades
- Do NOT report patterns that ARE covered by the 30 detectors above
- Do NOT report obvious observations (e.g., "winners have positive PnL")

## OUTPUT FORMAT
For each discovery, provide:
- Pattern description (1-2 sentences)
- Evidence: list of specific trade references (date, trade_id, PnL)
- Proposed root cause (from existing taxonomy or flag as "novel")
- Testable hypothesis that can be validated with future data
- Confidence rating (0.0-1.0) with justification
- Which automated detector relates (or "novel" if none)

## HYPOTHESIS CONTEXT
If existing_hypotheses data is present, check effectiveness scores before
proposing related structural changes. Prioritize hypotheses with positive
track records. Do NOT reinvestigate retired hypotheses (effectiveness <= 0).

## ACTIVE EXPERIMENTS
If active_experiments data is present, do NOT propose changes that conflict
with running experiments. Note any observations relevant to active experiments.

## STRATEGY IDEATION
Beyond pattern detection, consider whether the patterns you find suggest
an entirely new trading approach. A strategy idea should:
- Have a clear edge hypothesis (WHY would this work?)
- Be grounded in >= 10 supporting data points from the raw data
- Specify entry logic, exit logic, and applicable regimes
- NOT be a minor tweak to existing strategies — this is for genuinely new approaches

If strategy_discovery_packets data is present, treat it as diagnostics-only
context. New strategy ideas must cite recurring opportunity clusters, control
slices, after-cost estimates, and a replay or shadow plan. Do NOT present a
strategy_discovery_packet as approval-grade evidence.

Only propose a strategy idea if you found something truly novel and well-supported.
Most discovery runs will NOT produce a strategy idea — that's expected.

## STRUCTURED OUTPUT (REQUIRED)
At the END, emit a structured data block:
<!-- STRUCTURED_OUTPUT
{
  "discoveries": [
    {
      "pattern_description": "...",
      "evidence": [{"date": "...", "bot_id": "...", "trade_id": "...", "pnl": 0.0, "signal_strength": 0.0, "regime": "...", "note": "..."}],
      "proposed_root_cause": "...",
      "testable_hypothesis": "...",
      "confidence": 0.0,
      "detector_coverage": "novel|alpha_decay|signal_decay|component_signal_decay|factor_decay|exit_timing|correlation|time_of_day|drawdown_concentration|position_sizing|tight_stop|wide_stop|filter_cost|filter_interactions|regime_loss|regime_config_effectiveness|regime_transition_cost|stress_entry_pattern|microstructure|execution_bottleneck|sizing_methodology|portfolio_crowding|funding_impact|grade_selectivity|confluence_quality|leverage_utilization|mtf_alignment_drift|liquidation_proximity|symbol_concentration|session_patterns_24_7|funding_trend",
      "bot_id": "..."
    }
  ],
  "strategy_ideas": [
    {
      "title": "...",
      "description": "...",
      "edge_hypothesis": "...",
      "evidence": [{"date": "...", "bot_id": "...", "trade_id": "...", "pnl": 0.0, "regime": "...", "note": "..."}],
      "entry_logic": "...",
      "exit_logic": "...",
      "applicable_regimes": ["..."],
      "applicable_bots": ["..."],
      "confidence": 0.0
    }
  ],
  "structural_proposals": [
    {
      "hypothesis_id": "use id from existing_hypotheses if matching, else null",
      "bot_id": "...",
      "title": "...",
      "description": "...",
      "reversibility": "easy|moderate|hard",
      "evidence": "...",
      "estimated_complexity": "low|medium|high",
      "acceptance_criteria": [{"metric": "...", "direction": "improve|not_degrade", "minimum_change": 0.0, "observation_window_days": 14, "minimum_trade_count": 20}]
    }
  ]
}
-->"""


class DiscoveryPromptAssembler:
    """Assembles context for the discovery agent — raw data + exploration tools."""

    def __init__(
        self,
        date: str,
        bots: list[str],
        curated_dir: Path,
        memory_dir: Path,
        lookback_days: int = 30,
        bot_configs: dict | None = None,
        runs_dir: Path | None = None,
    ) -> None:
        self.date = date
        self.bots = bots
        self.curated_dir = curated_dir
        self.memory_dir = memory_dir
        self.runs_dir = runs_dir or memory_dir.parent / "runs"
        self.lookback_days = lookback_days
        self.bot_configs = bot_configs
        self._evidence = EvidenceMemory(memory_dir)
        self._ctx = ContextBuilder(
            memory_dir,
            curated_dir=curated_dir,
            evidence_memory=self._evidence,
        )

    def assemble(self, session_store=None) -> PromptPackage:
        """Build the complete prompt package for discovery analysis."""
        pkg = self._ctx.base_package(
            session_store=session_store,
            agent_type="discovery_analysis",
            bot_configs=self.bot_configs,
        )
        pkg.task_prompt = self._build_task_prompt()
        pkg.instructions = _DISCOVERY_INSTRUCTIONS
        pkg.data.update(self._load_discovery_context())
        pkg.context_files.extend(self._list_raw_data_files())
        pkg.metadata["bot_ids"] = ",".join(self.bots)
        pkg.metadata["date"] = self.date
        return pkg

    def _build_task_prompt(self) -> str:
        bot_list = ", ".join(self.bots)
        return (
            f"Explore the last {self.lookback_days} days of raw trade data for bots: {bot_list}.\n"
            f"Find patterns not covered by the 30 automated detectors.\n"
            f"Use Read/Grep/Glob tools to examine the JSONL trade files listed in context_files.\n"
            f"Reference date: {self.date}."
        )

    def _load_discovery_context(self) -> dict:
        """Load lightweight context for discovery — existing detector findings."""
        data: dict = {}

        # Load latest regime analysis for context
        for bot in self.bots:
            bot_dir = self.curated_dir / self.date / bot
            regime_path = bot_dir / "regime_analysis.json"
            if regime_path.exists():
                try:
                    data.setdefault("regime_context", {})[bot] = json.loads(
                        regime_path.read_text(encoding="utf-8")
                    )
                except (json.JSONDecodeError, OSError):
                    pass

        # Load hypothesis library for reference
        hypothesis_track = self._ctx.load_hypothesis_track_record()
        if hypothesis_track:
            data["existing_hypotheses"] = hypothesis_track

        # Load active experiments to avoid conflicts
        active_experiments = self._ctx.load_active_experiments()
        if active_experiments:
            data["active_experiments"] = active_experiments

        # Load experiment track record for calibration
        experiment_track_record = self._ctx.load_experiment_track_record()
        if experiment_track_record:
            data["experiment_track_record"] = experiment_track_record

        discovery_packets = self._load_strategy_discovery_packets()
        if discovery_packets:
            data["strategy_discovery_packets"] = discovery_packets

        return data

    def _list_raw_data_files(self) -> list[str]:
        """List paths to raw JSONL trade files for the lookback window."""
        files: list[str] = []
        end = datetime.strptime(self.date, "%Y-%m-%d")
        for d in range(self.lookback_days):
            date_str = (end - timedelta(days=d)).strftime("%Y-%m-%d")
            for bot in self.bots:
                trades_path = self.curated_dir / date_str / bot / "trades.jsonl"
                if trades_path.exists():
                    files.append(str(trades_path))
                missed_path = self.curated_dir / date_str / bot / "missed.jsonl"
                if missed_path.exists():
                    files.append(str(missed_path))
        for packet in self._strategy_discovery_packet_paths():
            files.append(str(packet))
        return files

    def _load_strategy_discovery_packets(self) -> list[dict]:
        packets: list[dict] = []
        for path in self._strategy_discovery_packet_paths()[:10]:
            try:
                packet = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(packet, dict):
                continue
            bot_id = str(packet.get("bot_id") or "")
            if self.bots and bot_id and bot_id not in self.bots:
                continue
            packets.append({
                "packet_path": str(path),
                "run_month": packet.get("run_month", ""),
                "bot_id": bot_id,
                "strategy_id": packet.get("strategy_id", ""),
                "authority": packet.get("authority", "diagnostics_only"),
                "evidence_authority": packet.get("evidence_authority", "diagnostics_only"),
                "approval_gate_eligible": bool(packet.get("approval_gate_eligible", False)),
                "missed_opportunity_clusters": packet.get("missed_opportunity_clusters", []),
                "denominator_clusters": packet.get("denominator_clusters", []),
                "control_slices": packet.get("control_slices", []),
                "after_cost_estimates": packet.get("after_cost_estimates", []),
                "replay_or_shadow_plan": packet.get("replay_or_shadow_plan", ""),
            })
        return packets

    def _strategy_discovery_packet_paths(self) -> list[Path]:
        if not self.runs_dir.exists():
            return []
        return sorted(
            self.runs_dir.rglob("strategy_discovery_packet.json"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
            reverse=True,
        )
