"""Process Quality Scorer — deterministic rules engine for trade quality.

Scores every trade's process quality independent of PnL and tags root
causes from a controlled taxonomy.  Pure rules-based, no LLMs.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger("instrumentation.process_scorer")

# Controlled root cause taxonomy — these are the ONLY valid tags
ROOT_CAUSES = [
    "regime_mismatch",
    "weak_signal",
    "strong_signal",
    "late_entry",
    "early_exit",
    "premature_stop",
    "slippage_spike",
    "good_execution",
    "filter_blocked_good",
    "filter_saved_bad",
    "risk_cap_hit",
    "data_gap",
    "order_reject",
    "latency_spike",
    "correlation_crowding",
    "funding_adverse",
    "funding_favorable",
    "regime_aligned",
    "normal_loss",
    "normal_win",
    "exceptional_win",
]


@dataclass
class ProcessScore:
    """Output of the process quality scorer for a single trade."""
    trade_id: str
    process_quality_score: int              # 0-100
    root_causes: List[str]
    evidence_refs: List[str]
    positive_factors: List[str]
    negative_factors: List[str]
    classification: str                     # "good_process" | "bad_process" | "neutral"

    def to_dict(self) -> dict:
        return asdict(self)


class ProcessScorer:
    """Deterministic rules engine for trade process quality.

    Usage::

        scorer = ProcessScorer("instrumentation/config/process_scoring_rules.yaml")
        score = scorer.score_trade(trade_event_dict, strategy_type="ATRSS")
    """

    def __init__(self, rules_path: str = "instrumentation/config/process_scoring_rules.yaml"):
        try:
            import yaml
            with open(rules_path) as f:
                self.rules = yaml.safe_load(f)
        except Exception:
            self.rules = {"global": {}, "strategies": {}}
        self.global_rules = self.rules.get("global", {})
        self.strategy_rules = self.rules.get("strategies", {})

    def _get_rules(self, strategy_type: str) -> dict:
        merged = dict(self.global_rules)
        if strategy_type in self.strategy_rules:
            merged.update(self.strategy_rules[strategy_type])
        return merged

    def score_trade(self, trade: dict, strategy_type: str = "default") -> ProcessScore:
        """Score a completed trade event (must have exit data)."""
        try:
            return self._score_trade_inner(trade, strategy_type)
        except Exception as e:
            logger.warning("Process scoring failed for %s: %s", trade.get("trade_id"), e)
            return ProcessScore(
                trade_id=trade.get("trade_id", "unknown"),
                process_quality_score=50,
                root_causes=[],
                evidence_refs=[],
                positive_factors=[],
                negative_factors=["scoring_error: " + str(e)],
                classification="neutral",
            )

    def _score_trade_inner(self, trade: dict, strategy_type: str) -> ProcessScore:
        rules = self._get_rules(strategy_type)
        score = 100
        root_causes: List[str] = []
        evidence: List[str] = []
        positive: List[str] = []
        negative: List[str] = []

        # --- REGIME FIT ---
        regime = trade.get("market_regime", "")
        preferred = rules.get("preferred_regimes", [])
        adverse = rules.get("adverse_regimes", [])

        if regime and adverse and regime in adverse:
            score -= 20
            root_causes.append("regime_mismatch")
            negative.append(f"Regime '{regime}' is adverse for {strategy_type}")
            evidence.append(f"market_regime={regime}, adverse_regimes={adverse}")
        elif regime and preferred and regime in preferred:
            root_causes.append("regime_aligned")
            positive.append(f"Regime '{regime}' is preferred for {strategy_type}")

        # --- SIGNAL STRENGTH ---
        strength = trade.get("entry_signal_strength", 0.5)
        min_strength = rules.get("min_signal_strength", 0.3)
        strong_threshold = rules.get("strong_signal_threshold", 0.7)

        if strength < min_strength:
            score -= 25
            root_causes.append("weak_signal")
            negative.append(f"Signal strength {strength:.2f} below threshold {min_strength}")
            evidence.append(f"entry_signal_strength={strength}")
        elif strength >= strong_threshold:
            root_causes.append("strong_signal")
            positive.append(f"Signal strength {strength:.2f} above strong threshold {strong_threshold}")

        # --- ENTRY LATENCY ---
        latency = trade.get("entry_latency_ms")
        max_latency = rules.get("max_entry_latency_ms", 5000)

        if latency is not None and latency > max_latency:
            score -= 15
            root_causes.append("late_entry")
            negative.append(f"Entry latency {latency}ms exceeds {max_latency}ms")
            evidence.append(f"entry_latency_ms={latency}")
        elif latency is not None and latency < max_latency * 0.5:
            positive.append(f"Fast entry: {latency}ms")

        # --- SLIPPAGE ---
        entry_slippage = trade.get("entry_slippage_bps")
        expected_slippage = rules.get("expected_slippage_bps", 5)
        max_slip_mult = rules.get("max_slippage_multiplier", 2.0)

        if entry_slippage is not None and entry_slippage > expected_slippage * max_slip_mult:
            score -= 10
            root_causes.append("slippage_spike")
            negative.append(f"Entry slippage {entry_slippage:.1f}bps vs expected {expected_slippage}bps")
            evidence.append(f"entry_slippage_bps={entry_slippage}")
        elif entry_slippage is not None and entry_slippage < expected_slippage * 0.5:
            root_causes.append("good_execution")
            positive.append(f"Below-average slippage: {entry_slippage:.1f}bps")

        exit_slippage = trade.get("exit_slippage_bps")
        if exit_slippage is not None and exit_slippage > expected_slippage * max_slip_mult:
            score -= 10
            if "slippage_spike" not in root_causes:
                root_causes.append("slippage_spike")
            negative.append(f"Exit slippage {exit_slippage:.1f}bps vs expected {expected_slippage}bps")
            evidence.append(f"exit_slippage_bps={exit_slippage}")

        # --- EXIT REASON ANALYSIS ---
        exit_reason = trade.get("exit_reason", "")
        pnl = trade.get("pnl", 0)

        if exit_reason == "MANUAL":
            score -= 10
            root_causes.append("early_exit")
            negative.append("Manual exit")
            evidence.append("exit_reason=MANUAL")

        if exit_reason == "STOP_LOSS":
            strategy_params = trade.get("strategy_params_at_entry", {})
            sl_mult = strategy_params.get("sl_atr_multiplier") or strategy_params.get("sl_atr_mult")
            if sl_mult is not None and sl_mult < 0.8:
                score -= 10
                root_causes.append("premature_stop")
                negative.append(f"SL multiplier {sl_mult}x ATR may be too tight")
                evidence.append(f"sl_atr_multiplier={sl_mult}")

        # --- FUNDING RATE ---
        funding = trade.get("funding_rate_at_entry")
        side = trade.get("side", "")
        if funding is not None and abs(funding) > 0.01:
            if (side == "LONG" and funding > 0.03) or (side == "SHORT" and funding < -0.03):
                score -= 5
                root_causes.append("funding_adverse")
                negative.append(f"Funding rate {funding:.4f} working against {side} position")
                evidence.append(f"funding_rate_at_entry={funding}")
            elif (side == "LONG" and funding < -0.01) or (side == "SHORT" and funding > 0.01):
                root_causes.append("funding_favorable")
                positive.append(f"Funding rate {funding:.4f} favorable for {side}")

        # --- FINAL CLASSIFICATION ---
        score = max(0, min(100, score))

        if score >= 80:
            if pnl and pnl > 0:
                pnl_pct = abs(trade.get("pnl_pct", 0))
                if pnl_pct > 3.0:
                    root_causes.append("exceptional_win")
                else:
                    root_causes.append("normal_win")
            elif pnl is not None and pnl <= 0:
                root_causes.append("normal_loss")

        if score >= 70:
            classification = "good_process"
        elif score >= 40:
            classification = "neutral"
        else:
            classification = "bad_process"

        return ProcessScore(
            trade_id=trade.get("trade_id", "unknown"),
            process_quality_score=score,
            root_causes=root_causes,
            evidence_refs=evidence,
            positive_factors=positive,
            negative_factors=negative,
            classification=classification,
        )

    def score_and_write(self, trade: dict, strategy_type: str, data_dir: str) -> ProcessScore:
        """Score a trade and write the result to the scores JSONL file."""
        ps = self.score_trade(trade, strategy_type)
        try:
            score_dir = Path(data_dir) / "scores"
            score_dir.mkdir(parents=True, exist_ok=True)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filepath = score_dir / f"scores_{today}.jsonl"
            with open(filepath, "a") as f:
                f.write(json.dumps(ps.to_dict()) + "\n")
        except Exception as e:
            logger.warning("Failed to write process score: %s", e)
        return ps
