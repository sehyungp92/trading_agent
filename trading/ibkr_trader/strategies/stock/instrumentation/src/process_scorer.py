import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger("instrumentation.process_scorer")

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
    process_quality_score: int
    root_causes: List[str]
    evidence_refs: List[str]
    positive_factors: List[str]
    negative_factors: List[str]
    classification: str

    def to_dict(self) -> dict:
        return asdict(self)


class ProcessScorer:
    """
    Deterministic rules engine for trade process quality.
    Pure rules-based scoring — no LLMs.

    Usage:
        scorer = ProcessScorer("instrumentation/config/process_scoring_rules.yaml")
        score = scorer.score_trade(trade_event_dict, strategy_type="helix")
    """

    def __init__(self, rules_path: str = "instrumentation/config/process_scoring_rules.yaml"):
        try:
            import yaml
            with open(rules_path) as f:
                self.rules = yaml.safe_load(f) or {}
        except Exception:
            self.rules = {}

        self.global_rules = self.rules.get("global", {})
        self.strategy_rules = self.rules.get("strategies", {})

    def _get_rules(self, strategy_type: str) -> dict:
        merged = dict(self.global_rules)
        if strategy_type in self.strategy_rules:
            merged.update(self.strategy_rules[strategy_type])
        return merged

    def score_trade(self, trade: dict, strategy_type: str = "default") -> ProcessScore:
        """
        Score a completed trade event (must have exit data).

        Args:
            trade: a TradeEvent.to_dict()
            strategy_type: key into strategy_rules

        Returns:
            ProcessScore with score, root causes, and evidence
        """
        try:
            return self._score_trade_inner(trade, strategy_type)
        except Exception as e:
            logger.warning("Process scoring failed for %s: %s", trade.get("trade_id", "?"), e)
            return ProcessScore(
                trade_id=trade.get("trade_id", "unknown"),
                process_quality_score=50,
                root_causes=[],
                evidence_refs=[],
                positive_factors=[],
                negative_factors=["scoring_error"],
                classification="neutral",
            )

    def _score_trade_inner(self, trade: dict, strategy_type: str) -> ProcessScore:
        rules = self._get_rules(strategy_type)
        score = 100
        root_causes = []
        evidence = []
        positive = []
        negative = []

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
            atr = trade.get("atr_at_entry")
            strategy_params = trade.get("strategy_params_at_entry", {}) or {}
            sl_mult = strategy_params.get("sl_atr_multiplier") or strategy_params.get("sl_atr_mult")

            if atr and sl_mult and sl_mult < 0.8:
                score -= 10
                root_causes.append("premature_stop")
                negative.append(f"SL multiplier {sl_mult}x ATR may be too tight")
                evidence.append(f"sl_atr_multiplier={sl_mult}, atr_at_entry={atr}")

        # --- MFE CAPTURE EFFICIENCY (V2/P14) ---
        mfe_r = trade.get("mfe_r", 0)
        _raw_realized_r = trade.get("realized_r")
        realized_r = _raw_realized_r if _raw_realized_r is not None else (pnl / max(abs(trade.get("atr_at_entry", 1.0)), 1e-9))
        if mfe_r >= 2.0 and pnl < 0:
            score -= 10
            root_causes.append("mfe_reversal")
            negative.append(f"Had {mfe_r:.1f}R MFE but closed with a loss -- full reversal")
            evidence.append(f"mfe_r={mfe_r}, pnl={pnl}")
        elif mfe_r >= 1.5 and realized_r is not None and realized_r < mfe_r * 0.2:
            score -= 5
            root_causes.append("poor_mfe_capture")
            negative.append(f"Captured only {realized_r:.2f}R of {mfe_r:.1f}R MFE")
            evidence.append(f"mfe_r={mfe_r}, realized_r={realized_r}")

        # --- TIME STOP WITH LOSS ---
        if exit_reason == "TIME_STOP" and pnl <= 0:
            score -= 5
            root_causes.append("time_stop_loss")
            negative.append("Time stop triggered with no profit -- weak signal")
            evidence.append(f"exit_reason=TIME_STOP, pnl={pnl}")

        # --- CARRY OVERNIGHT LOSS ---
        meta = trade.get("exit_meta", {}) or {}
        carry_path = meta.get("carry_decision_path", "")
        if carry_path and "flatten" not in carry_path and pnl < 0:
            score -= 5
            root_causes.append("carry_loss")
            negative.append(f"Overnight carry (path={carry_path}) resulted in loss")
            evidence.append(f"carry_decision_path={carry_path}, pnl={pnl}")

        # --- DISCIPLINED EXIT CREDIT ---
        if exit_reason in ("QUICK_EXIT", "QUICK_EXIT_S1", "FLOW_REVERSAL") and pnl < 0:
            positive.append(f"Disciplined early cut via {exit_reason}")
            if "disciplined_exit" not in root_causes:
                root_causes.append("disciplined_exit")

        # --- FINAL CLASSIFICATION ---
        score = max(0, min(100, score))

        if score >= 100 and not negative:
            if "good_execution" not in root_causes:
                root_causes.append("good_execution")
                positive.append("Perfect process execution — no deductions applied")
                evidence.append("score=100, negative_factors=[]")

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
