# skills/correction_pattern_extractor.py
"""Extract recurring patterns from human corrections.

Groups corrections by (type, target) and clusters by keyword similarity
to surface "you keep getting X wrong" meta-patterns.
"""
from __future__ import annotations

import hashlib
import string
from collections import defaultdict
from datetime import datetime, timezone

from trading_assistant.schemas.correction_patterns import CorrectionPattern, CorrectionPatternReport

# Domain-specific keywords for grouping corrections within a (type, target) bucket
_DOMAIN_KEYWORDS = {
    "regime", "stop", "filter", "allocation", "entry", "exit",
    "signal", "drawdown", "position", "sizing", "timing",
    "correlation", "hedge", "miss", "premature", "late",
    "threshold", "volume", "volatility", "momentum", "trend",
    "ranging", "breakout", "reversal", "slippage", "risk",
}


def _normalize(text: str) -> set[str]:
    """Normalize text to a set of lowercase word tokens."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return set(text.split())


def _extract_domain_words(words: set[str]) -> frozenset[str]:
    """Extract domain-specific keywords from a word set."""
    return frozenset(words & _DOMAIN_KEYWORDS)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class CorrectionPatternExtractor:
    """Extracts recurring correction patterns from a list of correction dicts."""

    def __init__(self, min_occurrences: int = 2) -> None:
        self._min_occurrences = min_occurrences

    def extract(self, corrections: list[dict]) -> CorrectionPatternReport:
        """Extract patterns from correction entries.

        Args:
            corrections: List of dicts with keys like correction_type,
                target_id, raw_text, timestamp.

        Returns:
            CorrectionPatternReport with patterns meeting min_occurrences.
        """
        if not corrections:
            return CorrectionPatternReport(
                extracted_at=datetime.now(timezone.utc).isoformat(),
                total_corrections_analyzed=0,
            )

        # Step 1: Group by (correction_type, target)
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for corr in corrections:
            ctype = corr.get("correction_type", "free_text")
            target = corr.get("target_id", "") or corr.get("bot_id", "") or "all"
            groups[(ctype, target)].append(corr)

        # Step 2: Within each group, cluster by keyword overlap
        patterns: list[CorrectionPattern] = []
        for (ctype, target), group_entries in groups.items():
            clusters = self._cluster_by_keywords(group_entries)
            for cluster in clusters:
                if len(cluster) < self._min_occurrences:
                    continue
                patterns.append(self._build_pattern(ctype, target, cluster))

        # Sort by count descending
        patterns.sort(key=lambda p: p.count, reverse=True)

        return CorrectionPatternReport(
            extracted_at=datetime.now(timezone.utc).isoformat(),
            total_corrections_analyzed=len(corrections),
            patterns=patterns,
        )

    def _cluster_by_keywords(self, entries: list[dict]) -> list[list[dict]]:
        """Cluster entries by domain keyword overlap (simple Jaccard)."""
        if not entries:
            return []

        # Extract domain keywords for each entry
        entry_keywords: list[tuple[dict, frozenset[str]]] = []
        for entry in entries:
            raw = entry.get("raw_text", "")
            words = _normalize(raw)
            domain = _extract_domain_words(words)
            entry_keywords.append((entry, domain))

        # Greedy clustering: assign each entry to the first cluster
        # whose representative has Jaccard > 0.3, or create a new cluster
        clusters: list[list[tuple[dict, frozenset[str]]]] = []
        representatives: list[frozenset[str]] = []

        for entry, keywords in entry_keywords:
            assigned = False
            for i, rep in enumerate(representatives):
                if _jaccard(keywords, rep) > 0.3:
                    clusters[i].append((entry, keywords))
                    # Update representative to intersection (tighten cluster)
                    representatives[i] = rep & keywords if rep & keywords else rep
                    assigned = True
                    break
            if not assigned:
                clusters.append([(entry, keywords)])
                representatives.append(keywords)

        return [[entry for entry, _ in cluster] for cluster in clusters]

    def _build_pattern(
        self, ctype: str, target: str, cluster: list[dict],
    ) -> CorrectionPattern:
        """Build a CorrectionPattern from a cluster of correction entries."""
        timestamps = []
        for entry in cluster:
            ts_str = entry.get("timestamp", "")
            if ts_str:
                timestamps.append(ts_str)
        timestamps.sort()

        # Generate description from common domain keywords
        all_words: list[set[str]] = []
        for entry in cluster:
            all_words.append(_normalize(entry.get("raw_text", "")))
        if all_words:
            common = all_words[0]
            for ws in all_words[1:]:
                common = common & ws
            domain_common = common & _DOMAIN_KEYWORDS
        else:
            domain_common = set()

        if domain_common:
            desc = (
                f"Recurring {ctype} correction for {target}: "
                f"{len(cluster)} instances involving {', '.join(sorted(domain_common))}"
            )
        else:
            desc = f"Recurring {ctype} correction for {target}: {len(cluster)} instances"

        # Generate pattern_id from (type, target, sorted keywords)
        id_input = f"{ctype}|{target}|{'|'.join(sorted(domain_common))}"
        pattern_id = hashlib.sha256(id_input.encode()).hexdigest()[:12]

        return CorrectionPattern(
            pattern_id=pattern_id,
            correction_type=ctype,
            target=target,
            description=desc,
            count=len(cluster),
            first_seen=timestamps[0] if timestamps else "",
            last_seen=timestamps[-1] if timestamps else "",
            example_texts=[
                entry.get("raw_text", "")[:200] for entry in cluster[:3]
            ],
        )
