"""Phase analysis placeholders."""

from __future__ import annotations


def summarize_rejections(rows: list[dict]) -> dict:
    return {
        "rejected_count": len(rows),
        "reasons": sorted({str(row.get("reason", "")) for row in rows}),
    }
