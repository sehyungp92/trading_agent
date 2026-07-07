from __future__ import annotations

from trading_assistant.analysis.triage_response_parser import parse_triage_response


def test_parse_triage_response_from_comment_block():
    response = """Summary
<!-- TRIAGE_RESULT
{
  "proposal_type": "bug_fix",
  "confidence": 0.8,
  "candidate_files": ["tests/test_bot.py"],
  "issue_title": "Fix missing import",
  "fix_plan": "Add a regression test.",
  "file_changes": [
    {
      "file_path": "tests/test_bot.py",
      "new_content": "def test_ok():\\n    assert True\\n"
    }
  ]
}
-->
"""
    proposal = parse_triage_response(response)
    assert proposal is not None
    assert proposal.proposal_type.value == "bug_fix"
    assert proposal.file_changes[0].file_path == "tests/test_bot.py"


def test_parse_triage_response_returns_none_for_plain_text():
    assert parse_triage_response("no structured output here") is None
