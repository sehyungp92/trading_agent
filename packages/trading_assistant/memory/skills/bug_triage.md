# Bug Triage Skill

## Purpose
Diagnose bot errors, classify severity, and propose fixes with appropriate depth based on complexity. Ensures instrumentation failures never affect trading.

## Trigger
HIGH or CRITICAL error events routed by orchestrator brain. Error rate >3/hour triggers immediate triage.

## Pipeline
1. **Error Classification** — deterministic severity classifier (pattern matching, no LLM):
   - CRITICAL: affects trading execution, data corruption, heartbeat loss
   - HIGH: affects data pipeline, missed events, degraded analysis
   - MEDIUM: non-blocking errors, retryable failures
   - LOW: warnings, deprecation notices
2. **Complexity Classification** — route by fix difficulty:
   - OBVIOUS_FIX: single line/config change → propose specific fix
   - SINGLE_FUNCTION: isolated to one function → investigate + propose
   - MULTI_FILE: spans multiple files → summarize findings, recommend human review
   - STATE_DEPENDENT: race conditions, timing issues → careful investigation
3. **Error Rate Tracking** — sliding window detection (>3 errors/hour = escalate)
4. **Context Build** — source code snippet, stack trace, past rejections from failure log
5. **Prompt Assembly** — 6-step error investigation framework
6. **Claude Invocation** — agent_type="triage" with Read/Bash/Grep/Glob tools
7. **Failure Log** — record outcome in failure-log.jsonl (Ralph Loop V2)

## Input Data
- ErrorEvent (bot_id, error_type, message, stack_trace, source_file, source_line, context)
- Source code snippet (if available)
- Past rejections from failure log (to prevent re-suggesting)
- Recent git log (if available)

## Output
- triage_result.json (outcome, affected_files, suggested_fix)
- Fix complexity determines depth:
  - OBVIOUS_FIX: exact file paths + code changes
  - SINGLE_FUNCTION: root cause + investigation steps
  - MULTI_FILE+: summary + recommendation for human review

## Quality Criteria
- Check past rejections — never repeat previously rejected fixes
- Distinguish code bugs from data issues from infrastructure problems
- Never propose fixes that could affect live trading without approval gates
- If root cause unclear, propose diagnostic steps rather than guessing
- Start with severity assessment and blast radius
