"""Stream and output parsing for agent CLI runtimes."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_PROGRESS_PREVIEW_LIMIT = 200


@dataclass
class StreamState:
    raw_lines: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    final_result: str = ""
    resolved_session_id: str = ""
    cost_usd: float = 0.0
    first_output_ms: int = 0
    stream_event_count: int = 0
    tool_call_count: int = 0


class StreamParser:
    """Parses streaming and buffered output from Claude CLI and Codex CLI."""

    def __init__(
        self,
        broadcast_fn: Callable[[str, dict], None] | None = None,
    ) -> None:
        self._broadcast = broadcast_fn or (lambda _type, _data: None)

    def parse_claude_stream_line(
        self,
        line: str,
        state: StreamState,
        run_id: str,
        agent_type: str,
        provider: str,
        runtime: str,
    ) -> None:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            self._broadcast(
                "agent_invocation_progress",
                {
                    "run_id": run_id,
                    "agent_type": agent_type,
                    "provider": provider,
                    "runtime": runtime,
                    "kind": "raw",
                    "preview": self._truncate_preview(line),
                },
            )
            return

        if not isinstance(parsed, dict):
            return

        self._capture_session_identifier(parsed, state)
        self._capture_cost(parsed, state)

        event_type = str(parsed.get("type", "")).lower()
        if event_type == "assistant":
            message = parsed.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        block_type = str(block.get("type", "")).lower()
                        if block_type == "text":
                            self._record_assistant_text(
                                state=state,
                                text=block.get("text"),
                                run_id=run_id,
                                agent_type=agent_type,
                                provider=provider,
                                runtime=runtime,
                            )
                        elif block_type == "tool_use":
                            state.tool_call_count += 1
                            tool_name = str(block.get("name", "")).strip()
                            tool_input = block.get("input")
                            preview = (
                                self._truncate_preview(json.dumps(tool_input, default=str))
                                if tool_input is not None
                                else ""
                            )
                            self._broadcast(
                                "agent_invocation_progress",
                                {
                                    "run_id": run_id,
                                    "agent_type": agent_type,
                                    "provider": provider,
                                    "runtime": runtime,
                                    "kind": "tool_call",
                                    "tool_name": tool_name,
                                    "preview": preview,
                                },
                            )
        elif event_type == "user":
            message = parsed.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if str(block.get("type", "")).lower() != "tool_result":
                            continue
                        self._broadcast(
                            "agent_invocation_progress",
                            {
                                "run_id": run_id,
                                "agent_type": agent_type,
                                "provider": provider,
                                "runtime": runtime,
                                "kind": "tool_result",
                                "preview": self._extract_tool_result_preview(block),
                            },
                        )
        elif event_type == "result":
            result_text = parsed.get("result")
            if isinstance(result_text, str) and result_text.strip():
                state.final_result = result_text.strip()
                self._broadcast(
                    "agent_invocation_progress",
                    {
                        "run_id": run_id,
                        "agent_type": agent_type,
                        "provider": provider,
                        "runtime": runtime,
                        "kind": "result",
                        "preview": self._truncate_preview(result_text),
                    },
                )

        item = parsed.get("item")
        if isinstance(item, dict):
            item_text = item.get("text")
            item_type = str(item.get("type", "")).lower()
            if isinstance(item_text, str) and item_text.strip() and (
                not item_type or "message" in item_type
            ):
                self._record_assistant_text(
                    state=state,
                    text=item_text,
                    run_id=run_id,
                    agent_type=agent_type,
                    provider=provider,
                    runtime=runtime,
                )

    def parse_codex_stream_line(
        self,
        line: str,
        state: StreamState,
        run_id: str,
        agent_type: str,
        provider: str,
        runtime: str,
    ) -> None:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            self._broadcast(
                "agent_invocation_progress",
                {
                    "run_id": run_id,
                    "agent_type": agent_type,
                    "provider": provider,
                    "runtime": runtime,
                    "kind": "raw",
                    "preview": self._truncate_preview(line),
                },
            )
            return

        if not isinstance(parsed, dict):
            return

        self._capture_session_identifier(parsed, state)
        self._capture_cost(parsed, state)

        result_text = parsed.get("result")
        if isinstance(result_text, str) and result_text.strip():
            state.final_result = result_text.strip()
            self._broadcast(
                "agent_invocation_progress",
                {
                    "run_id": run_id,
                    "agent_type": agent_type,
                    "provider": provider,
                    "runtime": runtime,
                    "kind": "result",
                    "preview": self._truncate_preview(result_text),
                },
            )

        item = parsed.get("item")
        if not isinstance(item, dict):
            return

        item_type = str(item.get("type", "")).lower()
        item_text = item.get("text")
        if isinstance(item_text, str) and item_text.strip() and (
            not item_type or "message" in item_type
        ):
            self._record_assistant_text(
                state=state,
                text=item_text,
                run_id=run_id,
                agent_type=agent_type,
                provider=provider,
                runtime=runtime,
            )

        if "tool" not in item_type:
            return

        kind = "tool_result" if "result" in item_type or "output" in item_type else "tool_call"
        if kind == "tool_call":
            state.tool_call_count += 1
        self._broadcast(
            "agent_invocation_progress",
            {
                "run_id": run_id,
                "agent_type": agent_type,
                "provider": provider,
                "runtime": runtime,
                "kind": kind,
                "tool_name": str(item.get("name") or item.get("tool_name") or "").strip(),
                "preview": self._extract_codex_item_preview(item),
            },
        )

    def _record_assistant_text(
        self,
        state: StreamState,
        text: object,
        run_id: str,
        agent_type: str,
        provider: str,
        runtime: str,
    ) -> None:
        if not isinstance(text, str):
            return
        cleaned = text.strip()
        if not cleaned:
            return
        state.texts.append(cleaned)
        self._broadcast(
            "agent_invocation_progress",
            {
                "run_id": run_id,
                "agent_type": agent_type,
                "provider": provider,
                "runtime": runtime,
                "kind": "assistant_text",
                "preview": self._truncate_preview(cleaned),
            },
        )

    def _capture_session_identifier(self, parsed: dict, state: StreamState) -> None:
        for key in ("session_id", "thread_id"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                state.resolved_session_id = value.strip()
                break

    def _capture_cost(self, parsed: dict, state: StreamState) -> None:
        direct_cost = parsed.get("cost_usd") or parsed.get("total_cost_usd")
        if isinstance(direct_cost, (int, float)):
            state.cost_usd = float(direct_cost)
            return
        usage = parsed.get("usage")
        if isinstance(usage, dict):
            total = usage.get("total_cost_usd") or usage.get("cost_usd")
            if isinstance(total, (int, float)):
                state.cost_usd = float(total)

    def _extract_tool_result_preview(self, block: dict) -> str:
        content = block.get("content")
        if isinstance(content, str):
            return self._truncate_preview(content)
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text.strip())
            return self._truncate_preview(" ".join(chunks))
        return self._truncate_preview(json.dumps(content, default=str))

    def _extract_codex_item_preview(self, item: dict) -> str:
        for key in ("arguments", "args", "input", "output", "content", "text"):
            value = item.get(key)
            if value in (None, ""):
                continue
            if isinstance(value, str):
                return self._truncate_preview(value)
            return self._truncate_preview(json.dumps(value, default=str))
        return self._truncate_preview(json.dumps(item, default=str))

    def parse_output(
        self,
        stdout: str,
        run_dir: Path,
        elapsed_ms: int,
        session_id: str,
        parse_mode: str,
        provider: str,
        runtime: str,
        requested_model: str | None,
        effective_model: str,
        auth_mode: str,
    ) -> dict:
        """Parse buffered output. Returns dict with AgentResult fields."""
        if parse_mode == "jsonl":
            return self._parse_jsonl_output(
                stdout=stdout, run_dir=run_dir, elapsed_ms=elapsed_ms,
                session_id=session_id, provider=provider, runtime=runtime,
                requested_model=requested_model, effective_model=effective_model,
                auth_mode=auth_mode,
            )
        return self._parse_json_output(
            stdout=stdout, run_dir=run_dir, elapsed_ms=elapsed_ms,
            session_id=session_id, provider=provider, runtime=runtime,
            requested_model=requested_model, effective_model=effective_model,
            auth_mode=auth_mode,
        )

    def _parse_json_output(
        self,
        stdout: str,
        run_dir: Path,
        elapsed_ms: int,
        session_id: str,
        provider: str,
        runtime: str,
        requested_model: str | None,
        effective_model: str,
        auth_mode: str,
    ) -> dict:
        try:
            data = json.loads(stdout)
            return {
                "response": data.get("result", stdout),
                "run_dir": run_dir,
                "cost_usd": data.get("cost_usd", 0.0),
                "duration_ms": data.get("duration_ms", elapsed_ms),
                "session_id": session_id,
                "success": True,
                "provider": provider,
                "runtime": runtime,
                "requested_model": requested_model,
                "effective_model": effective_model,
                "auth_mode": auth_mode,
            }
        except (json.JSONDecodeError, TypeError):
            return {
                "response": stdout.strip(),
                "run_dir": run_dir,
                "cost_usd": 0.0,
                "duration_ms": elapsed_ms,
                "session_id": session_id,
                "success": True,
                "provider": provider,
                "runtime": runtime,
                "requested_model": requested_model,
                "effective_model": effective_model,
                "auth_mode": auth_mode,
            }

    def _parse_jsonl_output(
        self,
        stdout: str,
        run_dir: Path,
        elapsed_ms: int,
        session_id: str,
        provider: str,
        runtime: str,
        requested_model: str | None,
        effective_model: str,
        auth_mode: str,
    ) -> dict:
        texts: list[str] = []
        resolved_session_id = session_id
        usage_total = 0.0
        stream_event_count = 0
        tool_call_count = 0
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            stream_event_count += 1
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                thread_id = parsed.get("thread_id")
                if isinstance(thread_id, str) and thread_id.strip():
                    resolved_session_id = thread_id.strip()
                usage = parsed.get("usage")
                if isinstance(usage, dict):
                    total = usage.get("total_cost_usd") or usage.get("cost_usd")
                    if isinstance(total, (int, float)):
                        usage_total = float(total)
                item = parsed.get("item")
                if isinstance(item, dict):
                    item_type = str(item.get("type", "")).lower()
                    if "tool" in item_type:
                        tool_call_count += 1
                    text = item.get("text")
                    if isinstance(text, str) and (
                        not item_type or "message" in item_type
                    ):
                        texts.append(text)

        response = "\n".join(texts).strip() or stdout.strip()
        return {
            "response": response,
            "run_dir": run_dir,
            "cost_usd": usage_total,
            "duration_ms": elapsed_ms,
            "session_id": resolved_session_id,
            "success": True,
            "provider": provider,
            "runtime": runtime,
            "requested_model": requested_model,
            "effective_model": effective_model,
            "stream_event_count": stream_event_count,
            "tool_call_count": tool_call_count,
            "auth_mode": auth_mode,
        }

    def _truncate_preview(self, text: str) -> str:
        preview = text.strip().replace("\r", " ").replace("\n", " ")
        if len(preview) <= _PROGRESS_PREVIEW_LIMIT:
            return preview
        return preview[: _PROGRESS_PREVIEW_LIMIT - 3] + "..."
