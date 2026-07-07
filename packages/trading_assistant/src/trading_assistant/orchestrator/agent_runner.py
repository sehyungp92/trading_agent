"""Agent runner for configured CLI-backed analysis providers."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from trading_assistant.orchestrator.agent_preferences import AgentPreferencesManager
from trading_assistant.orchestrator.cost_tracker import CostTracker
from trading_assistant.orchestrator.provider_cooldown import ProviderCooldownTracker
from trading_assistant.orchestrator.event_stream import EventStream
from trading_assistant.orchestrator.invocation_builder import InvocationBuilder, InvocationSpec
from trading_assistant.orchestrator.provider_auth import ProviderAuthChecker
from trading_assistant.orchestrator.session_store import SessionStore
from trading_assistant.orchestrator.skills_registry import SkillsRegistry
from trading_assistant.orchestrator.stream_parser import StreamParser, StreamState
from trading_assistant.schemas.agent_preferences import (
    AgentPreferences,
    AgentPreferencesView,
    AgentProvider,
    AgentSelection,
    AgentWorkflow,
    ProviderReadiness,
)
from trading_assistant.schemas.prompt_package import PromptPackage

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 600
_PREVIEW_LIMIT = 200


@dataclass
class AgentResult:
    response: str
    run_dir: Path
    cost_usd: float = 0.0
    duration_ms: int = 0
    session_id: str = ""
    success: bool = True
    error: str = ""
    provider: str = ""
    runtime: str = ""
    requested_model: str | None = None
    effective_model: str = ""
    first_output_ms: int = 0
    stream_event_count: int = 0
    tool_call_count: int = 0
    auth_mode: str = ""


class AgentRunner:
    """Invokes the configured provider runtime with a PromptPackage."""

    def __init__(
        self,
        runs_dir: Path,
        session_store: SessionStore,
        claude_command: str = "claude",
        claude_command_args: list[str] | None = None,
        codex_command: str = "codex",
        codex_command_args: list[str] | None = None,
        default_model: str = "sonnet",
        default_max_turns: int = 5,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        skills_registry: SkillsRegistry | None = None,
        enforce_skills: bool = False,
        preferences: AgentPreferences | None = None,
        zai_api_key: str = "",
        openrouter_api_key: str = "",
        event_stream: EventStream | None = None,
        cost_tracker: CostTracker | None = None,
        run_index: object | None = None,
    ) -> None:
        self._runs_dir = Path(runs_dir)
        self._session_store = session_store
        self._claude_command = claude_command
        self._claude_command_args = list(claude_command_args or [])
        self._codex_command = codex_command
        self._codex_command_args = list(codex_command_args or [])
        self._default_model = default_model
        self._default_max_turns = default_max_turns
        self._timeout_seconds = timeout_seconds
        self._skills_registry = skills_registry
        self._enforce_skills = enforce_skills
        self._event_stream = event_stream
        self._cost_tracker = cost_tracker
        self._run_index = run_index

        # Delegate: provider auth and command resolution
        self._auth_checker = ProviderAuthChecker(
            claude_command=claude_command,
            claude_command_args=claude_command_args,
            codex_command=codex_command,
            codex_command_args=codex_command_args,
            zai_api_key=zai_api_key,
            openrouter_api_key=openrouter_api_key,
        )

        # Delegate: CLI invocation building
        self._invocation_builder = InvocationBuilder(
            auth_checker=self._auth_checker,
            claude_command=claude_command,
            claude_command_args=claude_command_args,
            codex_command=codex_command,
            codex_command_args=codex_command_args,
            default_model=default_model,
        )

        # Delegate: stream and output parsing
        self._stream_parser = StreamParser(
            broadcast_fn=self._broadcast_runtime_event,
        )

        # Preferences manager
        self._preferences = AgentPreferencesManager(
            preferences
            or AgentPreferences(
                default=AgentSelection(
                    provider=AgentProvider.CLAUDE_MAX,
                    model=default_model,
                )
            ),
            findings_dir=(Path(runs_dir).parent / "memory" / "findings"),
        )
        self._preferences.set_provider_status_resolver(self.get_provider_statuses)

        # Cooldown tracker for provider fallback
        self._cooldown_tracker = ProviderCooldownTracker()


    @property
    def session_store(self) -> SessionStore:
        """Expose session store for assemblers that need session history."""
        return self._session_store

    # -- Preferences --

    def get_preferences(self) -> AgentPreferences:
        return self._preferences.get_preferences()

    def get_preferences_view(self) -> AgentPreferencesView:
        return self._preferences.build_view()

    def update_preferences(self, preferences: AgentPreferences) -> None:
        errors = self._preferences.unavailable_reasons(preferences)
        if errors:
            raise ValueError("; ".join(errors))
        self._preferences.set_preferences(preferences)

    def get_provider_statuses(self) -> list[ProviderReadiness]:
        return [self._get_provider_status(provider) for provider in AgentProvider]

    def invalidate_provider_status_cache(
        self,
        provider: AgentProvider | None = None,
    ) -> None:
        """Clear cached provider readiness results."""
        self._auth_checker.invalidate_cache(provider)

    # -- Invocation --

    async def invoke(
        self,
        agent_type: str,
        prompt_package: PromptPackage,
        run_id: str,
        model: str | None = None,
        max_turns: int | None = None,
        allowed_tools: list[str] | None = None,
    ) -> AgentResult:
        """Invoke the configured runtime with the given prompt package.

        If the primary provider fails and a fallback_chain is configured,
        tries each fallback in order, recording cooldowns on failure.
        """
        workflow = self._resolve_workflow(agent_type)

        # Resolve per-workflow tuning
        tuning = self._preferences.resolve_tuning(
            workflow,
            max_turns_override=max_turns,
            allowed_tools_override=allowed_tools,
        )
        effective_max_turns = tuning.max_turns
        effective_allowed_tools = tuning.allowed_tools
        effective_timeout = tuning.timeout_seconds

        candidates = self._preferences.resolve_with_fallbacks(
            workflow, model, cooldown_tracker=self._cooldown_tracker,
        )

        if not candidates:
            # Every configured candidate is unavailable or cooling down.
            self._check_skill_access(agent_type)
            run_dir = self._runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_run_files(run_dir, prompt_package, agent_type=agent_type)
            selection, requested_model = self._preferences.resolve_selection(
                workflow,
                model,
                cooldown_tracker=self._cooldown_tracker,
            )
            status = self._get_provider_status(selection.provider)
            reason = status.reason or "provider unavailable or cooling down"
            error = (
                f"No available providers for {agent_type} after availability "
                f"and cooldown filtering. Primary {selection.provider.value}: {reason}"
            )
            result = AgentResult(
                response="",
                run_dir=run_dir,
                success=False,
                error=error,
                provider=selection.provider.value,
                runtime=status.runtime,
                requested_model=requested_model,
                effective_model=selection.model,
            )
            self._broadcast_runtime_event(
                "agent_invocation_failed",
                {
                    "run_id": run_id,
                    "agent_type": agent_type,
                    "provider": result.provider,
                    "runtime": result.runtime,
                    "model": result.effective_model,
                    "error": error,
                },
            )
            self._record_cost(result, agent_type, run_id)
            return result

        last_result: AgentResult | None = None
        for i, (selection, requested_model) in enumerate(candidates):
            attempt_run_id = run_id if i == 0 else f"{run_id}-fallback-{i}"
            result = await self.invoke_with_selection(
                agent_type=agent_type,
                prompt_package=prompt_package,
                run_id=attempt_run_id,
                selection=selection,
                requested_model=requested_model,
                max_turns=effective_max_turns,
                allowed_tools=effective_allowed_tools,
                timeout_seconds=effective_timeout,
            )
            self._preferences.record_committed_route_change(
                workflow=workflow,
                selected=selection,
                model_override=model,
                agent_type=agent_type,
                run_id=attempt_run_id,
                success=result.success,
            )
            if result.success:
                return result

            # Record failure and try next
            self._cooldown_tracker.record_failure(selection.provider)
            last_result = result
            if i < len(candidates) - 1:
                next_sel = candidates[i + 1][0]
                logger.warning(
                    "Provider %s failed for %s (run_id=%s), falling back to %s",
                    selection.provider.value,
                    agent_type,
                    run_id,
                    next_sel.provider.value,
                )
                self._broadcast_runtime_event(
                    "provider_fallback",
                    {
                        "run_id": run_id,
                        "agent_type": agent_type,
                        "failed_provider": selection.provider.value,
                        "next_provider": next_sel.provider.value,
                        "error": result.error[:200],
                    },
                )

        return last_result  # type: ignore[return-value]

    async def invoke_with_selection(
        self,
        agent_type: str,
        prompt_package: PromptPackage,
        run_id: str,
        selection: AgentSelection,
        requested_model: str | None = None,
        max_turns: int | None = None,
        allowed_tools: list[str] | None = None,
        timeout_seconds: int | None = None,
    ) -> AgentResult:
        """Invoke the runtime with an explicit provider selection."""
        self._check_skill_access(agent_type)

        effective_timeout = timeout_seconds if timeout_seconds is not None else self._timeout_seconds

        result = await self._invoke_with_selection_inner(
            agent_type=agent_type,
            prompt_package=prompt_package,
            run_id=run_id,
            selection=selection,
            requested_model=requested_model,
            max_turns=max_turns,
            allowed_tools=allowed_tools,
            timeout_seconds=effective_timeout,
        )
        self._record_cost(result, agent_type, run_id)
        return result

    def _index_run(
        self,
        run_id: str,
        agent_type: str,
        run_dir: Path,
        result: AgentResult,
        invocation: InvocationSpec,
        prompt_package: "PromptPackage | None" = None,
    ) -> None:
        """Index the completed run in RunIndex (best-effort)."""
        if self._run_index is None:
            return
        try:
            _meta = self._run_index_metadata(prompt_package)
            self._run_index.index_run(
                run_id=run_id,
                agent_type=agent_type,
                run_dir=run_dir,
                provider=invocation.provider.value,
                model=invocation.effective_model,
                bot_ids=_meta.get("bot_ids", ""),
                date=_meta.get("date", ""),
                success=result.success,
                duration_ms=result.duration_ms,
                cost_usd=result.cost_usd,
                metadata=_meta,
            )
        except Exception:
            logger.debug("Failed to index run %s", run_id)

    def refresh_run_index(
        self,
        run_id: str,
        agent_type: str,
        run_dir: Path,
        *,
        provider: str = "",
        model: str = "",
        prompt_package: "PromptPackage | None" = None,
        success: bool = True,
        duration_ms: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Refresh a run index entry after handlers persist extra artifacts."""
        if self._run_index is None:
            return
        try:
            _meta = self._run_index_metadata(prompt_package)
            self._run_index.index_run(
                run_id=run_id,
                agent_type=agent_type,
                run_dir=run_dir,
                provider=provider,
                model=model,
                bot_ids=_meta.get("bot_ids", ""),
                date=_meta.get("date", ""),
                success=success,
                duration_ms=duration_ms,
                cost_usd=cost_usd,
                metadata=_meta,
            )
        except Exception:
            logger.debug("Failed to refresh run index for %s", run_id)

    @staticmethod
    def _run_index_metadata(prompt_package: "PromptPackage | None") -> dict:
        """Keep RunIndex metadata compact and exclude injected prompt blocks."""
        if prompt_package is None:
            return {}
        metadata = dict(prompt_package.metadata or {})
        metadata.pop("_learning_cards_text", None)
        metadata.pop("_generated_playbooks_text", None)
        return metadata

    def _record_cost(self, result: AgentResult, agent_type: str, run_id: str) -> None:
        if self._cost_tracker is None:
            return
        try:
            from trading_assistant.schemas.cost_tracking import CostRecord
            self._cost_tracker.record(CostRecord(
                provider=result.provider,
                workflow=agent_type,
                model=result.effective_model,
                cost_usd=result.cost_usd,
                duration_ms=result.duration_ms,
                success=result.success,
                run_id=run_id,
            ))
        except Exception:
            logger.debug("Failed to record cost for run_id=%s", run_id)

    async def _invoke_with_selection_inner(
        self,
        agent_type: str,
        prompt_package: PromptPackage,
        run_id: str,
        selection: AgentSelection,
        requested_model: str | None = None,
        max_turns: int | None = None,
        allowed_tools: list[str] | None = None,
        timeout_seconds: int | None = None,
    ) -> AgentResult:
        session_id = f"{run_id}-{uuid.uuid4().hex[:8]}"
        run_dir = self._runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_run_files(run_dir, prompt_package, agent_type=agent_type)

        status = self._get_provider_status(selection.provider)
        auth_mode = self._auth_checker.auth_mode_for_provider(selection.provider)
        if status.available and status.reason:
            logger.warning(
                "Provider %s diagnostics for %s: %s",
                selection.provider.value,
                agent_type,
                status.reason,
            )
        if not status.available:
            logger.error(
                "Provider %s unavailable for %s (run_id=%s): %s",
                selection.provider.value,
                agent_type,
                run_id,
                status.reason,
            )
            result = AgentResult(
                response="",
                run_dir=run_dir,
                duration_ms=0,
                session_id=session_id,
                success=False,
                error=status.reason or f"{selection.provider.value} is unavailable",
                provider=selection.provider.value,
                runtime=status.runtime,
                requested_model=requested_model,
                effective_model=selection.model or "",
                auth_mode=auth_mode,
            )
            self._broadcast_runtime_event(
                "agent_invocation_failed",
                {
                    "run_id": run_id,
                    "agent_type": agent_type,
                    "provider": result.provider,
                    "runtime": result.runtime,
                    "model": result.effective_model,
                    "error": result.error[:_PREVIEW_LIMIT],
                },
            )
            return result

        invocation = self._invocation_builder.build(
            prompt_package=prompt_package,
            selection=selection,
            requested_model=requested_model,
            max_turns=max_turns or self._default_max_turns,
            allowed_tools=allowed_tools,
        )
        timeout = timeout_seconds if timeout_seconds is not None else self._timeout_seconds

        logger.info(
            "Invoking agent runtime for %s (run_id=%s, provider=%s, runtime=%s, model=%s)",
            agent_type,
            run_id,
            invocation.provider.value,
            invocation.runtime,
            invocation.effective_model,
        )
        self._broadcast_runtime_event(
            "agent_invocation_started",
            {
                "run_id": run_id,
                "agent_type": agent_type,
                "provider": invocation.provider.value,
                "runtime": invocation.runtime,
                "model": invocation.effective_model,
                **({"diagnostics": status.reason} if status.reason else {}),
            },
        )

        try:
            process = await asyncio.create_subprocess_exec(
                invocation.command,
                *invocation.args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(run_dir),
                env=invocation.env,
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            logger.error("Could not start agent runtime for %s: %s", run_id, exc)
            result = AgentResult(
                response="",
                run_dir=run_dir,
                duration_ms=0,
                session_id=session_id,
                success=False,
                error=str(exc),
                provider=invocation.provider.value,
                runtime=invocation.runtime,
                requested_model=invocation.requested_model,
                effective_model=invocation.effective_model,
                auth_mode=auth_mode,
            )
            self._broadcast_runtime_event(
                "agent_invocation_failed",
                {
                    "run_id": run_id,
                    "agent_type": agent_type,
                    "provider": result.provider,
                    "runtime": result.runtime,
                    "model": result.effective_model,
                    "error": result.error[:_PREVIEW_LIMIT],
                },
            )
            return result

        if invocation.parse_mode == "stream-json":
            result = await self._invoke_claude_stream(
                process=process,
                run_id=run_id,
                agent_type=agent_type,
                run_dir=run_dir,
                session_id=session_id,
                provider=invocation.provider.value,
                runtime=invocation.runtime,
                requested_model=invocation.requested_model,
                effective_model=invocation.effective_model,
                auth_mode=auth_mode,
                timeout_seconds=timeout,
            )
        elif invocation.parse_mode == "jsonl-stream":
            result = await self._invoke_codex_stream(
                process=process,
                run_id=run_id,
                agent_type=agent_type,
                run_dir=run_dir,
                session_id=session_id,
                provider=invocation.provider.value,
                runtime=invocation.runtime,
                requested_model=invocation.requested_model,
                effective_model=invocation.effective_model,
                auth_mode=auth_mode,
                timeout_seconds=timeout,
            )
        else:
            result = await self._invoke_buffered_process(
                process=process,
                run_dir=run_dir,
                session_id=session_id,
                provider=invocation.provider.value,
                runtime=invocation.runtime,
                requested_model=invocation.requested_model,
                effective_model=invocation.effective_model,
                parse_mode=invocation.parse_mode,
                timeout_seconds=timeout,
                auth_mode=auth_mode,
            )

        if result.response:
            (run_dir / "response.md").write_text(result.response, encoding="utf-8")

        self._session_store.record_invocation(
            session_id=result.session_id or session_id,
            agent_type=agent_type,
            prompt_package=prompt_package.model_dump(mode="json"),
            response=result.response,
            token_usage={},
            duration_ms=result.duration_ms,
            metadata={
                "run_id": run_id,
                "provider": invocation.provider.value,
                "runtime": invocation.runtime,
                "requested_model": invocation.requested_model,
                "effective_model": invocation.effective_model,
                "cost_usd": result.cost_usd,
                "first_output_ms": result.first_output_ms,
                "stream_event_count": result.stream_event_count,
                "tool_call_count": result.tool_call_count,
                "auth_mode": result.auth_mode,
            },
        )

        self._index_run(
            run_id=run_id,
            agent_type=agent_type,
            run_dir=run_dir,
            result=result,
            invocation=invocation,
            prompt_package=prompt_package,
        )

        # Backfill metadata.json with actual provider/model (written empty pre-invocation)
        try:
            _meta_path = run_dir / "metadata.json"
            _meta = json.loads(_meta_path.read_text(encoding="utf-8"))
            _meta["provider"] = invocation.provider.value
            _meta["effective_model"] = invocation.effective_model
            _meta_path.write_text(json.dumps(_meta, default=str), encoding="utf-8")
        except Exception:
            pass

        self._broadcast_runtime_event(
            "agent_invocation_completed",
            {
                "run_id": run_id,
                "agent_type": agent_type,
                "provider": result.provider,
                "runtime": result.runtime,
                "model": result.effective_model,
                "duration_ms": result.duration_ms,
                "cost_usd": result.cost_usd,
                "first_output_ms": result.first_output_ms,
                "stream_event_count": result.stream_event_count,
                "tool_call_count": result.tool_call_count,
            },
        )
        logger.info(
            "Agent runtime completed for %s (run_id=%s, provider=%s, duration=%dms, cost=$%.4f)",
            agent_type,
            run_id,
            invocation.provider.value,
            result.duration_ms,
            result.cost_usd,
        )
        return result

    # -- Internal helpers --

    def _check_skill_access(self, agent_type: str) -> None:
        if self._skills_registry:
            check = self._skills_registry.check_action(agent_type, "generate_report")
            if not check.allowed:
                msg = f"SkillsRegistry denied {agent_type}: {check.reason}"
                if self._enforce_skills:
                    raise PermissionError(msg)
                logger.warning(msg)

    def _write_run_files(self, run_dir: Path, package: PromptPackage, agent_type: str = "") -> None:
        for key, value in package.data.items():
            file_path = run_dir / f"{key}.json"
            file_path.write_text(
                json.dumps(value, indent=2, default=str), encoding="utf-8"
            )

        if package.instructions:
            (run_dir / "instructions.md").write_text(package.instructions, encoding="utf-8")

        if package.system_prompt:
            (run_dir / "system_prompt.md").write_text(package.system_prompt, encoding="utf-8")

        if package.corrections:
            (run_dir / "corrections.json").write_text(
                json.dumps(package.corrections, indent=2, default=str), encoding="utf-8"
            )

        if package.skill_context:
            (run_dir / "skill_context.md").write_text(package.skill_context, encoding="utf-8")

        # Write metadata.json for reindex_from_directory() compatibility
        meta = {
            "agent_type": agent_type,
            "bot_ids": package.metadata.get("bot_ids", ""),
            "date": package.metadata.get("date", ""),
            "provider": "",
            "effective_model": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "metadata.json").write_text(
            json.dumps(meta, default=str), encoding="utf-8",
        )

    # -- Async subprocess lifecycle --

    async def _invoke_buffered_process(
        self,
        process: asyncio.subprocess.Process,
        run_dir: Path,
        session_id: str,
        provider: str,
        runtime: str,
        requested_model: str | None,
        effective_model: str,
        parse_mode: str,
        auth_mode: str,
        timeout_seconds: int | None = None,
    ) -> AgentResult:
        timeout = timeout_seconds if timeout_seconds is not None else self._timeout_seconds
        start = datetime.now(timezone.utc)
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            with suppress(ProcessLookupError):
                process.kill()
            with suppress(Exception):
                await process.wait()
            elapsed = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
            logger.error(
                "Agent runtime timed out after %ds for %s",
                timeout,
                run_dir.name,
            )
            return AgentResult(
                response="",
                run_dir=run_dir,
                duration_ms=elapsed,
                session_id=session_id,
                success=False,
                error=f"Timeout after {timeout}s",
                provider=provider,
                runtime=runtime,
                requested_model=requested_model,
                effective_model=effective_model,
                auth_mode=auth_mode,
            )

        elapsed_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if process.returncode != 0:
            logger.error(
                "Agent runtime exited with code %d for %s: %s",
                process.returncode,
                run_dir.name,
                stderr,
            )
            return AgentResult(
                response=stdout,
                run_dir=run_dir,
                duration_ms=elapsed_ms,
                session_id=session_id,
                success=False,
                error=f"Exit code {process.returncode}: {(stderr or stdout)[:500]}",
                provider=provider,
                runtime=runtime,
                requested_model=requested_model,
                effective_model=effective_model,
                auth_mode=auth_mode,
            )

        parsed = self._stream_parser.parse_output(
            stdout=stdout,
            run_dir=run_dir,
            elapsed_ms=elapsed_ms,
            session_id=session_id,
            parse_mode=parse_mode,
            provider=provider,
            runtime=runtime,
            requested_model=requested_model,
            effective_model=effective_model,
            auth_mode=auth_mode,
        )
        return AgentResult(**parsed)

    async def _invoke_claude_stream(
        self,
        process: asyncio.subprocess.Process,
        run_id: str,
        agent_type: str,
        run_dir: Path,
        session_id: str,
        provider: str,
        runtime: str,
        requested_model: str | None,
        effective_model: str,
        auth_mode: str,
        timeout_seconds: int | None = None,
    ) -> AgentResult:
        timeout = timeout_seconds if timeout_seconds is not None else self._timeout_seconds
        start = datetime.now(timezone.utc)
        stdout_path = run_dir / "claude-session.jsonl"
        stderr_path = run_dir / "claude-stderr.log"

        stdout_task = asyncio.create_task(
            self._consume_claude_stdout(
                stdout=process.stdout,
                output_path=stdout_path,
                started_at=start,
                run_id=run_id,
                agent_type=agent_type,
                provider=provider,
                runtime=runtime,
            )
        )
        stderr_task = asyncio.create_task(self._consume_stderr(process.stderr, stderr_path))

        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
            logger.error(
                "Agent runtime timed out after %ds for %s",
                timeout,
                run_id,
            )
            with suppress(ProcessLookupError):
                process.kill()
            with suppress(Exception):
                await process.wait()

        stream_state = await stdout_task
        stderr_text = await stderr_task
        elapsed_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        response = (
            stream_state.final_result.strip()
            or "\n".join(text for text in stream_state.texts if text).strip()
            or "\n".join(stream_state.raw_lines).strip()
        )
        resolved_session_id = stream_state.resolved_session_id or session_id

        if timed_out:
            return AgentResult(
                response=response,
                run_dir=run_dir,
                duration_ms=elapsed_ms,
                session_id=resolved_session_id,
                success=False,
                error=f"Timeout after {timeout}s",
                provider=provider,
                runtime=runtime,
                requested_model=requested_model,
                effective_model=effective_model,
                first_output_ms=stream_state.first_output_ms,
                stream_event_count=stream_state.stream_event_count,
                tool_call_count=stream_state.tool_call_count,
                auth_mode=auth_mode,
            )

        if process.returncode != 0:
            logger.error(
                "Agent runtime exited with code %d for %s: %s",
                process.returncode,
                run_id,
                stderr_text,
            )
            return AgentResult(
                response=response,
                run_dir=run_dir,
                duration_ms=elapsed_ms,
                session_id=resolved_session_id,
                success=False,
                error=f"Exit code {process.returncode}: {(stderr_text or response)[:500]}",
                provider=provider,
                runtime=runtime,
                requested_model=requested_model,
                effective_model=effective_model,
                first_output_ms=stream_state.first_output_ms,
                stream_event_count=stream_state.stream_event_count,
                tool_call_count=stream_state.tool_call_count,
                auth_mode=auth_mode,
            )

        return AgentResult(
            response=response,
            run_dir=run_dir,
            cost_usd=stream_state.cost_usd,
            duration_ms=elapsed_ms,
            session_id=resolved_session_id,
            success=True,
            provider=provider,
            runtime=runtime,
            requested_model=requested_model,
            effective_model=effective_model,
            first_output_ms=stream_state.first_output_ms,
            stream_event_count=stream_state.stream_event_count,
            tool_call_count=stream_state.tool_call_count,
            auth_mode=auth_mode,
        )

    async def _invoke_codex_stream(
        self,
        process: asyncio.subprocess.Process,
        run_id: str,
        agent_type: str,
        run_dir: Path,
        session_id: str,
        provider: str,
        runtime: str,
        requested_model: str | None,
        effective_model: str,
        auth_mode: str,
        timeout_seconds: int | None = None,
    ) -> AgentResult:
        timeout = timeout_seconds if timeout_seconds is not None else self._timeout_seconds
        start = datetime.now(timezone.utc)
        stdout_path = run_dir / "codex-session.jsonl"
        stderr_path = run_dir / "codex-stderr.log"

        stdout_task = asyncio.create_task(
            self._consume_codex_stdout(
                stdout=process.stdout,
                output_path=stdout_path,
                started_at=start,
                run_id=run_id,
                agent_type=agent_type,
                provider=provider,
                runtime=runtime,
            )
        )
        stderr_task = asyncio.create_task(self._consume_stderr(process.stderr, stderr_path))

        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
            logger.error(
                "Agent runtime timed out after %ds for %s",
                timeout,
                run_id,
            )
            with suppress(ProcessLookupError):
                process.kill()
            with suppress(Exception):
                await process.wait()

        stream_state = await stdout_task
        stderr_text = await stderr_task
        elapsed_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        response = (
            stream_state.final_result.strip()
            or "\n".join(text for text in stream_state.texts if text).strip()
            or "\n".join(stream_state.raw_lines).strip()
        )
        resolved_session_id = stream_state.resolved_session_id or session_id

        if timed_out:
            return AgentResult(
                response=response,
                run_dir=run_dir,
                duration_ms=elapsed_ms,
                session_id=resolved_session_id,
                success=False,
                error=f"Timeout after {timeout}s",
                provider=provider,
                runtime=runtime,
                requested_model=requested_model,
                effective_model=effective_model,
                first_output_ms=stream_state.first_output_ms,
                stream_event_count=stream_state.stream_event_count,
                tool_call_count=stream_state.tool_call_count,
                auth_mode=auth_mode,
            )

        if process.returncode != 0:
            logger.error(
                "Agent runtime exited with code %d for %s: %s",
                process.returncode,
                run_id,
                stderr_text,
            )
            return AgentResult(
                response=response,
                run_dir=run_dir,
                duration_ms=elapsed_ms,
                session_id=resolved_session_id,
                success=False,
                error=f"Exit code {process.returncode}: {(stderr_text or response)[:500]}",
                provider=provider,
                runtime=runtime,
                requested_model=requested_model,
                effective_model=effective_model,
                first_output_ms=stream_state.first_output_ms,
                stream_event_count=stream_state.stream_event_count,
                tool_call_count=stream_state.tool_call_count,
                auth_mode=auth_mode,
            )

        return AgentResult(
            response=response,
            run_dir=run_dir,
            cost_usd=stream_state.cost_usd,
            duration_ms=elapsed_ms,
            session_id=resolved_session_id,
            success=True,
            provider=provider,
            runtime=runtime,
            requested_model=requested_model,
            effective_model=effective_model,
            first_output_ms=stream_state.first_output_ms,
            stream_event_count=stream_state.stream_event_count,
            tool_call_count=stream_state.tool_call_count,
            auth_mode=auth_mode,
        )

    async def _consume_claude_stdout(
        self,
        stdout: asyncio.StreamReader | None,
        output_path: Path,
        started_at: datetime,
        run_id: str,
        agent_type: str,
        provider: str,
        runtime: str,
    ) -> StreamState:
        state = StreamState()
        if stdout is None:
            output_path.write_text("", encoding="utf-8")
            return state
        with output_path.open("w", encoding="utf-8") as raw_file:
            while True:
                line_bytes = await stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace").rstrip("\n\r")
                if not line:
                    continue
                if state.first_output_ms == 0:
                    state.first_output_ms = int(
                        (datetime.now(timezone.utc) - started_at).total_seconds() * 1000
                    )
                state.raw_lines.append(line)
                state.stream_event_count += 1
                raw_file.write(line + "\n")
                self._stream_parser.parse_claude_stream_line(
                    line=line,
                    state=state,
                    run_id=run_id,
                    agent_type=agent_type,
                    provider=provider,
                    runtime=runtime,
                )
        return state

    async def _consume_codex_stdout(
        self,
        stdout: asyncio.StreamReader | None,
        output_path: Path,
        started_at: datetime,
        run_id: str,
        agent_type: str,
        provider: str,
        runtime: str,
    ) -> StreamState:
        state = StreamState()
        if stdout is None:
            output_path.write_text("", encoding="utf-8")
            return state
        with output_path.open("w", encoding="utf-8") as raw_file:
            while True:
                line_bytes = await stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace").rstrip("\n\r")
                if not line:
                    continue
                if state.first_output_ms == 0:
                    state.first_output_ms = int(
                        (datetime.now(timezone.utc) - started_at).total_seconds() * 1000
                    )
                state.raw_lines.append(line)
                state.stream_event_count += 1
                raw_file.write(line + "\n")
                self._stream_parser.parse_codex_stream_line(
                    line=line,
                    state=state,
                    run_id=run_id,
                    agent_type=agent_type,
                    provider=provider,
                    runtime=runtime,
                )
        return state

    async def _consume_stderr(
        self,
        stderr: asyncio.StreamReader | None,
        output_path: Path,
    ) -> str:
        if stderr is None:
            output_path.write_text("", encoding="utf-8")
            return ""

        stderr_bytes = await stderr.read()
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        output_path.write_text(stderr_text, encoding="utf-8")
        return stderr_text

    # -- Internal helpers --

    def _get_provider_status(self, provider: AgentProvider) -> ProviderReadiness:
        return self._auth_checker.get_provider_status(provider)

    def _resolve_workflow(self, agent_type: str) -> AgentWorkflow | None:
        try:
            return AgentWorkflow(agent_type)
        except ValueError:
            return None

    def _broadcast_runtime_event(self, event_type: str, data: dict) -> None:
        if self._event_stream is None:
            return
        self._event_stream.broadcast(event_type, data)
