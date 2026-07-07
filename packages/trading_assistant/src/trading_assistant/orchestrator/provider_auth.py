"""Provider authentication and command resolution for agent runtimes."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from trading_assistant.schemas.agent_preferences import AgentProvider, ProviderReadiness

logger = logging.getLogger(__name__)

_CLAUDE_RUNTIME = "claude_cli"
_CODEX_RUNTIME = "codex_cli"
_CLI_PREVIEW_TIMEOUT_SECONDS = 5
_PROVIDER_STATUS_CACHE_TTL_SECONDS = 30
_CODEX_AUTH_MODE = "chatgpt"


@dataclass
class ClaudeAuthStatus:
    available: bool
    reason: str = ""
    auth_method: str = ""
    subscription_type: str = ""


@dataclass
class CodexAuthStatus:
    available: bool
    reason: str = ""
    auth_mode: str = ""
    diagnostics: list[str] = field(default_factory=list)


class ProviderAuthChecker:
    """Checks provider readiness: command resolution, auth status, caching."""

    def __init__(
        self,
        claude_command: str = "claude",
        claude_command_args: list[str] | None = None,
        codex_command: str = "codex",
        codex_command_args: list[str] | None = None,
        zai_api_key: str = "",
        openrouter_api_key: str = "",
    ) -> None:
        self._claude_command = claude_command
        self._claude_command_args = claude_command_args or []
        self._codex_command = codex_command
        self._codex_command_args = codex_command_args or []
        self._zai_api_key = zai_api_key
        self._openrouter_api_key = openrouter_api_key
        self._provider_status_cache: dict[AgentProvider, ProviderReadiness] = {}
        self._provider_status_checked_at: dict[AgentProvider, datetime] = {}

    @property
    def zai_api_key(self) -> str:
        return self._zai_api_key

    @zai_api_key.setter
    def zai_api_key(self, value: str) -> None:
        self._zai_api_key = value

    @property
    def openrouter_api_key(self) -> str:
        return self._openrouter_api_key

    @openrouter_api_key.setter
    def openrouter_api_key(self, value: str) -> None:
        self._openrouter_api_key = value

    def get_provider_status(self, provider: AgentProvider) -> ProviderReadiness:
        cached = self._provider_status_cache.get(provider)
        checked_at = self._provider_status_checked_at.get(provider)
        if cached is not None and (
            checked_at is None
            or (datetime.now(timezone.utc) - checked_at).total_seconds()
            < _PROVIDER_STATUS_CACHE_TTL_SECONDS
        ):
            return cached

        if provider == AgentProvider.CLAUDE_MAX:
            auth_status = self.claude_max_auth_status()
            status = ProviderReadiness(
                provider=provider,
                available=auth_status.available,
                runtime=_CLAUDE_RUNTIME,
                reason=auth_status.reason,
            )
        elif provider == AgentProvider.CODEX_PRO:
            available, reason = self.command_ready(
                self._codex_command,
                self._codex_command_args,
                strict_execution=True,
            )
            if not available:
                status = ProviderReadiness(
                    provider=provider,
                    available=False,
                    runtime=_CODEX_RUNTIME,
                    reason=reason,
                )
            else:
                auth_status = self.codex_auth_status()
                status = ProviderReadiness(
                    provider=provider,
                    available=auth_status.available,
                    runtime=_CODEX_RUNTIME,
                    reason=auth_status.reason or "; ".join(auth_status.diagnostics),
                )
        elif provider == AgentProvider.ZAI_CODING_PLAN:
            cli_available, cli_reason = self.command_ready(
                self._claude_command,
                self._claude_command_args,
                strict_execution=True,
            )
            status = ProviderReadiness(
                provider=provider,
                available=bool(self._zai_api_key.strip()) and cli_available,
                runtime=_CLAUDE_RUNTIME,
                reason=(
                    ""
                    if self._zai_api_key.strip() and cli_available
                    else (
                        "ZAI_API_KEY is not configured"
                        if not self._zai_api_key.strip()
                        else cli_reason
                    )
                ),
            )
        else:
            cli_available, cli_reason = self.command_ready(
                self._claude_command,
                self._claude_command_args,
                strict_execution=True,
            )
            status = ProviderReadiness(
                provider=provider,
                available=bool(self._openrouter_api_key.strip()) and cli_available,
                runtime=_CLAUDE_RUNTIME,
                reason=(
                    ""
                    if self._openrouter_api_key.strip() and cli_available
                    else (
                        "OPENROUTER_API_KEY is not configured"
                        if not self._openrouter_api_key.strip()
                        else cli_reason
                    )
                ),
            )

        self._provider_status_cache[provider] = status
        self._provider_status_checked_at[provider] = datetime.now(timezone.utc)
        return status

    def invalidate_cache(self, provider: AgentProvider | None = None) -> None:
        if provider is None:
            self._provider_status_cache.clear()
            self._provider_status_checked_at.clear()
            return
        self._provider_status_cache.pop(provider, None)
        self._provider_status_checked_at.pop(provider, None)

    def claude_max_auth_status(self) -> ClaudeAuthStatus:
        resolved = self.resolve_command(self._claude_command)
        if not resolved:
            return ClaudeAuthStatus(
                available=False,
                reason=f"Command not found: {self._claude_command}",
            )

        try:
            result = subprocess.run(
                [resolved, *self._claude_command_args, "auth", "status"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=_CLI_PREVIEW_TIMEOUT_SECONDS,
                check=False,
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            return ClaudeAuthStatus(
                available=False,
                reason=f"Claude auth check failed for {self._claude_command}: {exc}",
            )
        except subprocess.TimeoutExpired:
            return ClaudeAuthStatus(
                available=False,
                reason=f"Claude auth check timed out for {self._claude_command}",
            )

        if result.returncode != 0:
            details = (result.stderr or result.stdout).strip()
            return ClaudeAuthStatus(
                available=False,
                reason=(
                    "Claude Max login check failed: "
                    f"{details or f'exit code {result.returncode}'}"
                ),
            )

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return ClaudeAuthStatus(
                available=False,
                reason="Claude auth status returned invalid JSON",
            )

        logged_in = bool(payload.get("loggedIn"))
        auth_method = str(payload.get("authMethod", "")).strip()
        subscription_type = str(payload.get("subscriptionType", "")).strip().lower()

        if not logged_in:
            return ClaudeAuthStatus(
                available=False,
                reason="Claude Max login required: run 'claude auth login'",
                auth_method=auth_method,
                subscription_type=subscription_type,
            )
        if auth_method != "claude.ai":
            return ClaudeAuthStatus(
                available=False,
                reason=f"Claude Max requires claude.ai auth (found {auth_method or 'unknown'})",
                auth_method=auth_method,
                subscription_type=subscription_type,
            )
        if subscription_type != "max":
            found = subscription_type or "unknown"
            return ClaudeAuthStatus(
                available=False,
                reason=f"Claude Max subscription required (found {found})",
                auth_method=auth_method,
                subscription_type=subscription_type,
            )
        return ClaudeAuthStatus(
            available=True,
            auth_method=auth_method,
            subscription_type=subscription_type,
        )

    def codex_auth_status(self) -> CodexAuthStatus:
        codex_home = self.codex_home()
        diagnostics = self.codex_runtime_diagnostics(codex_home)
        auth_path = codex_home / "auth.json"
        if not auth_path.exists():
            return CodexAuthStatus(
                available=False,
                reason=f"Codex ChatGPT auth file not found: {auth_path}",
                diagnostics=diagnostics,
            )

        try:
            payload = json.loads(auth_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return CodexAuthStatus(
                available=False,
                reason=f"Codex auth check failed for {auth_path}: {exc}",
                diagnostics=diagnostics,
            )

        auth_mode = str(payload.get("auth_mode", "")).strip().lower()
        if auth_mode != _CODEX_AUTH_MODE:
            return CodexAuthStatus(
                available=False,
                reason=(
                    "Codex Pro requires chatgpt auth mode "
                    f"(found {auth_mode or 'unknown'})"
                ),
                auth_mode=auth_mode,
                diagnostics=diagnostics,
            )
        if payload.get("OPENAI_API_KEY"):
            return CodexAuthStatus(
                available=False,
                reason="Codex Pro requires ChatGPT auth without OPENAI_API_KEY fallback",
                auth_mode=auth_mode,
                diagnostics=diagnostics,
            )
        if not self.codex_has_tokens(payload):
            return CodexAuthStatus(
                available=False,
                reason=(
                    "Codex Pro requires local ChatGPT tokens in auth.json; "
                    "re-authenticate before using the subscription-backed profile"
                ),
                auth_mode=auth_mode,
                diagnostics=diagnostics,
            )
        return CodexAuthStatus(
            available=True,
            auth_mode=auth_mode,
            diagnostics=diagnostics,
        )

    def codex_has_tokens(self, payload: dict) -> bool:
        tokens = payload.get("tokens")
        if not isinstance(tokens, dict):
            return False
        for key in ("access_token", "id_token", "refresh_token"):
            value = tokens.get(key)
            if isinstance(value, str) and value.strip():
                return True
        return False

    def codex_runtime_diagnostics(self, codex_home: Path) -> list[str]:
        diagnostics: list[str] = []
        config_path = codex_home / "config.toml"
        if config_path.exists():
            try:
                config = tomllib.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError):
                config = {}
            effort = str(config.get("model_reasoning_effort", "")).strip().lower()
            if effort == "xhigh":
                diagnostics.append(
                    "Codex config uses model_reasoning_effort=xhigh, which can increase "
                    "orchestrator latency"
                )

        codex_log = self._find_latest_codex_log()
        if codex_log is not None:
            try:
                tail = self._read_text_tail(codex_log).lower()
            except OSError:
                tail = ""
            if "database is locked" in tail or "slow statement" in tail:
                diagnostics.append(
                    "Recent Codex logs show shared state lock/slow-write symptoms under "
                    f"{codex_home}"
                )
        return diagnostics

    def _find_latest_codex_log(self) -> Path | None:
        if os.name != "nt":
            return None
        logs_root = Path.home() / "AppData" / "Roaming" / "Code" / "logs"
        if not logs_root.exists():
            return None
        candidates = sorted(
            logs_root.glob("**/openai.chatgpt/Codex.log"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _read_text_tail(self, path: Path, max_bytes: int = 65536) -> str:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")

    def codex_home(self) -> Path:
        raw = os.environ.get("CODEX_HOME", "").strip()
        return Path(raw).expanduser() if raw else (Path.home() / ".codex")

    def auth_mode_for_provider(self, provider: AgentProvider) -> str:
        if provider == AgentProvider.CLAUDE_MAX:
            return "claude.ai:max"
        if provider == AgentProvider.CODEX_PRO:
            return self.codex_auth_status().auth_mode or "unknown"
        if provider == AgentProvider.ZAI_CODING_PLAN:
            return "anthropic-compat:zai"
        if provider == AgentProvider.OPENROUTER:
            return "anthropic-compat:openrouter"
        return "local-cli"

    def command_ready(
        self,
        command: str,
        command_args: list[str] | None = None,
        strict_execution: bool = False,
    ) -> tuple[bool, str]:
        resolved = self.resolve_command(command)
        if not resolved:
            return False, f"Command not found: {command}"
        if not strict_execution:
            return True, ""

        try:
            result = subprocess.run(
                [resolved, *(command_args or []), "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=_CLI_PREVIEW_TIMEOUT_SECONDS,
                check=False,
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            return False, f"Command preflight failed for {command}: {exc}"
        except subprocess.TimeoutExpired:
            return False, f"Command preflight timed out for {command}"

        if result.returncode != 0:
            details = (result.stderr or result.stdout).strip()
            return False, (
                f"Command preflight failed for {command}: "
                f"{details or f'exit code {result.returncode}'}"
            )
        return True, ""

    def resolve_command(self, command: str) -> str | None:
        candidate = command.strip()
        if not candidate:
            return None

        resolved = shutil.which(candidate)
        if resolved:
            resolved_path = Path(resolved)
            if os.name == "nt" and not resolved_path.suffix:
                for suffix in (".cmd", ".exe", ".bat"):
                    suffixed = Path(f"{resolved}{suffix}")
                    if suffixed.exists():
                        return str(suffixed)
            return str(resolved_path)

        direct = Path(candidate).expanduser()
        if direct.exists():
            return str(direct)

        if os.name == "nt" and not direct.suffix:
            for suffix in (".cmd", ".exe", ".bat"):
                suffixed = Path(f"{candidate}{suffix}")
                if suffixed.exists():
                    return str(suffixed)
        return None

    def require_resolved_command(self, command: str) -> str:
        resolved = self.resolve_command(command)
        if resolved is None:
            raise FileNotFoundError(f"Command not found: {command}")
        return resolved
