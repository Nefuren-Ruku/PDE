"""LLM API wrapper for the autonomous research agent."""

from __future__ import annotations

import json
from typing import Any

import httpx

try:
    import anthropic
except ImportError:  # pragma: no cover - handled at runtime.
    anthropic = None  # type: ignore[assignment]


class LLMInterface:
    """Thin wrapper around the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        base_url: str | None = None,
        provider: str = "anthropic",
        reasoning_effort: str = "high",
        thinking_enabled: bool = True,
    ) -> None:
        """Initialize the LLM interface.

        Args:
            api_key: API key for the Anthropic-compatible endpoint.
            model: Model name to request.
            base_url: Optional custom base URL for compatible providers.
            provider: API provider, either ``anthropic`` or ``deepseek``.
            reasoning_effort: Reasoning effort for providers that support it.
            thinking_enabled: Whether to request thinking mode where supported.
        """
        self.api_key = api_key
        self.model = model
        self.provider = provider.lower()
        self.base_url = base_url or self._default_base_url(self.provider)
        self.reasoning_effort = reasoning_effort
        self.thinking_enabled = thinking_enabled
        self._client: Any | None = None

    @staticmethod
    def _default_base_url(provider: str) -> str | None:
        """Return the default base URL for a provider."""
        if provider == "deepseek":
            return "https://api.deepseek.com"
        return None

    def _get_client(self) -> Any:
        """Create and cache the Anthropic client."""
        if anthropic is None:
            raise RuntimeError(
                "The 'anthropic' package is not installed. Install it before running the agent."
            )
        if self._client is None:
            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    @staticmethod
    def _split_system_messages(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
        """Convert system-role messages into the Anthropic top-level system field."""
        system_parts: list[str] = []
        chat_messages: list[dict[str, str]] = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role in {"user", "assistant"}:
                chat_messages.append({"role": role, "content": content})

        return ("\n\n".join(system_parts) if system_parts else None, chat_messages)

    @staticmethod
    def _normalize_tool_args(raw_input: Any) -> dict[str, Any]:
        """Return tool arguments as a dictionary."""
        if isinstance(raw_input, dict):
            return raw_input
        if isinstance(raw_input, str):
            try:
                parsed = json.loads(raw_input)
            except json.JSONDecodeError:
                return {"raw": raw_input}
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        return {"value": raw_input}

    @staticmethod
    def _to_openai_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Convert Anthropic-style tools to OpenAI-compatible tools."""
        if not tools:
            return None
        converted: list[dict[str, Any]] = []
        for tool in tools:
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {"type": "object"}),
                    },
                }
            )
        return converted

    def _chat_deepseek(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Send a request to DeepSeek's OpenAI-compatible chat endpoint."""
        if not self.base_url:
            return {"content": "[ERROR] DeepSeek base URL is not configured.", "tool_calls": None}

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": 4096,
        }
        openai_tools = self._to_openai_tools(tools)
        if openai_tools:
            payload["tools"] = openai_tools
            payload["tool_choice"] = "auto"
        if self.model.startswith("deepseek-v4"):
            payload["reasoning_effort"] = self.reasoning_effort
            payload["thinking"] = {"type": "enabled" if self.thinking_enabled else "disabled"}

        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        response = httpx.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return {"content": "[ERROR] DeepSeek returned no choices.", "tool_calls": None}

        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        reasoning_content = message.get("reasoning_content")
        if reasoning_content:
            content = f"[reasoning]\n{reasoning_content}\n\n[answer]\n{content}".strip()

        normalized_calls: list[dict[str, Any]] = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name")
            if not name:
                continue
            normalized_calls.append(
                {
                    "name": name,
                    "args": self._normalize_tool_args(function.get("arguments", "{}")),
                }
            )
        return {"content": content, "tool_calls": normalized_calls or None}

    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Send a chat request and return text plus normalized tool calls.

        Args:
            messages: Messages using roles ``system``, ``user``, and ``assistant``.
            tools: Optional Anthropic-style tool definitions.

        Returns:
            A dictionary with ``content`` and ``tool_calls`` keys. On failure,
            ``content`` contains a friendly ``[ERROR]`` message and
            ``tool_calls`` is ``None``.
        """
        try:
            if self.provider == "deepseek":
                return self._chat_deepseek(messages, tools)
            if self.provider != "anthropic":
                return {
                    "content": f"[ERROR] Unsupported LLM provider: {self.provider}",
                    "tool_calls": None,
                }

            system, chat_messages = self._split_system_messages(messages)
            if not chat_messages:
                chat_messages = [
                    {
                        "role": "user",
                        "content": "Begin by auditing the current repository and decide the next action.",
                    }
                ]

            create_kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": chat_messages,
            }
            if system:
                create_kwargs["system"] = system
            if tools:
                create_kwargs["tools"] = tools

            response = self._get_client().messages.create(**create_kwargs)

            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in getattr(response, "content", []):
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    text_parts.append(getattr(block, "text", ""))
                elif block_type == "tool_use":
                    tool_calls.append(
                        {
                            "name": getattr(block, "name", ""),
                            "args": self._normalize_tool_args(getattr(block, "input", {})),
                        }
                    )

            return {
                "content": "\n".join(part for part in text_parts if part).strip(),
                "tool_calls": tool_calls or None,
            }
        except Exception as exc:  # pragma: no cover - depends on external service.
            return {
                "content": f"[ERROR] LLM request failed: {exc}",
                "tool_calls": None,
            }
