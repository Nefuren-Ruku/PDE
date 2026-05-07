"""Conversation memory and experiment trace utilities."""

from __future__ import annotations

from collections import deque
from typing import Deque


class Memory:
    """Bounded message memory for the research agent."""

    def __init__(self, max_turns: int = 20) -> None:
        """Initialize the memory window.

        Args:
            max_turns: Approximate number of recent dialogue turns to retain.
        """
        self.max_turns = max_turns
        self.system_message: dict[str, str] | None = None
        self.messages: Deque[dict[str, str]] = deque()
        self.archived_messages: list[dict[str, str]] = []
        self._max_messages = max(2, max_turns * 2)

    def _append(self, role: str, content: str) -> None:
        """Append a message while keeping the active window bounded."""
        self.messages.append({"role": role, "content": content})
        while len(self.messages) > self._max_messages:
            self.archived_messages.append(self.messages.popleft())

    def add_system_message(self, content: str) -> None:
        """Save the system prompt, always kept at the front of context."""
        self.system_message = {"role": "system", "content": content}

    def add_user_message(self, content: str) -> None:
        """Append a user message."""
        self._append("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Append an assistant message."""
        self._append("assistant", content)

    def add_tool_result(self, tool_name: str, result: str) -> None:
        """Append a tool result as a user-visible observation."""
        content = f"Tool result from {tool_name}:\n{result}"
        self._append("user", content)

    def get_context(self) -> list[dict[str, str]]:
        """Return the active context messages."""
        context: list[dict[str, str]] = []
        if self.system_message is not None:
            context.append(self.system_message)
        if self.archived_messages:
            summary = self.summarize_old_experiments()
            if summary:
                context.append({"role": "user", "content": summary})
        context.extend(self.messages)
        return context

    def summarize_old_experiments(self) -> str:
        """Summarize messages that were evicted from the active window."""
        if not self.archived_messages:
            return ""

        lines = ["Summary of earlier context and experiments:"]
        for message in self.archived_messages[-20:]:
            role = message.get("role", "unknown")
            content = message.get("content", "").strip().replace("\n", " ")
            if len(content) > 240:
                content = content[:237] + "..."
            lines.append(f"- {role}: {content}")
        return "\n".join(lines)
