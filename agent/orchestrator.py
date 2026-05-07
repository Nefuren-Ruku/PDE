"""Main orchestration loop for the autonomous research agent."""

from __future__ import annotations

import re
import time
from typing import Any, Callable

from agent import parser
from agent.memory import Memory
from agent.prompts import AGENT_SYSTEM_PROMPT

ToolFunction = Callable[..., str]


class ResearchAgent:
    """Autonomous loop that asks an LLM to plan and call repository tools."""

    def __init__(
        self,
        llm: Any,
        memory: Memory,
        tools: dict[str, ToolFunction],
        max_steps: int = 50,
        max_time_minutes: int = 480,
    ) -> None:
        """Initialize the research agent."""
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.max_steps = max_steps
        self.max_time_minutes = max_time_minutes
        self.tool_definitions = self._build_tool_definitions()

    @staticmethod
    def _build_tool_definitions() -> list[dict[str, Any]]:
        """Build Anthropic-style tool schemas."""
        return [
            {
                "name": "run_python",
                "description": "Execute Python code in the project environment.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "timeout": {"type": "integer", "default": 300},
                    },
                    "required": ["code"],
                },
            },
            {
                "name": "file_io",
                "description": "Read, write, or list files inside the project directory.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["read", "write", "list_dir"]},
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["operation", "path"],
                },
            },
            {
                "name": "log_experiment",
                "description": "Append a structured experiment record.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "integer"},
                        "version": {"type": "string"},
                        "hypothesis": {"type": "string"},
                        "code_changes": {"type": "string"},
                        "metrics": {"type": "object"},
                        "conclusion": {"type": "string"},
                    },
                    "required": [
                        "task_id",
                        "version",
                        "hypothesis",
                        "code_changes",
                        "metrics",
                        "conclusion",
                    ],
                },
            },
            {
                "name": "submit_final",
                "description": "Package final deliverables using the repository packaging logic.",
                "input_schema": {
                    "type": "object",
                    "properties": {"task_id": {"type": "integer"}},
                    "required": ["task_id"],
                },
            },
        ]

    @staticmethod
    def _result_is_error(result: str) -> bool:
        """Return True when a tool result represents a failure."""
        if "[ERROR]" in result or "[TIMEOUT" in result:
            return True
        match = re.search(r"Exit code:\s*(-?\d+)", result)
        return bool(match and int(match.group(1)) != 0)

    def _execute_tool(self, call: dict[str, Any], task_id: int) -> tuple[str, str]:
        """Execute one parsed tool call."""
        name = str(call.get("name", ""))
        args = call.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        if name == "submit_final" and "task_id" not in args:
            args["task_id"] = task_id
        if name not in self.tools:
            return name, f"[ERROR] Unknown tool: {name}"
        try:
            result = self.tools[name](**args)
        except TypeError as exc:
            result = f"[ERROR] Invalid arguments for {name}: {exc}"
        except Exception as exc:
            result = f"[ERROR] Tool {name} failed: {exc}"
        return name, result

    def run(self, task_id: int = 1) -> None:
        """Run the autonomous research loop."""
        start_time = time.monotonic()
        print(f"[agent] Starting autonomous run for task_id={task_id}.")
        self.memory.add_system_message(AGENT_SYSTEM_PROMPT)
        self.memory.add_user_message(
            "Start an autonomous research run. First inspect the repository structure and available validation paths."
        )

        consecutive_no_progress = 0
        retry_count = 0

        for step in range(1, self.max_steps + 1):
            elapsed_minutes = (time.monotonic() - start_time) / 60.0
            if elapsed_minutes >= self.max_time_minutes:
                print("[agent] Time budget reached.")
                self.memory.add_user_message("Time budget reached. Package the best available result.")
                break

            print(f"[agent] Step {step}/{self.max_steps} elapsed={elapsed_minutes:.2f}m")
            response = self.llm.chat(self.memory.get_context(), tools=self.tool_definitions)
            content = str(response.get("content") or "")
            tool_calls = parser.parse_tool_calls(response)

            if content:
                self.memory.add_assistant_message(content)
                preview = content.replace("\n", " ")
                if len(preview) > 500:
                    preview = preview[:497] + "..."
                print(f"[assistant] {preview}")

            if content.startswith("[ERROR] LLM request failed") and not tool_calls:
                print("[agent] LLM call failed. Fix the API key, model, base URL, or network before retrying.")
                return

            if not tool_calls:
                print("[agent] No tool call returned.")
                consecutive_no_progress += 1
                if consecutive_no_progress >= 5:
                    self.memory.add_user_message(
                        "No tool progress has occurred for several turns. Use a repository inspection tool next."
                    )
                    consecutive_no_progress = 0
                continue

            consecutive_no_progress = 0
            for call in tool_calls:
                print(f"[tool] Calling {call.get('name', '<unknown>')}")
                name, result = self._execute_tool(call, task_id)
                self.memory.add_tool_result(name, result)
                result_preview = result.replace("\n", " ")
                if len(result_preview) > 700:
                    result_preview = result_preview[:697] + "..."
                print(f"[tool:{name}] {result_preview}")

                if name == "log_experiment" and not self._result_is_error(result):
                    retry_count = 0

                if name == "submit_final":
                    print("[agent] Final packaging requested; stopping.")
                    return

                if self._result_is_error(result):
                    retry_count += 1
                    self.memory.add_user_message(
                        "The previous tool call failed. Analyze the error and apply one targeted fix before retrying."
                    )
                    if retry_count >= 3:
                        self.memory.add_user_message(
                            "Three consecutive tool failures occurred. Inspect the smallest relevant file before any further change."
                        )
                        retry_count = 0

        if "submit_final" in self.tools:
            name, result = self._execute_tool({"name": "submit_final", "args": {"task_id": task_id}}, task_id)
            self.memory.add_tool_result(name, result)
            print(f"[tool:{name}] {result}")
