"""Parsing helpers for LLM replies."""

from __future__ import annotations

import json
import re
from typing import Any

TOOL_NAMES = {"run_python", "file_io", "log_experiment", "submit_final"}


def _coerce_args(value: Any) -> dict[str, Any]:
    """Normalize a parsed argument value to a dictionary."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {"value": value}


def extract_code_blocks(text: str) -> list[dict[str, str]]:
    """Extract Markdown Python code blocks from text."""
    blocks: list[dict[str, str]] = []
    pattern = re.compile(r"```(?P<lang>python|py)\s*\n(?P<code>.*?)```", re.IGNORECASE | re.DOTALL)
    for match in pattern.finditer(text):
        blocks.append({"language": "python", "code": match.group("code").strip()})
    return blocks


def _iter_json_objects(text: str) -> list[Any]:
    """Find JSON objects embedded in free-form text."""
    decoder = json.JSONDecoder()
    objects: list[Any] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        objects.append(obj)
    return objects


def parse_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse tool calls from structured responses or text fallbacks."""
    direct_calls = response.get("tool_calls")
    if direct_calls:
        normalized: list[dict[str, Any]] = []
        for call in direct_calls:
            if not isinstance(call, dict):
                continue
            name = call.get("name") or call.get("tool")
            args = call.get("args", call.get("arguments", {}))
            if name:
                normalized.append({"name": str(name), "args": _coerce_args(args)})
        return normalized

    content = str(response.get("content") or "")
    parsed_calls: list[dict[str, Any]] = []

    for obj in _iter_json_objects(content):
        if not isinstance(obj, dict):
            continue
        name = obj.get("tool") or obj.get("name")
        if not name:
            continue
        args = obj.get("args", obj.get("arguments", {}))
        parsed_calls.append({"name": str(name), "args": _coerce_args(args)})

    if parsed_calls:
        return parsed_calls

    if "run_python" in content:
        blocks = extract_code_blocks(content)
        if blocks:
            return [{"name": "run_python", "args": {"code": blocks[0]["code"]}}]

    for tool_name in TOOL_NAMES - {"run_python"}:
        if tool_name in content:
            return [{"name": tool_name, "args": {}}]

    return []
