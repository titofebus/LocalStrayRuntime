"""Stream filtering, sanitization, and structured tool-call extraction."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple
import uuid

STRUCTURED_TOOL_CALLS_V1 = "structured_tool_calls_v1"
MALFORMED_TOOL_CALL_MARKER = "[Malformed tool call omitted]"


def should_use_prefix_cache(tools: Any) -> bool:
    """Use token-prefix caching; rendered tool schemas participate in cache matching."""
    _ = tools
    return True


def extract_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract <tool_call> JSON blocks into structured tool calls and clean text."""
    if not text:
        return "", []

    tool_calls: List[Dict[str, Any]] = []

    def _replace_tool_call(match: re.Match) -> str:
        raw_content = match.group(1).strip()
        if not raw_content:
            return MALFORMED_TOOL_CALL_MARKER
        try:
            call_obj = json.loads(raw_content)
            if not isinstance(call_obj, dict):
                return MALFORMED_TOOL_CALL_MARKER
            name = call_obj.get("name")
            if not name or not isinstance(name, str):
                return MALFORMED_TOOL_CALL_MARKER
            args = call_obj.get("arguments", {})
            if isinstance(args, (dict, list)):
                args_str = json.dumps(args)
            elif isinstance(args, str):
                args_str = args
            else:
                args_str = json.dumps(args)

            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args_str,
                },
            })
            return ""
        except Exception:
            return MALFORMED_TOOL_CALL_MARKER

    clean_text = re.sub(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        _replace_tool_call,
        text,
        flags=re.DOTALL,
    )

    if "<tool_call>" in clean_text:
        clean_text = re.sub(
            r"<tool_call>.*$",
            MALFORMED_TOOL_CALL_MARKER,
            clean_text,
            flags=re.DOTALL,
        )

    clean_text = clean_text.replace("</tool_call>", "").replace("<tool_call>", "")

    if not clean_text.strip():
        clean_text = ""
    else:
        clean_text = clean_text.strip()

    return clean_text, tool_calls


class ControlTokenFilter:
    _MARKERS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")

    def __init__(self) -> None:
        self._pending = ""
        self._stopped = False

    def push(self, text: str) -> tuple[str, bool]:
        if self._stopped:
            return "", True

        combined = self._pending + text
        marker_positions = [
            position
            for marker in self._MARKERS
            if (position := combined.find(marker)) >= 0
        ]
        if marker_positions:
            self._pending = ""
            self._stopped = True
            return combined[: min(marker_positions)], True

        pending_length = 0
        for marker in self._MARKERS:
            for length in range(1, min(len(marker), len(combined)) + 1):
                if combined.endswith(marker[:length]):
                    pending_length = max(pending_length, length)

        if pending_length == 0:
            self._pending = ""
            return combined, False

        self._pending = combined[-pending_length:]
        return combined[:-pending_length], False

    def finish(self) -> str:
        if self._stopped:
            return ""
        pending = self._pending
        self._pending = ""
        return pending


class ToolMarkupFilter:
    _OPEN = "<tool_call>"
    _CLOSE = "</tool_call>"

    def __init__(self) -> None:
        self._pending = ""
        self._inside = False
        self._current = ""
        self._captured: list[str] = []

    def push(self, text: str) -> str:
        data = self._pending + text
        self._pending = ""
        visible: list[str] = []

        while data:
            if self._inside:
                captured_candidate = self._current + data
                close_position = captured_candidate.find(self._CLOSE)
                if close_position < 0:
                    self._current = captured_candidate
                    break
                close_end = close_position + len(self._CLOSE)
                self._captured.append(captured_candidate[:close_end])
                self._current = ""
                self._inside = False
                data = captured_candidate[close_end:]
                continue

            open_position = data.find(self._OPEN)
            if open_position >= 0:
                visible.append(data[:open_position])
                self._inside = True
                self._current = self._OPEN
                data = data[open_position + len(self._OPEN):]
                continue

            pending_length = 0
            for length in range(1, min(len(self._OPEN), len(data)) + 1):
                if data.endswith(self._OPEN[:length]):
                    pending_length = length
            if pending_length:
                visible.append(data[:-pending_length])
                self._pending = data[-pending_length:]
            else:
                visible.append(data)
            break

        return "".join(visible)

    def finish(self) -> str:
        if self._inside:
            return ""
        pending = self._pending
        self._pending = ""
        return pending

    @property
    def captured_text(self) -> str:
        return "".join(self._captured)
