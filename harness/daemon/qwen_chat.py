import json
from typing import Any, Mapping


def canonical_tool_schema_json(tools: Any) -> str:
    return json.dumps(tools, indent=2, sort_keys=True)


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def generation_limits(body: Mapping[str, Any]) -> tuple[int, int]:
    completion_value = body.get("max_tokens")
    if completion_value is None:
        completion_value = body.get("max_completion_tokens")
    return (
        _bounded_int(completion_value, 1024, 1, 4096),
        _bounded_int(body.get("max_reasoning_tokens"), 96, 16, 256),
    )


def runtime_reasoning_policy(
    enable_thinking: bool,
    max_reasoning_tokens: int,
) -> str:
    if enable_thinking:
        return (
            "# Runtime mode: REASONING MODE\n"
            f"Use at most {max_reasoning_tokens} tokens for private reasoning, then close </think> "
            "and provide the answer. Do not restart reasoning after the answer begins."
        )
    return (
        "# Runtime mode: DIRECT MODE\n"
        "Do not emit <think> tags or a reasoning preamble. Answer immediately and directly. "
        "This runtime instruction overrides any general prompt that encourages reasoning."
    )


def format_assistant_turn(content: str, reasoning_content: str | None) -> str:
    reasoning = reasoning_content or ""
    return f"<think>\n{reasoning}\n</think>\n\n{content}"
