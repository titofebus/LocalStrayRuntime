import importlib.util
from pathlib import Path


def test_assistant_turn_preserves_reasoning_for_prefix_cache_reuse():
    module_path = Path(__file__).parents[1] / "harness" / "daemon" / "qwen_chat.py"
    assert module_path.is_file(), "Qwen chat formatting helper is missing"
    spec = importlib.util.spec_from_file_location("qwen_chat", module_path)
    assert spec is not None and spec.loader is not None
    qwen_chat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qwen_chat)
    format_assistant_turn = getattr(qwen_chat, "format_assistant_turn", None)
    assert callable(format_assistant_turn)

    assert format_assistant_turn("final answer", "checked edge cases") == (
        "<think>\nchecked edge cases\n</think>\n\nfinal answer"
    )
    assert format_assistant_turn("direct answer", None) == (
        "<think>\n\n</think>\n\ndirect answer"
    )


def _load_qwen_chat():
    module_path = Path(__file__).parents[1] / "harness" / "daemon" / "qwen_chat.py"
    spec = importlib.util.spec_from_file_location("qwen_chat", module_path)
    assert spec is not None and spec.loader is not None
    qwen_chat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qwen_chat)
    return qwen_chat


def test_direct_policy_overrides_system_prompts_that_encourage_reasoning():
    qwen_chat = _load_qwen_chat()

    policy = qwen_chat.runtime_reasoning_policy(False, max_reasoning_tokens=96)

    assert "DIRECT MODE" in policy
    assert "Do not emit <think>" in policy
    assert "Answer immediately" in policy


def test_reasoning_policy_has_an_explicit_token_budget():
    qwen_chat = _load_qwen_chat()

    policy = qwen_chat.runtime_reasoning_policy(True, max_reasoning_tokens=96)

    assert "REASONING MODE" in policy
    assert "96 tokens" in policy
    assert "close </think>" in policy


def test_generation_limits_are_bounded_and_honor_client_values():
    qwen_chat = _load_qwen_chat()

    assert qwen_chat.generation_limits({}) == (1024, 96)
    assert qwen_chat.generation_limits({
        "max_completion_tokens": 768,
        "max_reasoning_tokens": 64,
    }) == (768, 64)
    assert qwen_chat.generation_limits({
        "max_tokens": 99999,
        "max_reasoning_tokens": 99999,
    }) == (4096, 256)


def test_tool_schema_json_is_stable_across_dictionary_key_order():
    qwen_chat = _load_qwen_chat()
    first = [{
        "type": "function",
        "function": {"name": "read", "description": "Read a file"},
    }]
    reordered = [{
        "function": {"description": "Read a file", "name": "read"},
        "type": "function",
    }]

    assert qwen_chat.canonical_tool_schema_json(first) == (
        qwen_chat.canonical_tool_schema_json(reordered)
    )


def test_prompt_fingerprint_is_stable_without_exposing_prompt_text():
    qwen_chat = _load_qwen_chat()

    first = qwen_chat.prompt_fingerprint("private prompt")
    repeated = qwen_chat.prompt_fingerprint("private prompt")
    changed = qwen_chat.prompt_fingerprint("different prompt")

    assert first == repeated
    assert first != changed
    assert "private" not in first
    assert len(first) == 16
