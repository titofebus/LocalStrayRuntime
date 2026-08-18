import json
from pathlib import Path

from harness.daemon.stream_sanitizer import (
    MALFORMED_TOOL_CALL_MARKER,
    STRUCTURED_TOOL_CALLS_V1,
    ToolMarkupFilter,
    extract_tool_calls,
    should_use_prefix_cache,
)


def test_capability_identity_constant():
    assert STRUCTURED_TOOL_CALLS_V1 == "structured_tool_calls_v1"


def test_tool_requests_use_rendered_prompt_prefix_cache():
    assert should_use_prefix_cache(None) is True
    assert should_use_prefix_cache([]) is True
    assert should_use_prefix_cache([{"type": "function"}]) is True


def test_stream_filter_extracts_structured_tool_call_without_raw_xml():
    stream_filter = ToolMarkupFilter()

    stream_filter.push("<tool_call>")
    stream_filter.push('{"name":"workspace_read_file",')
    stream_filter.push('"arguments":{"path":"Package.swift"}}')
    stream_filter.push("</tool_call>")

    captured = stream_filter.captured_text
    clean_text, tool_calls = extract_tool_calls(captured)

    # Raw XML tags must not leak into visible content
    assert "<tool_call>" not in clean_text
    assert "</tool_call>" not in clean_text
    assert clean_text == ""

    # OpenAI-compatible structured tool call output
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["type"] == "function"
    assert isinstance(tool_call["id"], str)
    assert tool_call["id"].startswith("call_")
    assert tool_call["function"]["name"] == "workspace_read_file"
    assert isinstance(tool_call["function"]["arguments"], str)
    assert json.loads(tool_call["function"]["arguments"]) == {"path": "Package.swift"}


def test_stream_filter_preserves_surrounding_explanatory_content():
    stream_filter = ToolMarkupFilter()

    p1 = stream_filter.push("I will check the manifest first.\n")
    p2 = stream_filter.push('<tool_call>{"name":"workspace_read_file","arguments":{"path":"Package.swift"}}</tool_call>\n')
    p3 = stream_filter.push("Let me know if you need anything else.")
    trailing = stream_filter.finish()

    visible_streamed = p1 + p2 + p3 + trailing
    clean_markup, tool_calls = extract_tool_calls(stream_filter.captured_text)

    # Clean visible text preserves surrounding explanations without XML
    assert "<tool_call>" not in visible_streamed
    assert "</tool_call>" not in visible_streamed
    assert "I will check the manifest first." in visible_streamed
    assert "Let me know if you need anything else." in visible_streamed

    # Structured tool call is cleanly extracted
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "workspace_read_file"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"path": "Package.swift"}


def test_stream_filter_malformed_tool_json_omits_raw_markup_and_inserts_sanitized_marker():
    raw_markup = '<tool_call>{"name": "workspace_read_file", "arguments": {invalid_json</tool_call>'
    clean_text, tool_calls = extract_tool_calls(raw_markup)

    # Malformed JSON must produce no tool_calls
    assert tool_calls == []

    # Raw <tool_call> markup and arguments must not appear in visible content
    assert "<tool_call>" not in clean_text
    assert "</tool_call>" not in clean_text
    assert "invalid_json" not in clean_text

    # Visible content must contain the sanitized marker
    assert MALFORMED_TOOL_CALL_MARKER in clean_text


def test_unified_server_wires_the_tool_stream_contract():
    source = (
        Path(__file__).parents[1] / "harness" / "daemon" / "unified_server.py"
    ).read_text(encoding="utf-8")

    assert "from harness.daemon.stream_sanitizer import" in source
    assert 'identity["capabilities"] = [STRUCTURED_TOOL_CALLS_V1]' in source
    assert "identity[\"runtime_features\"]" in source
    assert "should_use_prefix_cache(tools)" in source
    assert "use_prefix_cache=use_prefix_cache" in source
    assert "reasoning_filter = ControlTokenFilter()" in source
    assert "reasoning_filter.push(clean_txt)" in source
    assert "code_prompt = prompt + sanitized_thought_text" in source
    assert "code_prompt = prompt + thought_text" not in source
    assert "extract_tool_calls(tool_markup_text)" in source
