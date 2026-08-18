from pathlib import Path

from harness.daemon.stream_sanitizer import ControlTokenFilter, ToolMarkupFilter


def test_control_token_filter_stops_before_complete_marker():
    stream_filter = ControlTokenFilter()

    visible, stopped = stream_filter.push('print("Hello")<|im_end|>ignored')

    assert visible == 'print("Hello")'
    assert stopped is True
    assert stream_filter.finish() == ""


def test_control_token_filter_holds_split_marker_without_latency_for_other_text():
    stream_filter = ControlTokenFilter()

    first, first_stopped = stream_filter.push("answer<|im_")
    second, second_stopped = stream_filter.push("end|>ignored")

    assert first == "answer"
    assert first_stopped is False
    assert second == ""
    assert second_stopped is True


def test_control_token_filter_preserves_literal_angle_bracket_content():
    stream_filter = ControlTokenFilter()

    visible, stopped = stream_filter.push("Array<Element> and <div>")

    assert visible == "Array<Element> and <div>"
    assert stopped is False
    assert stream_filter.finish() == ""


def test_unified_server_filters_content_before_sse_serialization():
    server_source = (
        Path(__file__).parents[1] / "harness" / "daemon" / "unified_server.py"
    ).read_text(encoding="utf-8")

    assert "from harness.daemon.stream_sanitizer import ControlTokenFilter" in server_source
    assert "content_filter = ControlTokenFilter()" in server_source
    assert "clean_txt, content_stopped = content_filter.push(clean_txt)" in server_source


def test_tool_markup_filter_streams_prose_and_captures_split_tool_call():
    stream_filter = ToolMarkupFilter()

    first = stream_filter.push("Checking. <tool_")
    second = stream_filter.push('call>{"name":"read"}</tool_')
    third = stream_filter.push("call> Done.")

    assert first == "Checking. "
    assert second == ""
    assert third == " Done."
    assert stream_filter.captured_text == '<tool_call>{"name":"read"}</tool_call>'


def test_tool_markup_filter_does_not_delay_other_angle_bracket_content():
    stream_filter = ToolMarkupFilter()

    assert stream_filter.push("Use <div> immediately.") == "Use <div> immediately."


def test_unified_server_converts_captured_markup_to_structured_tool_calls():
    server_source = (
        Path(__file__).parents[1] / "harness" / "daemon" / "unified_server.py"
    ).read_text(encoding="utf-8")

    assert "tool_filter = ToolMarkupFilter() if tools else None" in server_source
    assert 'yield ("tool_markup", tool_filter.captured_text)' in server_source
    assert "extract_tool_calls(tool_markup_text)" in server_source
