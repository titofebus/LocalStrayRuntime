import ast
from pathlib import Path

import harness.dflash_runtime as dflash_runtime
import pytest


def test_production_generation_uses_tuned_four_token_mtp_blocks():
    source_path = (
        Path(__file__).parents[1] / "harness" / "executors" / "dflash_engine.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    engine = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "DFlashEngine"
    )
    initializer = next(
        node
        for node in engine.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    defaults = dict(
        zip(
            [argument.arg for argument in initializer.args.args[-len(initializer.args.defaults):]],
            initializer.args.defaults,
            strict=True,
        )
    )

    assert ast.literal_eval(defaults["block_tokens"]) == 4

    generation_calls = [
        node
        for node in ast.walk(engine)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "stream_dflash_generate"
    ]
    assert len(generation_calls) == 2
    for call in generation_calls:
        block_tokens = next(
            keyword.value
            for keyword in call.keywords
            if keyword.arg == "block_tokens"
        )
        assert isinstance(block_tokens, ast.Attribute)
        assert isinstance(block_tokens.value, ast.Name)
        assert block_tokens.value.id == "self"
        assert block_tokens.attr == "block_tokens"


def test_production_runtime_uses_measured_adaptive_mode_and_native_prefix_cache(monkeypatch):
    monkeypatch.delenv("QWEN_PRIME_VERIFY_MODE", raising=False)
    build_context = getattr(dflash_runtime, "build_dflash_runtime_context", None)
    assert callable(build_context)

    runtime = build_context().runtime

    assert runtime.prefix_cache is True
    assert runtime.prefix_cache_l2 is False
    assert runtime.target_fa_window == 0
    assert runtime.verify_mode == "adaptive"


def test_ddtree_can_be_enabled_explicitly_for_benchmarking(monkeypatch):
    monkeypatch.setenv("QWEN_PRIME_VERIFY_MODE", "ddtree")

    runtime = dflash_runtime.build_dflash_runtime_context().runtime

    assert runtime.verify_mode == "ddtree"


def test_unknown_verify_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("QWEN_PRIME_VERIFY_MODE", "turbo")

    with pytest.raises(ValueError, match="QWEN_PRIME_VERIFY_MODE"):
        dflash_runtime.build_dflash_runtime_context()


def test_stream_generation_uses_dflash_prefix_snapshots_and_prefill_events():
    source_path = (
        Path(__file__).parents[1] / "harness" / "executors" / "dflash_engine.py"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "PrefixCacheFlow.for_request" in source
    assert "prefix_snapshot=prefix_flow.snapshot" in source
    assert "snapshot_service=prefix_flow.snapshot_service" in source
    assert "prefix_cache_active=prefix_flow.cache_active" in source
    assert "PrefillCompleteEvent" in source


def test_unified_server_finishes_metal_warmup_before_becoming_ready():
    source_path = (
        Path(__file__).parents[1] / "harness" / "daemon" / "unified_server.py"
    )
    source = source_path.read_text(encoding="utf-8")

    warmup_position = source.index("self.engine.warmup()")
    ready_position = source.index("self.ready_event.set()")
    assert warmup_position < ready_position
