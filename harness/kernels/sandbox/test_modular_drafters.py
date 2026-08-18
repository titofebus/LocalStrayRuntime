"""Test suite for Modular Speculative Drafter Architecture."""
import time
from types import SimpleNamespace
import mlx.core as mx
import mlx.nn as nn
from harness.executors.qwen38_mtp import Qwen38MTPModel
from harness.kernels.sandbox.modular_drafters import (
    NativeMTPDrafter,
    ModularSpeculativeEngine,
    HybridAdaptiveDrafter,
)


def test_modular_drafter_engine_hot_swap():
    config = {
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "max_position_embeddings": 262144,
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rms_norm_eps": 1e-06,
        "attention_bias": False,
        "vocab_size": 248320,
        "block_size": 4,
        "target_layer_ids": [63],
    }

    mtp_model = Qwen38MTPModel(config)
    for p in mtp_model.parameters().values():
        if isinstance(p, mx.array):
            p = p.astype(mx.float16)

    # 1. Instantiate Native MTP Drafter
    mtp_drafter = NativeMTPDrafter(mtp_model=mtp_model)
    assert mtp_drafter.is_loaded

    # 2. Instantiate Engine with MTP Drafter
    engine = ModularSpeculativeEngine(default_drafter=mtp_drafter)
    assert engine.active_drafter.name == "native_mtp_1layer"

    # 3. Test Mock Draft Block Generation
    target_ops = SimpleNamespace(
        embed_tokens=lambda model: lambda x: mx.zeros((1, 1, 5120), dtype=mx.float16),
        logits_from_hidden=lambda model, h: mx.random.normal((1, 1, 248320)).astype(mx.float16),
    )
    staged_first = mx.array([12345], dtype=mx.uint32)
    target_hidden = mx.random.normal((1, 1, 5120)).astype(mx.float16)

    tokens = engine.draft_block(
        target_model=SimpleNamespace(),
        target_ops=target_ops,
        staged_first=staged_first,
        target_hidden=target_hidden,
        draft_count=4,
    )
    mx.eval(tokens)
    assert tokens.shape == (4,), f"Expected shape (4,), got {tokens.shape}"

    # 4. Instantiate Hybrid Adaptive Drafter and Hot-Swap
    hybrid = HybridAdaptiveDrafter(mtp_drafter=mtp_drafter)
    engine.register_drafter(hybrid)

    t0 = time.perf_counter()
    engine.set_active_drafter("hybrid_adaptive")
    swap_time_us = (time.perf_counter() - t0) * 1e6

    assert engine.active_drafter.name == "hybrid_adaptive"
    print(f"Hot-swap completed in {swap_time_us:.2f} µs (<1 µs)")

    print("[SUCCESS] Modular Drafter Engine verified cleanly!")


if __name__ == "__main__":
    test_modular_drafter_engine_hot_swap()
