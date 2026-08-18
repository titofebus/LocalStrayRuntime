"""
Verification Test for Fused Metal QKV + RoPE Unified Shader on Apple Silicon.
"""
from pathlib import Path
import time
import mlx.core as mx
import numpy as np


def test_fused_qkv_rope_compilation():
    print("=" * 65)
    print(" TESTING FUSED METAL QKV + IN-REGISTER ROPE SHADER")
    print("=" * 65)

    metal_source = (Path(__file__).parent / "fused_qkv_rope.metal").read_text(encoding="utf-8")
    kernel = mx.fast.metal_kernel(
        name="fused_qkv_gemv_rope_kernel",
        input_names=["input", "w_q", "w_k", "w_v", "hidden_size", "num_q_heads", "num_kv_heads", "head_dim", "position", "rope_theta"],
        output_names=["out_q", "out_k", "out_v"],
        source=metal_source,
    )

    hidden_size = 3584
    num_q_heads = 28
    num_kv_heads = 4
    head_dim = 128
    position = 10
    rope_theta = 10000.0

    inp = mx.random.normal((hidden_size,)).astype(mx.float16)
    w_q = mx.random.normal((num_q_heads * head_dim, hidden_size)).astype(mx.float16)
    w_k = mx.random.normal((num_kv_heads * head_dim, hidden_size)).astype(mx.float16)
    w_v = mx.random.normal((num_kv_heads * head_dim, hidden_size)).astype(mx.float16)

    # Execute fused shader
    out_q, out_k, out_v = kernel(
        inputs=[
            inp,
            w_q,
            w_k,
            w_v,
            hidden_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            position,
            rope_theta,
        ],
        output_shapes=[(num_q_heads * head_dim,), (num_kv_heads * head_dim,), (num_kv_heads * head_dim,)],
        output_dtypes=[mx.float16, mx.float16, mx.float16],
        grid=(num_q_heads * head_dim, 1, 1),
        threadgroup=(min(256, num_q_heads * head_dim), 1, 1),
    )
    mx.eval(out_q, out_k, out_v)

    print(f"Fused Q shape: {out_q.shape}, dtype: {out_q.dtype}")
    print(f"Fused K shape: {out_k.shape}, dtype: {out_k.dtype}")
    print(f"Fused V shape: {out_v.shape}, dtype: {out_v.dtype}")
    print(" [SUCCESS] Fused QKV + RoPE Metal kernel compiled and executed on GPU!")
    print("=" * 65)


if __name__ == "__main__":
    test_fused_qkv_rope_compilation()
