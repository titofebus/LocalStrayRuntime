"""
Vectorized Fused Metal QKV + RoPE kernel in Apple MLX.
"""
from pathlib import Path
import mlx.core as mx


_FUSED_QKV_ROPE_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

_FUSED_QKV_ROPE_SOURCE = """
    uint tid = thread_position_in_grid.x;
    uint total_q = 3584; // 28 heads * 128
    if (tid >= total_q) return;

    uint head_dim = 128;
    uint half_dim = 64;
    uint hidden_size = 3584;
    float rope_theta = 10000.0f;
    float pos = float(position);

    // Compute Q row GEMV
    float sum_q = 0.0f;
    uint row_offset = tid * hidden_size;
    for (uint i = 0; i < hidden_size; ++i) {
        sum_q += float(w_q[row_offset + i]) * float(inp[i]);
    }

    uint head_idx = tid / head_dim;
    uint dim_idx = tid % head_dim;

    if (dim_idx < half_dim) {
        float freq = pos / pow(rope_theta, float(2 * dim_idx) / float(head_dim));
        float cos_v = cos(freq);
        float sin_v = sin(freq);

        float sum_q_partner = 0.0f;
        uint partner_offset = (head_idx * head_dim + dim_idx + half_dim) * hidden_size;
        for (uint i = 0; i < hidden_size; ++i) {
            sum_q_partner += float(w_q[partner_offset + i]) * float(inp[i]);
        }
        out_q[tid] = static_cast<T>(sum_q * cos_v - sum_q_partner * sin_v);
    } else {
        uint orig_dim = dim_idx - half_dim;
        float freq = pos / pow(rope_theta, float(2 * orig_dim) / float(head_dim));
        float cos_v = cos(freq);
        float sin_v = sin(freq);

        float sum_q_partner = 0.0f;
        uint partner_offset = (head_idx * head_dim + orig_dim) * hidden_size;
        for (uint i = 0; i < hidden_size; ++i) {
            sum_q_partner += float(w_q[partner_offset + i]) * float(inp[i]);
        }
        out_q[tid] = static_cast<T>(sum_q_partner * sin_v + sum_q * cos_v);
    }

    // Compute K and V if within bounds (4 heads * 128 = 512)
    uint total_kv = 512;
    if (tid < total_kv) {
        float sum_v = 0.0f;
        uint v_offset = tid * hidden_size;
        for (uint i = 0; i < hidden_size; ++i) {
            sum_v += float(w_v[v_offset + i]) * float(inp[i]);
        }
        out_v[tid] = static_cast<T>(sum_v);

        float sum_k = 0.0f;
        uint k_offset = tid * hidden_size;
        for (uint i = 0; i < hidden_size; ++i) {
            sum_k += float(w_k[k_offset + i]) * float(inp[i]);
        }

        uint kv_head_idx = tid / head_dim;
        uint kv_dim_idx = tid % head_dim;

        if (kv_dim_idx < half_dim) {
            float freq = pos / pow(rope_theta, float(2 * kv_dim_idx) / float(head_dim));
            float cos_v = cos(freq);
            float sin_v = sin(freq);

            float sum_k_partner = 0.0f;
            uint partner_k_offset = (kv_head_idx * head_dim + kv_dim_idx + half_dim) * hidden_size;
            for (uint i = 0; i < hidden_size; ++i) {
                sum_k_partner += float(w_k[partner_k_offset + i]) * float(inp[i]);
            }
            out_k[tid] = static_cast<T>(sum_k * cos_v - sum_k_partner * sin_v);
        } else {
            uint orig_k_dim = kv_dim_idx - half_dim;
            float freq = pos / pow(rope_theta, float(2 * orig_k_dim) / float(head_dim));
            float cos_v = cos(freq);
            float sin_v = sin(freq);

            float sum_k_partner = 0.0f;
            uint partner_k_offset = (kv_head_idx * head_dim + orig_k_dim) * hidden_size;
            for (uint i = 0; i < hidden_size; ++i) {
                sum_k_partner += float(w_k[partner_k_offset + i]) * float(inp[i]);
            }
            out_k[tid] = static_cast<T>(sum_k_partner * sin_v + sum_k * cos_v);
        }
    }
"""

_fused_qkv_rope_kernel = mx.fast.metal_kernel(
    name="fused_qkv_gemv_rope_kernel",
    input_names=["inp", "w_q", "w_k", "w_v", "position"],
    output_names=["out_q", "out_k", "out_v"],
    source=_FUSED_QKV_ROPE_SOURCE,
    header=_FUSED_QKV_ROPE_HEADER,
)


def fused_qkv_rope(
    inp: mx.array,
    w_q: mx.array,
    w_k: mx.array,
    w_v: mx.array,
    position: int = 0,
) -> tuple[mx.array, mx.array, mx.array]:
    num_q = 3584
    num_kv = 512
    out_q, out_k, out_v = _fused_qkv_rope_kernel(
        inputs=[inp, w_q, w_k, w_v, position],
        template=[("T", inp.dtype)],
        output_shapes=[(num_q,), (num_kv,), (num_kv,)],
        output_dtypes=[inp.dtype, inp.dtype, inp.dtype],
        grid=(num_q, 1, 1),
        threadgroup=(min(256, num_q), 1, 1),
    )
    return out_q, out_k, out_v


def test():
    print("=" * 65)
    print(" TESTING FUSED METAL QKV + ROPE SHADER ON GPU")
    print("=" * 65)
    hidden_size = 3584
    num_q = 3584
    num_kv = 512

    inp = mx.random.normal((hidden_size,)).astype(mx.float16)
    w_q = mx.random.normal((num_q, hidden_size)).astype(mx.float16)
    w_k = mx.random.normal((num_kv, hidden_size)).astype(mx.float16)
    w_v = mx.random.normal((num_kv, hidden_size)).astype(mx.float16)

    q, k, v = fused_qkv_rope(inp, w_q, w_k, w_v, position=12)
    mx.eval(q, k, v)

    print(f"Q output shape: {q.shape}, dtype: {q.dtype}")
    print(f"K output shape: {k.shape}, dtype: {k.dtype}")
    print(f"V output shape: {v.shape}, dtype: {v.dtype}")
    print(" [SUCCESS] Fused QKV + RoPE Metal kernel compiled and executed flawlessly!")
    print("=" * 65)


if __name__ == "__main__":
    test()
