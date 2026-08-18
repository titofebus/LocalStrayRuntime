"""DeepSeek-optimized polymorphic Metal Shading Language (MSL) kernels for Qwen 3.8 MTP."""
import os
import mlx.core as mx

# ---------------------------------------------------------------------------
# 1. Vectorized Dual RMSNorm + Concatenation (Generic vec<T, 4> + SIMD warp shuffle)
# ---------------------------------------------------------------------------
_DUAL_RMSNORM_CONCAT_HEADER = """
#include <metal_stdlib>
using namespace metal;

template <typename T>
inline T simd_sum_all(T val) {
    for (uint offset = 16; offset > 0; offset >>= 1) {
        val += simd_shuffle_down(val, offset);
    }
    return val;
}
"""

_DUAL_RMSNORM_CONCAT_SOURCE = """
    // Grid: (256, B, 1), Threadgroup: (256, 1, 1)
    typedef typename metal::vec<T, 4> vec4_t;

    uint tid = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.y;
    uint tg_size = threads_per_threadgroup.x;

    uint D = 5120; // Qwen 3.8 Hidden dimension
    float eps = eps_value[0];
    uint row_offset = row * D;
    uint out_offset = row * (2 * D);

    // Vectorized pointers (4 elements per vector)
    device const vec4_t* emb4 = reinterpret_cast<device const vec4_t*>(emb + row_offset);
    device const vec4_t* hid4 = reinterpret_cast<device const vec4_t*>(hid + row_offset);
    device const vec4_t* w_emb4 = reinterpret_cast<device const vec4_t*>(w_emb);
    device const vec4_t* w_hid4 = reinterpret_cast<device const vec4_t*>(w_hid);
    device vec4_t* out4 = reinterpret_cast<device vec4_t*>(out + out_offset);

    uint D4 = D / 4; // 1280 4-element vectors
    float sum_sq_emb = 0.0f;
    float sum_sq_hid = 0.0f;

    // Vectorized sum of squares in FP32
    for (uint i = tid; i < D4; i += tg_size) {
        vec4_t e = emb4[i];
        vec4_t h = hid4[i];
        float4 e_f = float4(e);
        float4 h_f = float4(h);
        sum_sq_emb += dot(e_f, e_f);
        sum_sq_hid += dot(h_f, h_f);
    }

    // Warp-level SIMD reduction (32 lanes)
    sum_sq_emb = simd_sum_all(sum_sq_emb);
    sum_sq_hid = simd_sum_all(sum_sq_hid);

    // Threadgroup reduction across 8 SIMD warps (256 threads / 32)
    threadgroup float tg_sum_emb[32];
    threadgroup float tg_sum_hid[32];
    uint lane = tid & 31;
    uint warp_id = tid >> 5;

    if (lane == 0) {
        tg_sum_emb[warp_id] = sum_sq_emb;
        tg_sum_hid[warp_id] = sum_sq_hid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        float final_emb = (tid < (tg_size / 32)) ? tg_sum_emb[tid] : 0.0f;
        float final_hid = (tid < (tg_size / 32)) ? tg_sum_hid[tid] : 0.0f;
        final_emb = simd_sum_all(final_emb);
        final_hid = simd_sum_all(final_hid);
        if (tid == 0) {
            tg_sum_emb[0] = rsqrt(final_emb / float(D) + eps);
            tg_sum_hid[0] = rsqrt(final_hid / float(D) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms_emb = tg_sum_emb[0];
    float inv_rms_hid = tg_sum_hid[0];

    // Second pass: Vectorized normalization, scaling, and concatenated write
    for (uint i = tid; i < D4; i += tg_size) {
        vec4_t e = emb4[i];
        vec4_t h = hid4[i];
        vec4_t we = w_emb4[i];
        vec4_t wh = w_hid4[i];

        float4 e_norm = float4(e) * inv_rms_emb * float4(we);
        float4 h_norm = float4(h) * inv_rms_hid * float4(wh);

        out4[i] = vec4_t(e_norm);
        out4[D4 + i] = vec4_t(h_norm);
    }
"""

_vectorized_dual_rmsnorm_concat_kernel = mx.fast.metal_kernel(
    name="vectorized_dual_rmsnorm_concat",
    input_names=["emb", "hid", "w_emb", "w_hid", "eps_value"],
    output_names=["out"],
    source=_DUAL_RMSNORM_CONCAT_SOURCE,
    header=_DUAL_RMSNORM_CONCAT_HEADER,
    compile_options={"math_mode": "fast"},
)


def fused_dual_rmsnorm_concat(
    emb: mx.array,
    hid: mx.array,
    w_emb: mx.array,
    w_hid: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    """Vectorized in-kernel RMSNorm on emb and hid in parallel with concatenated output."""
    B, D = emb.shape
    assert D == 5120, f"Expected D=5120, got {D}"

    # Ensure robust dtype matching across all inputs
    target_dtype = emb.dtype
    if hid.dtype != target_dtype:
        hid = hid.astype(target_dtype)
    if w_emb.dtype != target_dtype:
        w_emb = w_emb.astype(target_dtype)
    if w_hid.dtype != target_dtype:
        w_hid = w_hid.astype(target_dtype)

    out = _vectorized_dual_rmsnorm_concat_kernel(
        inputs=[
            emb,
            hid,
            w_emb,
            w_hid,
            mx.array([eps], dtype=mx.float32),
        ],
        template=[("T", target_dtype)],
        grid=(256, B, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, 2 * D)],
        output_dtypes=[target_dtype],
    )[0]
    return out


# ---------------------------------------------------------------------------
# 2. Vectorized 248k Vocabulary Argmax (Generic vec<T, 4> unrolled + SIMD warp shuffle)
# ---------------------------------------------------------------------------
_ARGMAX_HEADER = """
#include <metal_stdlib>
using namespace metal;

template <typename T>
inline void simd_argmax(T val, uint idx, thread T& max_val, thread uint& max_idx) {
    for (uint offset = 16; offset > 0; offset >>= 1) {
        T other_val = simd_shuffle_down(val, offset);
        uint other_idx = simd_shuffle_down(idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
    max_val = val;
    max_idx = idx;
}
"""

_STAGE1_ARGMAX_SOURCE = """
    // Grid: (32 * 256, B, 1), Threadgroup: (256, 1, 1)
    typedef typename metal::vec<T, 4> vec4_t;

    uint block_id = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.y;
    uint tg_size = threads_per_threadgroup.x;

    uint V = 248320;
    uint num_blocks = 32;
    uint chunk_size = (V + num_blocks - 1) / num_blocks; // 7760 elements
    uint start_idx = block_id * chunk_size;
    uint end_idx = min(start_idx + chunk_size, V);
    uint row_offset = row * V;

    device const T* row_logits = logits + row_offset;
    device const vec4_t* logits4 = reinterpret_cast<device const vec4_t*>(row_logits);

    float local_max = -1e30f;
    uint local_idx = start_idx;

    uint start4 = start_idx / 4;
    uint end4 = end_idx / 4;

    // Vectorized 4-element load and unrolled comparison
    for (uint i = start4 + tid; i < end4; i += tg_size) {
        vec4_t v = logits4[i];
        float4 vf = float4(v);
        uint base_idx = i * 4;
        if (vf.x > local_max) { local_max = vf.x; local_idx = base_idx; }
        if (vf.y > local_max) { local_max = vf.y; local_idx = base_idx + 1; }
        if (vf.z > local_max) { local_max = vf.z; local_idx = base_idx + 2; }
        if (vf.w > local_max) { local_max = vf.w; local_idx = base_idx + 3; }
    }

    // SIMD Warp Reduction
    float s_max;
    uint s_idx;
    simd_argmax(local_max, local_idx, s_max, s_idx);

    threadgroup float tg_vals[32];
    threadgroup uint tg_idxs[32];
    uint lane = tid & 31;
    uint warp_id = tid >> 5;

    if (lane == 0) {
        tg_vals[warp_id] = s_max;
        tg_idxs[warp_id] = s_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final threadgroup reduction across warps
    if (tid < 32) {
        float w_max = (tid < (tg_size / 32)) ? tg_vals[tid] : -1e30f;
        uint w_idx = (tid < (tg_size / 32)) ? tg_idxs[tid] : 0;
        simd_argmax(w_max, w_idx, s_max, s_idx);
        if (tid == 0) {
            uint out_idx = row * num_blocks + block_id;
            block_max_vals[out_idx] = s_max;
            block_max_indices[out_idx] = s_idx;
        }
    }
"""

_STAGE2_ARGMAX_SOURCE = """
    // Grid: (32, B, 1), Threadgroup: (32, 1, 1)
    uint tid = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.y;
    uint num_blocks = 32;

    uint in_offset = row * num_blocks;

    float val = block_max_vals[in_offset + tid];
    uint idx = block_max_indices[in_offset + tid];

    float final_max;
    uint final_idx;
    simd_argmax(val, idx, final_max, final_idx);

    if (tid == 0) {
        final_tokens[row] = final_idx;
    }
"""

_vectorized_stage1_argmax_kernel = mx.fast.metal_kernel(
    name="vectorized_stage1_argmax",
    input_names=["logits"],
    output_names=["block_max_vals", "block_max_indices"],
    source=_STAGE1_ARGMAX_SOURCE,
    header=_ARGMAX_HEADER,
    compile_options={"math_mode": "fast"},
)

_vectorized_stage2_argmax_kernel = mx.fast.metal_kernel(
    name="vectorized_stage2_argmax",
    input_names=["block_max_vals", "block_max_indices"],
    output_names=["final_tokens"],
    source=_STAGE2_ARGMAX_SOURCE,
    header=_ARGMAX_HEADER,
    compile_options={"math_mode": "fast"},
)


def fast_vocab_argmax(logits: mx.array) -> mx.array:
    """Computes vectorized SIMD argmax reduction across 248,320 vocabulary logits."""
    B, V = logits.shape
    assert V == 248320, f"Expected V=248320, got {V}"

    # Stage 1: 32 threadgroups of 256 threads (vectorized vec<T, 4> + warp shuffles)
    block_vals, block_idxs = _vectorized_stage1_argmax_kernel(
        inputs=[logits],
        template=[("T", logits.dtype)],
        grid=(32 * 256, B, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, 32), (B, 32)],
        output_dtypes=[mx.float32, mx.uint32],
    )

    # Stage 2: 1 SIMD group of 32 threads with warp reduction
    final_tokens = _vectorized_stage2_argmax_kernel(
        inputs=[block_vals, block_idxs],
        grid=(32, B, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B,)],
        output_dtypes=[mx.uint32],
    )[0]

    return final_tokens


def is_fused_mtp_supported() -> bool:
    """Check if Metal GPU execution is available for fused kernels."""
    try:
        if not mx.metal.is_available():
            return False
        _ = mx.default_stream(mx.gpu)
        return True
    except Exception:
        return False


def is_fused_mtp_enabled() -> bool:
    """Check if fused MTP is enabled via environment variable and supported on device."""
    flag = os.environ.get("QWEN_PRIME_FUSED_MTP", "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    return is_fused_mtp_supported()
