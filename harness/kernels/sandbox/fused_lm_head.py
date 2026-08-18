"""Optimized Fused LM-Head GEMV + Argmax Kernel with Threadgroup SRAM Caching."""
import mlx.core as mx

MSL_HEADER = """
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

MSL_STAGE1_SOURCE = """
    // Grid: (32 * 256, B, 1), Threadgroup: (256, 1, 1)
    typedef typename metal::vec<T, 4> vec4_t;

    uint block_idx = threadgroup_position_in_grid.x;
    uint thread_idx = thread_position_in_threadgroup.x;
    uint batch_idx = threadgroup_position_in_grid.y;
    uint tg_size = threads_per_threadgroup.x;

    uint VOCAB_SIZE = 248320;
    uint HIDDEN_SIZE = 5120;
    uint NUM_BLOCKS = 32;
    uint VEC_ELEMENTS = HIDDEN_SIZE / 4; // 1280

    uint chunk_size = (VOCAB_SIZE + NUM_BLOCKS - 1) / NUM_BLOCKS; // 7760
    uint vocab_start = block_idx * chunk_size;
    uint vocab_end = metal::min(vocab_start + chunk_size, VOCAB_SIZE);

    const device T* hidden_row = hidden + batch_idx * HIDDEN_SIZE;
    const device vec4_t* hidden_vec4 = reinterpret_cast<const device vec4_t*>(hidden_row);

    // Cache hidden vector in fast 10KB Threadgroup SRAM
    threadgroup vec4_t tg_hidden[1280];
    for (uint i = thread_idx; i < VEC_ELEMENTS; i += tg_size) {
        tg_hidden[i] = hidden_vec4[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max_val = -1e30f;
    uint local_max_idx = vocab_start;

    // Strided loop over all vocab rows assigned to this block
    for (uint vocab_row = vocab_start + thread_idx; vocab_row < vocab_end; vocab_row += tg_size) {
        const device T* weight_row = weights + vocab_row * HIDDEN_SIZE;
        const device vec4_t* weight_vec4 = reinterpret_cast<const device vec4_t*>(weight_row);

        float sum = 0.0f;
        #pragma unroll 4
        for (uint i = 0; i < VEC_ELEMENTS; i++) {
            vec4_t h_v = tg_hidden[i];
            vec4_t w_v = weight_vec4[i];
            float4 h_f = float4(h_v);
            float4 w_f = float4(w_v);
            sum += dot(h_f, w_f);
        }

        if (sum > local_max_val) {
            local_max_val = sum;
            local_max_idx = vocab_row;
        }
    }

    // Warp-level SIMD reduction across 32 threads
    float s_max;
    uint s_idx;
    simd_argmax(local_max_val, local_max_idx, s_max, s_idx);

    threadgroup float tg_vals[32];
    threadgroup uint tg_idxs[32];
    uint lane = thread_idx & 31;
    uint warp_id = thread_idx >> 5;

    if (lane == 0) {
        tg_vals[warp_id] = s_max;
        tg_idxs[warp_id] = s_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final threadgroup reduction across 8 warps
    if (thread_idx < 32) {
        float w_max = (thread_idx < (tg_size / 32)) ? tg_vals[thread_idx] : -1e30f;
        uint w_idx = (thread_idx < (tg_size / 32)) ? tg_idxs[thread_idx] : 0;
        simd_argmax(w_max, w_idx, s_max, s_idx);
        if (thread_idx == 0) {
            uint out_idx = batch_idx * NUM_BLOCKS + block_idx;
            block_max_vals[out_idx] = s_max;
            block_max_indices[out_idx] = s_idx;
        }
    }
"""

MSL_STAGE2_SOURCE = """
    // Grid: (32, B, 1), Threadgroup: (32, 1, 1)
    uint thread_idx = thread_position_in_threadgroup.x;
    uint batch_idx = threadgroup_position_in_grid.y;
    uint NUM_BLOCKS = 32;

    uint in_offset = batch_idx * NUM_BLOCKS;
    float val = block_max_vals[in_offset + thread_idx];
    uint idx = block_max_indices[in_offset + thread_idx];

    float final_max;
    uint final_idx;
    simd_argmax(val, idx, final_max, final_idx);

    if (thread_idx == 0) {
        tokens[batch_idx] = final_idx;
    }
"""

_stage1_fused_lm_head_kernel = mx.fast.metal_kernel(
    name="stage1_fused_lm_head_sram",
    input_names=["hidden", "weights"],
    output_names=["block_max_vals", "block_max_indices"],
    source=MSL_STAGE1_SOURCE,
    header=MSL_HEADER,
    compile_options={"math_mode": "fast"},
)

_stage2_fused_lm_head_kernel = mx.fast.metal_kernel(
    name="stage2_fused_lm_head_sram",
    input_names=["block_max_vals", "block_max_indices"],
    output_names=["tokens"],
    source=MSL_STAGE2_SOURCE,
    header=MSL_HEADER,
    compile_options={"math_mode": "fast"},
)


def fused_lm_head_argmax(
    hidden: mx.array,
    lm_head_weights: mx.array,
) -> mx.array:
    """
    Fused LM-head matrix-vector projection and top-1 greedy token selection.
    
    Eliminates allocating or writing the (B, 248320) logits tensor to unified memory.
    
    Args:
        hidden: Shape (B, 5120) in float16 or float32
        lm_head_weights: Shape (248320, 5120) in float16 or float32
    Returns:
        tokens: Shape (B,) uint32 argmax token indices
    """
    if hidden.ndim == 3:
        hidden = hidden[:, -1, :]  # Take last token hidden state (B, 5120)

    B, D = hidden.shape
    V, D_w = lm_head_weights.shape
    assert D == 5120, f"Expected hidden dim 5120, got {D}"
    assert V == 248320, f"Expected vocab size 248320, got {V}"
    assert D_w == 5120, f"Expected weight dim 5120, got {D_w}"

    target_dtype = hidden.dtype
    if lm_head_weights.dtype != target_dtype:
        lm_head_weights = lm_head_weights.astype(target_dtype)

    # Stage 1: 32 threadgroups of 256 threads (Vectorized vec<T, 4> dot products + SIMD shuffles)
    block_vals, block_idxs = _stage1_fused_lm_head_kernel(
        inputs=[hidden, lm_head_weights],
        template=[("T", target_dtype)],
        grid=(32 * 256, B, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, 32), (B, 32)],
        output_dtypes=[mx.float32, mx.uint32],
    )

    # Stage 2: 1 SIMD group of 32 threads
    tokens = _stage2_fused_lm_head_kernel(
        inputs=[block_vals, block_idxs],
        grid=(32, B, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B,)],
        output_dtypes=[mx.uint32],
    )[0]

    return tokens
