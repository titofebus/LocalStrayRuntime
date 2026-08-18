```python
import mlx.core as mx
import mlx.fast as mf

# ---------------------------------------------------------------------------
# Dual RMSNorm + Concat kernel
# ---------------------------------------------------------------------------
_DUAL_RMSNORM_CONCAT_SOURCE = r"""
using namespace metal;

// SIMD-width reduction using shuffle
template <typename T>
T simd_sum_all(T val) {
    // Reduce across 32 lanes
    for (uint offset = 16; offset > 0; offset >>= 1) {
        val += simd_shuffle_down(val, offset);
    }
    return val;
}

kernel void dual_rmsnorm_concat(
    device const half* emb [[buffer(0)]],
    device const half* hid [[buffer(1)]],
    device const half* w_emb [[buffer(2)]],
    device const half* w_hid [[buffer(3)]],
    device half* out [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid.y]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint row_offset = row * D;
    const uint out_offset = row * (2 * D);
    
    // Vectorized pointers (half4 = 8 bytes)
    device const half4* emb4 = reinterpret_cast<device const half4*>(emb + row_offset);
    device const half4* hid4 = reinterpret_cast<device const half4*>(hid + row_offset);
    device const half4* w_emb4 = reinterpret_cast<device const half4*>(w_emb);
    device const half4* w_hid4 = reinterpret_cast<device const half4*>(w_hid);
    device half4* out4 = reinterpret_cast<device half4*>(out + out_offset);
    
    const uint D4 = D / 4;  // Number of half4 elements
    const uint vec_per_thread = (D4 + tg_size - 1) / tg_size;
    
    float sum_sq_emb = 0.0f;
    float sum_sq_hid = 0.0f;
    
    // Vectorized accumulation
    for (uint v = 0; v < vec_per_thread; v++) {
        uint idx = tid * vec_per_thread + v;
        if (idx < D4) {
            half4 e = emb4[idx];
            half4 h = hid4[idx];
            
            // Accumulate squares in FP32
            float4 e_f = float4(e);
            float4 h_f = float4(h);
            sum_sq_emb += dot(e_f, e_f);
            sum_sq_hid += dot(h_f, h_f);
        }
    }
    
    // SIMD reduction
    sum_sq_emb = simd_sum_all(sum_sq_emb);
    sum_sq_hid = simd_sum_all(sum_sq_hid);
    
    // Threadgroup reduction (only 32 lanes need to participate)
    threadgroup float tg_sum_emb[32];
    threadgroup float tg_sum_hid[32];
    
    uint lane = tid & 31;
    if (lane == 0) {
        tg_sum_emb[tid >> 5] = sum_sq_emb;
        tg_sum_hid[tid >> 5] = sum_sq_hid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Final reduction across threadgroups (assuming <= 32 threadgroups)
    if (tid < 32) {
        float final_emb = 0.0f;
        float final_hid = 0.0f;
        uint num_tg = tg_size / 32;
        for (uint i = 0; i < num_tg; i++) {
            final_emb += tg_sum_emb[i];
            final_hid += tg_sum_hid[i];
        }
        
        // Compute inverse RMS
        float inv_rms_emb = rsqrt(final_emb / float(D) + eps);
        float inv_rms_hid = rsqrt(final_hid / float(D) + eps);
        
        tg_sum_emb[tid] = inv_rms_emb;
        tg_sum_hid[tid] = inv_rms_hid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float inv_rms_emb = tg_sum_emb[0];
    float inv_rms_hid = tg_sum_hid[0];
    
    // Vectorized normalization and concat
    for (uint v = 0; v < vec_per_thread; v++) {
        uint idx = tid * vec_per_thread + v;
        if (idx < D4) {
            half4 e = emb4[idx];
            half4 h = hid4[idx];
            half4 we = w_emb4[idx];
            half4 wh = w_hid4[idx];
            
            float4 e_norm = float4(e) * inv_rms_emb * float4(we);
            float4 h_norm = float4(h) * inv_rms_hid * float4(wh);
            
            out4[idx] = half4(e_norm);
            out4[idx + D4] = half4(h_norm);
        }
    }
}
"""

# ---------------------------------------------------------------------------
# Stage 1 Argmax kernel (per-block reduction)
# ---------------------------------------------------------------------------
_STAGE1_ARGMAX_SOURCE = r"""
using namespace metal;

// SIMD argmax reduction
template <typename T>
void simd_argmax(T val, uint idx, thread T& max_val, thread uint& max_idx) {
    // Reduce across 32 lanes
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

kernel void stage1_argmax(
    device const half* logits [[buffer(0)]],
    device float* block_max_vals [[buffer(1)]],
    device uint* block_max_indices [[buffer(2)]],
    constant uint& V [[buffer(3)]],
    constant uint& num_blocks [[buffer(4)]],
    uint block_id [[threadgroup_position_in_grid.x]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid.y]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint chunk_size = (V + num_blocks - 1) / num_blocks;
    const uint start_idx = block_id * chunk_size;
    const uint end_idx = min(start_idx + chunk_size, V);
    const uint row_offset = row * V;
    
    // Vectorized pointers
    device const half4* logits4 = reinterpret_cast<device const half4*>(logits + row_offset);
    
    float local_max = -1e30f;
    uint local_idx = start_idx;
    
    // Process vectorized portion
    const uint start4 = (start_idx + 3) / 4;
    const uint end4 = end_idx / 4;
    
    for (uint i = start4 + tid; i < end4; i += tg_size) {
        half4 vals = logits4[i];
        float4 vals_f = float4(vals);
        
        // Unrolled comparisons
        if (vals_f.x > local_max) { local_max = vals_f.x; local_idx = i * 4; }
        if (vals_f.y > local_max) { local_max = vals_f.y; local_idx = i * 4 + 1; }
        if (vals_f.z > local_max) { local_max = vals_f.z; local_idx = i * 4 + 2; }
        if (vals_f.w > local_max) { local_max = vals_f.w; local_idx = i * 4 + 3; }
    }
    
    // Handle remainder elements
    const uint remainder_start = end4 * 4;
    for (uint i = remainder_start + tid; i < end_idx; i += tg_size) {
        float val = float(logits[row_offset + i]);
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }
    
    // SIMD reduction
    float simd_max;
    uint simd_idx;
    simd_argmax(local_max, local_idx, simd_max, simd_idx);
    
    // Threadgroup reduction
    threadgroup float tg_vals[256];
    threadgroup uint tg_idxs[256];
    
    uint lane = tid & 31;
    if (lane == 0) {
        tg_vals[tid >> 5] = simd_max;
        tg_idxs[tid >> 5] = simd_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Final reduction (assuming <= 8 threadgroups)
    if (tid < 8) {
        float best_val = tg_vals[tid];
        uint best_idx = tg_idxs[tid];
        
        for (uint i = tid + 8; i < tg_size / 32; i += 8) {
            if (tg_vals[i] > best_val) {
                best_val = tg_vals[i];
                best_idx = tg_idxs[i];
            }
        }
        
        tg_vals[tid] = best_val;
        tg_idxs[tid] = best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid == 0) {
        uint out_idx = row * num_blocks + block_id;
        block_max_vals[out_idx] = tg_vals[0];
        block_max_indices[out_idx] = tg_idxs[0];
    }
}
"""

# ---------------------------------------------------------------------------
# Stage 2 Argmax kernel (final reduction)
# ---------------------------------------------------------------------------
_STAGE2_ARGMAX_SOURCE = r"""
using namespace metal;

kernel void stage2_argmax(
    device const float* block_max_vals [[buffer(0)]],
    device const uint* block_max_indices [[buffer(1)]],
    device float* final_max_vals [[buffer(2)]],
    device uint* final_max_indices [[buffer(3)]],
    constant uint& num_blocks [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid.y]]
) {
    const uint row_offset = row * num_blocks;
    
    float best_val = -1e30f;
    uint best_idx = 0;
    
    for (uint i = tid; i < num_blocks; i += 32) {
        float val = block_max_vals[row_offset + i];
        if (val > best_val) {
            best_val = val;
            best_idx = block_max_indices[row_offset + i];
        }
    }
    
    // SIMD reduction
    for (uint offset = 16; offset > 0; offset >>= 1) {
        float other_val = simd_shuffle_down(best_val, offset);
        uint other_idx = simd_shuffle_down(best_idx, offset);
        
        if (other_val > best_val) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }
    
    if (tid == 0) {
        final_max_vals[row] = best_val;
        final_max_indices[row] = best_idx;
    }
}
"""

# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
def fused_dual_rmsnorm_concat(
    emb: mx.array,
    hid: mx.array,
    w_emb: mx.array,
    w_hid: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    """Fused dual RMSNorm + concatenation."""
    B, D = emb.shape
    out = mx.zeros((B, 2 * D), dtype=mx.float16)
    
    # Launch kernel with 256 threads per threadgroup
    kernel = mf.metal_kernel(
        _DUAL_RMSNORM_CONCAT_SOURCE,
        "dual_rmsnorm_concat",
        names=["emb", "hid", "w_emb", "w_hid", "out", "D", "eps"],
    )
    
    kernel(
        emb,
        hid,
        w_emb,
        w_hid,
        out,
        D,
        mx.float32(eps),
        grid=(1, B, 1),
        threadgroup=(256, 1, 1),
    )
    
    return out


def fast_vocab_argmax(logits: mx.array, num_blocks: int = 32) -> tuple[mx.array, mx.array]:
    """Two-stage parallel argmax over vocabulary dimension."""
    B, V = logits.shape
    
    # Stage 1: Per-block reduction
    block_max_vals = mx.zeros((B, num_blocks), dtype=mx.float32)
    block_max_indices = mx.zeros((B, num_blocks), dtype=mx.uint32)
    
    kernel1 = mf.metal_kernel(
        _STAGE1_ARGMAX_SOURCE,
        "stage1_argmax",
        names=["logits", "block_max_vals", "block_max_indices", "V", "num_blocks"],
    )
    
    kernel1(
        logits,
        block_max_vals,
        block_max_indices,
        V,
        num_blocks,
        grid=(num_blocks, B, 1),
        threadgroup=(256, 1, 1),
    )
    
    # Stage 2: Final reduction
    final_max_vals = mx.zeros((B,), dtype=mx.float32)
    final_max_indices = mx.zeros((B,), dtype=mx.uint32)
    
    kernel2 = mf.metal_kernel(
        _STAGE2_ARGMAX_SOURCE,
        "stage2_argmax",
        names=["block_max_vals", "block_max_indices", "final_max_vals", "final_max_indices", "num_blocks"],
    )
    
    kernel2(
        block_max_vals,
        block_max_indices,
        final_max_vals,
        final_max_indices,
        num_blocks,
        grid=(1, B, 1),
        threadgroup=(32, 1, 1),
    )
    
    return final_max_vals, final_max_indices
```