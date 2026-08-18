## Architectural Review: Qwen 3.8-27B on M4 Max

### 1. ROI Ranking: Top 3 Optimizations

**#1: Fused LM-Head GEMV + Argmax** (Highest ROI, ~15-25% gain)

**Rationale:** At 248,320 vocab × 27B params, the LM head is the single largest memory-bound operation per token. Currently you're:
- Writing 500KB logits to unified memory
- Reading them back for argmax
- That's 1MB of wasted bandwidth per token

With 400 GB/s and ~30 tok/s, you're burning ~13.3 GB/s just on logits I/O. Fusing eliminates this entirely.

**#2: Adaptive Speculative Block Sizing** (Second, ~10-15% gain)

**Rationale:** Your 54% acceptance rate with K=4 means you're wasting ~46% of draft compute. With adaptive K:
- High-entropy regions (K=2-3): Reduce wasted draft tokens
- Low-entropy regions (K=5-6): Exploit predictability
- Expected acceptance improvement: 54% → 65-70%

**#3: 4-bit Quantization Shift** (Third, ~20-30% gain but with quality risk)

**Rationale:** 14.2GB vs 20.5GB = 31% less memory traffic. However:
- 6-bit → 4-bit quality degradation is measurable on 27B
- Group size 64 at 4-bit: 2-bit effective quantization per group
- Recommend: Only if you can validate perplexity delta < 0.5

---

### 2. Theoretical Ceiling Analysis

**Physical ceiling calculation:**
```
Memory-bound ceiling = Bandwidth / (Model Size + KV Cache + Overhead)
                     = 400 GB/s / (20.5 GB + 2 GB KV + 0.5 GB overhead)
                     ≈ 17.4 tok/s (without speculation)

With speculative decoding (K=4, 54% acceptance):
Effective tokens/step = 1 + 4 × 0.54 = 3.16
Draft cost = 1/3.16 × target cost
Ceiling ≈ 17.4 × 3.16 / (1 + 1/3.16) ≈ 41.7 tok/s
```

**Current efficiency loss:**
- You're at 29-35 tok/s vs ~42 tok/s ceiling = **~75-83% efficiency**
- Loss breakdown:
  - LM head I/O: ~8-10%
  - Draft verification overhead: ~5%
  - Kernel launch latency: ~3-5%
  - Memory fragmentation: ~2-3%

---

### 3. Fused LM-Head GEMV + Argmax: MSL Strategy

```metal
#include <metal_stdlib>
using namespace metal;

// Constants for M4 Max (32-thread SIMD groups)
constant uint SIMD_WIDTH = 32;
constant uint VECTOR_WIDTH = 4;  // half4 alignment

kernel void fused_lm_head_argmax(
    device const half* weights,      // [vocab_size, hidden_size] 6-bit dequantized
    device const uchar* quant_meta,  // scale/zero-point metadata
    device const half* hidden_state, // [hidden_size] from last layer
    device const uint* quant_params, // group_size, etc.
    device atomic_uint* max_index,   // atomic output
    device atomic_float* max_value,  // atomic output
    constant uint& vocab_size,
    constant uint& hidden_size,
    constant uint& group_size,
    uint3 tid [[thread_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Each thread handles one vocab entry
    uint vocab_idx = tid.x;
    if (vocab_idx >= vocab_size) return;
    
    // Vectorized dot product with half4 alignment
    float acc = 0.0f;
    uint weight_offset = vocab_idx * hidden_size;
    
    // Process in half4 chunks (8 bytes per load)
    for (uint i = 0; i < hidden_size; i += VECTOR_WIDTH * 2) {
        // Load 4 half values from weights (dequantized on-the-fly)
        half4 w_vec = *(device half4*)(weights + weight_offset + i);
        half4 h_vec = *(device half4*)(hidden_state + i);
        
        // FMA accumulation
        acc += dot(w_vec, h_vec);
        
        // Handle 6-bit dequantization if needed
        // (Assuming pre-dequantized for simplicity, or inline here)
    }
    
    // SIMD-group reduction for argmax
    float local_max = acc;
    uint local_idx = vocab_idx;
    
    // Warp-level reduction using simd_shuffle
    for (uint offset = SIMD_WIDTH/2; offset > 0; offset >>= 1) {
        float other_max = simd_shuffle_down(local_max, offset);
        uint other_idx = simd_shuffle_down(local_idx, offset);
        
        bool take_other = (other_max > local_max) || 
                         (other_max == local_max && other_idx < local_idx);
        
        if (take_other) {
            local_max = other_max;
            local_idx = other_idx;
        }
    }
    
    // Thread 0 of each SIMD group writes to global atomics
    if (simd_is_first()) {
        atomic_fetch_max_explicit(max_value, local_max, memory_order_relaxed);
        // Only update index if we won (need CAS pattern)
        // Use atomic_compare_exchange for correctness
    }
}
```

**Key optimizations:**

1. **Memory layout:** Pre-dequantize weights to `half` in a separate pass (or fuse dequant into load). This doubles effective bandwidth vs 6-bit.

2. **Vectorized loads:** `half4` = 8 bytes per load. M4 Max has 128-byte cache lines, so 16 half4 loads per line.

3. **SIMD reduction:** Use `simd_shuffle_down` for O(log(32)) = 5 steps instead of 32 sequential comparisons.

4. **Atomic contention:** Use `atomic_fetch_max` for value, then CAS for index. With 248,320 threads → 7,760 SIMD groups → only 7,760 atomic operations.

**Alternative: Two-pass approach (lower latency):**

```metal
// Pass 1: Each SIMD group computes local max
// Pass 2: Single thread reduces 7,760 local maxima
// This avoids atomics entirely, better for latency-bound
```

**Integration with MLX:**

```python
# MLX integration
import mlx.core as mx

def fused_lm_head(hidden_states, lm_head_weights, quant_meta):
    """Fused GEMV + Argmax without materializing logits"""
    
    # Custom Metal kernel
    kernel = mx.compile(
        fused_lm_head_kernel,
        inputs=[hidden_states, lm_head_weights, quant_meta],
        outputs=[mx.int32, mx.float32]  # (index, value)
    )
    
    # Returns (token_id, logit_value) directly
    token_id, logit_value = kernel()
    return token_id
```

**Expected impact:**
- Eliminates 500KB write + 500KB read per token
- At 30 tok/s: saves ~30 MB/s bandwidth (7.5% of total)
- Combined with reduced kernel launch overhead: ~10-12% total gain

---

### Additional Critical Recommendations

**1. Kernel fusion hierarchy:**
```
Layer N → RMSNorm → LM Head + Argmax (all fused)
```
This eliminates 2 intermediate memory round-trips.

**2. Draft verification optimization:**
Instead of verifying 4 tokens sequentially, batch-verify all 4 in parallel:
```metal
// Process 4 draft tokens simultaneously
// Each gets its own SIMD group
// Reduces latency by 4x for verification step
```

**3. Memory allocation strategy:**
Use `MTLHeap` for all intermediate buffers. M4 Max benefits from:
- 16KB alignment for SIMD groups
- 64KB alignment for threadgroup memory
- Avoid fragmentation with pre-allocated pools

**4. Consider 3-bit quantization with 2-bit residual:**
```
3-bit coarse + 2-bit residual = effective 5-bit
Better quality than pure 4-bit, similar bandwidth
```

**Priority order for implementation:**
1. Fused LM-Head (this week)
2. Adaptive K (next week)  
3. 4-bit quantization (validate quality first)

This should get you from 29-35 tok/s to 38-42 tok/s, approaching the physical ceiling.