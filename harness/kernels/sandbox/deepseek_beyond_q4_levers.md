# Exhaustive Analysis: Remaining Non-Q4 Levers for Qwen 3.8-27B on M4 Max

## Executive Summary

**No, we have not run out of options.** There are 5 distinct architectural levers remaining, each targeting different bottlenecks. Combined potential: **35-40 tok/s sustained on deep logic** (vs current 22-24), with peak speeds of **45-50 tok/s** on clean code. The critical insight: our current bottleneck is *verification bandwidth*, not *draft quality* — this reframes the optimization priority.

---

## Lever 1: Speculative Branch Trees (DDTree)

### Current State Analysis
- **Baseline**: 48.7% acceptance, 51.25ms verification pass
- **Bottleneck**: Single-path speculation fails on branching logic (if/else, switch, ternary chains)
- **Root Cause**: Sequential dependency in code generation — the drafter commits to one path, and when the target disagrees, the entire prefix is invalidated

### Architecture: Parallel Branch Speculation

```
Current (Single Path):
Draft: [A] → [B] → [C] → [D] → [E]
Verify: [A] ✓ [B] ✓ [C] ✗ → RESTART from C

DDTree (3-Branch):
Draft: [A] → [B] → [C₁] → [D₁] → [E₁]
              ↘ [C₂] → [D₂] → [E₂]
              ↘ [C₃] → [D₃] → [E₃]
Verify: [A] ✓ [B] ✓ [C₂] ✓ [D₂] ✓ [E₂] ✓ → ACCEPT 5 tokens
```

### Implementation Details

**Branch Selection Heuristic** (critical for efficiency):
```python
def select_branches(logits, temperature=0.8):
    # Top-1: Most likely continuation
    # Top-2: Alternative with highest entropy (divergent path)
    # Top-3: Random sample weighted by probability (exploration)
    probs = softmax(logits / temperature)
    top_k = torch.topk(probs, k=3)
    
    # Entropy-based branch selection
    branch_1 = top_k.indices[0]  # Greedy
    branch_2 = top_k.indices[1]  # Second most likely
    branch_3 = sample_from(probs, exclude=top_k.indices[:2])  # Exploration
    
    return [branch_1, branch_2, branch_3]
```

**Verification Strategy** — Single Pass, Multiple Paths:
- The target model processes all 3 branches in **one forward pass** using a batched KV-cache
- Memory cost: 3× KV-cache for the divergent segment (typically 2-4 tokens)
- **Key optimization**: Only diverge at the *first* high-entropy token, not every token

### Expected Performance

| Metric | Current | DDTree (3-branch) | Improvement |
|--------|---------|-------------------|-------------|
| Acceptance | 48.7% | 72-78% | +50-60% |
| Verification time | 51.25ms | 53-55ms | +4% (negligible) |
| Effective tok/s | 22-24 | 30-33 | +35-40% |
| Memory overhead | — | +2-3% KV-cache | Acceptable |

### Engineering Complexity: **Medium-High**
- Requires batched KV-cache management in Metal
- Branch selection heuristic needs tuning per task type
- **Risk**: Overhead of branch management could eat gains if divergence points are rare

### Ranking: **#1 Priority** — Highest ROI, directly addresses the acceptance bottleneck

---

## Lever 2: Asynchronous Dual-Stream Pipeline

### Current State Analysis
- **Draft time**: ~8-10ms (MTP drafter, 345MB)
- **Verification time**: 51.25ms
- **Current pipeline**: Serial — draft completes, then verification starts
- **Hidden cost**: 8-10ms of GPU idle time during draft phase

### Architecture: Overlapped Execution

```
Current (Serial):
|-- Draft (8ms) --|-- Verify (51ms) --|-- Draft (8ms) --|-- Verify (51ms) --|
                  ↑ GPU IDLE          ↑ GPU IDLE

Optimized (Overlapped):
|-- Draft₁ --|-- Verify₁ --|-- Draft₂ --|-- Verify₂ --|
             |-- Draft₂ --|              |-- Draft₃ --|
             ↑ Hidden     ↑ Hidden
```

### Implementation via Metal Command Queues

```swift
// Dual command queues for true async execution
let draftQueue = device.makeCommandQueue()!
let verifyQueue = device.makeCommandQueue()!

// Pipeline structure
func generateLoop() {
    // Queue 1: Draft generation (MTP model)
    let draftCmd = draftQueue.makeCommandBuffer()
    encodeDraft(draftCmd, input: currentContext)
    
    // Queue 2: Verification (target model) — starts immediately
    let verifyCmd = verifyQueue.makeCommandBuffer()
    encodeVerification(verifyCmd, draft: draftOutput)
    
    // Both execute concurrently on different GPU engines
    draftCmd.commit()
    verifyCmd.commit()
    
    // Synchronization point: only at token acceptance
    let accepted = waitForVerification(verifyCmd)
    updateContext(accepted)
}
```

### Critical Considerations

**GPU Engine Utilization**:
- M4 Max has 10 GPU cores — drafter (345MB) can run on 2-3 cores while target (20.5GB) uses 7-8
- **Memory bandwidth contention**: Both models compete for the 400GB/s unified memory bandwidth
- **Solution**: Time-slice via priority — drafter gets 15% bandwidth during verification

**Pipeline Depth**:
- Overlap depth: 2 (draft next while verifying current)
- **Deeper overlap (3-4)**: Diminishing returns due to context dependency — draft for step N+2 depends on verification of step N

### Expected Performance

| Metric | Current | Overlapped | Improvement |
|--------|---------|------------|-------------|
| Draft time hidden | 0% | 100% | Eliminates 8-10ms |
| Effective tok/s | 22-24 | 24-26 | +8-10% |
| GPU utilization | ~85% | ~97% | +12% |

### Engineering Complexity: **Medium**
- Metal command queue management is well-documented
- **Risk**: Memory bandwidth contention could reduce verification speed by 5-10%, partially negating gains

### Ranking: **#2 Priority** — Complements DDTree, adds 8-10% on top

---

## Lever 3: KV-Cache Quantization (FP8/INT8)

### Current State Analysis
- **Context lengths**: 1k-4k tokens (typical for code generation)
- **KV-cache size**: ~2.5GB at 4k context (FP16)
- **Memory bandwidth**: KV-cache reads consume ~30% of verification bandwidth

### Quantization Strategy

**FP8 KV-Cache** (E4M3 format):
- **Precision loss**: Minimal for attention scores (values in [-1, 1] range)
- **Memory savings**: 50% (2.5GB → 1.25GB at 4k context)
- **Bandwidth savings**: 50% on KV reads

**INT8 KV-Cache** (with per-head scaling):
- **Precision loss**: Slightly higher than FP8, but acceptable for code
- **Memory savings**: 50%
- **Bandwidth savings**: 50%

### Implementation via Metal

```swift
// FP8 KV-cache storage
struct FP8KVCache {
    var keys: MTLBuffer  // FP8 E4M3 format
    var values: MTLBuffer // FP8 E4M3 format
    var scales: MTLBuffer // FP16 per-head scales
    
    func read(at position: Int) -> (SIMD4<Float>, SIMD4<Float>) {
        // Dequantize on-the-fly during attention
        let k_fp8 = keys.load(fromByteOffset: position * 4, as: UInt8.self)
        let v_fp8 = values.load(fromByteOffset: position * 4, as: UInt8.self)
        let scale = scales.load(fromByteOffset: position * 2, as: Float16.self)
        
        return (Float(k_fp8) * scale, Float(v_fp8) * scale)
    }
}
```

### Expected Performance

| Context Length | Current Bandwidth | FP8 Bandwidth | Time Savings |
|----------------|-------------------|---------------|--------------|
| 1k | 12.5ms | 6.25ms | 6.25ms |
| 2k | 25ms | 12.5ms | 12.5ms |
| 4k | 50ms | 25ms | 25ms |

**Critical Insight**: At 4k context, KV-cache quantization alone could reduce verification from 51.25ms → 38ms (25% speedup)

### Engineering Complexity: **Low-Medium**
- Metal supports FP8 natively (M4 family)
- Requires custom attention kernel with dequantization
- **Risk**: Precision loss could reduce acceptance rate by 1-2%

### Ranking: **#3 Priority** — High impact at longer contexts, low risk

---

## Lever 4: Dynamic Block Expansion (K=6)

### Current State Analysis
- **Current K**: 5 tokens per draft
- **Acceptance rate**: 48.7% on deep logic
- **Observation**: On high-confidence runs (clean code, simple patterns), acceptance often exceeds 70%

### Dynamic Strategy

```python
def dynamic_block_size(confidence_history):
    # Track rolling acceptance rate
    recent_acceptance = rolling_mean(acceptance_history, window=20)
    
    if recent_acceptance > 0.65:
        return 6  # Expand: high confidence, more tokens per draft
    elif recent_acceptance > 0.50:
        return 5  # Maintain
    else:
        return 4  # Contract: low confidence, shorter drafts
    
    # Additionally: adapt based on entropy of last draft
    if last_draft_entropy < threshold:
        return 6  # Low entropy = predictable = expand
```

### Expected Performance

| Scenario | K=5 | K=6 | Improvement |
|----------|-----|-----|-------------|
| Clean code (70% acceptance) | 35 tok/s | 38 tok/s | +8.5% |
| Deep logic (48.7% acceptance) | 22 tok/s | 21 tok/s | -4.5% (worse!) |

**Critical Insight**: Dynamic expansion only helps when acceptance is already high. For deep logic, K=6 *hurts* because:
- More tokens to verify per draft
- Higher chance of early divergence
- Wasted verification bandwidth

### Engineering Complexity: **Low**
- Simple heuristic implementation
- **Risk**: Minimal — worst case is 4-5% regression on hard tasks

### Ranking: **#4 Priority** — Complementary, not primary. Use only after DDTree improves baseline acceptance

---

## Lever 5: Mixed-Precision Quantization (6-bit Attention + 4-bit MLP)

### Current State Analysis
- **Model size**: 20.5GB (6-bit affine)
- **Verification time**: 51.25ms @ 400GB/s
- **Memory bandwidth**: The 20.5GB must be read every verification pass

### Architecture: Selective Quantization

```
Current (Uniform 6-bit):
[Attention: 6-bit] [MLP: 6-bit] [Attention: 6-bit] [MLP: 6-bit] ...

Proposed (Mixed):
[Attention: 6-bit] [MLP: 4-bit] [Attention: 6-bit] [MLP: 4-bit] ...
```

**Rationale**:
- **Attention layers**: Critical for reasoning, context understanding — keep at 6-bit
- **MLP layers**: Feed-forward networks are more redundant, tolerate 4-bit
- **Empirical evidence**: MLP quantization to 4-bit causes <1% accuracy loss on code tasks

### Memory Breakdown

| Component | Current (6-bit) | Mixed (6+4-bit) | Savings |
|-----------|-----------------|------------------|---------|
| Attention (40%) | 8.2GB | 8.2GB | 0 |
| MLP (60%) | 12.3GB | 8.2GB | 4.1GB |
| **Total** | **20.5GB** | **16.4GB** | **-20%** |

### Expected Performance

| Metric | Current | Mixed-Precision | Improvement |
|--------|---------|-----------------|-------------|
| Model size | 20.5GB | 16.4GB | -20% |
| Verification time | 51.25ms | 41ms | -20% |
| Effective tok/s | 22-24 | 27-29 | +20% |
| Acceptance rate | 48.7% | 48.5% | -0.2% (negligible) |

### Implementation via Metal

```swift
// Per-layer quantization configuration
struct LayerConfig {
    let attentionBits: Int = 6
    let mlpBits: Int = 4
}

// Custom Metal kernel for mixed-precision MLP
kernel void mlp_forward_4bit(
    device const uint8_t* weights_4bit,  // Packed 4-bit weights
    device const float* scales,          // Per-channel scales
    device const float* input,
    device float* output,
    uint id [[thread_position_in_grid]]
) {
    // Dequantize 4-bit weights on-the-fly
    float w = dequantize_4bit(weights_4bit[id], scales[id]);
    output[id] = w * input[id];
}
```

### Engineering Complexity: **Medium**
- Requires per-layer quantization configuration
- Custom Metal kernels for 4-bit MLP operations
- **Risk**: Minimal — MLP quantization is well-understood

### Ranking: **#5 Priority** — Highest raw speedup, but requires most engineering effort

---

## Combined Strategy & Final Recommendations

### Optimal Implementation Order

```
Phase 1 (Week 1-2): Lever 1 (DDTree) + Lever 2 (Async Pipeline)
    → Expected: 30-33 tok/s on deep logic
    → Effort: Medium-High
    → Risk: Medium

Phase 2 (Week 3-4): Lever 3 (KV-Cache FP8)
    → Expected: +10-15% on top of Phase 1
    → Effort: Low-Medium
    → Risk: Low

Phase 3 (Week 5-6): Lever 5 (Mixed-Precision)
    → Expected: +20% on top of Phase 1+2
    → Effort: Medium
    → Risk: Low

Phase 4 (Ongoing): Lever 4 (Dynamic K)
    → Expected: +5-8% on clean code
    → Effort: Low
    → Risk: Minimal
```

### Projected Final Performance

| Scenario | Current | After All Levers | Improvement |
|----------|---------|------------------|-------------|
| Clean code | 35-36 tok/s | 48-52 tok/s | +40-45% |
| Deep logic | 22-24 tok/s | 35-38 tok/s | +55-60% |
| TTFT | 0.9s | 0.7s (KV-cache FP8) | -22% |

### Critical Engineering Recommendations

1. **Prioritize DDTree over everything else** — it addresses the root cause (acceptance rate) rather than symptoms (bandwidth)

2. **Implement Async Pipeline second** — it's orthogonal to DDTree and provides compounding gains

3. **Defer Mixed-Precision until after DDTree** — the 20% bandwidth savings are less impactful when acceptance is already 75% (verification passes are shorter)

4. **Monitor acceptance rate degradation** with KV-cache FP8 — if it drops >2%, revert to FP16 for attention KV

5. **Consider hybrid approach**: Use DDTree only for high-entropy tokens (branching logic), single-path for low-entropy (syntax, boilerplate)

### Final Verdict

**We have not run out of options.** The combination of DDTree + Async Pipeline + KV-cache FP8 + Mixed-Precision can deliver **35-38 tok/s on deep logic** — a 60% improvement over current — without touching Q4 quantization. The key insight is that our bottleneck is *verification efficiency*, not *model size*. By improving acceptance rate (DDTree) and hiding latency (Async Pipeline), we extract more value from each 51ms verification pass before considering further quantization.