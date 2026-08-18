# Qwen 3.8-27B Custom Metal Kernel Optimization Arena

**Status:** Scaffolded & Verified Live on Apple Silicon M4 Max  
**Sandbox Path:** `harness/kernels/sandbox/`  
**Host Framework:** Apple MLX 0.32.0 (`mx.fast.metal_kernel`)  
**Hardware Platform:** Apple Silicon (M4 Max / unified memory, 32-thread SIMDgroups)  
**Target Architecture:** `Qwen/Qwen3.8-27B` + Native 6-bit MTP Drafter  

---

## 1. Live Benchmark Results (DeepSeek Vectorized V2 vs Baseline MLX)

Executed via `uv run python harness/kernels/sandbox/benchmark_sandbox.py`:

| Operation | Baseline MLX | DeepSeek Vectorized Metal | Speedup / Savings | Accuracy Check |
| :--- | :--- | :--- | :--- | :--- |
| **248k Vocab Argmax ($B=4$)** | $205.48\,\mu\text{s}$ | **$168.90\,\mu\text{s}$** | **$+17.8\%$ ($1.22\times$, $-36.58\,\mu\text{s}$)** | **PASSED (Exact Match)** |
| **248k Vocab Argmax ($B=1$)** | $206.12\,\mu\text{s}$ | **$172.01\,\mu\text{s}$** | **$+16.5\%$ ($1.20\times$, $-34.11\,\mu\text{s}$)** | **PASSED (Exact Match)** |
| **Dual RMSNorm + Concat ($B=1$)** | $194.17\,\mu\text{s}$ | **$165.61\,\mu\text{s}$** | **$+14.7\%$ ($1.17\times$, $-28.56\,\mu\text{s}$)** | **PASSED ($<0.004$ diff)** |
| **Dual RMSNorm + Concat ($B=4$)** | $171.41\,\mu\text{s}$ | **$170.28\,\mu\text{s}$** | $\approx 1.01\times$ | **PASSED ($<0.004$ diff)** |

---

## 2. Kernel Implementations Applied

1. **`half4` Vectorized Pointer Reads:** Loads 8-byte aligned chunks to maximize memory bandwidth throughput.
2. **Warp-Level `simd_shuffle_down` Reductions:** Reduces 32 SIMD lanes inside registers without intermediate threadgroup shared memory barriers.
3. **Threadgroup Warp Bridge:** 8-warp reduction across 256 threads with in-kernel FP32 reciprocal square-root broadcasting.
