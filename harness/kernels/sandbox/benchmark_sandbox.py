"""Automated validation and benchmark suite for custom MLX/Metal kernels."""
import time
import mlx.core as mx
from harness.kernels.sandbox.fused_mtp_ops import fused_dual_rmsnorm_concat, fast_vocab_argmax

# Qwen 3.8-27B Exact Architectural Dimensions
HIDDEN_DIM = 5120
VOCAB_SIZE = 248320
BATCH_SIZES = [1, 4]  # 1 for single token, 4 for MTP speculative draft block
DTYPE = mx.float16
WARMUP_RUNS = 20
BENCH_RUNS = 200


def benchmark_dual_rmsnorm():
    print("=" * 65)
    print(" 1. BENCHMARK: Dual RMSNorm + Concatenate (MTP Draft Step)")
    print("=" * 65)
    print(f"Dimensions: Hidden Dim = {HIDDEN_DIM}, Dtype = {DTYPE}")
    
    eps = 1e-6
    w_emb = mx.random.normal((HIDDEN_DIM,)).astype(DTYPE)
    w_hid = mx.random.normal((HIDDEN_DIM,)).astype(DTYPE)
    
    def standard_mlx_baseline(emb: mx.array, hid: mx.array) -> mx.array:
        norm_emb = mx.fast.rms_norm(emb, w_emb, eps)
        norm_hid = mx.fast.rms_norm(hid, w_hid, eps)
        return mx.concatenate([norm_emb, norm_hid], axis=-1)

    for B in BATCH_SIZES:
        print(f"\n--- Batch Size B = {B} (Tokens per Step) ---")
        emb = mx.random.normal((B, HIDDEN_DIM)).astype(DTYPE)
        hid = mx.random.normal((B, HIDDEN_DIM)).astype(DTYPE)
        mx.eval(emb, hid, w_emb, w_hid)

        # Numerical Correctness
        baseline_out = standard_mlx_baseline(emb, hid)
        custom_out = fused_dual_rmsnorm_concat(emb, hid, w_emb, w_hid, eps)
        mx.eval(baseline_out, custom_out)

        max_diff = mx.max(mx.abs(baseline_out - custom_out)).item()
        is_close = mx.allclose(baseline_out, custom_out, atol=5e-3, rtol=5e-3)
        print(f"Accuracy Check: {'[PASSED]' if is_close else '[FAILED]'} (Max absolute diff: {max_diff:.6e})")

        # Warmup
        for _ in range(WARMUP_RUNS):
            res_b = standard_mlx_baseline(emb, hid)
            res_c = fused_dual_rmsnorm_concat(emb, hid, w_emb, w_hid, eps)
            mx.eval(res_b, res_c)

        # Benchmark Standard MLX
        start_b = time.perf_counter_ns()
        for _ in range(BENCH_RUNS):
            res_b = standard_mlx_baseline(emb, hid)
            mx.eval(res_b)
        end_b = time.perf_counter_ns()
        avg_b_us = (end_b - start_b) / (BENCH_RUNS * 1000)

        # Benchmark Custom Fused Metal
        start_c = time.perf_counter_ns()
        for _ in range(BENCH_RUNS):
            res_c = fused_dual_rmsnorm_concat(emb, hid, w_emb, w_hid, eps)
            mx.eval(res_c)
        end_c = time.perf_counter_ns()
        avg_c_us = (end_c - start_c) / (BENCH_RUNS * 1000)

        speedup = (avg_b_us / avg_c_us) if avg_c_us > 0 else 0
        diff_us = avg_b_us - avg_c_us
        pct = (diff_us / avg_b_us) * 100 if avg_b_us > 0 else 0

        print(f"Baseline Standard MLX: {avg_b_us:8.2f} µs/call")
        print(f"Custom Fused Metal:   {avg_c_us:8.2f} µs/call")
        print(f"Speedup:              {speedup:8.2f}x ({pct:+.1f}% faster, saving {diff_us:.2f} µs/step)")


def benchmark_vocab_argmax():
    print("\n" + "=" * 65)
    print(" 2. BENCHMARK: 248,320-Vocab Argmax Reduction (LM Head)")
    print("=" * 65)
    print(f"Dimensions: Vocab Size = {VOCAB_SIZE}, Dtype = {DTYPE}")

    for B in BATCH_SIZES:
        print(f"\n--- Batch Size B = {B} (Tokens per Step) ---")
        logits = mx.random.normal((B, VOCAB_SIZE)).astype(DTYPE)
        # Inject known maximums to guarantee distinct argmax values
        for b in range(B):
            target_idx = (b * 12345 + 777) % VOCAB_SIZE
            logits[b, target_idx] = 100.0
        mx.eval(logits)

        # Numerical Correctness
        baseline_tokens = mx.argmax(logits, axis=-1).astype(mx.uint32)
        custom_tokens = fast_vocab_argmax(logits)
        mx.eval(baseline_tokens, custom_tokens)

        is_equal = mx.array_equal(baseline_tokens, custom_tokens).item()
        print(f"Accuracy Check: {'[PASSED]' if is_equal else '[FAILED]'} (Values: baseline={baseline_tokens.tolist()}, custom={custom_tokens.tolist()})")

        # Warmup
        for _ in range(WARMUP_RUNS):
            res_b = mx.argmax(logits, axis=-1)
            res_c = fast_vocab_argmax(logits)
            mx.eval(res_b, res_c)

        # Benchmark Standard MLX
        start_b = time.perf_counter_ns()
        for _ in range(BENCH_RUNS):
            res_b = mx.argmax(logits, axis=-1)
            mx.eval(res_b)
        end_b = time.perf_counter_ns()
        avg_b_us = (end_b - start_b) / (BENCH_RUNS * 1000)

        # Benchmark Custom Fused Metal
        start_c = time.perf_counter_ns()
        for _ in range(BENCH_RUNS):
            res_c = fast_vocab_argmax(logits)
            mx.eval(res_c)
        end_c = time.perf_counter_ns()
        avg_c_us = (end_c - start_c) / (BENCH_RUNS * 1000)

        speedup = (avg_b_us / avg_c_us) if avg_c_us > 0 else 0
        diff_us = avg_b_us - avg_c_us
        pct = (diff_us / avg_b_us) * 100 if avg_b_us > 0 else 0

        print(f"Baseline Standard MLX: {avg_b_us:8.2f} µs/call")
        print(f"Custom Fused Metal:   {avg_c_us:8.2f} µs/call")
        print(f"Speedup:              {speedup:8.2f}x ({pct:+.1f}% faster, saving {diff_us:.2f} µs/step)")


if __name__ == "__main__":
    print(f"Running MLX Kernel Sandbox Benchmark on Apple Silicon...")
    print(f"MLX Version: {mx.__version__} | GPU Device: {mx.default_device()}\n")
    benchmark_dual_rmsnorm()
    benchmark_vocab_argmax()
    print("\n" + "=" * 65)
    print("All Sandbox Benchmarks Completed Successfully!")
    print("=" * 65)
