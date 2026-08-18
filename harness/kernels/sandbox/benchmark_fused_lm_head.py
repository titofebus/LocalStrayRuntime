"""Benchmark and verification suite for Fused LM-Head GEMV + Argmax Kernel."""
import time
import mlx.core as mx
from harness.kernels.sandbox.fused_lm_head import fused_lm_head_argmax

def run_fused_lm_head_benchmark(warmup: int = 20, iterations: int = 100):
    print("=" * 65)
    print(" FUSED LM-HEAD GEMV + ARGMAX BENCHMARK (Qwen 3.8-27B)")
    print("=" * 65)
    print(f"Device: {mx.default_device()} | MLX: {mx.__version__}")
    print("Dimensions: Hidden Dim = 5120, Vocab Size = 248,320, Dtype = float16\n")

    D = 5120
    V = 248320
    weights = mx.random.normal((V, D)).astype(mx.float16)
    mx.eval(weights)

    for B in [1, 4]:
        hidden = mx.random.normal((B, D)).astype(mx.float16)
        mx.eval(hidden)

        # Baseline: Matrix multiplication followed by argmax
        def baseline_gemv_argmax(h, w):
            logits = mx.matmul(h, w.T)
            return mx.argmax(logits, axis=-1).astype(mx.uint32)

        # Fused: In-kernel dot products + reduction (zero logits allocation)
        def fused_gemv_argmax(h, w):
            return fused_lm_head_argmax(h, w)

        # Accuracy verification
        base_tok = baseline_gemv_argmax(hidden, weights)
        fused_tok = fused_gemv_argmax(hidden, weights)
        mx.eval(base_tok, fused_tok)

        is_exact = mx.array_equal(base_tok, fused_tok).item()
        print(f"--- Batch Size B = {B} ({'Single Step' if B == 1 else 'Draft Block'}) ---")
        print(f"Accuracy Check: [{'PASSED' if is_exact else 'FAILED'}] (Baseline: {base_tok.tolist()}, Fused: {fused_tok.tolist()})")

        # Warmup
        for _ in range(warmup):
            b_out = baseline_gemv_argmax(hidden, weights)
            f_out = fused_gemv_argmax(hidden, weights)
            mx.eval(b_out, f_out)

        # Benchmark Baseline
        t0 = time.perf_counter()
        for _ in range(iterations):
            out = baseline_gemv_argmax(hidden, weights)
            mx.eval(out)
        t_base = (time.perf_counter() - t0) / iterations * 1e6

        # Benchmark Fused
        t0 = time.perf_counter()
        for _ in range(iterations):
            out = fused_gemv_argmax(hidden, weights)
            mx.eval(out)
        t_fused = (time.perf_counter() - t0) / iterations * 1e6

        speedup = t_base / t_fused
        saving = t_base - t_fused

        print(f"Baseline Standard GEMV+Argmax: {t_base:8.2f} µs/call")
        print(f"Custom Fused LM-Head:          {t_fused:8.2f} µs/call")
        print(f"Speedup:                       {speedup:8.2f}x ({'+' if speedup > 1 else ''}{(speedup-1)*100:.1f}% faster, saving {saving:.2f} µs/step)\n")

if __name__ == "__main__":
    run_fused_lm_head_benchmark()
