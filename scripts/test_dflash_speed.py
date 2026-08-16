import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.executors.dflash_engine import DFlashEngine

def run_test():
    prompt = "Write a high-performance Swift 6 actor-isolated Cache with TTL eviction, task cancellation, and Sendable error propagation. Include concise architectural notes."

    print("=" * 60)
    print("🔮 Qwen 3.8 27B + DFlash Drafter Live Performance Benchmark")
    print("=" * 60)
    print(f"Prompt: {prompt}\n")

    engine = DFlashEngine()

    print("Warming up engine & compiling Metal kernels...")
    # Warmup
    _ = engine.run_inference(
        prompt="Briefly define actor isolation in Swift 6 in one sentence.",
        max_tokens=64,
        enable_thinking=False,
        direct=True
    )

    print("\nWarmup complete. Running benchmark prompt...\n")
    start_time = time.perf_counter()
    first_token_time = None

    def on_token(count, tps):
        nonlocal first_token_time
        if first_token_time is None:
            first_token_time = time.perf_counter() - start_time
        if count % 20 == 0:
            print(f"[Streaming] Generated {count} tokens | Current speed: {tps:.1f} tok/s", flush=True)

    output = engine.run_inference(
        prompt=prompt,
        max_tokens=512,
        language="swift",
        enable_thinking=True,
        on_token_callback=on_token,
        direct=True
    )

    total_time = time.perf_counter() - start_time
    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Output Tokens:       {output.token_count}")
    print(f"Total Time:          {total_time:.2f} seconds")
    if first_token_time:
        print(f"Time to First Token: {first_token_time * 1000.0:.1f} ms")
    print(f"Effective Speed:     {output.tokens_per_second:.1f} tokens/second ⚡")
    print("=" * 60)
    print("\n📝 Generated Output:\n")
    if output.thinking:
        print(f"🧠 Thinking:\n{output.thinking}\n")
    print(f"💻 Code Output:\n{output.text}")

if __name__ == "__main__":
    run_test()
