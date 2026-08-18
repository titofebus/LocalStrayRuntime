"""Benchmark long-sequence token generation speed on live server."""
import json
import time
import urllib.request

url = "http://127.0.0.1:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

prompt = "Write a production-grade Swift 6 Actor called `AsyncDebouncedWorker` that coalesces high-frequency incoming events. Include complete implementation, cancellation tokens, and detailed docstrings."

payload = {
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": prompt}],
    "stream": True,
    "thinking": {"type": "disabled"},
    "max_tokens": 300,
}

print("=" * 65)
print(" BENCHMARKING LONG SEQUENCE GENERATION (300 TOKENS)")
print("=" * 65)

req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)

tokens = 0
start_time = time.perf_counter()
first_token_time = None
checkpoints = [50, 100, 150, 200, 250, 300]
last_cp_time = None
last_cp_tokens = 0

with urllib.request.urlopen(req) as resp:
    for line in resp:
        line = line.decode("utf-8").strip()
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            break
        data = json.loads(data_str)
        delta = data["choices"][0].get("delta", {})
        content = delta.get("content", "")
        if content:
            tokens += 1
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
                last_cp_time = now
                print(f"TTFT (Time to First Token): {(first_token_time - start_time):.3f}s")

            for cp in checkpoints:
                if tokens == cp:
                    chunk_time = now - last_cp_time
                    chunk_tokens = tokens - last_cp_tokens
                    chunk_speed = chunk_tokens / chunk_time
                    total_speed = (tokens - 1) / (now - first_token_time)
                    print(f"Tokens {last_cp_tokens:3d} -> {tokens:3d}: {chunk_speed:5.1f} t/s | Cumulative: {total_speed:5.1f} t/s")
                    last_cp_time = now
                    last_cp_tokens = tokens

        usage = data.get("usage")
        if usage:
            print("\n" + "=" * 65)
            print(" FINAL USAGE REPORT:")
            print(f" Total Tokens:        {usage.get('completion_tokens')}")
            print(f" Overall Speed:       {usage.get('tokens_per_second'):.1f} t/s")
            print(f" Accepted From Draft: {usage.get('accepted_from_draft')} ({usage.get('acceptance_ratio', 0):.1%})")
            print(f" Prefill Speed:       {usage.get('prefill_tokens_per_second'):.1f} t/s")
            print("=" * 65)
