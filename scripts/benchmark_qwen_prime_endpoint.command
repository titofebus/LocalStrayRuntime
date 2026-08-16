#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
HARNESS_DIR="${SCRIPT_DIR:h}"
LOG_PATH="/private/tmp/qwen-prime-endpoint-benchmark.log"
HEALTH_URL="http://127.0.0.1:8000/v1/models"
IDENTITY_URL="http://127.0.0.1:8000/v1/engine"

cd "$HARNESS_DIR"

if ! curl --fail --silent --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
    echo "Qwen Prime server is not running. Start it with scripts/launch_qwen_prime.command." >&2
    exit 1
fi

echo "QwenPrime live-endpoint benchmark" | tee "$LOG_PATH"
PYTHONUNBUFFERED=1 .venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG_PATH"
import json
import time
from urllib.request import Request, urlopen


with urlopen("http://127.0.0.1:8000/v1/engine", timeout=10) as response:
    engine_identity = json.load(response)

if (
    engine_identity.get("runtime_id") != "qwen38-native-mtp-v1"
    or engine_identity.get("prefix_cache_enabled") is not True
    or engine_identity.get("warmup_complete") is not True
):
    raise RuntimeError(f"Unexpected Qwen Prime runtime: {engine_identity}")

print(
    "engine_identity="
    f"{engine_identity['target_model_id']} + {engine_identity['draft_model_id']}, "
    f"target_bits={engine_identity['target_quantization_bits']}, "
    f"draft_bits={engine_identity['draft_quantization_bits']}, "
    f"block_tokens={engine_identity['block_tokens']}, "
    f"prefix_cache={engine_identity['prefix_cache_enabled']}, "
    f"warmup={engine_identity['warmup_complete']}, "
    f"draft_sha256={engine_identity['draft_weights_sha256']}"
)


body = {
    "model": "qwen3.8-27b",
    "messages": [
        {
            "role": "system",
            "content": (
                "You are Qwen Prime, an elite AI systems and software engineering "
                "assistant running natively on Apple Silicon with MLX and DFlash "
                "speculative acceleration.\n\nGuidelines:\n"
                "1. Provide precise, production-grade implementations with clean explanations.\n"
                "2. In Swift code, strictly enforce Swift 6 concurrency safety, actor "
                "isolation, and Sendable conformance. Avoid force-unwrapping.\n"
                "3. In Rust and Python, follow zero-cost abstractions, idiomatic design, "
                "and proper error handling."
            ),
        },
        {
            "role": "user",
            "content": (
                "Implement a production Swift 6 actor named "
                "BoundedTaskQueue<Element: Sendable>. It must support bounded capacity, "
                "FIFO ordering, suspending producers when full, suspending consumers when "
                "empty, cancellation without leaked continuations, and graceful shutdown "
                "that drains queued elements before returning nil. Return only compilable "
                "Swift code."
            ),
        },
    ],
    "temperature": 0.1,
    "max_completion_tokens": 256,
    "stream": True,
    "thinking": {"type": "disabled"},
}

request = Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(body).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
started = time.perf_counter()
first_token_at = None
estimated_tokens = 0
server_usage = None

with urlopen(request, timeout=3600) as response:
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        event = json.loads(payload)
        choices = event.get("choices") or []
        delta = choices[0].get("delta", {}) if choices else {}
        for key in ("reasoning_content", "content"):
            text = delta.get(key)
            if text:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                estimated_tokens += max(1, len(text) // 4)
        if event.get("usage"):
            server_usage = event["usage"]

finished = time.perf_counter()
client_generation_seconds = max(0.001, finished - (first_token_at or started))
client_estimated_tps = estimated_tokens / client_generation_seconds

print("\n=== LIVE ENDPOINT RESULT ===")
print(f"client_estimated_tps={client_estimated_tps:.2f}")
print(f"client_estimated_tokens={estimated_tokens}")
print(f"wall_seconds={finished - started:.2f}")
if server_usage is None:
    raise RuntimeError("Server omitted final usage telemetry")
else:
    required_speculative_fields = {
        "accepted_from_draft",
        "acceptance_ratio",
        "cycles_completed",
        "adaptive_block_reductions",
        "adaptive_block_min",
        "prefill_seconds",
        "prefill_tokens_per_second",
        "prefill_tokens_computed",
        "prefill_tokens_restored",
        "prefix_cache_hit_tokens",
        "prefix_cache_lookup_ms",
    }
    missing_speculative_fields = sorted(required_speculative_fields - server_usage.keys())
    if missing_speculative_fields:
        raise RuntimeError(
            "Server omitted speculative telemetry fields: "
            + ", ".join(missing_speculative_fields)
        )
    print(f"server_tps={float(server_usage.get('tokens_per_second', 0.0)):.2f}")
    print(f"server_completion_tokens={server_usage.get('completion_tokens')}")
    print(f"accepted_from_draft={server_usage.get('accepted_from_draft')}")
    print(f"acceptance_ratio={float(server_usage.get('acceptance_ratio', 0.0)):.1%}")
    print(f"cycles_completed={server_usage.get('cycles_completed')}")
    print(f"adaptive_block_reductions={server_usage.get('adaptive_block_reductions')}")
    print(f"adaptive_block_min={server_usage.get('adaptive_block_min')}")
    print(f"prefill_seconds={float(server_usage.get('prefill_seconds', 0.0)):.3f}")
    print(f"prefill_tps={float(server_usage.get('prefill_tokens_per_second', 0.0)):.2f}")
    print(f"prefill_tokens_computed={server_usage.get('prefill_tokens_computed')}")
    print(f"prefill_tokens_restored={server_usage.get('prefill_tokens_restored')}")
    print(f"prefix_cache_hit_tokens={server_usage.get('prefix_cache_hit_tokens')}")
    print(f"prefix_cache_lookup_ms={float(server_usage.get('prefix_cache_lookup_ms', 0.0)):.3f}")
PY

echo "Log: $LOG_PATH"
echo
read -k 1 "?Press any key to close..."
