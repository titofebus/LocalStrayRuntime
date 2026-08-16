#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
HARNESS_DIR="${SCRIPT_DIR:h}"
LOG_PATH="/private/tmp/qwen38-mtp-benchmark.log"

cd "$HARNESS_DIR"
echo "Qwen3.8 native-MTP smoke benchmark" | tee "$LOG_PATH"
echo "Started: $(date -Iseconds)" | tee -a "$LOG_PATH"

PYTHONUNBUFFERED=1 .venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG_PATH"
import time

from harness.executors.dflash_engine import DFlashEngine
from harness.config import DEFAULT_MLX_MODEL_PATH, DEFAULT_MTP_MODEL_PATH

engine = DFlashEngine(
    target_path=DEFAULT_MLX_MODEL_PATH,
    draft_ref=DEFAULT_MTP_MODEL_PATH,
)

started = time.perf_counter()
output = engine.run_inference(
    prompt=(
        "Write a production Swift 6 actor implementing a bounded asynchronous "
        "channel with send, receive, cancellation, and clean shutdown."
    ),
    language="swift",
    enable_thinking=False,
    max_tokens=256,
    temperature=0.0,
    direct=True,
)
elapsed = time.perf_counter() - started

print("\n=== BENCHMARK RESULT ===")
print(f"model={output.model_name}")
print(f"completion_tokens={output.completion_tokens}")
print(f"generation_tps={output.tokens_per_sec:.2f}")
print(f"wall_seconds_including_load={elapsed:.2f}")
print("response_preview=")
print(output.raw_response[:500])
PY

echo "Finished: $(date -Iseconds)" | tee -a "$LOG_PATH"
echo "Log: $LOG_PATH"
echo
read -k 1 "?Press any key to close..."
