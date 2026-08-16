#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
HARNESS_DIR="${SCRIPT_DIR:h}"
LOG_PATH="/private/tmp/qwen38-mtp-block-sweep.log"

cd "$HARNESS_DIR"
echo "Qwen3.8 native-MTP block-size sweep" | tee "$LOG_PATH"
echo "Started: $(date -Iseconds)" | tee -a "$LOG_PATH"

PYTHONUNBUFFERED=1 .venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG_PATH"
import gc
import time

import mlx.core as mx

from dflash_mlx.generate import (
    SummaryEvent,
    generation_tps_from_summary,
    stream_dflash_generate,
)
from harness.executors.dflash_engine import DFlashEngine
from harness.config import DEFAULT_MLX_MODEL_PATH, DEFAULT_MTP_MODEL_PATH


PROMPT = (
    "<|im_start|>user\n"
    "Write a production Swift 6 actor implementing a bounded asynchronous "
    "channel with send, receive, cancellation, and clean shutdown."
    "<|im_end|>\n<|im_start|>assistant\n"
)

engine = DFlashEngine(
    target_path=DEFAULT_MLX_MODEL_PATH,
    draft_ref=DEFAULT_MTP_MODEL_PATH,
)
engine._ensure_loaded()
bundle = engine._bundle
runtime_context = engine._runtime_context


def generate(block_tokens: int, max_new_tokens: int):
    summary = None
    started = time.perf_counter()
    stream = stream_dflash_generate(
        target_model=bundle.target_model,
        target_ops=bundle.target_ops,
        tokenizer=bundle.tokenizer,
        draft_model=bundle.draft_model,
        draft_backend=bundle.draft_backend,
        prompt=PROMPT,
        max_new_tokens=max_new_tokens,
        block_tokens=block_tokens,
        stop_token_ids=[],
        runtime_context=runtime_context,
    )
    try:
        for event in stream:
            if isinstance(event, SummaryEvent):
                summary = event
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    if summary is None:
        raise RuntimeError("DFlash did not emit a summary")
    return summary, time.perf_counter() - started


print("\nWarming and measuring each block size...")
for block_tokens in (1, 2, 4, 8):
    generate(block_tokens, 16)
    summary, wall_seconds = generate(block_tokens, 256)
    output_per_cycle = summary.generation_tokens / max(summary.cycles_completed, 1)
    print(
        f"BLOCK_RESULT block={block_tokens} "
        f"tps={generation_tps_from_summary(summary):.2f} "
        f"generated={summary.generation_tokens} "
        f"accepted={summary.accepted_from_draft} "
        f"acceptance={summary.acceptance_ratio:.1%} "
        f"cycles={summary.cycles_completed} "
        f"output_per_cycle={output_per_cycle:.3f} "
        f"wall_seconds={wall_seconds:.2f}"
    )
    gc.collect()
    mx.clear_cache()
PY

echo "Finished: $(date -Iseconds)" | tee -a "$LOG_PATH"
echo "Log: $LOG_PATH"
echo
read -k 1 "?Press any key to close..."
