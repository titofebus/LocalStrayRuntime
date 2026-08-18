"""Verify DenseMiniDrafter loading real Qwen2.5-Coder-0.5B model from SSD."""
import time
from pathlib import Path
import mlx.core as mx
from harness.kernels.sandbox.modular_drafters import DenseMiniDrafter

model_path = "/Volumes/Studio Storage/LLMs/Qwen2.5-Coder-0.5B-Instruct-MLX-4bit"
assert Path(model_path).exists(), f"Model path {model_path} not found!"

print(f"Loading DenseMiniDrafter from {model_path}...")
t0 = time.perf_counter()
drafter = DenseMiniDrafter(model_path=model_path)
t_load = time.perf_counter() - t0
print(f"Loaded DenseMiniDrafter in {t_load:.2f}s (is_loaded={drafter.is_loaded})")

# Test drafting 4 tokens from a prompt token
staged_first = mx.array([12345], dtype=mx.uint32)
drafted_tokens = drafter.draft_block(
    target_model=None,
    target_ops=None,
    staged_first=staged_first,
    target_hidden=None,
    draft_count=4,
)
mx.eval(drafted_tokens)

print(f"Generated 4 draft tokens: {drafted_tokens.tolist()} in {drafter.last_draft_time_ms:.2f} ms")
print("[SUCCESS] DenseMiniDrafter with Qwen2.5-Coder-0.5B verified successfully!")
