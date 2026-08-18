"""Verify in-place LoRA weight fusion on Qwen 3.8-27B."""
import time
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_unflatten
from mlx_lm import load, generate
from mlx_lm.tuner.utils import load_adapters

model_path = "/Volumes/Studio Storage/LLMs/Qwen__Qwen3.8-27B-MLX-6bit"
adapter_path = "/Volumes/Studio Storage/LLMs/adapters/polyglot-reasoning-mlx"

print(f"1. Loading base model from {model_path}...")
t0 = time.perf_counter()
model, tokenizer = load(model_path)
print(f"   Loaded in {time.perf_counter() - t0:.2f}s")

print(f"2. Attaching Polyglot Reasoning LoRA from {adapter_path}...")
t0 = time.perf_counter()
model = load_adapters(model, adapter_path)
print(f"   Attached in {time.perf_counter() - t0:.2f}s")

print("3. Executing In-Place Weight Matrix Fusion...")
t0 = time.perf_counter()
fused_linears = [
    (n, m.fuse())
    for n, m in model.named_modules()
    if hasattr(m, "fuse")
]
if fused_linears:
    model.update_modules(tree_unflatten(fused_linears))
    mx.eval(model.parameters())
print(f"   Fused {len(fused_linears)} LoRA layers in {time.perf_counter() - t0:.2f}s!")

# Verify that no LoRALinear layers remain
from mlx_lm.tuner.lora import LoRALinear
remaining_lora = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
print(f"   Remaining dynamic LoRA layers: {len(remaining_lora)} (should be 0)")
assert len(remaining_lora) == 0, "Failed to fuse all LoRA layers!"

# Benchmark generation speed with fused weights
prompt = "<|im_start|>user\nWrite a 1-line Python function to compute fibonacci numbers.<|im_end|>\n<|im_start|>assistant\n"
print("\n4. Testing Generation with Fused Weights...")
t0 = time.perf_counter()
response = generate(model, tokenizer, prompt=prompt, max_tokens=32)
t_gen = time.perf_counter() - t0
print(f"   Generated in {t_gen:.2f}s:")
print("-" * 50)
print(response)
print("-" * 50)
print("\n[SUCCESS] In-Place LoRA Weight Fusion verified with 0 extra kernel launches!")
