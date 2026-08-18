"""Test attaching MLX LoRA adapter to Qwen 3.8-27B model."""
import time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner.utils import load_adapters

model_path = "/Volumes/Studio Storage/LLMs/Qwen__Qwen3.8-27B-MLX-6bit"
adapter_path = "/Volumes/Studio Storage/LLMs/adapters/polyglot-reasoning-mlx"

print(f"Loading base model from {model_path}...")
t0 = time.perf_counter()
model, tokenizer = load(model_path)
print(f"Loaded base model in {time.perf_counter() - t0:.2f}s")

print(f"Attaching Polyglot Reasoning LoRA from {adapter_path}...")
t0 = time.perf_counter()
model = load_adapters(model, adapter_path)
print(f"Attached LoRA adapter in {time.perf_counter() - t0:.2f}s")

# Test generation with prompt
prompt = "<|im_start|>user\nWhat is the integral of x*exp(x) dx?<|im_end|>\n<|im_start|>assistant\n"
tokens = tokenizer.encode(prompt)
print("Generating response with active LoRA adapter...")
from mlx_lm import generate
response = generate(model, tokenizer, prompt=prompt, max_tokens=64)
print("\nResponse from Adapted Model:")
print("-" * 50)
print(response)
print("-" * 50)
print("\n[SUCCESS] Polyglot Reasoning Adapter attached and generated successfully!")
