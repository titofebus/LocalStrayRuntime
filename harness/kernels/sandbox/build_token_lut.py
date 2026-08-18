"""Build and test 0.5B -> 27B Token ID Mapping LUT."""
import json
import time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load

path_27b = "/Volumes/Studio Storage/LLMs/Qwen__Qwen3.8-27B-MLX-6bit"
path_05b = "/Volumes/Studio Storage/LLMs/Qwen2.5-Coder-0.5B-Instruct-MLX-4bit"

print("Loading tokenizers...")
_, tok_27b = load(path_27b)
_, tok_05b = load(path_05b)

# Build LUT: For each token id in 0.5B, decode to string, encode in 27B
vocab_05b_size = 152064
lut = [0] * vocab_05b_size

print("Building vocabulary translation lookup table...")
t0 = time.perf_counter()
for i in range(vocab_05b_size):
    try:
        text = tok_05b.decode([i])
        encoded = tok_27b.encode(text)
        if len(encoded) == 1:
            lut[i] = encoded[0]
        else:
            lut[i] = i # fallback
    except Exception:
        lut[i] = i

t_build = time.perf_counter() - t0
print(f"Built 152k token translation LUT in {t_build:.2f}s")

# Test translation on code tokens
sample_code = "import Foundation\npublic actor PriorityQueue {\n  let id: UUID\n}"
ids_05b = tok_05b.encode(sample_code)
expected_27b = tok_27b.encode(sample_code)

lut_mx = mx.array(lut, dtype=mx.uint32)
ids_05b_mx = mx.array(ids_05b, dtype=mx.uint32)
translated_27b = lut_mx[ids_05b_mx]
mx.eval(translated_27b)

print("0.5B Raw IDs:     ", ids_05b)
print("Translated to 27B:", translated_27b.tolist())
print("Expected 27B IDs: ", expected_27b)
match_rate = sum(1 for a, b in zip(translated_27b.tolist(), expected_27b) if a == b) / len(expected_27b)
print(f"Token Translation Match Rate: {match_rate:.1%}")
