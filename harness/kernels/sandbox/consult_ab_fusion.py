"""Consult DeepSeek: Modular Architecture for FP8 KV-Cache + Fused QKV RoPE + Dynamic K=6."""
import os
import json
import urllib.request
from pathlib import Path

# Load DEEPSEEK_API_KEY from ~/.agent/.deepseek_env
if "DEEPSEEK_API_KEY" not in os.environ:
    env_file = Path.home() / ".agent" / ".deepseek_env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("export DEEPSEEK_API_KEY="):
                os.environ["DEEPSEEK_API_KEY"] = line.split("=", 1)[1].strip('"\'')

api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("DEEPSEEK_API_KEY not found in environment or ~/.agent/.deepseek_env")

url = "https://api.deepseek.com/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}

prompt = """You are an elite Apple Silicon Metal GPU Kernel Engineer and MLX Core Contributor.

TASK:
Design and implement a modular acceleration package for Qwen 3.8 on Apple Silicon M4 Max:

1. **Module 1: Native FP8 (E4M3) KV-Cache for MLX:**
   - Implement `FP8KVCache` in MLX Python with on-the-fly per-head scaling.
   - Halves memory bandwidth traffic on long context reads.
   - Includes standalone unit test verifying exact numerical alignment with standard FP16 KV-cache.

2. **Module 2: Fused Metal QKV + RoPE Unified Kernel:**
   - Write a Metal Shading Language (MSL) kernel `fused_qkv_gemv_rope` that takes input hidden states and QKV quantized weights, performs GEMV in threadgroup SRAM, and applies Rotary Position Embedding (RoPE) in-register before writing Q, K, V.
   - Provide Python MLX bindings (`mx.fast.metal_kernel` or C++/Metal bridge).

3. **Module 3: Dynamic Speculative Horizon Gating (K=6 Expansion):**
   - Implements dynamic expansion from K=4 -> K=6 when rolling draft entropy is low (<0.8), and contracts to K=2 when entropy is high (>1.8).

Provide clean, modular, fully executable Python and Metal code with zero placeholder code.
"""

payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "You are the world's leading Apple Silicon GPU kernel engineer. Provide complete, modular, production-ready code."},
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.1,
    "max_tokens": 4096,
}

req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
print("[DeepSeek Consultation] Generating Modular FP8 KV Cache + Fused QKV RoPE + Dynamic K=6 package...")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]

output_file = Path(__file__).parent / "deepseek_ab_modular_package.md"
output_file.write_text(content)
print(f"[DeepSeek Consultation] Saved modular package design to {output_file}")
