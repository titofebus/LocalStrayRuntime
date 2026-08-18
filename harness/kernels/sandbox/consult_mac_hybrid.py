"""Consult DeepSeek: Mac-Specific / Apple Silicon Optimized Hybrid Quantization Paths."""
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

prompt = """You are a Principal Apple Silicon Kernel Architect and Apple MLX Framework Contributor.

CONTEXT:
We are implementing a Hybrid Quantization architecture for Qwen 3.8-27B on Apple Silicon (M4 Max, 400 GB/s Unified Memory).
The user asks:
"i know the hybrid path has a lot of different options in how to approach... any mac specific paths that would be best when we do this?"

TASK:
Provide an authoritative guide on Apple Silicon / macOS / MLX specific optimizations for Hybrid Quantization:

1. **Hardware-Level Apple Silicon Primitives (M4 Architecture):**
   - How Apple M4's Unified Memory Architecture (UMA) and Metal SIMDgroup matrix instructions handle mixed-bitwidth layers.
   - Why mixed 8-bit / 4-bit layers execute with ZERO memory alignment penalty on Apple Silicon (unlike CUDA warp bank conflicts).
   - Group size selection: Why `group_size=64` is the optimal Sweet Spot for Metal cachelines (128-byte cachelines / threadgroup memory).

2. **MLX Native Implementation Architecture:**
   - How to construct the layer-wise quantization map in MLX (`nn.QuantizedLinear` with per-module `bits` and `group_size`).
   - Attention projections (`self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj`) -> 8-bit (or 6-bit), group_size=64.
   - MLP projections (`mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`) -> 4-bit, group_size=64.
   - Embeddings & LM-Head -> 8-bit or unquantized float16 for maximum vocab precision.

3. **Step-by-Step Conversion & Fusion Pipeline:**
   - The fastest, lowest-memory streaming conversion script to produce the hybrid safetensors directly on external SSD without swapping RAM.
   - How to ensure 100% compatibility with DFlash speculative decoding and MTP drafter.

Provide clean, production-grade MLX Python code examples.
"""

payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "You are the world's leading expert on Apple Silicon MLX GPU kernel optimization."},
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.1,
    "max_tokens": 4096,
}

req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
print("[DeepSeek Consultation] Analyzing Mac-specific hybrid quantization architecture...")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]

output_file = Path(__file__).parent / "deepseek_mac_hybrid_paths.md"
output_file.write_text(content)
print(f"[DeepSeek Consultation] Saved analysis to {output_file}")
