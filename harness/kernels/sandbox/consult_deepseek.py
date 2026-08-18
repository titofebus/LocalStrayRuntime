"""DeepSeek Consultation on Qwen 3.8 Optimization Roadmap."""
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

prompt = """You are a Principal GPU Systems Architect specializing in Apple Silicon (M-series unified memory) and LLM inference optimization (MLX / Metal).

CURRENT SYSTEM ARCHITECTURE:
- Platform: Apple Silicon M4 Max (400 GB/s Unified Memory Bandwidth, 36 GB RAM, 32-thread SIMDgroups).
- Target Model: Qwen 3.8-27B quantized to 6-bit affine (group_size 64, weights resident in VRAM: ~20.5 GB).
- Speculative Drafter: Native 1-layer MTP predictor (6-bit, ~345 MB) with DFlash 4-token block verification.
- Runtime: Apple MLX 0.32.0 Python runtime with native Swift 6 / SwiftUI client (QwenPrime.app).
- Recent Kernel Update: We just implemented vectorized Metal kernels (`half4` aligned loads + warp-level `simd_shuffle_down`) for the 248,320-vocab Argmax and Dual-RMSNorm+Concat, measuring ~29-35 tok/s sustained with ~54% draft acceptance.

WE ARE CONSIDERING THE FOLLOWING NEXT STEPS:
1. Fused LM-Head GEMV + Argmax: Fusing matrix-vector multiplication with top-1 reduction so the 248,320-element float16 logits tensor (~500 KB) is NEVER allocated or written to unified memory.
2. Adaptive Speculative Block Sizing: Dynamically adjusting draft block count K in [2, 6] based on recent token entropy or acceptance history.
3. Quantization Shift: 4-bit affine (~14.2 GB) vs current 6-bit affine (~20.5 GB) for the 27B target model.
4. Fused SwiGLU / MLP Activation: In-kernel fused `(x * silu(x)) * y` in the 1-layer drafter.
5. Fused RoPE + QKV Projection in the MTP drafter.

QUESTIONS FOR YOUR ARCHITECTURAL REVIEW:
1. Which optimization offers the HIGHEST return-on-investment (ROI) for real user perceived tokens/sec on an M4 Max? Rank your top 3.
2. What is the theoretical physical ceiling (tok/s) on an M4 Max (400 GB/s) for 27B with speculative decoding, and where are we currently losing efficiency?
3. For your #1 recommended move, provide the exact architectural design or pseudocode / MSL strategy we should use.
"""

payload = {
    "model": "deepseek-chat",
    "messages": [
        {
            "role": "system",
            "content": "You are an elite Apple Silicon GPU / MLX systems architect. Provide direct, highly technical, actionable guidance."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    "temperature": 0.2,
    "max_tokens": 4096,
}

req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
print("[DeepSeek Consultation] Consulting DeepSeek architecture advisor...")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]

output_file = Path(__file__).parent / "deepseek_advice.md"
output_file.write_text(content)
print(f"[DeepSeek Consultation] Advice received and saved to {output_file}")
