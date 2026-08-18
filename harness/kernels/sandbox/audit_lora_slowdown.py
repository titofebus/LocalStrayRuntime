"""DeepSeek Consultation: Audit LoRA Inference Slowdown on Apple Silicon."""
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

prompt = """You are a Principal GPU Kernel & Inference Optimization Architect for Apple Silicon / MLX.

AUDIT REPORT & PERFORMANCE PROBLEM:
We attached a 32MB LoRA adapter (rank=8 across 64 layers / 256 projection matrices) to Qwen 3.8-27B (6-bit affine quantized) using `mlx_lm.tuner.utils.load_adapters` (dynamic `LoRALinear` layers).

THE SYMPTOM:
Inference generation speed noticeably dropped:
- Baseline generation speed before LoRA: ~29-30 tok/s.
- Speed after loading dynamic `LoRALinear` layers: ~20-21 tok/s (a 30% latency penalty!).
- Prefill time also increased significantly.

AUDIT QUESTIONS:
1. ROOT CAUSE: Why does dynamic `LoRALinear` (`y = W(x) + scale * (x @ A.T @ B.T)`) cause such a severe slowdown on Apple Silicon GPU (Metal kernel launch overhead, dispatch serialization, memory bandwidth)?
2. FUSED WEIGHTS VS DYNAMIC LORA: If we fuse the LoRA weights directly into the base weights (weight matrix fusion: W_new = W + Delta_W), does it completely eliminate the 30% latency penalty and restore 100% native kernel speed?
3. FUSION SCRIPT FOR QUANTIZED MODELS: How to correctly fuse a LoRA adapter into 6-bit quantized weights in MLX or float16 scales so that inference has 0 extra kernel launches and zero speed penalty?

Provide a clear audit diagnosis and exact Python/MLX code to fuse the adapter.
"""

payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "You are an elite Apple Silicon GPU performance auditor. Provide exact profiling root-cause analysis and the permanent fusion fix."},
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.1,
    "max_tokens": 4096,
}

req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
print("[DeepSeek Audit] Consulting DeepSeek on LoRA latency audit...")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]

output_file = Path(__file__).parent / "deepseek_lora_slowdown_audit.md"
output_file.write_text(content)
print(f"[DeepSeek Audit] Saved audit to {output_file}")
