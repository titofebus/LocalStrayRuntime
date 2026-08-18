"""DeepSeek Consultation: Best Path for LoRA Integration on Qwen Prime."""
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

prompt = """You are a Principal Inference Systems Architect on Apple Silicon / MLX.

CURRENT ENVIRONMENT:
- Platform: Apple Silicon M4 Max (400 GB/s unified memory bandwidth, 36GB RAM).
- Base Target Model: Qwen 3.8-27B (6-bit affine, 20.5 GB).
- Speculative Drafter: Native MTP (1-layer, 345 MB) with custom MSL kernels (Fused Dual RMSNorm + Fused LM-Head + SIMD Vocab Argmax).
- Downloaded LoRA Adapters ready on disk (`/Volumes/Studio Storage/LLMs/adapters/`):
  1. `qwen-limo-reasoning-32b` (LIMO mathematical & algorithmic reasoning)
  2. `qwen-s1k-reasoning-32b` (s1K chain-of-thought & planning)
  3. `qwen-lean-formal-math-32b` (Formal verification & proofs)
  4. `qwen-coder-reasoning-3b` (Code logic & reasoning)

THE QUESTION:
Which path is the superior architectural choice for Qwen Prime right now?
- Path A: Dynamic Multi-LoRA Hot-Swapping (In-memory adapter routing based on reasoning vs direct mode).
- Path B: TIES-DARE Merging & Static Fusion (Mathematically merging s1K + LIMO reasoning adapters into a single Polyglot Reasoning Adapter fused directly into weights with 0 MB extra RAM).
- Path C: Hybrid Dual-Track (Static fusion for base model reasoning + Dynamic drafter routing).

PROVIDE:
1. The definitive recommended path with architectural justification on M4 Max.
2. Step-by-step implementation code / script in MLX to execute the chosen path immediately.
"""

payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "You are an elite MLX / GPU systems architect. Provide the definitive best path and concrete execution code."},
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.1,
    "max_tokens": 4096,
}

req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
print("[DeepSeek Consultation] Consulting DeepSeek on the optimal integration path...")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]

output_file = Path(__file__).parent / "deepseek_best_path_decision.md"
output_file.write_text(content)
print(f"[DeepSeek Consultation] Saved decision to {output_file}")
