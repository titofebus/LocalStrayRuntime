"""Consult DeepSeek: Exhaustive Optimization Levers Beyond Q4 on Apple Silicon."""
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

prompt = """You are a Principal Inference Systems Architect on Apple Silicon (M4 Max / MLX).

CONTEXT:
We have optimized Qwen 3.8-27B (6-bit affine, 20.5 GB) with:
1. Native 1-layer MTP drafter (345 MB) with Fused Metal LM-Head + SIMD Vocab Argmax + Dual RMSNorm.
2. Prefix caching (0.9s TTFT).
3. Peak generation: 35-36 tok/s on syntax/clean code.
4. Sustained generation on deep logic (48.7% acceptance): 22-24 tok/s due to 51.25ms verification pass (20.5 GB @ 400 GB/s).

USER QUESTION:
"Have we run out of options other than Q4?"

TASK:
Provide a rigorous, exhaustive analysis of ALL remaining architectural levers to increase generation speed and acceptance on 6-bit weights WITHOUT converting to Q4.

Evaluate and rank:
1. Speculative Branch Trees (DDTree / Tree-based verification): How drafting 2-3 branches in parallel raises acceptance on branching logic from 48% -> 75% in a single 51ms verification pass.
2. Asynchronous Dual-Stream Pipeline (Overlapping drafter Metal queue with target verification Metal queue so draft time is 100% hidden).
3. KV-Cache Quantization (FP8 / INT8 KV cache): Memory bandwidth savings for 1k-4k context lengths.
4. Dynamic Block Expansion (K=6 on high confidence runs).
5. Mixed-Precision Quantization (Keeping attention at 6-bit / FP16 and quantizing MLP feed-forward to 4-bit, dropping model from 20.5 GB -> 16.5 GB with zero logic degradation).

Provide an honest feasibility ranking, expected speedups, and engineering recommendations.
"""

payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "You are an elite Apple Silicon MLX performance architect. Provide a complete, rigorous breakdown of all remaining non-Q4 levers."},
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.1,
    "max_tokens": 4096,
}

req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
print("[DeepSeek Consultation] Analyzing remaining optimization levers beyond Q4...")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]

output_file = Path(__file__).parent / "deepseek_beyond_q4_levers.md"
output_file.write_text(content)
print(f"[DeepSeek Consultation] Saved analysis to {output_file}")
