"""Consult DeepSeek: Hybrid Quantization (Q6-Q4 vs Q8-Q4) Quality & Speed Audit."""
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

prompt = """You are a Principal AI Researcher and LLM Quantization Specialist specializing in Code Generation & Reasoning models (Qwen 2.5 / Qwen 3.8 / DeepSeek-Coder).

CONTEXT:
We are evaluating quantization strategies for Qwen 3.8-27B on Apple Silicon M4 Max (400 GB/s unified memory).
The user asks:
"is that the way to go or a q8 - q4, and what will the code quality be if we do that?"

TASK:
Provide a comprehensive, authoritative comparison between:
1. Uniform Q6 (Current baseline, 20.5 GB)
2. Hybrid Q6-Attention + Q4-MLP (16.4 GB)
3. Hybrid Q8-Attention + Q4-MLP (18.2 GB)
4. Uniform Q4 (14.2 GB)

Analyze:
A. Mathematical & Architectural Sensitivity: Why Attention (Q, K, V, Out) preserves long-range context, token routing, and syntax structures, while MLP (Gate, Up, Down, 60% of params) acts as high-capacity associative memory that is highly resilient to 4-bit quantization.
B. Code Quality & Benchmark Impact: What happens to:
   - Complex type systems & strict concurrency (Swift 6 actors, Rust lifetimes/borrow checker, C++ templates)
   - Edge case logic, boundary checks, off-by-one errors
   - Formal math / algorithmic reasoning (HumanEval, MultiPL-E, LiveCodeBench)
C. Physical Throughput & Latency on M4 Max:
   - Exact memory bandwidth transfer time per verification pass
   - Steady-state tok/s on complex logic vs syntax
D. Definitive Recommendation: Which hybrid configuration gives the ultimate "sweet spot" of indistinguishable FP16 code quality + maximum generation velocity.
"""

payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "You are a world-class quantization researcher. Provide an exhaustive, evidence-based breakdown."},
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.1,
    "max_tokens": 4096,
}

req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
print("[DeepSeek Consultation] Analyzing Hybrid Quantization (Q6-Q4 vs Q8-Q4) quality & speed...")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]

output_file = Path(__file__).parent / "deepseek_hybrid_quant_audit.md"
output_file.write_text(content)
print(f"[DeepSeek Consultation] Saved analysis to {output_file}")
