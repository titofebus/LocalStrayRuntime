"""DeepSeek Consultation: LoRA Stacking, Merging, and Language Bundles on Apple Silicon."""
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

prompt = """You are an elite LLM Fine-Tuning & Multi-LoRA Systems Architect on Apple Silicon / MLX.

The user is asking:
"while wrap up testing, do research on the loras we want. and if they're that small... could we stack them? including basically all languages in bundles or a bundle?"

TARGET DOMAINS FOR QWEN PRIME:
- Swift 6 (Strict Concurrency, Actor isolation, Transferable, async/await)
- Rust 2021 (no_std, lock-free atomics, zero-copy parsing, borrow checker correctness)
- Go 1.22 (Channels, Worker Pools, context cancellation, goroutine backpressure)
- Python 3.12 (Typed, AsyncIO, PyDantic v2, high-perf data pipelines)
- TypeScript 5 / Modern Web (Strict typing, state machines, React/Next architecture)
- C++23 (Concepts, std::ranges, coroutines, memory safety)

QUESTIONS FOR YOUR ARCHITECTURAL BLUEPRINT:
1. CAN WE STACK / BUNDLE THEM?
   Compare the 3 stacking architectures:
   - A. Static Fusion / Merging (TIES / DARE / Linear Soup into 1 Unified Polyglot LoRA or direct base fusion via `mlx_lm.fuse` with 0 MB extra RAM).
   - B. Dynamic Multi-LoRA Serving (MOLA / S-LoRA in-memory pointer routing where 8 language LoRAs = ~130MB total).
   - C. Drafter-Specific Multi-LoRA (Applying language LoRAs to the speculative drafter).
2. INTERFERENCE & CATASTROPHIC FORGETTING:
   How do we prevent interference when merging 6+ language LoRAs together? What is the mathematical recipe (e.g. DARE with sparsity 0.7, TIES sign resolution)?
3. HOW TO CREATE / TRAIN THESE LORAS LOCALLY ON APPLE SILICON:
   - Dataset sources (synthetic high-quality pairs, The Stack v2 filtered, Evol-Instruct-Code).
   - Recommended rank, alpha, and target modules (`q_proj`, `v_proj`, `gate_proj`, `up_proj`, `down_proj`).
   - Training time on M4 Max using MLX LoRA (`mlx_lm.lora.train`).
"""

payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "You are a world-class MLX fine-tuning and LoRA merging architect. Provide deep, rigorous, actionable guidance."},
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.2,
    "max_tokens": 4096,
}

req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
print("[DeepSeek Consultation] Researching LoRA Stacking, Merging, and Language Bundles...")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]

output_file = Path(__file__).parent / "deepseek_lora_stacking_research.md"
output_file.write_text(content)
print(f"[DeepSeek Consultation] Saved research to {output_file}")
