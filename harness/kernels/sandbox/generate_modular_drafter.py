"""DeepSeek CodeGen: Modular Pluggable Drafter Architecture for Qwen Prime."""
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

prompt = """You are a Principal Software Architect for Apple Silicon / MLX Speculative Inference.

OBJECTIVE:
Build a clean, production-grade Modular Speculative Drafter Architecture in Python / MLX for Qwen Prime.

REQUIREMENTS:
1. `BaseSpeculativeDrafter` (Abstract Protocol / Class):
   - `def load(self, model_path: str, **kwargs) -> None`
   - `def draft_block(self, target_model, target_ops, staged_first, target_hidden, draft_count, **kwargs) -> mx.array`
   - `def reset_cache(self) -> None`

2. Implement Three Pluggable Drafter Backends:
   - `NativeMTPDrafter`: Uses the 1-layer MTP head with our fused Metal kernels (`fused_dual_rmsnorm_concat`, `fused_lm_head_argmax`). Ultra-low latency (~0.2ms/token).
   - `DenseMiniDrafter`: Uses a 0.5B dense coding model (e.g. `Qwen2.5-Coder-0.5B-Instruct-MLX-4bit`). Deep 24-layer reasoning for complex logic branches.
   - `HybridAdaptiveDrafter`: Routes to MTP for fast boilerplate syntax and swaps to 0.5B Dense drafter when token entropy / logic complexity spikes.

3. Implement `ModularSpeculativeEngine`:
   - Allows hot-swapping the active drafter backend at runtime via `engine.set_drafter(backend_name_or_instance)` with sub-millisecond switching and zero reload of the 20.5GB target model.

4. Include clean unit test demonstrating loading, swapping between MTP and Dense drafter, and generating a test block.

Deliver the complete, working implementation in `modular_drafters.py`.
"""

payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "You are an elite MLX & Python systems architect. Provide complete, fully working, clean modular code."},
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.1,
    "max_tokens": 4096,
}

req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
print("[DeepSeek CodeGen] Generating Modular Drafter Architecture...")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]

output_file = Path(__file__).parent / "deepseek_modular_drafter_raw.py"
output_file.write_text(content)
print(f"[DeepSeek CodeGen] Saved generated code to {output_file}")
