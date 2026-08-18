"""DeepSeek Code Generation for Dynamic Entropy-Gated Drafting in MLX."""
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

prompt = """You are an elite MLX & Apple Silicon speculative decoding engineer.

OBJECTIVE:
Implement Dynamic Entropy-Gated Drafting for Qwen 3.8 MTP in Apple MLX.

CONTEXT:
In `Qwen38MTPModel.predict_block`, we currently draft a fixed count of tokens (default draft_count=4).
We want an ultra-fast, zero-overhead function `calculate_dynamic_draft_count(logits: mx.array, min_k: int = 2, max_k: int = 6, default_k: int = 4) -> int` that:
1. Takes the logits (or top logits) from the last verified token of shape (1, 248320) or (1, D).
2. Computes the prediction confidence / entropy (e.g. using top-k softmax entropy, top-1 vs top-2 logit margin, or softmax temperature thresholding) efficiently in MLX without allocating large temporary arrays.
3. Maps confidence to dynamic draft length K in [min_k, max_k]:
   - High confidence (predictable syntax/types): K = 5 or 6.
   - Normal confidence: K = 4.
   - Low confidence / high entropy (branching logic/arithmetic): K = 2.
4. Also implement early-exit in the draft loop: if inside the draft loop a drafted token's intermediate logit entropy is high (confidence drops below a threshold), break early instead of drafting subsequent low-probability tokens.

REQUIREMENTS:
- Clean, drop-in Python function using `mlx.core` (mx).
- Fast and numerically stable.
- Provide the complete code for `dynamic_entropy_drafting.py` with unit test.
"""

payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "You are an elite MLX / GPU systems engineer. Provide clean, fully working code."},
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.1,
    "max_tokens": 4096,
}

req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
print("[DeepSeek CodeGen] Generating Dynamic Entropy-Gated Drafting module...")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]

output_file = Path(__file__).parent / "deepseek_entropy_gated_raw.py"
output_file.write_text(content)
print(f"[DeepSeek CodeGen] Saved generated code to {output_file}")
