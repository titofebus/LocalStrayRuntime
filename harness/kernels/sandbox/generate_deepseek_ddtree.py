"""DeepSeek Implementation: DDTree (Speculative Branch Trees) for Qwen on MLX."""
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

prompt = """You are an elite Apple Silicon MLX GPU Kernel Engineer and Speculative Decoding Architect.

TASK:
Implement a high-performance **Speculative Branch Tree (DDTree / Medusa-style tree decoding)** module for Qwen 3.8-27B + Native MTP drafter in MLX.

REQUIREMENTS:
1. `TreeCandidateGenerator`:
   - Takes staged initial token and target hidden state.
   - At high-entropy divergence points, branches into 3 candidate paths (Depth 3, total 6-8 candidate tokens).
   - Generates the packed tree token tensor and the corresponding 2D Tree Attention Mask matrix (boolean / additive mask) so all candidates can be verified simultaneously in ONE forward pass.
2. `TreeVerifier`:
   - Executes single target model forward pass over the packed tree.
   - Evaluates all paths in the tree in parallel.
   - Selects the longest accepted prefix branch.
   - Returns the accepted token array and the exact length accepted.
3. `TreeKVCacheManager`:
   - Prunes the non-accepted branches from the KV cache in <0.01ms (pointer / index slicing in unified memory).
4. `benchmark_ddtree.py`:
   - Standalone test comparing single-path linear speculation vs 3-branch DDTree on branching logic tokens.

Write production-grade, clean, fully executable MLX Python code with zero placeholder code.
"""

payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "You are a world-class MLX engineer. Provide complete, executable, production-ready Python code."},
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.1,
    "max_tokens": 4096,
}

req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
print("[DeepSeek Consultation] Generating DDTree Speculative Branch Tree implementation...")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]

output_file = Path(__file__).parent / "ddtree_speculation_raw.py"
output_file.write_text(content)
print(f"[DeepSeek Consultation] Saved DDTree implementation to {output_file}")
