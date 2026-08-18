"""Direct API caller for DeepSeek v4 / Reasoner to bypass agent tool-loop limits."""
import os
import json
import urllib.request
from pathlib import Path

# Load DEEPSEEK_API_KEY from ~/.agent/.deepseek_env if not in os.environ
if "DEEPSEEK_API_KEY" not in os.environ:
    env_file = Path.home() / ".agent" / ".deepseek_env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("export DEEPSEEK_API_KEY="):
                os.environ["DEEPSEEK_API_KEY"] = line.split("=", 1)[1].strip('"\'')

api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("DEEPSEEK_API_KEY not found in environment or ~/.agent/.deepseek_env")

prompt_file = Path(__file__).parent / "deepseek_prompt.txt"
prompt_text = prompt_file.read_text()

url = "https://api.deepseek.com/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}

payload = {
    "model": "deepseek-chat",
    "messages": [
        {
            "role": "system",
            "content": "You are a world-class Apple Silicon Metal Shading Language (MSL) and MLX kernel optimization engineer. Provide clean, working, fully vectorized Python/MSL code."
        },
        {
            "role": "user",
            "content": prompt_text
        }
    ],
    "temperature": 0.2,
    "max_tokens": 4096,
}

req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
print("[DeepSeek Direct] Sending request to DeepSeek API...")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    
output_file = Path(__file__).parent / "deepseek_result.py"
output_file.write_text(content)
print(f"[DeepSeek Direct] Response successfully written to {output_file}")
