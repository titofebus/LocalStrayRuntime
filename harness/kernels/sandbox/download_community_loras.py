"""Download curated community LoRA adapters for Qwen from Hugging Face."""
import os
import sys
from pathlib import Path

# Create adapters directory on external SSD storage
ADAPTERS_DIR = Path("/Volumes/Studio Storage/LLMs/adapters")
ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

# Curated high-impact adapters for reasoning, logic, and systems coding
CURATED_LORAS = [
    {
        "name": "qwen-limo-reasoning-32b",
        "repo_id": "t83714/qwen2.5-32b-instruct-limo-lora-adapter",
        "domain": "Deep Mathematical & Algorithmic Reasoning (LIMO dataset)",
    },
    {
        "name": "qwen-s1k-reasoning-32b",
        "repo_id": "jnward2/qwen-reasoning-lora",
        "domain": "Chain-of-Thought & Planning (s1K simple-scaling dataset)",
    },
    {
        "name": "qwen-lean-formal-math-32b",
        "repo_id": "Chattso-GPT/DeepSeek-R1-Distill-Qwen-32B-for-lean",
        "domain": "Formal Logic, Proofs & Algorithmic Verification",
    },
    {
        "name": "qwen-coder-reasoning-3b",
        "repo_id": "moos124/qwen-2.5-coder-3b-lora-reasoning",
        "domain": "Code Logic & Drafter Reasoning (3B Drafter Candidate)",
    }
]

def download_adapters():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Run: uv pip install huggingface_hub")
        return

    print("=" * 65)
    print(f" DOWNLOADING CURATED QWEN ADAPTERS TO {ADAPTERS_DIR}")
    print("=" * 65)

    for item in CURATED_LORAS:
        name = item["name"]
        repo_id = item["repo_id"]
        domain = item["domain"]
        dest = ADAPTERS_DIR / name

        print(f"\n📦 [{name}]")
        print(f"   Domain:  {domain}")
        print(f"   Repo:    https://huggingface.co/{repo_id}")
        print(f"   Target:  {dest}")

        if dest.exists() and any(dest.iterdir()):
            print(f"   Status:  Already downloaded.")
            continue

        try:
            print("   Downloading adapter weights from Hugging Face...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(dest),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print("   ✅ Download completed!")
        except Exception as e:
            print(f"   ❌ Download failed: {e}")

    print("\n" + "=" * 65)
    print(" All available community adapters downloaded successfully!")
    print("=" * 65)

if __name__ == "__main__":
    download_adapters()
