"""Verify unified Polyglot Reasoning adapter loading in MLX."""
from pathlib import Path
import mlx.core as mx

adapter_path = Path("/Volumes/Studio Storage/LLMs/adapters/polyglot-reasoning-merged")
weights_path = adapter_path / "adapter_model.safetensors"
cfg_path = adapter_path / "adapter_config.json"

assert weights_path.exists(), f"Missing {weights_path}"
assert cfg_path.exists(), f"Missing {cfg_path}"

weights = mx.load(str(weights_path))
print(f"Loaded {len(weights)} LoRA parameter matrices from {weights_path}")
print(f"Total file size: {weights_path.stat().st_size / (1024*1024):.2f} MB")

sample_key = list(weights.keys())[0]
print(f"Sample layer: {sample_key}")
print(f"Sample shape: {weights[sample_key].shape}, dtype: {weights[sample_key].dtype}")

print("\n[SUCCESS] Unified Polyglot Reasoning Adapter verified perfectly!")
