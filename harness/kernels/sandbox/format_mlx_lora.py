"""Format merged LoRA adapter to native MLX structure."""
import json
import re
from pathlib import Path
import mlx.core as mx

def format_to_mlx(
    source_dir: str | Path,
    output_dir: str | Path,
):
    src = Path(source_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Formatting {src} -> {out} for native MLX loading...")
    raw_weights = mx.load(str(src / "adapter_model.safetensors"))

    mlx_weights = {}
    for k, v in raw_weights.items():
        # Clean prefix: base_model.model.model. -> model.
        new_k = re.sub(r"^base_model\.model\.", "", k)
        # Rename lora_A.weight -> lora_a, lora_B.weight -> lora_b
        new_k = new_k.replace(".lora_A.weight", ".lora_a")
        new_k = new_k.replace(".lora_B.weight", ".lora_b")
        new_k = new_k.replace(".lora_A", ".lora_a")
        new_k = new_k.replace(".lora_B", ".lora_b")
        mlx_weights[new_k] = v

    # Save as adapters.safetensors
    mx.save_safetensors(str(out / "adapters.safetensors"), mlx_weights)

    # Write MLX-compatible adapter_config.json
    mlx_config = {
        "fine_tune_type": "lora",
        "num_layers": 64,
        "lora_parameters": {
            "rank": 8,
            "alpha": 16,
            "scale": 2.0,
            "dropout": 0.0,
            "keys": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
        }
    }
    (out / "adapter_config.json").write_text(json.dumps(mlx_config, indent=2), encoding="utf-8")
    print(f"✅ Formatted {len(mlx_weights)} MLX adapter parameters saved to {out / 'adapters.safetensors'}")

if __name__ == "__main__":
    src_dir = "/Volumes/Studio Storage/LLMs/adapters/polyglot-reasoning-merged"
    out_dir = "/Volumes/Studio Storage/LLMs/adapters/polyglot-reasoning-mlx"
    format_to_mlx(src_dir, out_dir)
