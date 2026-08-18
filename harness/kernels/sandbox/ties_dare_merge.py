"""TIES-DARE Mathematical Fusion of Multi-Rank LoRA Adapters for Qwen."""
import json
import re
import time
from pathlib import Path
import mlx.core as mx
import numpy as np

def load_lora_deltas(adapter_path: Path) -> dict[str, np.ndarray]:
    """Compute full delta matrices Delta_W = (alpha/r) * (B @ A) for each layer."""
    cfg = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    r = int(cfg.get("r", 8))
    alpha = float(cfg.get("lora_alpha", 16))
    scale = alpha / r

    weights = {}
    for f in adapter_path.glob("*.safetensors"):
        loaded = mx.load(str(f))
        for k, v in loaded.items():
            weights[k] = np.array(v.astype(mx.float32))

    modules = set()
    for k in weights.keys():
        m = re.sub(r"\.lora_[AB](\.weight)?$", "", k)
        modules.add(m)

    deltas = {}
    for m in modules:
        a_key = f"{m}.lora_A.weight" if f"{m}.lora_A.weight" in weights else f"{m}.lora_A"
        b_key = f"{m}.lora_B.weight" if f"{m}.lora_B.weight" in weights else f"{m}.lora_B"
        if a_key in weights and b_key in weights:
            A = weights[a_key]
            B = weights[b_key]
            delta = scale * (B @ A)
            deltas[m] = delta

    return deltas

def ties_dare_merge_adapters(
    adapter_paths: list[str | Path],
    output_path: str | Path,
    target_rank: int = 8,
    sparsity: float = 0.7,
    scale_factor: float = 0.8,
) -> Path:
    t0 = time.perf_counter()
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" TIES-DARE MULTI-RANK LO-RA ADAPTER FUSION")
    print(f" Adapters:     {[Path(p).name for p in adapter_paths]}")
    print(f" Target Rank:  {target_rank}")
    print(f" Sparsity:     {sparsity:.0%}")
    print(f" Output Path:  {out_dir}")
    print("=" * 65)

    print("\n1. Computing full rank Delta_W matrices for each adapter...")
    adapter_deltas = [load_lora_deltas(Path(p)) for p in adapter_paths]

    all_keys = set(adapter_deltas[0].keys())
    for ad in adapter_deltas[1:]:
        all_keys = all_keys.intersection(ad.keys())
    print(f"   Found {len(all_keys)} common linear modules.")

    merged_safetensors = {}
    print("\n2. Executing DARE Drop-and-Rescale + TIES Sign Election...")

    for idx, key in enumerate(sorted(all_keys)):
        # Stack full deltas: [N, D_out, D_in]
        stacked = np.stack([ad[key] for ad in adapter_deltas], axis=0)

        # 1. DARE Sparsity
        mask = (np.random.uniform(size=stacked.shape) > sparsity).astype(np.float32)
        dare_deltas = stacked * mask / (1.0 - sparsity)

        # 2. TIES Sign Election
        pos_mass = np.sum(np.maximum(dare_deltas, 0), axis=0)
        neg_mass = np.sum(np.maximum(-dare_deltas, 0), axis=0)
        elected_sign = np.where(pos_mass >= neg_mass, 1.0, -1.0)

        # 3. Disjoint Merge
        sign_matches = (np.sign(dare_deltas) == elected_sign[None, ...]).astype(np.float32)
        agreement_count = np.maximum(np.sum(sign_matches, axis=0), 1.0)
        merged_delta = (np.sum(dare_deltas * sign_matches, axis=0) / agreement_count) * scale_factor

        # 4. Truncated SVD via NumPy
        U, S, Vt = np.linalg.svd(merged_delta, full_matrices=False)
        U_r = U[:, :target_rank]
        S_r = S[:target_rank]
        Vt_r = Vt[:target_rank, :]

        sqrt_S = np.sqrt(S_r)
        B_r = U_r * sqrt_S[None, :]
        A_r = sqrt_S[:, None] * Vt_r

        merged_safetensors[f"{key}.lora_A.weight"] = mx.array(A_r, dtype=mx.bfloat16)
        merged_safetensors[f"{key}.lora_B.weight"] = mx.array(B_r, dtype=mx.bfloat16)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(all_keys):
            print(f"   Processed {idx + 1}/{len(all_keys)} layers...")

    print("\n3. Saving unified Polyglot Reasoning adapter...")
    mx.save_safetensors(str(out_dir / "adapter_model.safetensors"), merged_safetensors)

    # Write unified adapter_config.json
    cfg = {
        "base_model_name_or_path": "Qwen/Qwen3.8-27B",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": target_rank,
        "lora_alpha": target_rank * 2,
        "lora_dropout": 0.0,
        "fusion_method": "TIES-DARE-SVD",
        "merged_sources": [str(p) for p in adapter_paths],
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    t_total = time.perf_counter() - t0
    print(f"\n✅ Created Unified Polyglot Reasoning Adapter in {t_total:.2f}s!")
    print(f"   Saved to: {out_dir / 'adapter_model.safetensors'}")
    return out_dir

if __name__ == "__main__":
    adapter_1 = "/Volumes/Studio Storage/LLMs/adapters/qwen-limo-reasoning-32b"
    adapter_2 = "/Volumes/Studio Storage/LLMs/adapters/qwen-s1k-reasoning-32b"
    out_path = "/Volumes/Studio Storage/LLMs/adapters/polyglot-reasoning-merged"
    ties_dare_merge_adapters([adapter_1, adapter_2], out_path, target_rank=8)
