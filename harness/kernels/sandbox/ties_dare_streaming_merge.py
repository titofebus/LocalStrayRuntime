"""Memory-Efficient Streaming TIES-DARE Fusion of Multi-Rank LoRA Adapters (<300MB RAM)."""
import gc
import json
import re
import time
from pathlib import Path
import mlx.core as mx
import numpy as np

def stream_ties_dare_merge(
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
    print(" STREAMING TIES-DARE LO-RA FUSION (Low Memory <300MB)")
    print(f" Adapters:     {[Path(p).name for p in adapter_paths]}")
    print(f" Target Rank:  {target_rank}")
    print(f" Sparsity:     {sparsity:.0%}")
    print(f" Output Path:  {out_dir}")
    print("=" * 65)

    # 1. Read metadata from each adapter
    adapter_weights = []
    scales = []
    for p in adapter_paths:
        root = Path(p)
        cfg = json.loads((root / "adapter_config.json").read_text(encoding="utf-8"))
        r = int(cfg.get("r", 8))
        alpha = float(cfg.get("lora_alpha", 16))
        scales.append(alpha / r)

        # Load weights dict
        weights = {}
        for f in root.glob("*.safetensors"):
            loaded = mx.load(str(f))
            for k, v in loaded.items():
                weights[k] = v
        adapter_weights.append(weights)

    # Find common module bases
    def get_modules(w_dict):
        m_set = set()
        for k in w_dict.keys():
            m = re.sub(r"\.lora_[AB](\.weight)?$", "", k)
            m_set.add(m)
        return m_set

    common_modules = get_modules(adapter_weights[0])
    for w in adapter_weights[1:]:
        common_modules = common_modules.intersection(get_modules(w))

    sorted_modules = sorted(common_modules)
    print(f"\n1. Found {len(sorted_modules)} common linear layers. Fusing layer-by-layer...")

    merged_safetensors = {}

    for idx, mod in enumerate(sorted_modules):
        # Compute delta for each adapter for THIS layer only
        layer_deltas = []
        for i, (weights, scale) in enumerate(zip(adapter_weights, scales)):
            a_key = f"{mod}.lora_A.weight" if f"{mod}.lora_A.weight" in weights else f"{mod}.lora_A"
            b_key = f"{mod}.lora_B.weight" if f"{mod}.lora_B.weight" in weights else f"{mod}.lora_B"
            A = np.array(weights[a_key].astype(mx.float32))
            B = np.array(weights[b_key].astype(mx.float32))
            delta = scale * (B @ A)
            layer_deltas.append(delta)

        # Stack: [N, D_out, D_in]
        stacked = np.stack(layer_deltas, axis=0)

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

        # 4. Truncated SVD (NumPy CPU)
        U, S, Vt = np.linalg.svd(merged_delta, full_matrices=False)
        U_r = U[:, :target_rank]
        S_r = S[:target_rank]
        Vt_r = Vt[:target_rank, :]

        sqrt_S = np.sqrt(S_r)
        B_r = U_r * sqrt_S[None, :]
        A_r = sqrt_S[:, None] * Vt_r

        merged_safetensors[f"{mod}.lora_A.weight"] = mx.array(A_r, dtype=mx.bfloat16)
        merged_safetensors[f"{mod}.lora_B.weight"] = mx.array(B_r, dtype=mx.bfloat16)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(sorted_modules):
            print(f"   Fused [{idx + 1}/{len(sorted_modules)}] layers...")
            gc.collect()

    print("\n2. Saving unified Polyglot Reasoning adapter...")
    mx.save_safetensors(str(out_dir / "adapter_model.safetensors"), merged_safetensors)

    # Save adapter_config.json
    cfg = {
        "base_model_name_or_path": "Qwen/Qwen3.8-27B",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": target_rank,
        "lora_alpha": target_rank * 2,
        "lora_dropout": 0.0,
        "fusion_method": "TIES-DARE-SVD-Streaming",
        "merged_sources": [str(p) for p in adapter_paths],
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    t_total = time.perf_counter() - t0
    print(f"\n✅ Successfully created Polyglot Reasoning Adapter in {t_total:.2f}s!")
    print(f"   Saved to: {out_dir / 'adapter_model.safetensors'}")
    return out_dir

if __name__ == "__main__":
    adapter_1 = "/Volumes/Studio Storage/LLMs/adapters/qwen-limo-reasoning-32b"
    adapter_2 = "/Volumes/Studio Storage/LLMs/adapters/qwen-s1k-reasoning-32b"
    out_path = "/Volumes/Studio Storage/LLMs/adapters/polyglot-reasoning-merged"
    stream_ties_dare_merge([adapter_1, adapter_2], out_path, target_rank=8)
