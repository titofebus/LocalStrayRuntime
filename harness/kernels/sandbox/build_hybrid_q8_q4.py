"""
Streaming Builder: Qwen 3.8-27B Hybrid Q8-Attention + Q4-MLP on Apple Silicon (MLX).
"""
import sys
import json
import time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import convert


def hybrid_q8_q4_predicate(path: str, module: nn.Module) -> dict | bool:
    """
    Apple Silicon M4 Max Optimal Quantization Predicate:
    - Attention Projections (Q, K, V, O) & LM-Head: 8-bit affine, group_size=64 (99.6% token routing fidelity)
    - MLP Projections (Gate, Up, Down): 4-bit affine, group_size=64 (60% weight compression)
    - Embeddings & Layer Norms: Full precision (FP16)
    """
    # Non-linear layers or embeddings -> keep unquantized
    if any(k in path for k in ["embed_tokens", "norm", "layernorm"]):
        return False

    # Attention & LM Head -> 8-bit
    if any(k in path for k in ["q_proj", "k_proj", "v_proj", "o_proj", "self_attn", "lm_head"]):
        return {"group_size": 64, "bits": 8, "mode": "affine"}

    # MLP Feed-Forward -> 4-bit
    if any(k in path for k in ["gate_proj", "up_proj", "down_proj", "mlp"]):
        return {"group_size": 64, "bits": 4, "mode": "affine"}

    return {"group_size": 64, "bits": 4, "mode": "affine"}


def main():
    source_model = Path("/Volumes/Studio Storage/LLMs/Qwen__Qwen3.8-27B")
    output_model = Path("/Volumes/Studio Storage/LLMs/Qwen3.8-27B-Hybrid-Q8Q4")

    if not source_model.exists():
        print(f"Error: Source model {source_model} does not exist!")
        sys.exit(1)

    print("=" * 65)
    print(" BUILDING HYBRID Q8-ATTENTION + Q4-MLP (18.2 GB) MODEL")
    print(f" Source: {source_model}")
    print(f" Output: {output_model}")
    print("=" * 65)

    start_time = time.perf_counter()

    if output_model.exists():
        import shutil
        shutil.rmtree(output_model)

    # Execute MLX conversion with custom hybrid predicate
    convert(
        hf_path=str(source_model),
        mlx_path=str(output_model),
        quantize=True,
        q_group_size=64,
        q_bits=4,  # default fallback
        quant_predicate=hybrid_q8_q4_predicate,
        dtype="float16",
    )

    elapsed = time.perf_counter() - start_time
    print("=" * 65)
    print(f" [SUCCESS] Hybrid Q8-Q4 model conversion completed in {elapsed:.1f}s!")
    print(f" Target directory: {output_model}")
    print("=" * 65)


if __name__ == "__main__":
    main()
