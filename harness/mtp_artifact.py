"""Metadata and source validation for Qwen3.8's native MTP head."""

from __future__ import annotations

import json
from pathlib import Path


MTP_WEIGHT_NAMES = (
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
)

MTP_NORM_WEIGHT_NAMES = frozenset(
    {
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    }
)


def convert_mtp_weight_for_mlx(name: str, weight):
    return weight + 1.0 if name in MTP_NORM_WEIGHT_NAMES else weight


def locate_mtp_weights(index_path: str | Path) -> dict[str, str]:
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    missing = [name for name in MTP_WEIGHT_NAMES if name not in weight_map]
    if missing:
        raise ValueError(
            "Source is missing Qwen3.8 MTP tensors: " + ", ".join(missing)
        )
    return {name: str(weight_map[name]) for name in MTP_WEIGHT_NAMES}


def build_mtp_artifact_config(*, source_revision: str) -> dict:
    return {
        "architectures": ["Qwen3_8MTPModel"],
        "model_type": "qwen3_8_mtp",
        "source_model_id": "Qwen/Qwen3.8-27B",
        "source_revision": source_revision,
        "target_model_id": "Qwen/Qwen3.8-27B",
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 1,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "max_position_embeddings": 262144,
        "partial_rotary_factor": 0.25,
        "rope_theta": 10_000_000,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "vocab_size": 248320,
        "block_size": 8,
        "target_layer_ids": [63],
        "norm_weight_offset": 1.0,
        "quantization": {"group_size": 64, "bits": 6, "mode": "affine"},
    }
