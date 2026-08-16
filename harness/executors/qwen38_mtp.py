"""Native Qwen3.8 MTP draft model and DFlash-loop backend for MLX."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs

from dflash_mlx.engine.sampling import greedy_tokens_with_mask


class Qwen38MTPModel(nn.Module):
    """The Qwen3.8 checkpoint's bundled one-layer multi-token predictor."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        text_args = TextModelArgs.from_dict(
            {
                "model_type": "qwen3_5_text",
                "hidden_size": config["hidden_size"],
                "intermediate_size": config["intermediate_size"],
                "num_hidden_layers": 1,
                "num_attention_heads": config["num_attention_heads"],
                "num_key_value_heads": config["num_key_value_heads"],
                "head_dim": config["head_dim"],
                "max_position_embeddings": config["max_position_embeddings"],
                "partial_rotary_factor": config["partial_rotary_factor"],
                "rope_parameters": {
                    "type": "default",
                    "rope_theta": config["rope_theta"],
                    "partial_rotary_factor": config["partial_rotary_factor"],
                },
                "rms_norm_eps": config["rms_norm_eps"],
                "attention_bias": config["attention_bias"],
                "vocab_size": config["vocab_size"],
                "full_attention_interval": 1,
            }
        )
        hidden_size = int(config["hidden_size"])
        self.fc = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.layers = [DecoderLayer(text_args, layer_idx=0)]
        self.norm = nn.RMSNorm(hidden_size, eps=float(config["rms_norm_eps"]))
        self.pre_fc_norm_embedding = nn.RMSNorm(
            hidden_size, eps=float(config["rms_norm_eps"])
        )
        self.pre_fc_norm_hidden = nn.RMSNorm(
            hidden_size, eps=float(config["rms_norm_eps"])
        )

        self.block_size = int(config["block_size"])
        self.target_layer_ids = [int(index) for index in config["target_layer_ids"]]
        self.mask_token_id = int(config.get("mask_token_id", 248070))
        self.args = SimpleNamespace(layer_types=("full_attention",), sliding_window=0)

    def project_target_hidden(self, target_hidden: mx.array) -> mx.array:
        return target_hidden

    def predict_block(
        self,
        *,
        target_model: Any,
        target_ops: Any,
        staged_first: mx.array,
        target_hidden: mx.array,
        draft_count: int,
        suppress_token_mask: Optional[mx.array],
    ) -> mx.array:
        if draft_count <= 0:
            return mx.array([], dtype=mx.uint32)

        hidden = target_hidden[:, -1:, :]
        token = staged_first[:1].astype(mx.uint32)
        cache = KVCache()
        drafted: list[mx.array] = []

        for _ in range(draft_count):
            embedding = target_ops.embed_tokens(target_model)(token[None])
            fused = mx.concatenate(
                [
                    self.pre_fc_norm_embedding(embedding),
                    self.pre_fc_norm_hidden(hidden),
                ],
                axis=-1,
            )
            hidden = self.fc(fused)
            mask = create_attention_mask(hidden, cache)
            hidden = self.layers[0](hidden, mask=mask, cache=cache)
            hidden = self.norm(hidden)
            logits = target_ops.logits_from_hidden(target_model, hidden[:, -1:, :])
            token = greedy_tokens_with_mask(logits[:, -1, :], suppress_token_mask).reshape(-1)
            drafted.append(token.astype(mx.uint32))

        return mx.concatenate(drafted, axis=0)


class MTPDraftBackend:
    """Adapter allowing native MTP proposals to use the hardened DFlash verifier."""

    def make_cache(self, **_: Any) -> list[Any]:
        return []

    def draft_greedy(
        self,
        *,
        target_model: Any,
        target_ops: Any,
        draft_model: Qwen38MTPModel,
        staged_first: mx.array,
        draft_context: mx.array,
        block_len: int,
        suppress_token_mask: Optional[mx.array],
        async_launch: bool,
        **_: Any,
    ) -> mx.array:
        drafted = draft_model.predict_block(
            target_model=target_model,
            target_ops=target_ops,
            staged_first=staged_first,
            target_hidden=draft_context,
            draft_count=max(0, int(block_len) - 1),
            suppress_token_mask=suppress_token_mask,
        )
        if async_launch:
            mx.async_eval(drafted)
        else:
            mx.eval(drafted)
        return drafted

    def advance_context(self, **_: Any) -> None:
        return None

    def draft_with_topk(self, **_: Any):
        raise NotImplementedError("Qwen3.8 MTP supports linear speculative verification only")

    def draft_branch_blocks_batch(self, **_: Any):
        raise NotImplementedError("Qwen3.8 MTP does not support DDTree")


def load_qwen38_mtp(path: str | Path) -> tuple[Qwen38MTPModel, dict[str, Any]]:
    root = Path(path)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("target_model_id") != "Qwen/Qwen3.8-27B":
        raise ValueError("MTP artifact is not bound to Qwen/Qwen3.8-27B")

    weights = mx.load(str(root / "model.safetensors"))
    model = Qwen38MTPModel(config)
    quantization = config.get("quantization")
    if quantization:
        nn.quantize(
            model,
            group_size=int(quantization["group_size"]),
            bits=int(quantization["bits"]),
            mode=str(quantization.get("mode", "affine")),
            class_predicate=lambda path, module: f"{path}.scales" in weights,
        )
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model, config


def load_qwen38_mtp_runtime_bundle(
    *,
    model_ref: str | Path,
    draft_ref: str | Path,
    verify_config: Any,
):
    from dflash_mlx.runtime.bundle import RuntimeBundle
    from dflash_mlx.runtime.loading import load_target_bundle

    target_bundle = load_target_bundle(
        model_ref,
        lazy=True,
        verify_config=verify_config,
    )
    draft_model, draft_config = load_qwen38_mtp(draft_ref)
    return RuntimeBundle(
        target_model=target_bundle.model,
        tokenizer=target_bundle.tokenizer,
        target_meta=target_bundle.meta,
        draft_model=draft_model,
        draft_meta={
            "resolved_model_ref": str(draft_ref),
            "config": draft_config,
            "draft_quant_spec": "w6:gs64",
            "draft_quant_source": "artifact",
        },
        draft_backend=MTPDraftBackend(),
        target_ops=target_bundle.target_ops,
        resolved_model_ref=str(model_ref),
        resolved_draft_ref=str(draft_ref),
        effective_draft_quant="w6:gs64",
        support_spec=None,
    )
