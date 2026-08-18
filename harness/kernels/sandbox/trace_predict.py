"""Trace all 4 steps of predict_block."""
from types import SimpleNamespace
import mlx.core as mx
import mlx.nn as nn
from harness.executors.qwen38_mtp import Qwen38MTPModel
from harness.kernels.sandbox.fused_mtp_ops import fused_dual_rmsnorm_concat, fast_vocab_argmax
from mlx_lm.models.cache import KVCache
from mlx_lm.models.base import create_attention_mask

config = {
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "max_position_embeddings": 262144,
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000,
    "rms_norm_eps": 1e-06,
    "attention_bias": False,
    "vocab_size": 248320,
    "block_size": 4,
    "target_layer_ids": [63],
}

model = Qwen38MTPModel(config)
for p in model.parameters().values():
    if isinstance(p, mx.array):
        p = p.astype(mx.float16)

embed = nn.Embedding(248320, 5120)
lm_head = nn.Linear(5120, 248320, bias=False)

staged_first = mx.array([12345], dtype=mx.uint32)
target_hidden = mx.random.normal((1, 1, 5120)).astype(mx.float16)
mx.eval(staged_first, target_hidden)

# Standard step trace
hidden_std = target_hidden[:, -1:, :]
token_std = staged_first[:1].astype(mx.uint32)
cache_std = KVCache()
std_tokens = []

for step in range(4):
    e = embed(token_std[None]).astype(mx.float16)
    fused_std = mx.concatenate([model.pre_fc_norm_embedding(e), model.pre_fc_norm_hidden(hidden_std)], axis=-1)
    hidden_std = model.fc(fused_std)
    mask = create_attention_mask(hidden_std, cache_std)
    hidden_std = model.layers[0](hidden_std, mask=mask, cache=cache_std)
    hidden_std = model.norm(hidden_std)
    logits_std = lm_head(hidden_std[:, -1:, :]).astype(mx.float16)
    token_std = mx.argmax(logits_std[:, -1, :], axis=-1).astype(mx.uint32)
    std_tokens.append(token_std.item())

print("Standard tokens:", std_tokens)

# Fused step trace
hidden_fused = target_hidden[:, -1:, :]
token_fused = staged_first[:1].astype(mx.uint32)
cache_fused = KVCache()
fused_tokens = []
w_emb = model.pre_fc_norm_embedding.weight.astype(mx.float16)
w_hid = model.pre_fc_norm_hidden.weight.astype(mx.float16)
eps = float(model.norm.eps)

for step in range(4):
    e = embed(token_fused[None]).astype(mx.float16)
    fused_f = fused_dual_rmsnorm_concat(
        e.reshape(1, -1),
        hidden_fused.reshape(1, -1),
        w_emb,
        w_hid,
        eps=eps,
    ).reshape(1, 1, -1)
    hidden_fused = model.fc(fused_f)
    mask = create_attention_mask(hidden_fused, cache_fused)
    hidden_fused = model.layers[0](hidden_fused, mask=mask, cache=cache_fused)
    hidden_fused = model.norm(hidden_fused)
    logits_f = lm_head(hidden_fused[:, -1:, :]).astype(mx.float16)
    token_fused = fast_vocab_argmax(logits_f[:, -1, :])
    print(f"Step {step}: logits shape = {logits_f[:, -1, :].shape}, token_fused = {token_fused.item()}")
    fused_tokens.append(token_fused.item())

print("Fused tokens:   ", fused_tokens)
