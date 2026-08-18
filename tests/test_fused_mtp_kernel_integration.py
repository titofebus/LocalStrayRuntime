"""Integration test comparing standard Qwen3.8 MTP predict_block vs Fused Metal Shaders."""
import inspect
from types import SimpleNamespace
import pytest
import mlx.core as mx

try:
    mx.eval(mx.zeros((1,)))
except RuntimeError:
    pytest.skip("Metal device not available in environment", allow_module_level=True)

import mlx.nn as nn
from dflash_mlx.engine.sampling import greedy_tokens_with_mask
from harness.executors.qwen38_mtp import Qwen38MTPModel
from harness.kernels.fused_mtp_ops import fast_vocab_argmax, fused_dual_rmsnorm_concat


@pytest.fixture
def mtp_config():
    return {
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


class MockTargetOps:
    def __init__(self, hidden_size=5120, vocab_size=248320, dtype=mx.float16):
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.dtype = dtype

    def embed_tokens(self, model):
        def _embed(x):
            return self.embed(x).astype(self.dtype)
        return _embed

    def logits_from_hidden(self, model, hidden):
        return self.lm_head(hidden.astype(self.dtype)).astype(self.dtype)


def test_fused_metal_vs_standard_mtp_block_prediction(mtp_config):
    """Verify that fused Metal kernels produce matching tokens on Qwen 3.8 MTP draft block."""
    mtp_model = Qwen38MTPModel(mtp_config, fused_mtp=True)

    # Cast all parameters to float16 (exact production model state)
    for p in mtp_model.parameters().values():
        if isinstance(p, mx.array):
            p = p.astype(mx.float16)

    target_ops = MockTargetOps(
        hidden_size=mtp_config["hidden_size"],
        vocab_size=mtp_config["vocab_size"],
        dtype=mx.float16,
    )
    target_model = SimpleNamespace()

    staged_first = mx.array([12345], dtype=mx.uint32)
    target_hidden = mx.random.normal((1, 1, mtp_config["hidden_size"])).astype(mx.float16)
    mx.eval(staged_first, target_hidden)

    def standard_predict_block(
        model: Qwen38MTPModel,
        *,
        target_model,
        target_ops,
        staged_first,
        target_hidden,
        draft_count=4,
    ):
        hidden = target_hidden[:, -1:, :]
        token = staged_first[:1].astype(mx.uint32)
        from mlx_lm.models.cache import KVCache
        from mlx_lm.models.base import create_attention_mask

        cache = KVCache()
        drafted = []

        for _ in range(draft_count):
            emb = target_ops.embed_tokens(target_model)(token[None])
            fused = mx.concatenate(
                [
                    model.pre_fc_norm_embedding(emb),
                    model.pre_fc_norm_hidden(hidden),
                ],
                axis=-1,
            )
            hidden = model.fc(fused)
            mask = create_attention_mask(hidden, cache)
            hidden = model.layers[0](hidden, mask=mask, cache=cache)
            hidden = model.norm(hidden)
            logits = target_ops.logits_from_hidden(target_model, hidden[:, -1:, :])

            token = greedy_tokens_with_mask(logits[:, -1, :], None).reshape(-1)
            drafted.append(token.astype(mx.uint32))

        return mx.concatenate(drafted, axis=0)

    standard_tokens = standard_predict_block(
        mtp_model,
        target_model=target_model,
        target_ops=target_ops,
        staged_first=staged_first,
        target_hidden=target_hidden,
        draft_count=4,
    )
    fused_tokens = mtp_model.predict_block(
        target_model=target_model,
        target_ops=target_ops,
        staged_first=staged_first,
        target_hidden=target_hidden,
        draft_count=4,
        suppress_token_mask=None,
    )
    mx.eval(standard_tokens)
    mx.eval(fused_tokens)

    assert mx.array_equal(standard_tokens, fused_tokens), (
        f"Mismatch: standard={standard_tokens.tolist()} vs fused={fused_tokens.tolist()}"
    )


def test_fused_rmsnorm_honors_runtime_epsilon():
    hidden_size = 5120
    epsilon = 1e-2
    embedding = mx.full((1, hidden_size), 1e-2, dtype=mx.float16)
    hidden = mx.full((1, hidden_size), 2e-2, dtype=mx.float16)
    embedding_weight = mx.ones((hidden_size,), dtype=mx.float16)
    hidden_weight = mx.ones((hidden_size,), dtype=mx.float16)

    expected = mx.concatenate(
        [
            mx.fast.rms_norm(embedding, embedding_weight, epsilon),
            mx.fast.rms_norm(hidden, hidden_weight, epsilon),
        ],
        axis=-1,
    )
    actual = fused_dual_rmsnorm_concat(
        embedding,
        hidden,
        embedding_weight,
        hidden_weight,
        eps=epsilon,
    )
    mx.eval(expected, actual)

    assert mx.allclose(expected, actual, atol=5e-3, rtol=5e-3)


def test_mtp_hot_path_does_not_retry_failed_kernels_per_token():
    source = inspect.getsource(Qwen38MTPModel.predict_block)

    assert "except Exception" not in source


if __name__ == "__main__":
    pytest.main(["-s", __file__])
