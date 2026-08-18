import mlx.core as mx
import pytest

try:
    mx.eval(mx.zeros((1,)))
except RuntimeError:
    pytest.skip("Metal device not available in environment", allow_module_level=True)

from harness.kernels.fp8_kv_cache import NativeFP8KVCache


def test_fp8_cache_preserves_earlier_tokens_when_later_scale_is_larger():
    cache = NativeFP8KVCache(num_heads=2, head_dim=8, max_seq_len=4)
    first_keys = mx.full((1, 2, 1, 8), 0.01, dtype=mx.float16)
    first_values = mx.full((1, 2, 1, 8), -0.02, dtype=mx.float16)
    later_keys = mx.full((1, 2, 1, 8), 10_000.0, dtype=mx.float16)
    later_values = mx.full((1, 2, 1, 8), -12_000.0, dtype=mx.float16)

    cache.update(first_keys, first_values)
    cache.update(later_keys, later_values)
    restored_keys, restored_values = cache.get(0, 1)
    mx.eval(restored_keys, restored_values)

    assert mx.allclose(restored_keys, first_keys, atol=2e-3, rtol=5e-2)
    assert mx.allclose(restored_values, first_values, atol=2e-3, rtol=5e-2)


def test_fp8_cache_rejects_capacity_overflow():
    cache = NativeFP8KVCache(num_heads=2, head_dim=8, max_seq_len=1)
    values = mx.zeros((1, 2, 2, 8), dtype=mx.float16)

    with pytest.raises(ValueError, match="capacity"):
        cache.update(values, values)
