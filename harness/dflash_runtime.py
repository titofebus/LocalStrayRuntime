import os


def initialize_mlx_streams() -> None:
    """Initialize thread-local MLX streams before entering DFlash internals."""
    import mlx.core as mx

    _ = mx.default_stream(mx.cpu)
    try:
        _ = mx.default_stream(mx.gpu)
    except (IndexError, RuntimeError):
        # Headless test sessions may expose MLX without an available Metal device.
        pass


def build_dflash_runtime_context():
    initialize_mlx_streams()
    from dflash_mlx.runtime.config import runtime_config_from_defaults
    from dflash_mlx.runtime.context import build_runtime_context

    verify_mode = os.environ.get("QWEN_PRIME_VERIFY_MODE", "adaptive")
    if verify_mode not in {"adaptive", "ddtree"}:
        raise ValueError(
            "QWEN_PRIME_VERIFY_MODE must be 'adaptive' or 'ddtree'"
        )

    return build_runtime_context(
        runtime_config_from_defaults(
            prefix_cache=True,
            prefix_cache_l2=False,
            target_fa_window=0,
            verify_len_cap=0,
            verify_mode=verify_mode,
        )
    )
