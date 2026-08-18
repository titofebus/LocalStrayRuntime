"""Dynamic Entropy-Gated Speculative Drafting for Qwen 3.8 MTP in Apple MLX."""
import mlx.core as mx


def calculate_dynamic_draft_count(
    logits: mx.array,
    min_k: int = 2,
    max_k: int = 5,
    default_k: int = 4,
) -> int:
    """
    Dynamically determine draft count K based on top-2 logit margin confidence.
    
    When token prediction confidence is high (boilerplate/syntax), returns max_k (5).
    When token prediction is uncertain (branching/math logic), returns min_k (2).
    """
    if logits.ndim == 1:
        logits = logits[None, :]

    try:
        # Fast top-2 logit extraction
        top2 = mx.topk(logits, 2, axis=-1)
        margin = top2[..., 0] - top2[..., 1]
        confidence = float(mx.sigmoid(margin * 1.5).item())

        if confidence >= 0.75:
            return max_k
        elif confidence <= 0.45:
            return min_k
        else:
            return default_k
    except Exception:
        return default_k


def should_early_exit_drafting(
    logits: mx.array,
    confidence_threshold: float = 0.35,
) -> bool:
    """Check if the draft loop should early-exit on high-entropy intermediate tokens."""
    if logits.ndim == 1:
        logits = logits[None, :]

    try:
        top2 = mx.topk(logits, 2, axis=-1)
        margin = top2[..., 0] - top2[..., 1]
        confidence = float(mx.sigmoid(margin * 1.5).item())
        return confidence < confidence_threshold
    except Exception:
        return False
