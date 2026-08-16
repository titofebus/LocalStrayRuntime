from typing import Any, Dict, Iterable


def generation_usage(
    *,
    summary: Dict[str, Any],
    prefill: Dict[str, Any] | None = None,
    cache_lookup_ms: float = 0.0,
    prompt_tokens: int,
    total_time: float,
) -> Dict[str, Any]:
    prefill = prefill or {}
    phase_timings = summary.get("phase_timings_us") or {}
    generation_seconds = max(
        0.0,
        (float(summary.get("elapsed_us", 0.0)) - float(phase_timings.get("prefill", 0.0))) / 1_000_000.0,
    )
    completion_tokens = int(summary.get("generation_tokens", 0))
    tokens_per_second = completion_tokens / generation_seconds if generation_seconds > 0 else 0.0
    prefill_seconds = float(prefill.get("prefill_us", 0.0)) / 1_000_000.0
    prefill_tokens_computed = int(prefill.get("prefill_tokens_computed", prompt_tokens))

    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": completion_tokens,
        "tokens_per_second": tokens_per_second,
        "generation_seconds": generation_seconds,
        "latency": float(total_time),
        "accepted_from_draft": int(summary.get("accepted_from_draft", 0)),
        "acceptance_ratio": float(summary.get("acceptance_ratio", 0.0)),
        "cycles_completed": int(summary.get("cycles_completed", 0)),
        "fallback_ar": bool(summary.get("fallback_ar", False)),
        "adaptive_block_reductions": int(summary.get("adaptive_block_reductions", 0)),
        "adaptive_block_min": summary.get("adaptive_block_min"),
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": (
            prefill_tokens_computed / prefill_seconds if prefill_seconds > 0 else 0.0
        ),
        "physical_prefill_tokens": int(
            prefill.get("physical_prefill_tokens", prefill_tokens_computed)
        ),
        "prefill_tokens_restored": int(prefill.get("prefill_tokens_restored", 0)),
        "prefill_tokens_computed": prefill_tokens_computed,
        "prefix_cache_hit_tokens": int(prefill.get("snap_prefix_len", 0)),
        "prefix_cache_lookup_ms": float(cache_lookup_ms),
    }


def combine_generation_usage(
    phases: Iterable[Dict[str, Any]],
    reasoning_phase_count: int = 0,
) -> Dict[str, Any]:
    phase_list = list(phases)
    if not phase_list:
        return {}

    reasoning_phases = phase_list[:reasoning_phase_count]

    completion_tokens = sum(int(phase.get("completion_tokens", 0)) for phase in phase_list)
    generation_seconds = sum(float(phase.get("generation_seconds", 0.0)) for phase in phase_list)
    accepted_from_draft = sum(int(phase.get("accepted_from_draft", 0)) for phase in phase_list)
    adaptive_minima = [
        int(phase["adaptive_block_min"])
        for phase in phase_list
        if phase.get("adaptive_block_min") is not None
    ]
    prefill_seconds = sum(float(phase.get("prefill_seconds", 0.0)) for phase in phase_list)
    prefill_tokens_computed = sum(
        int(phase.get("prefill_tokens_computed", 0)) for phase in phase_list
    )

    return {
        "prompt_tokens": int(phase_list[0].get("prompt_tokens", 0)),
        "completion_tokens": completion_tokens,
        "tokens_per_second": completion_tokens / generation_seconds if generation_seconds > 0 else 0.0,
        "generation_seconds": generation_seconds,
        "reasoning_tokens": sum(
            int(phase.get("completion_tokens", 0)) for phase in reasoning_phases
        ),
        "reasoning_seconds": sum(
            float(phase.get("generation_seconds", 0.0)) for phase in reasoning_phases
        ),
        "latency": sum(float(phase.get("latency", 0.0)) for phase in phase_list),
        "accepted_from_draft": accepted_from_draft,
        "acceptance_ratio": accepted_from_draft / completion_tokens if completion_tokens > 0 else 0.0,
        "cycles_completed": sum(int(phase.get("cycles_completed", 0)) for phase in phase_list),
        "phase_count": len(phase_list),
        "fallback_ar": any(bool(phase.get("fallback_ar", False)) for phase in phase_list),
        "adaptive_block_reductions": sum(
            int(phase.get("adaptive_block_reductions", 0)) for phase in phase_list
        ),
        "adaptive_block_min": min(adaptive_minima) if adaptive_minima else None,
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": (
            prefill_tokens_computed / prefill_seconds if prefill_seconds > 0 else 0.0
        ),
        "physical_prefill_tokens": sum(
            int(phase.get("physical_prefill_tokens", 0)) for phase in phase_list
        ),
        "prefill_tokens_restored": sum(
            int(phase.get("prefill_tokens_restored", 0)) for phase in phase_list
        ),
        "prefill_tokens_computed": prefill_tokens_computed,
        "prefix_cache_hit_tokens": sum(
            int(phase.get("prefix_cache_hit_tokens", 0)) for phase in phase_list
        ),
        "prefix_cache_lookup_ms": sum(
            float(phase.get("prefix_cache_lookup_ms", 0.0)) for phase in phase_list
        ),
    }
