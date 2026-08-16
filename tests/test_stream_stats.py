from harness.daemon.stream_stats import combine_generation_usage, generation_usage


def test_generation_usage_exposes_speculative_summary():
    usage = generation_usage(
        summary={
            "elapsed_us": 3_000_000,
            "phase_timings_us": {"prefill": 1_000_000},
            "generation_tokens": 40,
            "accepted_from_draft": 30,
            "acceptance_ratio": 0.75,
            "cycles_completed": 10,
            "fallback_ar": False,
            "adaptive_block_reductions": 3,
            "adaptive_block_min": 2,
        },
        prefill={
            "prefill_us": 500_000,
            "physical_prefill_tokens": 80,
            "prefill_tokens_restored": 120,
            "prefill_tokens_computed": 80,
            "snap_prefix_len": 120,
        },
        cache_lookup_ms=2.5,
        prompt_tokens=100,
        total_time=3.2,
    )

    assert usage == {
        "prompt_tokens": 100,
        "completion_tokens": 40,
        "tokens_per_second": 20.0,
        "generation_seconds": 2.0,
        "latency": 3.2,
        "accepted_from_draft": 30,
        "acceptance_ratio": 0.75,
        "cycles_completed": 10,
        "fallback_ar": False,
        "adaptive_block_reductions": 3,
        "adaptive_block_min": 2,
        "prefill_seconds": 0.5,
        "prefill_tokens_per_second": 160.0,
        "physical_prefill_tokens": 80,
        "prefill_tokens_restored": 120,
        "prefill_tokens_computed": 80,
        "prefix_cache_hit_tokens": 120,
        "prefix_cache_lookup_ms": 2.5,
    }


def test_combine_generation_usage_weights_phases_by_generation_time():
    combined = combine_generation_usage([
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "tokens_per_second": 10.0,
            "generation_seconds": 2.0,
            "latency": 2.5,
            "accepted_from_draft": 5,
            "acceptance_ratio": 0.25,
            "cycles_completed": 10,
            "fallback_ar": False,
            "adaptive_block_reductions": 4,
            "adaptive_block_min": 1,
            "prefill_seconds": 0.5,
            "physical_prefill_tokens": 100,
            "prefill_tokens_restored": 0,
            "prefill_tokens_computed": 100,
            "prefix_cache_hit_tokens": 0,
            "prefix_cache_lookup_ms": 1.0,
        },
        {
            "prompt_tokens": 130,
            "completion_tokens": 60,
            "tokens_per_second": 30.0,
            "generation_seconds": 2.0,
            "latency": 2.8,
            "accepted_from_draft": 45,
            "acceptance_ratio": 0.75,
            "cycles_completed": 20,
            "fallback_ar": False,
            "adaptive_block_reductions": 1,
            "adaptive_block_min": 2,
            "prefill_seconds": 0.25,
            "physical_prefill_tokens": 20,
            "prefill_tokens_restored": 80,
            "prefill_tokens_computed": 20,
            "prefix_cache_hit_tokens": 80,
            "prefix_cache_lookup_ms": 0.5,
        },
    ])

    assert combined["prompt_tokens"] == 100
    assert combined["completion_tokens"] == 80
    assert combined["tokens_per_second"] == 20.0
    assert combined["generation_seconds"] == 4.0
    assert combined["latency"] == 5.3
    assert combined["accepted_from_draft"] == 50
    assert combined["acceptance_ratio"] == 0.625
    assert combined["cycles_completed"] == 30
    assert combined["phase_count"] == 2
    assert combined["adaptive_block_reductions"] == 5
    assert combined["adaptive_block_min"] == 1
    assert combined["fallback_ar"] is False
    assert combined["prefill_seconds"] == 0.75
    assert combined["prefill_tokens_per_second"] == 160.0
    assert combined["physical_prefill_tokens"] == 120
    assert combined["prefill_tokens_restored"] == 80
    assert combined["prefill_tokens_computed"] == 120
    assert combined["prefix_cache_hit_tokens"] == 80
    assert combined["prefix_cache_lookup_ms"] == 1.5


def test_combine_generation_usage_exposes_reasoning_phase_metrics():
    combined = combine_generation_usage([
        {
            "prompt_tokens": 100,
            "completion_tokens": 96,
            "generation_seconds": 4.0,
        },
        {
            "prompt_tokens": 200,
            "completion_tokens": 500,
            "generation_seconds": 20.0,
        },
    ], reasoning_phase_count=1)

    assert combined["reasoning_tokens"] == 96
    assert combined["reasoning_seconds"] == 4.0
