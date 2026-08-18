from pathlib import Path

import harness.model_provenance as model_provenance


def test_qwen_prime_runtime_identity_proves_exact_speculative_pair(tmp_path: Path):
    qwen_prime_runtime_identity = getattr(
        model_provenance, "qwen_prime_runtime_identity", None
    )
    assert callable(qwen_prime_runtime_identity)

    target_path = tmp_path / "target"
    draft_path = tmp_path / "draft"
    target_path.mkdir()
    draft_path.mkdir()
    (target_path / "README.md").write_text(
        "---\nbase_model: Qwen/Qwen3.8-27B\n---\n",
        encoding="utf-8",
    )
    (target_path / "config.json").write_text(
        """{
            "quantization": {
                "bits": 4,
                "group_size": 64,
                "mode": "affine",
                "language_model.model.layers.3.self_attn.q_proj": {
                    "bits": 8,
                    "group_size": 64,
                    "mode": "affine"
                }
            }
        }""",
        encoding="utf-8",
    )
    (draft_path / "config.json").write_text(
        """{
            "model_type": "qwen3_8_mtp",
            "norm_weight_offset": 1.0,
            "quantization": {"bits": 6, "group_size": 64, "mode": "affine"}
        }""",
        encoding="utf-8",
    )
    (draft_path / "draft_provenance.json").write_text(
        """{
            "source_model_id": "Qwen/Qwen3.8-27B#native-mtp",
            "target_model_id": "Qwen/Qwen3.8-27B",
            "weights_sha256": "0af49c2c931f9f98c6beb2a85cae899a1c839647aae28b18e9187af53340fab2"
        }""",
        encoding="utf-8",
    )

    identity = qwen_prime_runtime_identity(target_path, draft_path, block_tokens=4)

    assert identity == {
        "runtime_id": "qwen38-native-mtp-v2",
        "target_model_id": "Qwen/Qwen3.8-27B",
        "draft_model_id": "Qwen/Qwen3.8-27B#native-mtp",
        "target_path": str(target_path),
        "draft_path": str(draft_path),
        "block_tokens": 4,
        "target_quantization": {
            "scheme": "mixed",
            "bits": [4, 8],
            "default_bits": 4,
            "group_size": 64,
            "mode": "affine",
        },
        "draft_quantization": {
            "scheme": "uniform",
            "bits": [6],
            "default_bits": 6,
            "group_size": 64,
            "mode": "affine",
        },
        "draft_model_type": "qwen3_8_mtp",
        "draft_norm_weight_offset": 1.0,
        "draft_weights_sha256": (
            "0af49c2c931f9f98c6beb2a85cae899a1c839647aae28b18e9187af53340fab2"
        ),
    }


def test_launcher_requires_runtime_identity_not_only_model_health():
    launcher = (
        Path(__file__).parents[1] / "scripts" / "launch_qwen_prime.command"
    ).read_text(encoding="utf-8")

    assert 'IDENTITY_URL="http://127.0.0.1:8000/v1/engine"' in launcher
    assert '"runtime_id": "qwen38-native-mtp-v2"' in launcher
    assert '"scheme": "mixed"' in launcher
    assert '"bits": [4, 8]' in launcher
    assert '"default_bits": 4' in launcher
    assert '"scheme": "uniform"' in launcher
    assert '"bits": [6]' in launcher
    assert '"warmup_complete": True' in launcher
    assert "required.items() <= identity.items()" in launcher
    assert "runtime_is_ready" in launcher
    assert "--restart)" in launcher
    assert "FORCE_RESTART" in launcher
    assert "Refusing to stop PID" in launcher
    assert "pkill" not in launcher


def test_endpoint_benchmark_rejects_missing_speculative_telemetry():
    benchmark = (
        Path(__file__).parents[1]
        / "scripts"
        / "benchmark_qwen_prime_endpoint.command"
    ).read_text(encoding="utf-8")

    assert 'IDENTITY_URL="http://127.0.0.1:8000/v1/engine"' in benchmark
    assert 'engine_identity.get("prefix_cache_enabled") is not True' in benchmark
    assert 'engine_identity.get("warmup_complete") is not True' in benchmark
    assert "required_speculative_fields" in benchmark
    assert "missing_speculative_fields" in benchmark
    assert '"prefill_seconds"' in benchmark
    assert '"prefix_cache_hit_tokens"' in benchmark
    assert "engine_identity['verify_mode']" in benchmark
    assert 'if [[ -t 0 ]]; then' in benchmark


def test_engine_identity_exposes_active_verification_mode():
    server = (
        Path(__file__).parents[1] / "harness" / "daemon" / "unified_server.py"
    ).read_text(encoding="utf-8")

    assert 'identity["verify_mode"] = str(' in server
