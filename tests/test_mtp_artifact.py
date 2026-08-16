import json
from pathlib import Path

import pytest

import harness.mtp_artifact as mtp_artifact
from harness.release_metadata import write_mtp_release_metadata
from harness.mtp_artifact import (
    MTP_WEIGHT_NAMES,
    build_mtp_artifact_config,
    locate_mtp_weights,
)


def test_locates_all_qwen38_mtp_weights_across_shards(tmp_path: Path):
    index_path = tmp_path / "model.safetensors.index.json"
    weight_map = {
        name: f"model-{index % 2 + 1:05d}-of-00002.safetensors"
        for index, name in enumerate(MTP_WEIGHT_NAMES)
    }
    index_path.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")

    located = locate_mtp_weights(index_path)

    assert set(located) == set(MTP_WEIGHT_NAMES)
    assert set(located.values()) == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }


def test_rejects_source_missing_native_mtp_tensor(tmp_path: Path):
    index_path = tmp_path / "model.safetensors.index.json"
    weight_map = {name: "model.safetensors" for name in MTP_WEIGHT_NAMES[:-1]}
    index_path.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing Qwen3.8 MTP tensors"):
        locate_mtp_weights(index_path)


def test_builds_exact_qwen38_six_bit_artifact_config():
    config = build_mtp_artifact_config(
        source_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    )

    assert config["target_model_id"] == "Qwen/Qwen3.8-27B"
    assert config["model_type"] == "qwen3_8_mtp"
    assert config["block_size"] == 8
    assert config["target_layer_ids"] == [63]
    assert config["norm_weight_offset"] == 1.0
    assert config["quantization"] == {
        "group_size": 64,
        "bits": 6,
        "mode": "affine",
    }


def test_converts_qwen_checkpoint_norm_offsets_for_mlx():
    convert_weight = getattr(mtp_artifact, "convert_mtp_weight_for_mlx", None)

    assert callable(convert_weight), "MTP export must convert Qwen norm offsets"
    for name in (
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    ):
        assert convert_weight(name, 0.25) == 1.25
    assert convert_weight("mtp.fc.weight", 0.25) == 0.25


def test_export_metadata_carries_license_provenance_and_checksum(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "LICENSE").write_text("Apache License 2.0\n", encoding="utf-8")

    write_mtp_release_metadata(source, output, "source-revision", "abc123")

    assert (output / "LICENSE").read_text(encoding="utf-8") == "Apache License 2.0\n"
    assert "not a separately trained DFlash" in (output / "README.md").read_text(
        encoding="utf-8"
    )
    assert "source-revision" in (output / "NOTICE").read_text(encoding="utf-8")
    assert (output / "SHA256SUMS").read_text(encoding="utf-8") == (
        "abc123  model.safetensors\n"
    )
