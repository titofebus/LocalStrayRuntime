"""Identity checks for target and speculative draft model pairs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class SpeculativePairIdentity:
    target_model_id: str
    draft_model_id: str
    draft_target_model_id: str
    weights_sha256: str


def _target_model_id(target_path: Path) -> str:
    readme_path = target_path / "README.md"
    if not readme_path.is_file():
        raise ValueError(f"Target model is missing {readme_path}")

    readme = readme_path.read_text(encoding="utf-8")
    match = re.search(r"^base_model:\s*([^\n#]+)", readme, re.MULTILINE)
    if match is None:
        raise ValueError(f"Target model README has no base_model identity: {readme_path}")
    return match.group(1).strip().strip("'\"")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as weights:
        for chunk in iter(lambda: weights.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantization_identity(config: dict[str, object]) -> dict[str, object]:
    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError("Model config has no quantization metadata")

    default_bits = quantization.get("bits")
    group_size = quantization.get("group_size")
    mode = quantization.get("mode")
    if not isinstance(default_bits, int) or not isinstance(group_size, int):
        raise ValueError("Model quantization metadata is incomplete")
    if not isinstance(mode, str) or not mode:
        raise ValueError("Model quantization mode is missing")

    bits = {default_bits}
    for value in quantization.values():
        if isinstance(value, dict) and isinstance(value.get("bits"), int):
            bits.add(value["bits"])

    sorted_bits = sorted(bits)
    return {
        "scheme": "mixed" if len(sorted_bits) > 1 else "uniform",
        "bits": sorted_bits,
        "default_bits": default_bits,
        "group_size": group_size,
        "mode": mode,
    }


def validate_speculative_pair(
    target_path: str | Path,
    draft_path: str | Path,
    *,
    verify_weights: bool = True,
) -> SpeculativePairIdentity:
    target_root = Path(target_path)
    draft_root = Path(draft_path)
    target_model_id = _target_model_id(target_root)

    provenance_path = draft_root / "draft_provenance.json"
    if not provenance_path.is_file():
        raise ValueError(
            f"Speculative draft is untrusted: missing {provenance_path}. "
            "A renamed config is not evidence that draft weights match the target."
        )

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    draft_model_id = str(provenance.get("source_model_id", "")).strip()
    draft_target_model_id = str(provenance.get("target_model_id", "")).strip()
    expected_sha256 = str(provenance.get("weights_sha256", "")).strip().lower()

    if not draft_model_id or not draft_target_model_id:
        raise ValueError(f"Incomplete draft identity in {provenance_path}")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError(f"Invalid weights_sha256 in {provenance_path}")
    if draft_target_model_id != target_model_id:
        raise ValueError(
            f"Draft targets {draft_target_model_id}, but the loaded target is "
            f"{target_model_id}. Refusing mismatched speculative decoding."
        )

    if verify_weights:
        weights_path = draft_root / "model.safetensors"
        if not weights_path.is_file():
            raise ValueError(f"Speculative draft is missing {weights_path}")
        actual_sha256 = _sha256(weights_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Draft weight hash mismatch for {weights_path}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

    return SpeculativePairIdentity(
        target_model_id=target_model_id,
        draft_model_id=draft_model_id,
        draft_target_model_id=draft_target_model_id,
        weights_sha256=expected_sha256,
    )


def qwen_prime_runtime_identity(
    target_path: str | Path,
    draft_path: str | Path,
    *,
    block_tokens: int,
) -> dict[str, object]:
    target_root = Path(target_path)
    draft_root = Path(draft_path)
    pair = validate_speculative_pair(
        target_root,
        draft_root,
        verify_weights=False,
    )
    target_config = json.loads(
        (target_root / "config.json").read_text(encoding="utf-8")
    )
    draft_config = json.loads(
        (draft_root / "config.json").read_text(encoding="utf-8")
    )

    return {
        "runtime_id": "qwen38-native-mtp-v2",
        "target_model_id": pair.target_model_id,
        "draft_model_id": pair.draft_model_id,
        "target_path": str(target_root),
        "draft_path": str(draft_root),
        "block_tokens": block_tokens,
        "target_quantization": _quantization_identity(target_config),
        "draft_quantization": _quantization_identity(draft_config),
        "draft_model_type": draft_config.get("model_type"),
        "draft_norm_weight_offset": draft_config.get("norm_weight_offset"),
        "draft_weights_sha256": pair.weights_sha256,
    }
