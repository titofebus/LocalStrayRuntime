"""Export Qwen3.8's native MTP head as a standalone 6-bit MLX artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from harness.executors.qwen38_mtp import Qwen38MTPModel
from harness.mtp_artifact import (
    build_mtp_artifact_config,
    convert_mtp_weight_for_mlx,
    locate_mtp_weights,
)
from harness.config import QWEN_PRIME_DATA_DIR, SOURCE_MODEL_PATH
from harness.release_metadata import write_mtp_release_metadata


DEFAULT_SOURCE = SOURCE_MODEL_PATH
DEFAULT_OUTPUT = QWEN_PRIME_DATA_DIR / "Models" / "Qwen3.8-27B-MTP-MLX-6bit"
DEFAULT_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_mtp(source: Path, output: Path, source_revision: str) -> None:
    index_path = source / "model.safetensors.index.json"
    locations = locate_mtp_weights(index_path)
    source_weights: dict[str, mx.array] = {}
    for shard_name in sorted(set(locations.values())):
        shard = mx.load(str(source / shard_name))
        for source_name, located_shard in locations.items():
            if located_shard == shard_name:
                source_weights[source_name.removeprefix("mtp.")] = (
                    convert_mtp_weight_for_mlx(source_name, shard[source_name])
                )

    config = build_mtp_artifact_config(source_revision=source_revision)
    model = Qwen38MTPModel(config)
    model.load_weights(list(source_weights.items()), strict=True)
    mx.eval(model.parameters())
    nn.quantize(
        model,
        group_size=64,
        bits=6,
        mode="affine",
        class_predicate=lambda _path, module: hasattr(module, "to_quantized"),
    )
    mx.eval(model.parameters())

    output.mkdir(parents=True, exist_ok=False)
    weights_path = output / "model.safetensors"
    mx.save_safetensors(
        str(weights_path),
        dict(tree_flatten(model.parameters())),
        metadata={"format": "mlx", "target_model": "Qwen/Qwen3.8-27B"},
    )
    weights_hash = _sha256(weights_path)
    config["weights_sha256"] = weights_hash
    (output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (output / "draft_provenance.json").write_text(
        json.dumps(
            {
                "source_model_id": "Qwen/Qwen3.8-27B#native-mtp",
                "source_revision": source_revision,
                "target_model_id": "Qwen/Qwen3.8-27B",
                "weights_sha256": weights_hash,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_mtp_release_metadata(source, output, source_revision, weights_hash)
    print(f"Exported Qwen3.8 native MTP to {output}")
    print(f"SHA-256: {weights_hash}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    export_mtp(args.source, args.output, args.source_revision)


if __name__ == "__main__":
    main()
