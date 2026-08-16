"""Release metadata helpers that do not require loading MLX."""

from __future__ import annotations

import shutil
from pathlib import Path


def write_mtp_release_metadata(
    source: Path, output: Path, source_revision: str, weights_hash: str
) -> None:
    license_path = source / "LICENSE"
    if not license_path.is_file():
        raise FileNotFoundError(f"Source model license is missing: {license_path}")
    shutil.copy2(license_path, output / "LICENSE")
    (output / "NOTICE").write_text(
        "Qwen3.8 native MTP MLX artifact\n\n"
        "Derived from Qwen/Qwen3.8-27B under the Apache License 2.0.\n"
        f"Source revision: {source_revision}\n"
        "Qwen is a trademark of its respective owner. This artifact is not "
        "affiliated with or endorsed by the Qwen team.\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# Qwen3.8-27B native MTP for MLX (6-bit)\n\n"
        "This is the native multi-token prediction head extracted from "
        "`Qwen/Qwen3.8-27B`, quantized to 6-bit affine weights with group size "
        "64 for MLX. It is a draft artifact for speculative decoding and is "
        "not a separately trained DFlash diffusion model.\n\n"
        f"- Source revision: `{source_revision}`\n"
        f"- Weights SHA-256: `{weights_hash}`\n"
        "- Target model: `Qwen/Qwen3.8-27B`\n\n"
        "## Reproduction\n\n"
        "```bash\n"
        "uv run python -m harness.trainer.export_qwen38_mtp \\\n"
        "  --source /path/to/Qwen3.8-27B \\\n"
        "  --output /path/to/Qwen3.8-27B-MTP-MLX-6bit \\\n"
        f"  --source-revision {source_revision}\n"
        "```\n\n"
        "See `draft_provenance.json` for machine-readable provenance and "
        "`LICENSE` for the source license.\n",
        encoding="utf-8",
    )
    (output / "SHA256SUMS").write_text(
        f"{weights_hash}  model.safetensors\n", encoding="utf-8"
    )
