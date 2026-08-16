import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


def _validator():
    spec = importlib.util.find_spec("harness.model_provenance")
    assert spec is not None, "harness.model_provenance must validate speculative model identity"
    module = importlib.import_module("harness.model_provenance")
    return module.validate_speculative_pair


def _write_target(root: Path, model_id: str = "Qwen/Qwen3.8-27B") -> None:
    root.mkdir()
    (root / "README.md").write_text(
        f"---\nbase_model: {model_id}\n---\n",
        encoding="utf-8",
    )
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}),
        encoding="utf-8",
    )


def _write_draft(root: Path, target_model_id: str) -> None:
    root.mkdir()
    (root / "draft_provenance.json").write_text(
        json.dumps(
            {
                "source_model_id": "local/Qwen3.8-27B-DFlash",
                "target_model_id": target_model_id,
                "weights_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )


def test_accepts_draft_trained_for_exact_target_identity(tmp_path: Path):
    validate_speculative_pair = _validator()
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    _write_target(target)
    _write_draft(draft, "Qwen/Qwen3.8-27B")

    pair = validate_speculative_pair(target, draft, verify_weights=False)

    assert pair.target_model_id == "Qwen/Qwen3.8-27B"
    assert pair.draft_target_model_id == "Qwen/Qwen3.8-27B"


def test_rejects_qwen36_draft_relabelled_for_qwen38(tmp_path: Path):
    validate_speculative_pair = _validator()
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    _write_target(target)
    _write_draft(draft, "Qwen/Qwen3.6-27B")

    with pytest.raises(ValueError, match="Qwen/Qwen3.6-27B.*Qwen/Qwen3.8-27B"):
        validate_speculative_pair(target, draft, verify_weights=False)


def test_rejects_unprovenanced_draft(tmp_path: Path):
    validate_speculative_pair = _validator()
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    _write_target(target)
    draft.mkdir()

    with pytest.raises(ValueError, match="draft_provenance.json"):
        validate_speculative_pair(target, draft, verify_weights=False)


def test_engine_validates_pair_before_loading_metal(tmp_path: Path, monkeypatch):
    fake_mlx = types.ModuleType("mlx")
    fake_mlx_core = types.ModuleType("mlx.core")
    fake_mlx.core = fake_mlx_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_core)
    from harness.executors.dflash_engine import DFlashEngine

    target = tmp_path / "target"
    draft = tmp_path / "draft"
    _write_target(target)
    draft.mkdir()
    DFlashEngine._bundle = None

    with pytest.raises(ValueError, match="draft_provenance.json"):
        DFlashEngine(target_path=str(target), draft_ref=str(draft))._ensure_loaded()
