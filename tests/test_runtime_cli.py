import argparse
import json
from pathlib import Path

import harness.runtime_cli as runtime_cli


def test_configure_writes_portable_runtime_paths(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "runtime.json"
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    monkeypatch.setattr(runtime_cli, "RUNTIME_CONFIG", config_path)

    result = runtime_cli.configure(
        argparse.Namespace(target=str(target), draft=str(draft))
    )

    assert result == 0
    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "target_model": str(target),
        "draft_model": str(draft),
    }


def test_prime_agent_configuration_preserves_existing_providers(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    prime_dir = tmp_path / ".prime" / "agent"
    prime_dir.mkdir(parents=True)
    models_path = prime_dir / "models.json"
    models_path.write_text(
        json.dumps({"providers": {"existing": {"name": "Existing"}}}),
        encoding="utf-8",
    )

    result = runtime_cli.configure_prime_agent(argparse.Namespace())

    assert result == 0
    providers = json.loads(models_path.read_text(encoding="utf-8"))["providers"]
    assert providers["existing"] == {"name": "Existing"}
    assert providers["local-mlx"]["models"][0]["id"] == "qwen3.8-27b"


def test_remote_server_bind_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(runtime_cli, "_apply_runtime_config", lambda: {})

    result = runtime_cli.serve(
        argparse.Namespace(host="0.0.0.0", port=8000, allow_remote=False)
    )

    assert result == 2
