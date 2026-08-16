import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER = PROJECT_ROOT / "scripts" / "build_embedded_runtime.command"


def test_embedded_runtime_builder_has_a_read_only_verifier(tmp_path: Path):
    payload = tmp_path / "runtime"
    launcher = payload / "bin" / "qwen-prime-runtime"
    python = payload / "python" / "bin" / "python3.12"
    packages = payload / "site-packages" / "harness"
    launcher.parent.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    packages.mkdir(parents=True)
    launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launcher.chmod(0o755)
    python.chmod(0o755)

    result = subprocess.run(
        [str(BUILDER), "--verify", str(payload)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "UV_CACHE_DIR": str(tmp_path / "uv-cache")},
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_embedded_launcher_resolves_python_relative_to_moved_payload(tmp_path: Path):
    payload = tmp_path / "runtime"
    moved = tmp_path / "moved-runtime"
    python = payload / "python" / "bin" / "python3.12"
    python.parent.mkdir(parents=True)
    python.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$0\" \"$PYTHONPATH\"\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    (payload / "site-packages" / "harness").mkdir(parents=True)

    create = subprocess.run(
        [str(BUILDER), "--write-launcher", str(payload)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert create.returncode == 0, create.stderr

    payload.rename(moved)
    result = subprocess.run(
        [str(moved / "bin" / "qwen-prime-runtime"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert str(moved / "python" / "bin" / "python3.12") in result.stdout
    assert str(moved / "site-packages") in result.stdout
    assert str(payload) not in result.stdout
    launcher = (moved / "bin" / "qwen-prime-runtime").read_text(encoding="utf-8")
    assert "PYTHONDONTWRITEBYTECODE=1" in launcher
