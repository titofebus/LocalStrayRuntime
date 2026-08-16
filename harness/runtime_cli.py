"""Portable command-line entry point for the Qwen Prime local runtime."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


RUNTIME_CONFIG = (
    Path.home() / "Library" / "Application Support" / "QwenPrime" / "runtime.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any], *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    if private:
        temporary.chmod(0o600)
    temporary.replace(path)


def _apply_runtime_config() -> dict[str, Any]:
    config = _read_json(RUNTIME_CONFIG)
    target = config.get("target_model")
    draft = config.get("draft_model")
    if isinstance(target, str) and target:
        os.environ.setdefault("QWEN_PRIME_TARGET_MODEL", target)
    if isinstance(draft, str) and draft:
        os.environ.setdefault("QWEN_PRIME_DRAFT_MODEL", draft)
    return config


def configure(args: argparse.Namespace) -> int:
    current = _read_json(RUNTIME_CONFIG)
    current.update(
        {
            "target_model": str(Path(args.target).expanduser().resolve()),
            "draft_model": str(Path(args.draft).expanduser().resolve()),
        }
    )
    _write_json(RUNTIME_CONFIG, current)
    print(f"Wrote {RUNTIME_CONFIG}")
    return 0


def doctor(_: argparse.Namespace) -> int:
    _apply_runtime_config()
    from harness.config import DEFAULT_MLX_MODEL_PATH, DEFAULT_MTP_MODEL_PATH
    from harness.model_provenance import validate_speculative_pair

    problems: list[str] = []
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        problems.append("MLX runtime requires Apple Silicon macOS")

    for label, raw_path in (
        ("target", DEFAULT_MLX_MODEL_PATH),
        ("draft", DEFAULT_MTP_MODEL_PATH),
    ):
        if not Path(raw_path).is_dir():
            problems.append(f"{label} model directory is missing: {raw_path}")

    if not problems:
        try:
            identity = validate_speculative_pair(
                DEFAULT_MLX_MODEL_PATH,
                DEFAULT_MTP_MODEL_PATH,
            )
            print(
                f"Validated {identity.target_model_id} + {identity.draft_model_id} "
                f"({identity.weights_sha256})"
            )
        except Exception as error:
            problems.append(str(error))

    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 1
    print("Qwen Prime runtime configuration is ready.")
    return 0


def serve(args: argparse.Namespace) -> int:
    _apply_runtime_config()
    if args.host != "127.0.0.1" and not args.allow_remote:
        print("Refusing a non-loopback bind without --allow-remote", file=sys.stderr)
        return 2
    from harness.daemon.unified_server import app
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def configure_prime_agent(_: argparse.Namespace) -> int:
    prime_dir = Path.home() / ".prime" / "agent"
    models_path = prime_dir / "models.json"
    auth_path = prime_dir / "auth.json"

    models = _read_json(models_path)
    providers = models.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise ValueError(f"Expected providers to be an object in {models_path}")
    providers["local-mlx"] = {
        "name": "Qwen Prime Local Runtime",
        "baseUrl": "http://127.0.0.1:8000/v1",
        "api": "openai-completions",
        "models": [
            {
                "id": "qwen3.8-27b",
                "name": "Qwen3.8 27B + native MTP (MLX 6-bit)",
                "reasoning": True,
                "contextWindow": 262144,
                "maxTokens": 131072,
                "cost": {"input": 0, "output": 0},
            }
        ],
    }
    _write_json(models_path, models)

    auth = _read_json(auth_path)
    auth.setdefault("local-mlx", "local-runtime")
    _write_json(auth_path, auth, private=True)
    print(f"Configured local-mlx in {models_path} without replacing other providers.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen-prime-runtime")
    subcommands = parser.add_subparsers(dest="command", required=True)

    configure_parser = subcommands.add_parser("configure")
    configure_parser.add_argument("--target", required=True)
    configure_parser.add_argument("--draft", required=True)
    configure_parser.set_defaults(handler=configure)

    doctor_parser = subcommands.add_parser("doctor")
    doctor_parser.set_defaults(handler=doctor)

    serve_parser = subcommands.add_parser("serve")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--allow-remote", action="store_true")
    serve_parser.set_defaults(handler=serve)

    prime_parser = subcommands.add_parser("configure-prime-agent")
    prime_parser.set_defaults(handler=configure_prime_agent)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(args.handler(args))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
