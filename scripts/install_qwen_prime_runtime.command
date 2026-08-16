#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required: https://docs.astral.sh/uv/" >&2
    exit 1
fi

uv tool install --force "$PROJECT_DIR"
echo "Installed qwen-prime-runtime into $(uv tool dir --bin)"
echo "Next: qwen-prime-runtime configure --target /path/to/target --draft /path/to/draft"
