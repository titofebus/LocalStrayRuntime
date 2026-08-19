#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"

write_launcher() {
    local payload="$1"
    mkdir -p "$payload/bin"
    cat > "$payload/bin/qwen-prime-runtime" <<'EOF'
#!/bin/sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RUNTIME_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
export PYTHONPATH="$RUNTIME_ROOT/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
exec "$RUNTIME_ROOT/python/bin/python3.12" -m harness.runtime_cli "$@"
EOF
    chmod 755 "$payload/bin/qwen-prime-runtime"
}

verify_payload() {
    local payload="$1"
    local launcher="$payload/bin/qwen-prime-runtime"
    local python="$payload/python/bin/python3.12"

    [[ -x "$launcher" ]] || { echo "Missing executable launcher: $launcher" >&2; return 1; }
    [[ -x "$python" ]] || { echo "Missing embedded CPython: $python" >&2; return 1; }
    [[ -d "$payload/site-packages/harness" ]] || {
        echo "Missing qwen-prime-runtime package in $payload/site-packages" >&2
        return 1
    }
    if find "$payload" -type f \( -name '*.safetensors' -o -name '*.gguf' -o -name '*.mlx' \) -print -quit | grep -q .; then
        echo "Embedded runtime must not contain model weights." >&2
        return 1
    fi
    local absolute_link
    absolute_link="$(find "$payload" -type l -exec sh -c '
        for link do
            target=$(readlink "$link")
            case "$target" in /*) printf "%s -> %s\n" "$link" "$target"; exit 0;; esac
        done
    ' sh {} + | head -1)"
    if [[ -n "$absolute_link" ]]; then
        echo "$absolute_link" >&2
        echo "Embedded runtime contains an absolute symlink." >&2
        return 1
    fi
    echo "Embedded runtime payload verified: $payload"
}

case "${1:-}" in
    --write-launcher)
        [[ $# -eq 2 ]] || { echo "Usage: $0 --write-launcher PAYLOAD" >&2; exit 2; }
        write_launcher "$2"
        exit 0
        ;;
    --verify)
        [[ $# -eq 2 ]] || { echo "Usage: $0 --verify PAYLOAD" >&2; exit 2; }
        verify_payload "$2"
        exit 0
        ;;
esac

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required to build the embedded runtime." >&2
    exit 1
fi
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "The embedded MLX runtime must be built on Apple Silicon macOS." >&2
    exit 1
fi

OUTPUT="${1:-${QWEN_PRIME_RUNTIME_OUTPUT:-$PROJECT_DIR/dist/QwenPrimeRuntime}}"
case "$OUTPUT" in
    /|""|"$PROJECT_DIR")
        echo "Refusing unsafe runtime output path: $OUTPUT" >&2
        exit 1
        ;;
esac

UV_CACHE_DIR="${UV_CACHE_DIR:-/private/tmp/qwen-prime-uv-cache}"
export UV_CACHE_DIR
PYTHON_BIN="$(uv python find 3.12)"
PYTHON_ROOT="$($PYTHON_BIN -c 'from pathlib import Path; import sys; print(Path(sys.executable).resolve().parents[1])')"
BUILD_DIR="$(mktemp -d /private/tmp/qwenprime-runtime.XXXXXX)"
cleanup() {
    [[ "$BUILD_DIR" == /private/tmp/qwenprime-runtime.* ]] && rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

PAYLOAD="$BUILD_DIR/QwenPrimeRuntime"
mkdir -p "$PAYLOAD"
echo "Copying relocatable CPython 3.12..."
ditto "$PYTHON_ROOT" "$PAYLOAD/python"

echo "Installing the locked runtime environment..."
uv export --project "$PROJECT_DIR" --frozen --no-dev --no-editable --no-hashes \
    --output-file "$BUILD_DIR/requirements.txt"
(
    cd "$PROJECT_DIR"
    uv pip install \
        --python "$PAYLOAD/python/bin/python3.12" \
        --target "$PAYLOAD/site-packages" \
        --refresh-package qwen-prime-runtime \
        --requirements "$BUILD_DIR/requirements.txt"
)

find "$PAYLOAD" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$PAYLOAD" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
write_launcher "$PAYLOAD"

cat > "$PAYLOAD/runtime-build.json" <<EOF
{
  "runtime": "qwen-prime-runtime",
  "runtime_version": "0.1.1",
  "python": "3.12",
  "platform": "macos-arm64",
  "uv_lock_sha256": "$(shasum -a 256 "$PROJECT_DIR/uv.lock" | awk '{print $1}')"
}
EOF

verify_payload "$PAYLOAD"
"$PAYLOAD/bin/qwen-prime-runtime" --help >/dev/null

mkdir -p "${OUTPUT:h}"
if [[ -e "$OUTPUT" ]]; then
    rm -rf "$OUTPUT"
fi
ditto "$PAYLOAD" "$OUTPUT"
verify_payload "$OUTPUT"
echo "Embedded runtime created at $OUTPUT"
