#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"
LOG_PATH="${QWEN_PRIME_LOG_PATH:-$HOME/Library/Logs/QwenPrime/runtime.log}"
PID_PATH="${QWEN_PRIME_PID_PATH:-$HOME/Library/Application Support/QwenPrime/runtime.pid}"
IDENTITY_URL="http://127.0.0.1:8000/v1/engine"
FORCE_RESTART=false
SERVER_ONLY=false

for argument in "$@"; do
    case "$argument" in
        --restart) FORCE_RESTART=true ;;
        --server-only) SERVER_ONLY=true ;;
        *) echo "Unknown option: $argument" >&2; exit 2 ;;
    esac
done

mkdir -p "${LOG_PATH:h}" "${PID_PATH:h}"

runtime_is_ready() {
    local identity_json
    identity_json="$(curl --fail --silent --max-time 2 "$IDENTITY_URL")" || return 1
    IDENTITY_JSON="$identity_json" /usr/bin/python3 - <<'PY'
import json
import os
import sys

identity = json.loads(os.environ["IDENTITY_JSON"])
required = {
    "runtime_id": "qwen38-native-mtp-v2",
    "target_model_id": "Qwen/Qwen3.8-27B",
    "draft_model_id": "Qwen/Qwen3.8-27B#native-mtp",
    "target_quantization": {
        "scheme": "mixed",
        "bits": [4, 8],
        "default_bits": 4,
        "group_size": 64,
        "mode": "affine",
    },
    "draft_quantization": {
        "scheme": "uniform",
        "bits": [6],
        "default_bits": 6,
        "group_size": 64,
        "mode": "affine",
    },
    "draft_model_type": "qwen3_8_mtp",
    "warmup_complete": True,
}
sys.exit(0 if required.items() <= identity.items() else 1)
PY
}

stop_owned_runtime() {
    [[ -f "$PID_PATH" ]] || return 0
    local pid
    pid="$(<"$PID_PATH")"
    [[ "$pid" == <-> ]] || { rm -f "$PID_PATH"; return 0; }
    if kill -0 "$pid" 2>/dev/null; then
        local command
        command="$(/bin/ps -p "$pid" -o command=)"
        if [[ "$command" != *"qwen-prime-runtime"* \
            && "$command" != *"harness.runtime_cli"* \
            && "$command" != *"harness.daemon.unified_server"* ]]; then
            echo "Refusing to stop PID $pid because it is not a Qwen Prime runtime: $command" >&2
            exit 1
        fi
        kill -TERM "$pid"
        for _ in {1..30}; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        kill -0 "$pid" 2>/dev/null && {
            echo "Runtime did not stop within 30 seconds; refusing to force-kill it." >&2
            exit 1
        }
    fi
    rm -f "$PID_PATH"
}

if [[ "$FORCE_RESTART" == true ]]; then
    stop_owned_runtime
fi

if ! runtime_is_ready; then
    if curl --fail --silent --max-time 1 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
        echo "Port 8000 is occupied by an unexpected runtime; refusing to replace it." >&2
        exit 1
    fi

    if command -v qwen-prime-runtime >/dev/null 2>&1; then
        RUNTIME_COMMAND=(qwen-prime-runtime serve)
    elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
        RUNTIME_COMMAND=("$PROJECT_DIR/.venv/bin/python" -m harness.runtime_cli serve)
    else
        echo "Install the runtime with scripts/install_qwen_prime_runtime.command" >&2
        exit 1
    fi

    echo "Starting Qwen3.8 27B + native MTP runtime..."
    cd "$PROJECT_DIR"
    nohup env PYTHONUNBUFFERED=1 "${RUNTIME_COMMAND[@]}" >"$LOG_PATH" 2>&1 &
    runtime_pid=$!
    echo "$runtime_pid" > "$PID_PATH"

    for attempt in {1..180}; do
        if runtime_is_ready; then
            echo "Qwen Prime runtime ready and identity verified (PID $runtime_pid)."
            break
        fi
        if ! kill -0 "$runtime_pid" 2>/dev/null; then
            echo "Runtime exited during startup. Log: $LOG_PATH" >&2
            tail -80 "$LOG_PATH" >&2
            exit 1
        fi
        [[ "$attempt" -eq 180 ]] && {
            echo "Runtime did not become ready within 180 seconds. Log: $LOG_PATH" >&2
            exit 1
        }
        sleep 1
    done
else
    echo "Qwen Prime runtime is already ready and identity verified."
fi

if [[ "$SERVER_ONLY" != true ]]; then
    if [[ -n "${QWEN_PRIME_APP_PATH:-}" ]]; then
        open "$QWEN_PRIME_APP_PATH"
    else
        open -a "Qwen Prime"
    fi
fi

echo "Runtime log: $LOG_PATH"
