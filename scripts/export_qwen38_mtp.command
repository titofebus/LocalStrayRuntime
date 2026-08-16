#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
HARNESS_DIR="${SCRIPT_DIR:h}"
cd "$HARNESS_DIR"
LOG_PATH="/private/tmp/qwen38-mtp-export.log"
OUTPUT_PATH="${QWEN_PRIME_DRAFT_MODEL:-$HOME/Library/Application Support/QwenPrime/Models/Qwen3.8-27B-MTP-MLX-6bit}"

echo "Starting Qwen3.8 native MTP 6-bit export..." | tee "$LOG_PATH"
if [[ -d "$OUTPUT_PATH" ]]; then
    BACKUP_PATH="${OUTPUT_PATH}-invalid-norm-offset-$(date +%Y%m%d-%H%M%S)"
    mv "$OUTPUT_PATH" "$BACKUP_PATH"
    echo "Preserved previous artifact: $BACKUP_PATH" | tee -a "$LOG_PATH"
fi
.venv/bin/python -m harness.trainer.export_qwen38_mtp 2>&1 | tee -a "$LOG_PATH"
echo "Export complete. Log: $LOG_PATH" | tee -a "$LOG_PATH"
