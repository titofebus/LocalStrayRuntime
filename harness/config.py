"""Configuration constants and portable runtime paths."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SUITES_DIR = BASE_DIR / "harness" / "suites"
RESULTS_DIR = BASE_DIR / "results"
SANDBOX_TMP_DIR = BASE_DIR / ".sandbox_tmp"

QWEN_PRIME_DATA_DIR = Path(
    os.environ.get(
        "QWEN_PRIME_DATA_DIR",
        Path.home() / "Library" / "Application Support" / "QwenPrime",
    )
).expanduser()
DEFAULT_MLX_MODEL_PATH = os.environ.get(
    "QWEN_PRIME_TARGET_MODEL",
    str(QWEN_PRIME_DATA_DIR / "Models" / "Qwen3.8-27B-MLX-6bit"),
)
DEFAULT_MTP_MODEL_PATH = os.environ.get(
    "QWEN_PRIME_DRAFT_MODEL",
    str(QWEN_PRIME_DATA_DIR / "Models" / "Qwen3.8-27B-MTP-MLX-6bit"),
)
SOURCE_MODEL_PATH = Path(
    os.environ.get(
        "QWEN_PRIME_SOURCE_MODEL",
        QWEN_PRIME_DATA_DIR / "Models" / "Qwen3.8-27B",
    )
).expanduser()
TRAINING_DIR = Path(
    os.environ.get("QWEN_PRIME_TRAINING_DIR", QWEN_PRIME_DATA_DIR / "Training")
).expanduser()
TRAINING_OUTPUT_DIR = Path(
    os.environ.get(
        "QWEN_PRIME_TRAINING_OUTPUT",
        QWEN_PRIME_DATA_DIR / "Models" / "Qwen3.8-27B-DFlash-Experimental",
    )
).expanduser()
LEGACY_DRAFT_MODEL_PATH = os.environ.get(
    "QWEN_PRIME_LEGACY_DRAFT_MODEL",
    str(QWEN_PRIME_DATA_DIR / "Models" / "Qwen2.5-Coder-0.5B-Instruct-MLX-4bit"),
)

# Evaluation defaults
DEFAULT_MAX_TOKENS = 3072
DEFAULT_TEMPERATURE = 0.7
DEFAULT_EXECUTION_TIMEOUT = 10  # seconds per test run

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SANDBOX_TMP_DIR.mkdir(parents=True, exist_ok=True)
