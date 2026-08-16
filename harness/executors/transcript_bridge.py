"""Bridge that automatically detects and reads completions from your open Opus Antigravity window."""
import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple
from harness.core.models import ModelOutput

BRAIN_DIR = Path(
    os.environ.get(
        "ANTIGRAVITY_BRAIN_DIR",
        Path.home() / ".gemini" / "antigravity" / "brain",
    )
).expanduser()
CURRENT_CONVERSATION_ID = os.environ.get("ANTIGRAVITY_CONVERSATION_ID", "")

class AntigravityTranscriptBridge:
    @staticmethod
    def get_opus_conversation_dirs() -> list[Path]:
        """Find recent conversations that are not the current harness conversation."""
        if not BRAIN_DIR.is_dir():
            return []
        dirs = []
        for p in BRAIN_DIR.iterdir():
            if p.is_dir() and p.name != CURRENT_CONVERSATION_ID and p.name != "tempmediaStorage":
                transcript = p / ".system_generated" / "logs" / "transcript.jsonl"
                if transcript.exists():
                    dirs.append(p)
        # Sort by latest modification
        dirs.sort(key=lambda d: (d / ".system_generated" / "logs" / "transcript.jsonl").stat().st_mtime, reverse=True)
        return dirs

    @staticmethod
    def read_latest_opus_response(timeout_seconds: int = 120, min_mtime: float = 0.0) -> Optional[ModelOutput]:
        """Poll the open Opus window transcript for the next completed response."""
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            dirs = AntigravityTranscriptBridge.get_opus_conversation_dirs()
            if not dirs:
                time.sleep(1.0)
                continue

            target_dir = dirs[0]
            transcript_path = target_dir / ".system_generated" / "logs" / "transcript.jsonl"

            if transcript_path.stat().st_mtime > min_mtime:
                # Read transcript lines
                try:
                    with open(transcript_path, "r", encoding="utf-8") as f:
                        lines = [line.strip() for line in f if line.strip()]

                    # Look backwards for the last PLANNER_RESPONSE
                    for line in reversed(lines):
                        data = json.loads(line)
                        if data.get("type") == "PLANNER_RESPONSE" and data.get("status") == "DONE":
                            content = data.get("content", "")
                            thinking = data.get("thinking", "")

                            # Extract code blocks
                            import re
                            code = None
                            code_matches = re.findall(r"```(?:python|py)?\s*\n(.*?)```", content, re.DOTALL)
                            if code_matches:
                                code = "\n\n".join(c.strip() for c in code_matches)
                            elif "def " in content or "class " in content:
                                code = content.strip()

                            return ModelOutput(
                                model_name="Claude-4.6-Opus",
                                raw_response=content,
                                thinking_content=thinking,
                                code_content=code,
                                completion_tokens=len(content.split()),
                            )
                except Exception:
                    pass

            time.sleep(1.0)
        return None
