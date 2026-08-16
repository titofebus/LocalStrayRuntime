"""Results recorder persisting evaluation runs, test logs, and scoreboards."""
import json
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from harness.config import RESULTS_DIR
from harness.core.models import ComparisonRecord, VerificationResult, ModelOutput

class ResultsRecorder:
    @staticmethod
    def get_run_file(challenge_id: str) -> Path:
        return RESULTS_DIR / f"run_{challenge_id}.json"

    @staticmethod
    def save_record(record: ComparisonRecord) -> Path:
        """Save a comparison record to JSON."""
        file_path = ResultsRecorder.get_run_file(record.challenge_id)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(record.model_dump_json(indent=2))
        return file_path

    @staticmethod
    def load_record(challenge_id: str) -> Optional[ComparisonRecord]:
        """Load a specific challenge run."""
        file_path = ResultsRecorder.get_run_file(challenge_id)
        if not file_path.exists():
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ComparisonRecord(**data)
        except Exception:
            return None

    @staticmethod
    def load_all_records() -> List[ComparisonRecord]:
        """Load all saved comparison records."""
        records = []
        for p in RESULTS_DIR.glob("run_*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                records.append(ComparisonRecord(**data))
            except Exception:
                continue
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records

    @staticmethod
    def compute_leaderboard() -> Dict[str, Any]:
        """Aggregate stats across all runs for Qwen and Opus."""
        records = ResultsRecorder.load_all_records()

        qwen_runs = 0
        qwen_passes = 0
        qwen_constraint_passes = 0
        qwen_total_tokens = 0
        qwen_avg_tps = 0.0

        opus_runs = 0
        opus_passes = 0
        opus_constraint_passes = 0

        qwen_wins = 0
        opus_wins = 0
        ties = 0

        tps_list = []

        for r in records:
            # Qwen stats
            qwen_runs += 1
            if r.qwen_verification.test_passed:
                qwen_passes += 1
            if r.qwen_verification.constraint_passed:
                qwen_constraint_passes += 1
            qwen_total_tokens += r.qwen_output.completion_tokens
            if r.qwen_output.tokens_per_sec > 0:
                tps_list.append(r.qwen_output.tokens_per_sec)

            # Opus stats
            if r.opus_verification is not None:
                opus_runs += 1
                if r.opus_verification.test_passed:
                    opus_passes += 1
                if r.opus_verification.constraint_passed:
                    opus_constraint_passes += 1

                # Head to head winner
                q_ok = r.qwen_verification.test_passed and r.qwen_verification.constraint_passed
                o_ok = r.opus_verification.test_passed and r.opus_verification.constraint_passed
                if q_ok and not o_ok:
                    qwen_wins += 1
                elif o_ok and not q_ok:
                    opus_wins += 1
                elif q_ok and o_ok:
                    ties += 1

        avg_tps = round(sum(tps_list) / max(len(tps_list), 1), 2)

        return {
            "total_challenges_evaluated": len(records),
            "qwen": {
                "runs": qwen_runs,
                "passed_tests": qwen_passes,
                "pass_rate": f"{(qwen_passes/max(qwen_runs,1))*100:.1f}%",
                "constraint_pass_rate": f"{(qwen_constraint_passes/max(qwen_runs,1))*100:.1f}%",
                "avg_tokens_per_sec": avg_tps,
                "wins": qwen_wins,
            },
            "opus": {
                "runs": opus_runs,
                "passed_tests": opus_passes,
                "pass_rate": f"{(opus_passes/max(opus_runs,1))*100:.1f}%" if opus_runs > 0 else "N/A",
                "constraint_pass_rate": f"{(opus_constraint_passes/max(opus_runs,1))*100:.1f}%" if opus_runs > 0 else "N/A",
                "wins": opus_wins,
            },
            "ties": ties,
        }
