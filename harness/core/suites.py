"""Challenge suite loader parsing YAML benchmarks."""
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from harness.config import SUITES_DIR
from harness.core.models import Challenge, Constraint

class SuiteLoader:
    @staticmethod
    def load_all_challenges() -> List[Challenge]:
        """Load all challenges from all YAML and JSON files in SUITES_DIR."""
        challenges = []
        # 1. YAML files
        for yaml_file in sorted(SUITES_DIR.glob("*.yaml")):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                suite_name = data.get("suite", yaml_file.stem)
                for item in data.get("challenges", []):
                    constraints = [
                        Constraint(**c) for c in item.get("constraints", [])
                    ]
                    challenges.append(Challenge(
                        id=item["id"],
                        suite=suite_name,
                        title=item["title"],
                        difficulty=item.get("difficulty", "medium"),
                        category=item.get("category", "engineering"),
                        language=item.get("language", "python"),
                        description=item.get("description", ""),
                        prompt=item["prompt"].strip(),
                        constraints=constraints,
                        test_code=item["test_code"].strip(),
                        expected_output_pattern=item.get("expected_output_pattern"),
                    ))
            except Exception as e:
                print(f"[Warning] Failed to parse {yaml_file}: {e}")

        # 2. JSON files
        import json
        for json_file in sorted(SUITES_DIR.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    items = json.load(f)
                if isinstance(items, dict):
                    items = items.get("challenges", [items])
                for item in items:
                    constraints = [
                        Constraint(**c) for c in item.get("constraints", [])
                    ]
                    challenges.append(Challenge(
                        id=item["id"],
                        suite=item.get("suite", json_file.stem),
                        title=item["title"],
                        difficulty=item.get("difficulty", "medium"),
                        category=item.get("category", "engineering"),
                        language=item.get("language", "python"),
                        description=item.get("description", ""),
                        prompt=item["prompt"].strip(),
                        constraints=constraints,
                        test_code=item["test_code"].strip(),
                        expected_output_pattern=item.get("expected_output_pattern"),
                    ))
            except Exception as e:
                print(f"[Warning] Failed to parse {json_file}: {e}")

        return challenges

    @staticmethod
    def get_challenge(challenge_id: str) -> Optional[Challenge]:
        """Find a specific challenge by ID."""
        for c in SuiteLoader.load_all_challenges():
            if c.id.lower() == challenge_id.lower():
                return c
        return None
