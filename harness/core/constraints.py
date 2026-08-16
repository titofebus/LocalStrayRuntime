"""Constraint validator checking strict negative rules, AST rules, and formatting."""
import re
import ast
from typing import List, Tuple
from harness.core.models import Constraint

class ConstraintChecker:
    @staticmethod
    def check_constraints(code_or_text: str, constraints: List[Constraint]) -> Tuple[bool, List[str]]:
        """Validate generated output against specified constraints."""
        violations = []

        for c in constraints:
            rule_type = c.rule_type.lower()
            val = c.value

            if rule_type == "forbidden_import":
                # Check Python AST for imports
                try:
                    tree = ast.parse(code_or_text)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                if alias.name == val or alias.name.startswith(f"{val}."):
                                    violations.append(f"Forbidden import used: '{alias.name}' ({c.description})")
                        elif isinstance(node, ast.ImportFrom):
                            if node.module == val or (node.module and node.module.startswith(f"{val}.")):
                                violations.append(f"Forbidden 'from {node.module} import ...' used ({c.description})")
                except SyntaxError:
                    # Fallback to regex check if code has surrounding text
                    if re.search(rf"\bimport\s+{re.escape(val)}\b", code_or_text) or re.search(rf"\bfrom\s+{re.escape(val)}\b", code_or_text):
                        violations.append(f"Forbidden import found: '{val}' ({c.description})")

            elif rule_type == "forbidden_pattern":
                if re.search(val, code_or_text, re.MULTILINE):
                    violations.append(f"Violated pattern constraint '{val}': {c.description}")

            elif rule_type == "required_pattern":
                if not re.search(val, code_or_text, re.MULTILINE):
                    violations.append(f"Missing required pattern '{val}': {c.description}")

            elif rule_type == "max_lines":
                try:
                    max_l = int(val)
                    lines = [l for l in code_or_text.strip().split("\n") if l.strip()]
                    if len(lines) > max_l:
                        violations.append(f"Output exceeded maximum {max_l} lines (got {len(lines)} lines)")
                except ValueError:
                    pass

        return len(violations) == 0, violations
