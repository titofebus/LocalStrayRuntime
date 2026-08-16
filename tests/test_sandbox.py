"""Unit test for sandbox and constraint validator."""
import pytest
from harness.core.models import Challenge, Constraint
from harness.core.sandbox import CodeSandbox

def test_sandbox_pass():
    challenge = Challenge(
        id="test_01",
        suite="test",
        title="Add Function",
        difficulty="easy",
        category="math",
        description="Add two numbers",
        prompt="Write a function add(a, b)",
        constraints=[Constraint(name="no_math", rule_type="forbidden_import", value="math", description="No math module")],
        test_code="assert add(2, 3) == 5\nassert add(-1, 1) == 0",
    )

    code = "```python\ndef add(a, b):\n    return a + b\n```"
    result = CodeSandbox.run_verification("test_model", code, challenge)
    assert result.test_passed is True
    assert result.constraint_passed is True

def test_sandbox_forbidden_import_fail():
    challenge = Challenge(
        id="test_02",
        suite="test",
        title="Add Function",
        difficulty="easy",
        category="math",
        description="Add two numbers",
        prompt="Write a function add(a, b)",
        constraints=[Constraint(name="no_math", rule_type="forbidden_import", value="math", description="No math module")],
        test_code="assert add(2, 3) == 5",
    )

    code = "```python\nimport math\ndef add(a, b):\n    return a + b\n```"
    result = CodeSandbox.run_verification("test_model", code, challenge)
    assert result.test_passed is False
    assert result.constraint_passed is False
    assert len(result.constraint_violations) == 1
