"""AST & Structural Invariant Validation Engine for Polyglot Code Generation."""
import ast
import re
from typing import Optional, List, Tuple
from dataclasses import dataclass

@dataclass
class ASTValidationResult:
    is_valid: bool
    language: str
    error_message: Optional[str] = None
    extracted_symbols: List[str] = None
    warnings: List[str] = None

class ASTEngine:
    """Polyglot AST and structural code invariant validator."""

    @staticmethod
    def validate_code(code: str, language: str) -> ASTValidationResult:
        lang = language.lower()
        if "python" in lang or "algo" in lang or "arch" in lang:
            return ASTEngine._validate_python(code)
        elif "rust" in lang:
            return ASTEngine._validate_rust(code)
        elif "swift" in lang:
            return ASTEngine._validate_swift(code)
        elif "ts" in lang or "typescript" in lang or "js" in lang:
            return ASTEngine._validate_typescript(code)
        elif "go" in lang or "golang" in lang:
            return ASTEngine._validate_go(code)
        elif "cpp" in lang or "c++" in lang:
            return ASTEngine._validate_cpp(code)
        elif "bash" in lang or "shell" in lang:
            return ASTEngine._validate_bash(code)
        return ASTValidationResult(is_valid=True, language=language, extracted_symbols=[])

    @staticmethod
    def _validate_go(code: str) -> ASTValidationResult:
        warnings = []
        if code.count("{") != code.count("}"):
            return ASTValidationResult(is_valid=False, language="go", error_message="Unbalanced curly braces in Go code.")
        symbols = re.findall(r"\bfunc\s+(?:\([^)]+\)\s+)?([A-Za-z0-9_]+)", code)
        return ASTValidationResult(is_valid=True, language="go", extracted_symbols=symbols, warnings=warnings)

    @staticmethod
    def _validate_cpp(code: str) -> ASTValidationResult:
        warnings = []
        if code.count("{") != code.count("}"):
            return ASTValidationResult(is_valid=False, language="cpp", error_message="Unbalanced curly braces in C++ code.")
        symbols = re.findall(r"\b(?:class|struct|template)\s+([A-Za-z0-9_]+)", code)
        return ASTValidationResult(is_valid=True, language="cpp", extracted_symbols=symbols, warnings=warnings)

    @staticmethod
    def _validate_python(code: str) -> ASTValidationResult:
        try:
            tree = ast.parse(code)
            symbols = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))]
            return ASTValidationResult(is_valid=True, language="python", extracted_symbols=symbols, warnings=[])
        except SyntaxError as e:
            return ASTValidationResult(
                is_valid=False,
                language="python",
                error_message=f"Python SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}",
                extracted_symbols=[]
            )

    @staticmethod
    def _validate_rust(code: str) -> ASTValidationResult:
        warnings = []
        # Check balanced braces
        if code.count("{") != code.count("}"):
            return ASTValidationResult(is_valid=False, language="rust", error_message="Unbalanced curly braces in Rust code.")
        if code.count("(") != code.count(")"):
            return ASTValidationResult(is_valid=False, language="rust", error_message="Unbalanced parentheses in Rust code.")

        # Check illegal derives on io::Error
        if re.search(r"#\[derive\(.*?(?:Clone|PartialEq|Eq).*?\)\].*?enum\s+\w+\s*\{[^}]*?io::Error", code, re.DOTALL):
            return ASTValidationResult(
                is_valid=False,
                language="rust",
                error_message="Illegal derive: Cannot derive Clone or PartialEq on enum containing std::io::Error."
            )

        # Extract structs, enums, functions
        symbols = re.findall(r"\b(?:pub\s+)?(?:struct|enum|trait|fn)\s+([A-Za-z0-9_]+)", code)
        return ASTValidationResult(is_valid=True, language="rust", extracted_symbols=symbols, warnings=warnings)

    @staticmethod
    def _validate_swift(code: str) -> ASTValidationResult:
        warnings = []
        if code.count("{") != code.count("}"):
            return ASTValidationResult(is_valid=False, language="swift", error_message="Unbalanced curly braces in Swift code.")

        # Check hallucinated generic labels like <Key: String, Value: Data>()
        if re.search(r"\b[A-Za-z0-9_]+<[A-Za-z0-9_]+:\s*[A-Za-z0-9_]+>\s*\(", code):
            return ASTValidationResult(
                is_valid=False,
                language="swift",
                error_message="Illegal generic label instantiation in Swift (e.g. `Type<Key: A>()` instead of `Type<A>()`)."
            )

        # Extract actors, structs, classes
        symbols = re.findall(r"\b(?:actor|struct|class|protocol|func)\s+([A-Za-z0-9_]+)", code)
        return ASTValidationResult(is_valid=True, language="swift", extracted_symbols=symbols, warnings=warnings)

    @staticmethod
    def _validate_typescript(code: str) -> ASTValidationResult:
        warnings = []
        if code.count("{") != code.count("}"):
            return ASTValidationResult(is_valid=False, language="typescript", error_message="Unbalanced curly braces in TypeScript code.")

        # Check while-loop off-by-one infinite loop hazard
        if re.search(r"while\s*\([^)]*\)\s*\{[^}]*?\.slice\(\s*lineEnd\s*\)", code):
            warnings.append("Potential infinite while loop: buffer.slice(lineEnd) without advancing index.")

        symbols = re.findall(r"\b(?:interface|class|type|function|const|enum)\s+([A-Za-z0-9_]+)", code)
        return ASTValidationResult(is_valid=True, language="typescript", extracted_symbols=symbols, warnings=warnings)

    @staticmethod
    def _validate_bash(code: str) -> ASTValidationResult:
        warnings = []
        if "set -euo pipefail" not in code and "set -e" not in code:
            warnings.append("Missing `set -euo pipefail` safety header in Bash script.")

        # Check interactive commands
        for bad_cmd in ["sudo", "read -p", "nano", "vim"]:
            if re.search(rf"\b{bad_cmd}\b", code):
                return ASTValidationResult(
                    is_valid=False,
                    language="bash",
                    error_message=f"Forbidden interactive command in non-interactive CLI script: `{bad_cmd}`"
                )

        symbols = re.findall(r"\b(?:function\s+)?([A-Za-z0-9_]+)\s*\(\)\s*\{", code)
        return ASTValidationResult(is_valid=True, language="bash", extracted_symbols=symbols, warnings=warnings)
