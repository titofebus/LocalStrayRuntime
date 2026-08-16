"""Multi-language sandbox executor for Swift, Rust, TypeScript, and Python."""
import re
import sys
import time
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from harness.config import SANDBOX_TMP_DIR, DEFAULT_EXECUTION_TIMEOUT
from harness.core.models import Challenge, VerificationResult
from harness.core.constraints import ConstraintChecker

class CodeSandbox:
    @staticmethod
    def extract_multifile(raw_text: str, default_filename: str = "main") -> dict[str, str]:
        """Extract multi-file blocks demarcated by '// FILE: <name>' or '### File: <name>'."""
        files = {}
        pattern = r"(?://|#|/{2,3})\s*(?:FILE|File):\s*([A-Za-z0-9_\-\./]+)\s*\n(.*?)(?=(?://|#|/{2,3})\s*(?:FILE|File):|\Z)"
        matches = list(re.finditer(pattern, raw_text, re.DOTALL))
        if matches:
            for m in matches:
                filename = m.group(1).strip()
                content = m.group(2).strip()
                content = re.sub(r"^```\w*\s*\n", "", content)
                content = re.sub(r"\n```\s*$", "", content)
                files[filename] = content.strip()
        else:
            code = CodeSandbox.extract_code(raw_text)
            files[default_filename] = code
        return files

    @staticmethod
    def extract_code(raw_text: str, language: str = "python") -> str:
        """Extract code blocks for the specified language, handling both closed and open fences."""
        lang_pattern = r"(?:python|py)" if language == "python" else (
            r"swift" if language == "swift" else (
                r"rust" if language == "rust" else (
                    r"(?:typescript|ts|javascript|js)" if language in ("typescript", "ts") else (
                        r"(?:golang|go)" if language in ("go", "golang") else (
                            r"(?:cpp|c\+\+|cxx)" if language in ("cpp", "c++") else (
                                r"(?:bash|sh|shell)" if language in ("bash", "shell", "sh") else r"\w+"
                            )
                        )
                    )
                )
            )
        )
        matches = re.findall(rf"```{lang_pattern}\s*\n(.*?)```", raw_text, re.DOTALL | re.IGNORECASE)
        if matches:
            code = max(matches, key=lambda m: len(m.strip())).strip()
        else:
            generic = re.findall(r"```(?:\w+)?\s*\n(.*?)```", raw_text, re.DOTALL)
            if generic:
                code = max(generic, key=lambda g: len(g.strip())).strip()
            else:
                unclosed = re.search(rf"```{lang_pattern}?\s*\n(.*)", raw_text, re.DOTALL | re.IGNORECASE)
                if unclosed:
                    code = unclosed.group(1).strip()
                else:
                    code = raw_text.strip()

        clean_lines = [l for l in code.split("\n") if not l.strip().startswith("```")]
        return "\n".join(clean_lines).strip()

    @staticmethod
    def run_verification(
        model_name: str,
        raw_output: str,
        challenge: Challenge,
        timeout: int = DEFAULT_EXECUTION_TIMEOUT,
    ) -> VerificationResult:
        """Execute multi-language code against test suites in an isolated sandbox."""
        lang = getattr(challenge, "language", "python").lower()
        code = CodeSandbox.extract_code(raw_output, language=lang)

        # 1. Constraint checks
        constraint_passed, violations = ConstraintChecker.check_constraints(
            code_or_text=code,
            constraints=challenge.constraints,
        )

        if not code.strip():
            return VerificationResult(
                model_name=model_name,
                passed_tests=0,
                total_tests=1,
                test_passed=False,
                test_output="No executable code block found in response.",
                constraint_passed=constraint_passed,
                constraint_violations=violations,
                execution_time_seconds=0.0,
                error_message="Missing code block",
            )

        SANDBOX_TMP_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)

        if lang == "swift":
            return CodeSandbox._run_swift_tests(model_name, code, challenge, timestamp, timeout, constraint_passed, violations)
        elif lang == "rust":
            return CodeSandbox._run_rust_tests(model_name, code, challenge, timestamp, timeout, constraint_passed, violations)
        elif lang in ("typescript", "ts"):
            return CodeSandbox._run_ts_tests(model_name, code, challenge, timestamp, timeout, constraint_passed, violations)
        elif lang in ("go", "golang"):
            return CodeSandbox._run_go_tests(model_name, code, challenge, timestamp, timeout, constraint_passed, violations)
        elif lang in ("cpp", "c++"):
            return CodeSandbox._run_cpp_tests(model_name, code, challenge, timestamp, timeout, constraint_passed, violations)
        elif lang in ("bash", "shell", "sh"):
            return CodeSandbox._run_bash_tests(model_name, code, challenge, timestamp, timeout, constraint_passed, violations)
        else:
            return CodeSandbox._run_python_tests(model_name, code, challenge, timestamp, timeout, constraint_passed, violations)

    @staticmethod
    def _run_swift_tests(model_name, code, challenge, timestamp, timeout, constraint_passed, violations) -> VerificationResult:
        """Compile and execute Swift code using swiftc."""
        src_file = SANDBOX_TMP_DIR / f"test_{challenge.id}_{timestamp}.swift"
        bin_file = SANDBOX_TMP_DIR / f"test_{challenge.id}_{timestamp}"

        full_source = f"{code}\n\n// --- TEST SUITE ---\n{challenge.test_code}\n"
        with open(src_file, "w", encoding="utf-8") as f:
            f.write(full_source)

        start_time = time.perf_counter()
        try:
            # Compile Swift
            compile_res = subprocess.run(
                ["swiftc", "-parse-as-library", "-O", str(src_file), "-o", str(bin_file)],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if compile_res.returncode != 0:
                return VerificationResult(
                    model_name=model_name,
                    passed_tests=0,
                    total_tests=1,
                    test_passed=False,
                    test_output=f"Swift Compilation Error:\n{compile_res.stderr}",
                    constraint_passed=constraint_passed,
                    constraint_violations=violations,
                    execution_time_seconds=round(time.perf_counter() - start_time, 3),
                    error_message=compile_res.stderr.strip()[:300],
                )

            # Run binary
            run_res = subprocess.run(
                [str(bin_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            exec_time = round(time.perf_counter() - start_time, 3)
            passed = (run_res.returncode == 0)
            return VerificationResult(
                model_name=model_name,
                passed_tests=1 if passed else 0,
                total_tests=1,
                test_passed=passed and constraint_passed,
                test_output=run_res.stdout if passed else (run_res.stderr or run_res.stdout),
                constraint_passed=constraint_passed,
                constraint_violations=violations,
                execution_time_seconds=exec_time,
                error_message=None if passed else (run_res.stderr or "Tests failed"),
            )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                model_name=model_name,
                passed_tests=0,
                total_tests=len(challenge.test_cases) or 1,
                test_passed=False,
                test_output=f"Execution timed out after {timeout}s",
                constraint_passed=constraint_passed,
                constraint_violations=violations,
                execution_time_seconds=float(timeout),
                error_message="TimeoutExpired",
            )
        finally:
            for p in (src_file, bin_file):
                if p.exists():
                    p.unlink(missing_ok=True)

    @staticmethod
    def _run_rust_tests(model_name, code, challenge, timestamp, timeout, constraint_passed, violations) -> VerificationResult:
        """Compile and execute Rust code using rustc or cargo."""
        full_source = f"{code}\n\n// --- TEST SUITE ---\n{challenge.test_code}\n"
        start_time = time.perf_counter()

        # If external crates like tokio are requested, use cargo release build
        if "tokio" in full_source or "serde" in full_source:
            crate_dir = SANDBOX_TMP_DIR / f"cargo_{challenge.id}_{timestamp}"
            crate_dir.mkdir(parents=True, exist_ok=True)
            (crate_dir / "Cargo.toml").write_text("""[package]
name = "sandbox_test"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1", features = ["full"] }
""")
            src_dir = crate_dir / "src"
            src_dir.mkdir(exist_ok=True)
            (src_dir / "main.rs").write_text(full_source)

            try:
                compile_res = subprocess.run(
                    ["cargo", "build", "--release", "--manifest-path", str(crate_dir / "Cargo.toml")],
                    capture_output=True,
                    text=True,
                    timeout=45,
                )
                if compile_res.returncode != 0:
                    return VerificationResult(
                        model_name=model_name,
                        passed_tests=0,
                        total_tests=1,
                        test_passed=False,
                        test_output=f"Rust Compilation Error:\n{compile_res.stderr}",
                        constraint_passed=constraint_passed,
                        constraint_violations=violations,
                        execution_time_seconds=round(time.perf_counter() - start_time, 3),
                        error_message=compile_res.stderr.strip()[:300],
                    )

                bin_file = crate_dir / "target" / "release" / "sandbox_test"
                run_res = subprocess.run([str(bin_file)], capture_output=True, text=True, timeout=timeout)
                exec_time = round(time.perf_counter() - start_time, 3)
                passed = (run_res.returncode == 0)
                return VerificationResult(
                    model_name=model_name,
                    passed_tests=1 if passed else 0,
                    total_tests=1,
                    test_passed=passed and constraint_passed,
                    test_output=run_res.stdout if passed else (run_res.stderr or run_res.stdout),
                    constraint_passed=constraint_passed,
                    constraint_violations=violations,
                    execution_time_seconds=exec_time,
                    error_message=None if passed else (run_res.stderr or "Tests failed"),
                )
            finally:
                if crate_dir.exists():
                    shutil.rmtree(crate_dir, ignore_errors=True)

        src_file = SANDBOX_TMP_DIR / f"test_{challenge.id}_{timestamp}.rs"
        bin_file = SANDBOX_TMP_DIR / f"test_{challenge.id}_{timestamp}"
        with open(src_file, "w", encoding="utf-8") as f:
            f.write(full_source)

        start_time = time.perf_counter()
        try:
            compile_res = subprocess.run(
                ["rustc", "--edition", "2021", "-O", str(src_file), "-o", str(bin_file)],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if compile_res.returncode != 0:
                return VerificationResult(
                    model_name=model_name,
                    passed_tests=0,
                    total_tests=1,
                    test_passed=False,
                    test_output=f"Rust Compilation Error:\n{compile_res.stderr}",
                    constraint_passed=constraint_passed,
                    constraint_violations=violations,
                    execution_time_seconds=round(time.perf_counter() - start_time, 3),
                    error_message=compile_res.stderr.strip()[:300],
                )

            run_res = subprocess.run(
                [str(bin_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            exec_time = round(time.perf_counter() - start_time, 3)
            passed = (run_res.returncode == 0)
            return VerificationResult(
                model_name=model_name,
                passed_tests=1 if passed else 0,
                total_tests=1,
                test_passed=passed and constraint_passed,
                test_output=run_res.stdout if passed else (run_res.stderr or run_res.stdout),
                constraint_passed=constraint_passed,
                constraint_violations=violations,
                execution_time_seconds=exec_time,
                error_message=None if passed else (run_res.stderr or "Tests failed"),
            )
        except Exception as e:
            return VerificationResult(
                model_name=model_name,
                passed_tests=0,
                total_tests=len(challenge.test_cases) or 1,
                test_passed=False,
                test_output=str(e),
                constraint_passed=constraint_passed,
                constraint_violations=violations,
                execution_time_seconds=round(time.perf_counter() - start_time, 3),
                error_message=str(e),
            )
        finally:
            for p in (src_file, bin_file):
                if p.exists():
                    p.unlink(missing_ok=True)

    @staticmethod
    def _run_ts_tests(model_name, code, challenge, timestamp, timeout, constraint_passed, violations) -> VerificationResult:
        """Execute TypeScript natively using Node.js v25 --experimental-strip-types."""
        src_file = SANDBOX_TMP_DIR / f"test_{challenge.id}_{timestamp}.ts"
        # Strip relative imports and export statements for unified single-module test execution
        clean_code = re.sub(r"import\s+.*?from\s+['\"]./.*?['\"];?", "", code)
        clean_code = re.sub(r"export\s+default\s+.*?;?", "", clean_code)
        clean_code = re.sub(r"\bexport\s+(?:interface|class|type|function|const|let|var|enum)\b", lambda m: m.group(0).replace("export ", ""), clean_code)

        clean_test = re.sub(r"import\s+.*?from\s+['\"]./.*?['\"];?", "", challenge.test_code)
        full_source = f"{clean_code}\n\n// --- TEST SUITE ---\n{clean_test}\n"
        with open(src_file, "w", encoding="utf-8") as f:
            f.write(full_source)

        start_time = time.perf_counter()
        try:
            run_res = subprocess.run(
                ["node", "--experimental-strip-types", str(src_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            exec_time = round(time.perf_counter() - start_time, 3)
            passed = (run_res.returncode == 0)
            return VerificationResult(
                model_name=model_name,
                passed_tests=1 if passed else 0,
                total_tests=1,
                test_passed=passed and constraint_passed,
                test_output=run_res.stdout if passed else (run_res.stderr or run_res.stdout),
                constraint_passed=constraint_passed,
                constraint_violations=violations,
                execution_time_seconds=exec_time,
                error_message=None if passed else (run_res.stderr or "Tests failed"),
            )
        except Exception as e:
            return VerificationResult(
                model_name=model_name,
                passed_tests=0,
                total_tests=1,
                test_passed=False,
                test_output=str(e),
                constraint_passed=constraint_passed,
                constraint_violations=violations,
                execution_time_seconds=round(time.perf_counter() - start_time, 3),
                error_message=str(e),
            )
        finally:
            if src_file.exists():
                src_file.unlink(missing_ok=True)

    @staticmethod
    def _run_python_tests(model_name, code, challenge, timestamp, timeout, constraint_passed, violations) -> VerificationResult:
        """Execute Python code."""
        test_script_path = SANDBOX_TMP_DIR / f"test_run_{challenge.id}_{timestamp}.py"
        full_code = f"{code}\n\n# --- TEST SUITE ---\n{challenge.test_code}\n"
        with open(test_script_path, "w", encoding="utf-8") as f:
            f.write(full_code)

        start_time = time.perf_counter()
        try:
            res = subprocess.run(
                [sys.executable, str(test_script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            exec_time = round(time.perf_counter() - start_time, 3)
            passed = (res.returncode == 0)
            return VerificationResult(
                model_name=model_name,
                passed_tests=1 if passed else 0,
                total_tests=1,
                test_passed=passed and constraint_passed,
                test_output=res.stdout if passed else (res.stderr or res.stdout),
                constraint_passed=constraint_passed,
                constraint_violations=violations,
                execution_time_seconds=exec_time,
                error_message=None if passed else (res.stderr or "Tests failed"),
            )
        except Exception as e:
            return VerificationResult(
                model_name=model_name,
                passed_tests=0,
                total_tests=1,
                test_passed=False,
                test_output=str(e),
                constraint_passed=constraint_passed,
                constraint_violations=violations,
                execution_time_seconds=round(time.perf_counter() - start_time, 3),
                error_message=str(e),
            )
        finally:
            if test_script_path.exists():
                test_script_path.unlink(missing_ok=True)

    @staticmethod
    def _run_go_tests(model_name, code, challenge, timestamp, timeout, constraint_passed, violations) -> VerificationResult:
        """Compile and execute Go code with cleanly merged imports."""
        src_file = SANDBOX_TMP_DIR / f"test_{challenge.id}_{timestamp}.go"
        raw_combined = f"{code}\n\n{challenge.test_code}"

        import_lines = set()
        for m in re.finditer(r'import\s+"([^"]+)"', raw_combined):
            import_lines.add(f'"{m.group(1)}"')
        for m in re.finditer(r'import\s*\((.*?)\)', raw_combined, re.DOTALL):
            for line in m.group(1).splitlines():
                pkg = line.strip()
                if pkg and not pkg.startswith("//"):
                    import_lines.add(pkg)

        clean_code = re.sub(r'package\s+\w+', '', code)
        clean_code = re.sub(r'import\s*\([^)]*\)', '', clean_code, flags=re.DOTALL)
        clean_code = re.sub(r'import\s+"[^"]+"', '', clean_code)

        clean_test = re.sub(r'package\s+\w+', '', challenge.test_code)
        clean_test = re.sub(r'import\s*\([^)]*\)', '', clean_test, flags=re.DOTALL)
        clean_test = re.sub(r'import\s+"[^"]+"', '', clean_test)

        imports_block = "\n".join(f"\t{imp}" for imp in sorted(import_lines))
        full_source = f"package main\n\nimport (\n{imports_block}\n)\n\n{clean_code}\n\n// --- TEST SUITE ---\n{clean_test}\n"
        with open(src_file, "w", encoding="utf-8") as f:
            f.write(full_source)

        start_time = time.perf_counter()
        try:
            run_res = subprocess.run(
                ["go", "run", str(src_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            exec_time = round(time.perf_counter() - start_time, 3)
            passed = (run_res.returncode == 0)
            return VerificationResult(
                model_name=model_name,
                passed_tests=1 if passed else 0,
                total_tests=1,
                test_passed=passed and constraint_passed,
                test_output=run_res.stdout if passed else (run_res.stderr or run_res.stdout),
                constraint_passed=constraint_passed,
                constraint_violations=violations,
                execution_time_seconds=exec_time,
                error_message=None if passed else (run_res.stderr or "Go execution failed"),
            )
        except Exception as e:
            return VerificationResult(
                model_name=model_name,
                passed_tests=0,
                total_tests=1,
                test_passed=False,
                test_output=str(e),
                constraint_passed=constraint_passed,
                constraint_violations=violations,
                execution_time_seconds=round(time.perf_counter() - start_time, 3),
                error_message=str(e),
            )
        finally:
            if src_file.exists():
                src_file.unlink(missing_ok=True)

    @staticmethod
    def _run_cpp_tests(model_name, code, challenge, timestamp, timeout, constraint_passed, violations) -> VerificationResult:
        """Compile and execute C++20 code."""
        src_file = SANDBOX_TMP_DIR / f"test_{challenge.id}_{timestamp}.cpp"
        bin_file = SANDBOX_TMP_DIR / f"test_{challenge.id}_{timestamp}"
        full_source = f"{code}\n\n// --- TEST SUITE ---\n{challenge.test_code}\n"
        with open(src_file, "w", encoding="utf-8") as f:
            f.write(full_source)

        start_time = time.perf_counter()
        try:
            compile_res = subprocess.run(
                ["clang++", "-std=c++20", "-O3", str(src_file), "-o", str(bin_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if compile_res.returncode != 0:
                return VerificationResult(
                    model_name=model_name,
                    passed_tests=0,
                    total_tests=1,
                    test_passed=False,
                    test_output=compile_res.stderr,
                    constraint_passed=constraint_passed,
                    constraint_violations=violations,
                    execution_time_seconds=round(time.perf_counter() - start_time, 3),
                    error_message=f"C++ Compilation Error: {compile_res.stderr[:200]}",
                )

            run_res = subprocess.run([str(bin_file)], capture_output=True, text=True, timeout=timeout)
            exec_time = round(time.perf_counter() - start_time, 3)
            passed = (run_res.returncode == 0)
            return VerificationResult(
                model_name=model_name,
                passed_tests=1 if passed else 0,
                total_tests=1,
                test_passed=passed and constraint_passed,
                test_output=run_res.stdout if passed else (run_res.stderr or run_res.stdout),
                constraint_passed=constraint_passed,
                constraint_violations=violations,
                execution_time_seconds=exec_time,
                error_message=None if passed else (run_res.stderr or "C++ Test failed"),
            )
        except Exception as e:
            return VerificationResult(
                model_name=model_name,
                passed_tests=0,
                total_tests=1,
                test_passed=False,
                test_output=str(e),
                constraint_passed=constraint_passed,
                constraint_violations=violations,
                execution_time_seconds=round(time.perf_counter() - start_time, 3),
                error_message=str(e),
            )
        finally:
            for p in (src_file, bin_file):
                if p.exists():
                    p.unlink(missing_ok=True)

    @staticmethod
    def _run_bash_tests(model_name, code, challenge, timestamp, timeout, constraint_passed, violations) -> VerificationResult:
        """Execute Bash script."""
        src_file = SANDBOX_TMP_DIR / f"test_{challenge.id}_{timestamp}.sh"
        full_source = f"{code}\n\n# --- TEST SUITE ---\n{challenge.test_code}\n"
        with open(src_file, "w", encoding="utf-8") as f:
            f.write(full_source)

        start_time = time.perf_counter()
        try:
            run_res = subprocess.run(["bash", str(src_file)], capture_output=True, text=True, timeout=timeout)
            exec_time = round(time.perf_counter() - start_time, 3)
            passed = (run_res.returncode == 0)
            return VerificationResult(
                model_name=model_name,
                passed_tests=1 if passed else 0,
                total_tests=1,
                test_passed=passed and constraint_passed,
                test_output=run_res.stdout if passed else (run_res.stderr or run_res.stdout),
                constraint_passed=constraint_passed,
                constraint_violations=violations,
                execution_time_seconds=exec_time,
                error_message=None if passed else (run_res.stderr or "Bash script failed"),
            )
        except Exception as e:
            return VerificationResult(
                model_name=model_name,
                passed_tests=0,
                total_tests=1,
                test_passed=False,
                test_output=str(e),
                constraint_passed=constraint_passed,
                constraint_violations=violations,
                execution_time_seconds=round(time.perf_counter() - start_time, 3),
                error_message=str(e),
            )
        finally:
            if src_file.exists():
                src_file.unlink(missing_ok=True)
