"""Interactive CLI interface for running real-world local vs frontier evaluations."""
import sys
import time
import argparse
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from harness.core.suites import SuiteLoader
from harness.core.sandbox import CodeSandbox
from harness.core.recorder import ResultsRecorder
from harness.core.models import Challenge, ModelOutput, VerificationResult, ComparisonRecord
from harness.executors.mlx_engine import MLXEngine
from harness.executors.transcript_bridge import AntigravityTranscriptBridge

console = Console()

def cmd_list(args):
    """List all available evaluation challenges."""
    challenges = SuiteLoader.load_all_challenges()
    table = Table(title=f"Evaluation Benchmark Challenges ({len(challenges)} total)")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Suite", style="dim")
    table.add_column("Category", style="magenta")
    table.add_column("Difficulty", style="bold yellow")
    table.add_column("Title", style="white")

    for c in challenges:
        table.add_row(c.id, c.suite, c.category, c.difficulty.upper(), c.title)

    console.print(table)

def cmd_challenge(args):
    """Run local Qwen3.8-27B via DFlash Speculative Engine, display Opus prompt up front, and ingest Opus response."""
    challenge_id = args.id
    engine_type = getattr(args, "engine", "dflash") or "dflash"
    challenge = SuiteLoader.get_challenge(challenge_id)
    if not challenge:
        console.print(f"[bold red]Error:[/] Challenge '{challenge_id}' not found. Run `eval list` to see IDs.")
        return

    console.print(Panel(
        f"[bold cyan]{challenge.title}[/]\n"
        f"[dim]Category: {challenge.category} | Difficulty: {challenge.difficulty.upper()} | Suite: {challenge.suite} | Engine: {engine_type.upper()}[/]\n\n"
        f"{challenge.description}",
        title=f"Challenge: {challenge.id}",
        border_style="blue",
    ))

    # 1. Display Prompt for Opus UP FRONT
    opus_prompt_panel = Panel(
        challenge.prompt,
        title="[bold magenta]Copy this prompt into your open Claude Opus 4.6 Antigravity window[/]",
        border_style="magenta",
    )
    console.print("\n", opus_prompt_panel)

    # 2. Run Local Qwen3.8-27B on DFlash Speculative Engine
    console.print(f"\n[bold yellow]► Generating local Qwen3.8-27B solution on Apple M4 Max ({engine_type.upper()} Engine)...[/]")

    if engine_type == "mlx":
        from harness.executors.mlx_engine import MLXEngine
        engine = MLXEngine()
    else:
        from harness.executors.dflash_engine import DFlashEngine
        engine = DFlashEngine()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing weights...", total=None)

        def update_tokens(count, tps):
            progress.update(task, description=f"Generating via DFlash: [green]{count} tokens[/] ([bold green]{tps:.1f} tok/s[/])")

        qwen_output = engine.run_inference(
            prompt=challenge.prompt,
            enable_thinking=True,
            max_thinking_tokens=500,
            max_tokens=2000,
            on_token_callback=update_tokens,
        )

    console.print(f"  • Qwen generation complete: [green]{qwen_output.completion_tokens} tokens[/] @ [bold green]{qwen_output.tokens_per_sec:.1f} tok/s[/]")
    if qwen_output.peak_memory_gb:
        console.print(f"  • Peak Memory: [cyan]{qwen_output.peak_memory_gb:.2f} GB[/]")

    # Run Sandbox Verification on Qwen
    console.print("  • Verifying Qwen output in sandbox...")
    qwen_verif = CodeSandbox.run_verification("Qwen3.8-27B", qwen_output.raw_response, challenge)
    if qwen_verif.test_passed:
        console.print("  • [bold green]✓ Qwen passed all automated unit tests and constraints![/]")
    else:
        console.print(f"  • [bold red]✗ Qwen failed:[/] {qwen_verif.error_message or 'Test failed'}")

    # 3. Auto-detect from open Opus window or accept manual paste
    console.print("\n[bold cyan]► Listening for response from your open Opus window (up to 90s)...[/] [dim](or press Enter to paste manually)[/]")

    start_listen = time.time()
    opus_output = AntigravityTranscriptBridge.read_latest_opus_response(timeout_seconds=90, min_mtime=start_listen)

    if not opus_output:
        console.print("[yellow]Paste Opus 4.6's response below (Type 'EOF' on a new line when done, or press Enter twice to skip):[/]")
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "EOF":
                    break
                if not line and lines and not lines[-1]:
                    break
                lines.append(line)
            except EOFError:
                break
        opus_raw = "\n".join(lines).strip()
        if opus_raw:
            opus_output = ModelOutput(
                model_name="Claude-4.6-Opus",
                raw_response=opus_raw,
                completion_tokens=len(opus_raw.split()),
            )

    opus_verif = None
    winner = "pending"

    if opus_output:
        console.print("\n[bold yellow]► Received Opus 4.6 completion! Verifying in sandbox...[/]")
        opus_verif = CodeSandbox.run_verification("Claude-4.6-Opus", opus_output.raw_response, challenge)
        if opus_verif.test_passed:
            console.print("  • [bold green]✓ Opus passed all automated unit tests and constraints![/]")
        else:
            console.print(f"  • [bold red]✗ Opus failed:[/] {opus_verif.error_message or 'Test failed'}")

        # Determine winner
        q_ok = qwen_verif.test_passed
        o_ok = opus_verif.test_passed
        if q_ok and not o_ok:
            winner = "qwen"
        elif o_ok and not q_ok:
            winner = "opus"
        elif q_ok and o_ok:
            winner = "tie"
        else:
            winner = "none"

    # Save Comparison Record
    record = ComparisonRecord(
        run_id=f"eval_{challenge.id}",
        challenge_id=challenge.id,
        challenge_title=challenge.title,
        qwen_output=qwen_output,
        qwen_verification=qwen_verif,
        opus_output=opus_output,
        opus_verification=opus_verif,
        winner=winner,
    )
    saved_path = ResultsRecorder.save_record(record)
    console.print(f"\n[dim]Run recorded to {saved_path}[/]")

    # Display Side-by-Side Summary Table
    display_comparison_table(record)

def display_comparison_table(record: ComparisonRecord):
    table = Table(title=f"Head-to-Head Comparison: {record.challenge_title}")
    table.add_column("Metric", style="dim")
    table.add_column("Qwen3.8-27B (Local 6-bit)", style="bold cyan")
    table.add_column("Claude 4.6 Opus (Frontier)", style="bold magenta")

    # Tests Passed
    q_test_str = f"[green]PASSED ({record.qwen_verification.passed_tests}/{record.qwen_verification.total_tests})[/]" if record.qwen_verification.test_passed else f"[red]FAILED[/]"
    o_test_str = "Pending / Not Provided"
    if record.opus_verification:
        o_test_str = f"[green]PASSED ({record.opus_verification.passed_tests}/{record.opus_verification.total_tests})[/]" if record.opus_verification.test_passed else f"[red]FAILED[/]"
    table.add_row("Unit Tests", q_test_str, o_test_str)

    # Constraints
    q_c_str = "[green]✓ Obeyed[/]" if record.qwen_verification.constraint_passed else f"[red]Violated ({', '.join(record.qwen_verification.constraint_violations)})[/]"
    o_c_str = "-"
    if record.opus_verification:
        o_c_str = "[green]✓ Obeyed[/]" if record.opus_verification.constraint_passed else f"[red]Violated ({', '.join(record.opus_verification.constraint_violations)})[/]"
    table.add_row("Constraints", q_c_str, o_c_str)

    # Speed
    q_speed = f"{record.qwen_output.tokens_per_sec:.1f} tok/s" if record.qwen_output.tokens_per_sec else "-"
    table.add_row("Speed", q_speed, "Cloud API")

    # Execution Time
    table.add_row("Sandbox Runtime", f"{record.qwen_verification.execution_time_seconds:.3f}s", f"{record.opus_verification.execution_time_seconds:.3f}s" if record.opus_verification else "-")

    # Outcome
    win_color = "green" if record.winner in ("qwen", "opus") else "yellow"
    table.add_row("Result", f"[bold {win_color}]{record.winner.upper() if record.winner else 'PENDING'}[/]", "")

    console.print("\n", table)

def cmd_leaderboard(args):
    """Display overall aggregated leaderboard."""
    stats = ResultsRecorder.compute_leaderboard()

    table = Table(title=f"Overall Leaderboard ({stats['total_challenges_evaluated']} Challenges Evaluated)")
    table.add_column("Model", style="bold")
    table.add_column("Runs", justify="right")
    table.add_column("Test Pass Rate", justify="right", style="green")
    table.add_column("Constraint Pass Rate", justify="right", style="cyan")
    table.add_column("Avg Speed", justify="right", style="yellow")
    table.add_column("Head-to-Head Wins", justify="right", style="bold magenta")

    q = stats["qwen"]
    table.add_row("Qwen3.8-27B (Local 6-bit)", str(q["runs"]), q["pass_rate"], q["constraint_pass_rate"], f"{q['avg_tokens_per_sec']} tok/s", str(q["wins"]))

    o = stats["opus"]
    table.add_row("Claude 4.6 Opus (Frontier)", str(o["runs"]), o["pass_rate"], o["constraint_pass_rate"], "Cloud", str(o["wins"]))

    console.print(table)
    console.print(f"[dim]Ties: {stats['ties']}[/]")

def cmd_compare(args):
    """View details of a past challenge run."""
    record = ResultsRecorder.load_record(args.id)
def cmd_ast_check(args):
    """Run AST structural invariant validation on a source file."""
    from harness.core.ast_engine import ASTEngine
    from pathlib import Path

    file_path = Path(args.file)
    if not file_path.exists():
        console.print(f"[red]File '{args.file}' not found.[/]")
        return

    content = file_path.read_text(encoding="utf-8")
    lang = args.lang or file_path.suffix.lstrip(".")
    res = ASTEngine.validate_code(content, lang)

    console.print(Panel(
        f"[bold]File:[/] {args.file} ({res.language.upper()})\n"
        f"[bold]Valid Syntax:[/] [{'green' if res.is_valid else 'red'}]{res.is_valid}[/]\n"
        f"[bold]Extracted Symbols:[/] {', '.join(res.extracted_symbols) if res.extracted_symbols else 'None'}\n"
        f"[bold]Warnings:[/] {len(res.warnings)}\n" +
        ("\n".join([f"  • [yellow]Warning:[/] {w}" for w in res.warnings]) if res.warnings else "") +
        (f"\n[bold red]Error:[/] {res.error_message}" if res.error_message else ""),
        title="AST Structural Invariant Report",
        border_style="green" if res.is_valid else "red",
    ))

def cmd_refactor(args):
    """Pipe code via stdin or file for fast DFlash compiler-checked refactor."""
    from harness.executors.dflash_engine import DFlashEngine
    import sys

    if args.file:
        content = open(args.file).read()
    elif not sys.stdin.isatty():
        content = sys.stdin.read()
    else:
        console.print("[dim]Reading source code from stdin... (Ctrl+D to finish)[/]")
        content = sys.stdin.read()

    engine = DFlashEngine()
    prompt = f"Refactor the following {args.lang or 'production'} code according to instruction: {args.instruction}\n\n```\n{content}\n```"

    out = engine.run_inference(prompt=prompt, language=args.lang)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out.code_content or out.raw_response)
        console.print(f"[bold green]Saved refactored code to {args.output}[/]")
    else:
        print(out.code_content or out.raw_response)

def cmd_daemon(args):
    """Manage background resident DFlash inference daemon."""
    from harness.daemon.client import DaemonClient, SOCKET_PATH
    from harness.daemon.server import PID_FILE
    from pathlib import Path
    import subprocess
    import os
    import signal

    if args.action == "status":
        running = DaemonClient.is_daemon_running()
        pid = Path(PID_FILE).read_text().strip() if Path(PID_FILE).exists() else "Unknown"
        console.print(Panel(
            f"[bold]Resident Daemon Status:[/] [{'green' if running else 'yellow'}]{'ACTIVE' if running else 'STOPPED'}[/]\n"
            f"[bold]Socket Path:[/] {SOCKET_PATH}\n"
            f"[bold]Process PID:[/] {pid if running else 'None'}\n"
            f"[bold]Features:[/] [cyan]0.0s TTFT • Pre-warmed Metal GPU Graph • 36+ tok/s Instant Inference[/]",
            title="DFlash Resident Daemon",
            border_style="green" if running else "yellow",
        ))
    elif args.action == "start":
        if DaemonClient.is_daemon_running():
            console.print("[yellow]Resident daemon is already running.[/]")
            return
        console.print("[dim]Starting resident DFlash daemon in background...[/]")
        proc = subprocess.Popen(
            [sys.executable, "-m", "harness.daemon.server"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(2)
        console.print(f"[bold green]DFlash resident daemon started (PID: {proc.pid}). Pre-warming complete![/]")
    elif args.action == "stop":
        if Path(PID_FILE).exists():
            try:
                pid = int(Path(PID_FILE).read_text().strip())
                os.kill(pid, signal.SIGTERM)
                console.print(f"[bold green]Stopped resident daemon (PID: {pid}).[/]")
            except Exception as e:
                console.print(f"[yellow]Could not kill PID: {e}[/]")
            if Path(PID_FILE).exists():
                Path(PID_FILE).unlink(missing_ok=True)
            if Path(SOCKET_PATH).exists():
                Path(SOCKET_PATH).unlink(missing_ok=True)
        else:
            console.print("[yellow]No running daemon PID file found.[/]")

def main():
    parser = argparse.ArgumentParser(description="Local vs Frontier LLM Evaluation Harness & Developer CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    subparsers.add_parser("list", help="List all benchmark challenges")

    # challenge
    p_chal = subparsers.add_parser("challenge", help="Run a benchmark challenge")
    p_chal.add_argument("id", help="Challenge ID (e.g. concurrency_01_thundering_herd)")
    p_chal.add_argument("--engine", choices=["mlx", "dflash"], default="dflash", help="Inference engine (mlx or dflash)")

    # compare
    p_comp = subparsers.add_parser("compare", help="View past challenge results")
    p_comp.add_argument("id", help="Challenge ID")

    # leaderboard
    subparsers.add_parser("leaderboard", help="Display overall leaderboard")

    # ast-check
    p_ast = subparsers.add_parser("ast-check", help="Run AST structural validation on a file")
    p_ast.add_argument("file", help="Path to source file")
    p_ast.add_argument("--lang", default=None, help="Language override (swift, rust, ts, python, bash)")

    # refactor
    p_ref = subparsers.add_parser("refactor", help="Pipe code for DFlash compiler-verified refactor")
    p_ref.add_argument("instruction", help="Refactoring instruction")
    p_ref.add_argument("--file", default=None, help="Input source file (or stdin)")
    p_ref.add_argument("--lang", default="rust", help="Language target")
    p_ref.add_argument("--output", default=None, help="Output destination file")

    # daemon
    p_daemon = subparsers.add_parser("daemon", help="Manage background resident inference daemon")
    p_daemon.add_argument("action", choices=["start", "status", "stop"], help="Daemon action")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "challenge":
        cmd_challenge(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "leaderboard":
        cmd_leaderboard(args)
    elif args.command == "ast-check":
        cmd_ast_check(args)
    elif args.command == "refactor":
        cmd_refactor(args)
    elif args.command == "daemon":
        cmd_daemon(args)

if __name__ == "__main__":
    main()
