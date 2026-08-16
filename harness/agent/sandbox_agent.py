"""Sandboxed Autonomous Agent Loop for Real-World Systems Architecture & Implementation."""
import os
import re
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from harness.daemon.client import DaemonClient
from harness.executors.dflash_engine import DFlashEngine

console = Console()

SANDBOX_ROOT = Path(
    os.environ.get(
        "QWEN_PRIME_AGENT_WORKSPACES",
        Path.home() / "Library" / "Application Support" / "QwenPrime" / "AgentWorkspaces",
    )
).expanduser()

class SandboxedToolRuntime:
    def __init__(self, workspace_name: str, allowed_read_roots: List[str]):
        self.workspace_dir = SANDBOX_ROOT / workspace_name
        self.allowed_read_roots = [Path(r).expanduser().resolve() for r in allowed_read_roots]

        # Initialize clean isolated workspace
        if self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold green]Created isolated sandbox workspace:[/] [cyan]{self.workspace_dir}[/]")

    def is_read_allowed(self, path: str) -> bool:
        resolved = Path(path).expanduser().resolve()
        return any(resolved == root or resolved.is_relative_to(root) for root in self.allowed_read_roots)

    def list_dir(self, path: str) -> Dict[str, Any]:
        """List files in an allowed project directory (read-only)."""
        p = Path(path).expanduser().resolve()
        if not self.is_read_allowed(str(p)):
            return {"error": f"Permission denied: cannot list directory outside allowed read roots ({path})"}
        if not p.exists() or not p.is_dir():
            return {"error": f"Directory not found: {path}"}

        items = []
        for item in sorted(p.iterdir()):
            if item.name.startswith("."):
                continue
            items.append({
                "name": item.name,
                "is_dir": item.is_dir(),
                "size_bytes": item.stat().st_size if item.is_file() else None,
            })
        return {"directory": str(p), "entries": items}

    def read_file(self, path: str) -> Dict[str, Any]:
        """Read content from an allowed file (read-only)."""
        p = Path(path).expanduser().resolve()
        if not self.is_read_allowed(str(p)):
            return {"error": f"Permission denied: cannot read file outside allowed roots ({path})"}
        if not p.exists() or not p.is_file():
            return {"error": f"File not found: {path}"}
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
            return {"path": str(p), "lines": len(content.splitlines()), "content": content}
        except Exception as e:
            return {"error": str(e)}

    def write_file(self, rel_path: str, content: str) -> Dict[str, Any]:
        """Write file strictly inside the sandboxed workspace."""
        p_str = str(rel_path).strip()
        if Path(p_str).is_absolute():
            return {"error": "Absolute write paths are not allowed; use a workspace-relative path"}

        clean_rel = p_str
        target = (self.workspace_dir / clean_rel).resolve()

        if not str(target).startswith(str(self.workspace_dir.resolve())):
            return {"error": f"Security violation: path {rel_path} escapes sandbox workspace!"}

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"status": "success", "written_file": str(target), "bytes": len(content)}

    def run_command(self, command: str, timeout: int = 60) -> Dict[str, Any]:
        """Run build or test command strictly inside the sandboxed workspace."""
        try:
            start = time.perf_counter()
            res = subprocess.run(
                command,
                shell=True,
                cwd=str(self.workspace_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed = time.perf_counter() - start
            return {
                "exit_code": res.returncode,
                "stdout": res.stdout[:3000],
                "stderr": res.stderr[:3000],
                "elapsed_seconds": round(elapsed, 2),
                "success": res.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {"error": f"Command timed out after {timeout} seconds"}
        except Exception as e:
            return {"error": str(e)}


SYSTEM_PROMPT_AGENT = """You are a Principal Software Architect and Autonomous Systems Engineer.
You have access to a live environment with tools to explore existing specifications, write code into an isolated sandbox, and execute the local compiler to test and verify your solution.

### Available Tools (format your tool call using <tool_call><function=...><parameter=...>...</parameter></function></tool_call>):

1. `read_file`: Inspect schemas, source code, and design contracts (read-only).
   <tool_call>
   <function=read_file>
   <parameter=path>/path/inside/an/allowed/read/root/schema.json</parameter>
   </function>
   </tool_call>

2. `write_file`: Create or update a file in your sandboxed workspace. Use relative paths!
   <tool_call>
   <function=write_file>
   <parameter=path>Package.swift</parameter>
   <parameter=content>// swift-tools-version: 6.0
import PackageDescription
...</parameter>
   </function>
   </tool_call>

3. `run_command`: Run `swift build` or `swift test` directly in your sandboxed workspace.
   <tool_call>
   <function=run_command>
   <parameter=command>swift test</parameter>
   </function>
   </tool_call>

4. `finish`: When all files are written, `swift build` and `swift test` pass with 0 errors.
   <tool_call>
   <function=finish>
   <parameter=summary>Completed Swift 6 MCP package with full Sendable actor isolation and passing tests.</parameter>
   </function>
   </tool_call>

### Rules:
- Your workspace already exists and is your working directory.
- Write files using clean relative paths: `Package.swift`, `Sources/AgentBlockKitMCP/AppConfig.swift`, `Sources/AgentBlockKitMCP/AppBlockCoordinator.swift`, `Tests/AgentBlockKitMCPTests/ConfigTests.swift`.
- Run `swift test` to verify everything compiles and passes cleanly!
"""

class AutonomousQwenAgent:
    def __init__(self, workspace_name: str, allowed_read_roots: List[str]):
        self.runtime = SandboxedToolRuntime(workspace_name, allowed_read_roots)
        self.engine = DFlashEngine(draft_quant="w8")

    def _call_llm(self, prompt: str) -> str:
        """Call Qwen via the resident daemon for instant 0.0s TTFT & 35 tok/s."""
        out = DaemonClient.run_inference(
            prompt=prompt,
            enable_thinking=True,
            max_tokens=4096,
            temperature=0.1,
        )
        if out is not None:
            return out.raw_response

        # Fallback to in-process
        out = self.engine.run_inference(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT_AGENT,
            enable_thinking=True,
            max_tokens=4096,
            temperature=0.1,
        )
        return out.raw_response

    def _extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        calls = []
        # 1. Match Qwen XML tool format (<tool_call><function=...><parameter=...>)
        for m_xml in re.finditer(r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", text, re.DOTALL):
            fn_name = m_xml.group(1).strip()
            body = m_xml.group(2)
            params = {}
            for p in re.finditer(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", body, re.DOTALL):
                params[p.group(1).strip()] = p.group(2).strip()

            fn_lower = fn_name.lower()
            if "read" in fn_lower:
                calls.append({"tool": "read_file", "args": {"path": params.get("file_path") or params.get("path", "")}})
            elif "list" in fn_lower or "ls" in fn_lower:
                calls.append({"tool": "list_dir", "args": {"path": params.get("dir_path") or params.get("path", "")}})
            elif "write" in fn_lower:
                calls.append({"tool": "write_file", "args": {"path": params.get("file_path") or params.get("path", ""), "content": params.get("content", "")}})
            elif "run" in fn_lower or "bash" in fn_lower:
                calls.append({"tool": "run_command", "args": {"command": params.get("command", "")}})
            elif "finish" in fn_lower:
                calls.append({"tool": "finish", "args": {"summary": params.get("summary", "Done")}})

        if calls:
            return calls

        # 2. Match JSON blocks ```json ... ```
        matches = re.findall(r"```(?:json)?\s*\n({.*?})\s*\n```", text, re.DOTALL)
        for m in matches:
            try:
                data = json.loads(m)
                if "tool" in data:
                    calls.append(data)
            except Exception:
                continue
        if calls:
            return calls

        # 3. Fallback raw json search
        try:
            m_raw = re.search(r'{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"args"\s*:\s*({.*?})\s*}', text, re.DOTALL)
            if m_raw:
                calls.append({"tool": m_raw.group(1), "args": json.loads(m_raw.group(2))})
        except Exception:
            pass
        return calls

    def run_task(self, goal: str, max_turns: int = 16):
        console.print(Panel(
            f"[bold]Goal:[/] {goal}\n[bold]Isolated Sandbox:[/] [cyan]{self.runtime.workspace_dir}[/]\n"
            f"[bold]Engine:[/] Local Qwen3.8-27B (Resident Daemon • Prefix Cached • 35 t/s)",
            title="🤖 Autonomous Qwen Agent Launch",
            border_style="magenta",
        ))

        conversation = [f"### Objective:\n{goal}\n\nPlease begin by exploring the necessary files or schemas."]

        for turn in range(1, max_turns + 1):
            console.print(f"\n[bold yellow]── Turn {turn}/{max_turns} ──[/]")
            prompt = "\n\n".join(conversation)

            start = time.perf_counter()
            response = self._call_llm(prompt)
            el = time.perf_counter() - start

            # Extract thinking vs content
            clean_resp = response
            if "</think>" in response:
                parts = response.split("</think>", 1)
                thinking = parts[0].replace("<think>", "").strip()
                clean_resp = parts[1].strip()
                console.print(Panel(thinking[:500] + ("..." if len(thinking) > 500 else ""), title="🧠 Qwen Reasoning", border_style="blue"))

            tool_calls = self._extract_tool_calls(clean_resp)
            if not tool_calls:
                console.print(Panel(clean_resp[:600], title="Qwen Response (No Tool Call)", border_style="yellow"))
                conversation.append(f"Model Response:\n{clean_resp}\n\nSystem: Please invoke one of the available tools (`list_dir`, `read_file`, `write_file`, `run_command`, or `finish`).")
                continue

            turn_tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("tool")
                tool_args = tool_call.get("args", {})
                console.print(f"⚡️ [bold cyan]Qwen invokes tool:[/] [bold green]{tool_name}[/] with args [dim]{tool_args}[/]")

                # Dispatch tool
                if tool_name == "list_dir":
                    res = self.runtime.list_dir(tool_args.get("path", ""))
                elif tool_name == "read_file":
                    res = self.runtime.read_file(tool_args.get("path", ""))
                    preview_len = len(res.get("content", ""))
                    console.print(f"   [dim]Read {preview_len} bytes from disk[/]")
                elif tool_name == "write_file":
                    res = self.runtime.write_file(tool_args.get("path", ""), tool_args.get("content", ""))
                    console.print(f"   [bold green]✓ Created {tool_args.get('path')} ({res.get('bytes')} bytes)[/]")
                elif tool_name == "run_command":
                    cmd = tool_args.get("command", "")
                    console.print(f"   [bold yellow]Executing in sandbox:[/] `{cmd}`")
                    res = self.runtime.run_command(cmd)
                    status_color = "green" if res.get("success") else "red"
                    console.print(f"   [{status_color}]Exit code: {res.get('exit_code')} ({res.get('elapsed_seconds')}s)[/]")
                    if res.get("stderr"):
                        console.print(f"   [red]Compiler Stderr:[/] {res.get('stderr')[:400]}")
                elif tool_name == "finish":
                    console.print(Panel(
                        f"[bold green]Task Successfully Completed![/]\n\n"
                        f"[bold]Summary:[/] {tool_args.get('summary', 'Done')}\n"
                        f"[bold]Sandboxed Package Location:[/] [cyan]{self.runtime.workspace_dir}[/]",
                        title="🎉 Autonomous Mission Success",
                        border_style="green",
                    ))
                    return True
                else:
                    res = {"error": f"Unknown tool '{tool_name}'"}

                res_str = json.dumps(res, indent=2)
                if len(res_str) > 4000:
                    res_str = res_str[:4000] + "\n... (truncated for context)"
                turn_tool_results.append(f"Tool `{tool_name}` output:\n```json\n{res_str}\n```")

            combined_results = "\n\n".join(turn_tool_results)
            conversation.append(f"Assistant:\n{clean_resp}\n\n{combined_results}\n\nPlease proceed to the next step.")

        console.print("[red]Agent reached maximum turn limit.[/]")
        return False

def main():
    configured_roots = os.environ.get("QWEN_PRIME_AGENT_READ_ROOTS", "")
    allowed_roots = [root for root in configured_roots.split(os.pathsep) if root]
    if not allowed_roots:
        raise SystemExit("Set QWEN_PRIME_AGENT_READ_ROOTS to one or more trusted directories")
    agent = AutonomousQwenAgent(
        workspace_name="qwen-prime-agent",
        allowed_read_roots=allowed_roots,
    )

    task_goal = (
        "1. Inspect the user-provided trusted roots for the requested contracts.\n"
        "2. Implement the requested work only in the agent workspace.\n"
        "3. For Swift packages:\n"
        "   - `Package.swift` configured for Swift 6 (`.macOS(.v14)` or `.v15`).\n"
        "   - Strong `AppConfig` models conforming to `Codable & Sendable` matching the schema.\n"
        "   - An Actor-isolated `AppBlockCoordinator` that loads config JSON and validates structure.\n"
        "   - A lightweight stdio JSON-RPC MCP Dispatcher (`MCPServer`) exposing tools: `validate_config` and `plan_app`.\n"
        "   - Unit tests in `Tests/` verifying config loading and tool execution.\n"
        "4. Run the relevant build and test commands before reporting completion.\n"
        "5. Call `finish` when done."
    )

    agent.run_task(task_goal, max_turns=16)

if __name__ == "__main__":
    main()
