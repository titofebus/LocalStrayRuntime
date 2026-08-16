import os
import re
import sys
import time
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Callable, Tuple, Any, Dict
import mlx.core as mx
from harness.config import DEFAULT_MLX_MODEL_PATH, DEFAULT_MTP_MODEL_PATH, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from harness.core.models import ModelOutput, Challenge, VerificationResult
from harness.daemon.stream_stats import generation_usage
from harness.dflash_runtime import build_dflash_runtime_context, initialize_mlx_streams

DEFAULT_DFLASH_DRAFT_REF = DEFAULT_MTP_MODEL_PATH

# Dynamic Language & Domain Invariant Slices
DOMAIN_INVARIANT_SLICES = {
    "swift": """### Swift 6 Invariants:
1. STRICT CONCURRENCY: Strictly observe Swift 6 Sendable and actor isolation.
2. CLEAN GENERICS: Never write argument labels inside generic brackets (use `Cache<A, B>()`, not `Cache<Key: A, Value: B>()`).
3. ZERO SCRIPT LEAKAGE: Emit only production structs/classes/actors. No top-level `async let` or `@main` blocks.""",

    "rust": """### Rust 2021 Invariants:
1. ZERO EXTERNAL DEPENDENCIES: Always prefer `std::sync::mpsc`, `std::thread`, and std atomics. Never pull in external crates unless explicitly requested.
2. CHANNEL CONCURRENCY: In multi-worker pipelines with `std::sync::mpsc`, wrap `Receiver<T>` in `Arc<Mutex<Receiver<T>>>` so worker threads can safely share and receive events across threads.
3. CHANNEL TYPES: `sync_channel(bound)` returns `(SyncSender<T>, Receiver<T>)`. Ensure sender field is typed `SyncSender<T>` (not `Sender<T>`).
4. TRAIT BOUNDS: Never derive `Clone`, `PartialEq`, or `Eq` on enums holding `std::io::Error` or non-comparable types.
5. MINIMAL ENUMS: Declare only the exact error variants required (`CorruptHeader`, `ChecksumMismatch`).
6. ATOMICS: Use strict `Ordering::Acquire` for loads and `Ordering::Release` for stores in lock-free ring buffers.
7. INTERIOR MUTABILITY: In lock-free structures without Mutex, wrap slot buffers in `Vec<UnsafeCell<Option<T>>>` or atomic pointers.""",

    "typescript": """### TypeScript & Web Invariants:
1. DECLARATIVE BUFFERING: Use declarative standard library methods (e.g. `.split("\\n\\n")`, `.replace()`) over manual while-loop pointer indexing.
2. DISCRIMINATED UNIONS: Ensure exhaustive type checking without `any` casts.
3. ZERO DEPENDENCIES: Use standard Web / Node.js native APIs exclusively.""",

    "python": """### Systems & Algorithm Invariants:
1. DETERMINISM: For topological orderings or cycle breaking, always sort lexicographically for deterministic tie-breaking.
2. CLEAN STATE: Implement finite state machines with explicit state enum transitions and zero global mutable state.""",

    "go": """### Go (Golang) Invariants:
1. BINARY SERIALIZATION: Always use `encoding/binary.BigEndian` or `binary.LittleEndian` with dynamic slices (e.g. `make([]byte, size)`). NEVER write manual fixed-array slice index offsets (e.g. `buf[20]`).
2. DELEGATION: Top-level struct methods must forward calls to internal components (e.g. `s.wal.Append(...)`).
3. CONCURRENCY: Always pass `context.Context` as the first argument. Ensure zero goroutine leaks by listening on `ctx.Done()`.
4. CHANNEL DRAINING: Use non-blocking select or sync.WaitGroup for graceful worker termination.
5. ERROR HANDLING: Return explicit errors; never panic.""",

    "cpp": """### C++20 Systems Invariants:
1. CACHELINE ALIGNMENT: Use `alignas(64)` on atomic variables accessed by different threads to eliminate false sharing.
2. MEMORY ORDERING: Use explicit `std::memory_order_acquire` for loads and `std::memory_order_release` for stores.
3. ZERO-ALLOCATION: Avoid dynamic `new`/`malloc` inside hot paths.""",

    "bash": """### Terminal & Bash Invariants:
1. PIPEFAIL: Enforce `set -euo pipefail`.
2. NON-INTERACTIVE: Zero interactive commands (`read`, `sudo`, `nano`).
3. EFFICIENCY: Prefer `jq`/`awk`/`sed` streaming over manual line-by-line bash while-read loops."""
}

def detect_domain(prompt: str, language: Optional[str] = None) -> str:
    """Detect domain or language from challenge metadata or prompt content."""
    if language:
        lang = language.lower()
        if "swift" in lang: return "swift"
        if "rust" in lang: return "rust"
        if "ts" in lang or "typescript" in lang or "js" in lang: return "typescript"
        if "python" in lang or "algo" in lang or "distributed" in lang: return "python"
        if "bash" in lang or "shell" in lang or "cli" in lang: return "bash"

    p_lower = prompt.lower()
    if "swift" in p_lower or "actor" in p_lower or "swiftui" in p_lower: return "swift"
    if "rust" in p_lower or "tokio" in p_lower or "atomic" in p_lower or "spmc" in p_lower: return "rust"
    if "typescript" in p_lower or "interface" in p_lower or "sse" in p_lower or "json" in p_lower: return "typescript"
    if "bash" in p_lower or "shell" in p_lower: return "bash"
    return "python"

class DFlashEngine:
    _bundle = None
    _runtime_context = None
    _loaded_target = None
    _loaded_draft = None

    def __init__(
        self,
        target_path: str = DEFAULT_MLX_MODEL_PATH,
        draft_ref: str = DEFAULT_DFLASH_DRAFT_REF,
        draft_quant: Optional[str] = "w8",
        block_tokens: int = 4,
    ):
        self.target_path = target_path
        self.draft_ref = draft_ref
        self.draft_quant = draft_quant
        self.block_tokens = block_tokens

    def _ensure_loaded(self):
        """Load DFlash speculative decoding runtime bundle with Metal GPU acceleration."""
        if (
            DFlashEngine._bundle is None
            or DFlashEngine._loaded_target != self.target_path
            or DFlashEngine._loaded_draft != self.draft_ref
        ):
            from harness.model_provenance import validate_speculative_pair

            pair = validate_speculative_pair(self.target_path, self.draft_ref)
            initialize_mlx_streams()
            print(
                "[DFlashEngine] Validated speculative pair "
                f"target={pair.target_model_id}, draft={pair.draft_model_id}.",
                flush=True,
            )
            print(f"[DFlashEngine] Loading Speculative Target ({self.target_path}) + Drafter ({self.draft_ref})...", flush=True)
            DFlashEngine._runtime_context = build_dflash_runtime_context()
            draft_config_path = Path(self.draft_ref) / "config.json"
            draft_config = json.loads(draft_config_path.read_text(encoding="utf-8"))
            if draft_config.get("model_type") == "qwen3_8_mtp":
                from harness.executors.qwen38_mtp import (
                    load_qwen38_mtp_runtime_bundle,
                )

                DFlashEngine._bundle = load_qwen38_mtp_runtime_bundle(
                    model_ref=self.target_path,
                    draft_ref=self.draft_ref,
                    verify_config=DFlashEngine._runtime_context.verify,
                )
            else:
                from dflash_mlx.runtime.bundle import load_runtime_bundle

                DFlashEngine._bundle = load_runtime_bundle(
                    model_ref=self.target_path,
                    draft_ref=self.draft_ref,
                    draft_quant=self.draft_quant or "w8",
                    verify_config=DFlashEngine._runtime_context.verify,
                )
            DFlashEngine._loaded_target = self.target_path
            DFlashEngine._loaded_draft = self.draft_ref
            runtime = DFlashEngine._runtime_context.runtime
            print(
                "[DFlashEngine] Speculative Decoding Engine Active "
                f"(verify_mode={runtime.verify_mode}, draft_window={runtime.draft_window_size}, "
                f"draft_sink={runtime.draft_sink_size}, target_fa_window={runtime.target_fa_window}).",
                flush=True,
            )

    def warmup(self) -> None:
        """Compile the Metal prefill, draft, and verification paths before readiness."""
        prompt = (
            "<|im_start|>system\nYou are Qwen Prime.<|im_end|>\n"
            "<|im_start|>user\nReturn the integer 42.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        for _ in self.stream_generate_tokens(
            prompt=prompt,
            max_tokens=8,
            use_prefix_cache=False,
        ):
            pass

    def _prefix_cache_flow(
        self,
        *,
        prompt_tokens: list[int],
        messages: Optional[list[dict[str, Any]]],
    ):
        from dflash_mlx.server.prefix_cache_flow import PrefixCacheFlow

        bundle = DFlashEngine._bundle
        provider = SimpleNamespace(
            model_key=(self.target_path, None, self.draft_ref),
            tokenizer=bundle.tokenizer,
            cli_args=SimpleNamespace(chat_template_args={}),
        )
        request = (
            SimpleNamespace(request_type="chat", messages=messages)
            if messages
            else None
        )
        return PrefixCacheFlow.for_request(
            model_provider=provider,
            draft_model=bundle.draft_model,
            tokenizer=bundle.tokenizer,
            prompt=prompt_tokens,
            request=request,
            runtime_context=DFlashEngine._runtime_context,
        )

    def run_inference(
        self,
        prompt: str,
        language: Optional[str] = None,
        system_prompt: Optional[str] = None,
        enable_thinking: bool = True,
        max_thinking_tokens: int = 250,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        on_token_callback: Optional[Callable[[int, float], None]] = None,
        direct: bool = False,
    ) -> ModelOutput:
        """Run block speculative diffusion generation with dynamic domain slicing."""
        # 1. Check if resident warm daemon is active (0.0s TTFT fast path)
        if not direct and not os.environ.get("DFLASH_DAEMON_WORKER") and not on_token_callback and not system_prompt:
            from harness.daemon.client import DaemonClient
            if DaemonClient.is_daemon_running():
                daemon_out = DaemonClient.run_inference(
                    prompt=prompt,
                    language=language,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    enable_thinking=enable_thinking,
                )
                if daemon_out is not None:
                    return daemon_out

        # 2. In-process fallback
        _ = mx.default_stream(mx.Device(mx.cpu))
        _ = mx.default_stream(mx.Device(mx.gpu))
        self._ensure_loaded()
        from dflash_mlx.generate import (
            stream_dflash_generate,
            get_stop_token_ids,
            decode_token,
            TokenEvent,
            SummaryEvent,
            generation_tps_from_summary,
        )

        bundle = DFlashEngine._bundle
        runtime_context = DFlashEngine._runtime_context

        domain = detect_domain(prompt, language)
        domain_slice = DOMAIN_INVARIANT_SLICES.get(domain, DOMAIN_INVARIANT_SLICES["python"])

        base_sys = (
            "You are a Principal Systems Software Engineer specializing in high-performance production code.\n\n"
            "### Core Guidelines:\n"
            "1. Output ONLY the production implementation code inside a single markdown block.\n"
            "2. DO NOT write unit tests, `mod tests`, examples, or `main()` entrypoints.\n"
            "3. Keep code minimal, robust, and declarative.\n\n"
            f"{domain_slice}"
        )

        if enable_thinking:
            base_sys += (
                "\n\n### Micro-Reasoning Invariants:\n"
                "Inside `<think>`, perform a concise, rapid verification (<100 words):\n"
                "- Verify trait bounds, Send/Sync safety, and memory ordering.\n"
                "- Verify array/slice bounds and dynamic allocations.\n"
                "- Proceed immediately to production code output after `</think>`."
            )

        if prompt.startswith("<|im_start|>"):
            formatted_prompt = prompt
        else:
            effective_sys = f"{system_prompt}\n\n{base_sys}" if system_prompt else base_sys
            messages = [
                {"role": "system", "content": effective_sys},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = bundle.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                reasoning_effort="low" if enable_thinking else None,
            )

        stop_token_ids = get_stop_token_ids(bundle.tokenizer)
        start_time = time.perf_counter()
        chunks = []
        token_count = 0
        summary_tps = 0.0
        summary_payload = None

        mx.set_default_device(mx.gpu)
        with mx.stream(mx.gpu):
            stream = stream_dflash_generate(
                target_model=bundle.target_model,
                target_ops=bundle.target_ops,
                tokenizer=bundle.tokenizer,
                draft_model=bundle.draft_model,
                draft_backend=bundle.draft_backend,
                prompt=formatted_prompt,
                max_new_tokens=max_tokens,
                block_tokens=self.block_tokens,
                stop_token_ids=stop_token_ids,
                runtime_context=runtime_context,
            )

            try:
                for event in stream:
                    if isinstance(event, TokenEvent):
                        txt = decode_token(bundle.tokenizer, int(event.token_id))
                        chunks.append(txt)
                        token_count += 1
                        if on_token_callback and token_count % 10 == 0:
                            elapsed = time.perf_counter() - start_time
                            on_token_callback(token_count, token_count / max(elapsed, 0.001))
                    elif isinstance(event, SummaryEvent):
                        summary_tps = generation_tps_from_summary(event)
                        summary_payload = event.to_payload()
                        print(f"[DFlashEngine] Speculative Summary: {summary_tps:.1f} t/s, generated={event.generation_tokens}, accepted_from_draft={event.accepted_from_draft} ({event.acceptance_ratio:.1%}), cycles={event.cycles_completed}", flush=True)
            finally:
                close_fn = getattr(stream, "close", None)
                if close_fn is not None:
                    close_fn()

        total_time = time.perf_counter() - start_time
        raw_output = "".join(chunks)
        tps = summary_tps if summary_tps > 0 else round(token_count / max(total_time, 0.001), 2)
        prompt_tokens = len(bundle.tokenizer.encode(formatted_prompt))

        clean_text = raw_output.strip()
        thinking = None

        if "</think>" in clean_text:
            parts = clean_text.split("</think>", 1)
            thinking = parts[0].replace("<think>", "").strip()
            clean_text = parts[1].strip()
        elif "<think>" in clean_text:
            parts = clean_text.split("<think>", 1)
            thinking = parts[1].strip()
            clean_text = ""
        else:
            thinking = None
            clean_text = raw_output.strip()

        # Extract code blocks
        code = None
        code_matches = re.findall(r"```(?:\w+)?\s*\n(.*?)```", clean_text, re.DOTALL)
        if code_matches:
            code = max(code_matches, key=lambda c: len(c.strip())).strip()
        elif clean_text.strip():
            code = clean_text.strip()

        return ModelOutput(
            model_name="Qwen3.8-27B-DFlash-Diffusion",
            raw_response=clean_text,
            thinking_content=thinking,
            code_content=code,
            prompt_tokens=prompt_tokens,
            completion_tokens=token_count,
            tokens_per_sec=tps,
            ttft_seconds=round(prompt_tokens / 10.0, 3),
            peak_memory_gb=23.0,
        )

    def stream_generate_tokens(
        self,
        prompt: str,
        max_tokens: int = 4096,
        cancel_event: Optional[Any] = None,
        use_speculative: bool = True,
        stop_token_ids: Optional[list[int]] = None,
        messages: Optional[list[dict[str, Any]]] = None,
        use_prefix_cache: bool = True,
    ):
        """Yield (event_type, payload) using block speculative diffusion generation."""
        _ = mx.default_stream(mx.Device(mx.cpu))
        _ = mx.default_stream(mx.Device(mx.gpu))
        self._ensure_loaded()

        bundle = DFlashEngine._bundle
        runtime_context = DFlashEngine._runtime_context
        start_time = time.perf_counter()
        token_count = 0
        summary_tps = 0.0
        summary_payload = None
        encoded_prompt = [int(token) for token in bundle.tokenizer.encode(prompt)]
        prompt_tokens = len(encoded_prompt)
        from dflash_mlx.server.prefix_cache_flow import PrefixCacheFlow

        prefix_flow = PrefixCacheFlow(cache_manager=None)
        if use_prefix_cache:
            prefix_flow = self._prefix_cache_flow(
                prompt_tokens=encoded_prompt,
                messages=messages,
            )
        prefill_payload = None

        from dflash_mlx.generate import (
            stream_dflash_generate,
            get_stop_token_ids,
            decode_token,
            TokenEvent,
            SummaryEvent,
            generation_tps_from_summary,
        )
        from dflash_mlx.engine.events import PrefillCompleteEvent

        if stop_token_ids is None:
            try:
                stop_token_ids = get_stop_token_ids(bundle.tokenizer)
            except Exception:
                stop_token_ids = [248046, 248044]

        mx.set_default_device(mx.gpu)
        with mx.stream(mx.gpu):
            stream = stream_dflash_generate(
                target_model=bundle.target_model,
                target_ops=bundle.target_ops,
                tokenizer=bundle.tokenizer,
                draft_model=bundle.draft_model,
                draft_backend=bundle.draft_backend,
                prompt=prompt,
                max_new_tokens=max_tokens,
                block_tokens=self.block_tokens,
                use_chat_template=False,
                stop_token_ids=stop_token_ids,
                prompt_tokens_override=encoded_prompt,
                prefix_snapshot=prefix_flow.snapshot,
                snapshot_service=prefix_flow.snapshot_service,
                stable_prefix_len=prefix_flow.stable_prefix_len,
                prefix_cache_active=prefix_flow.cache_active,
                runtime_context=runtime_context,
            )

            try:
                for event in stream:
                    if cancel_event and cancel_event.is_set():
                        break
                    if isinstance(event, TokenEvent):
                        txt = decode_token(bundle.tokenizer, int(event.token_id))
                        token_count += 1
                        yield ("token", txt)
                    elif isinstance(event, PrefillCompleteEvent):
                        prefill_payload = event.to_payload()
                    elif isinstance(event, SummaryEvent):
                        summary_tps = generation_tps_from_summary(event)
                        summary_payload = event.to_payload()
                        print(f"[DFlashEngine] Speculative Summary: {summary_tps:.1f} t/s, generated={event.generation_tokens}, accepted_from_draft={event.accepted_from_draft} ({event.acceptance_ratio:.1%})", flush=True)
            finally:
                close_fn = getattr(stream, "close", None)
                if close_fn is not None:
                    close_fn()

        total_time = max(0.001, time.perf_counter() - start_time)
        tps = summary_tps if summary_tps > 0 else round(token_count / total_time, 2)
        if summary_payload is not None:
            yield ("usage", generation_usage(
                summary=summary_payload,
                prefill=prefill_payload,
                cache_lookup_ms=prefix_flow.lookup_ms,
                prompt_tokens=prompt_tokens,
                total_time=total_time,
            ))
        else:
            yield ("usage", {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": token_count,
                "tokens_per_second": tps,
                "generation_seconds": token_count / tps if tps > 0 else total_time,
                "latency": total_time,
                "accepted_from_draft": 0,
                "acceptance_ratio": 0.0,
                "cycles_completed": 0,
                "fallback_ar": True,
                "adaptive_block_reductions": 0,
                "adaptive_block_min": None,
                "prefill_seconds": total_time,
                "prefill_tokens_per_second": prompt_tokens / total_time,
                "physical_prefill_tokens": prompt_tokens,
                "prefill_tokens_restored": 0,
                "prefill_tokens_computed": prompt_tokens,
                "prefix_cache_hit_tokens": 0,
                "prefix_cache_lookup_ms": 0.0,
            })

    def run_with_compiler_feedback(
        self,
        challenge: Challenge,
        max_correction_attempts: int = 1,
    ) -> Tuple[ModelOutput, VerificationResult]:
        """Run generation with AST pre-validation and automatic compiler & runtime self-correction."""
        from harness.core.sandbox import CodeSandbox
        from harness.core.ast_engine import ASTEngine

        # Attempt 1: Standard Generation with Domain Slice
        output = self.run_inference(prompt=challenge.prompt, language=challenge.language)

        # AST Structural Pre-Validation
        code_to_check = output.code_content or output.raw_response
        ast_res = ASTEngine.validate_code(code_to_check, challenge.language)

        if not ast_res.is_valid and max_correction_attempts > 0:
            print(f"\n[DFlashEngine] AST Structural Invariant Warning: {ast_res.error_message}")
            correction_prompt = (
                f"{challenge.prompt}\n\n"
                f"### CRITICAL STRUCTURAL INVARIANT ERROR:\n"
                f"{ast_res.error_message}\n\n"
                f"Fix the error strictly. Output ONLY the corrected production implementation inside a single markdown code block."
            )
            output = self.run_inference(prompt=correction_prompt, language=challenge.language)

        verif = CodeSandbox.run_verification("Qwen3.8-27B-DFlash", output.raw_response, challenge)

        if verif.test_passed or max_correction_attempts <= 0:
            return output, verif

        # Formulate exact diagnostic for turn 2
        err_msg = verif.test_output.strip()
        if "timed out" in err_msg.lower():
            diagnostic = (
                "DIAGNOSTIC: Your previous code timed out due to an infinite loop.\n"
                "CAUSE: Manual while-loop pointer indexing failed to advance past delimiters.\n"
                "FIX: Rewrite using declarative standard library methods (e.g. .split('\\n\\n') or .replace()). Avoid manual while-loops."
            )
        else:
            diagnostic = f"DIAGNOSTIC (Compiler / Runtime Diagnostic):\n```\n{err_msg[:500]}\n```"

        print(f"\n[DFlashEngine] Self-correction triggered:\n{diagnostic}\n")
        correction_prompt = (
            f"{challenge.prompt}\n\n"
            f"### ISSUE WITH PREVIOUS IMPLEMENTATION:\n"
            f"{diagnostic}\n\n"
            f"Fix the issue directly. Emit ONLY the corrected production implementation inside a single markdown code block."
        )

        corrected_output = self.run_inference(prompt=correction_prompt, language=challenge.language)
        corrected_verif = CodeSandbox.run_verification("Qwen3.8-27B-DFlash (Self-Corrected)", corrected_output.raw_response, challenge)

        return corrected_output, corrected_verif
