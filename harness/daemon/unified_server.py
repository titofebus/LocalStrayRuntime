"""Unified High-Performance OpenAI-Compatible HTTP Server with Embedded DFlash Engine and Tool-Calling."""
import os
import re
import json
import time
import uuid
import queue
import asyncio
import threading
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import mlx.core as mx
from harness.executors.dflash_engine import DFlashEngine
from harness.daemon.stream_stats import combine_generation_usage
from harness.daemon.stream_sanitizer import ControlTokenFilter, ToolMarkupFilter, extract_tool_calls, should_use_prefix_cache, STRUCTURED_TOOL_CALLS_V1
from harness.daemon.qwen_chat import (
    format_assistant_turn,
    generation_limits,
    runtime_reasoning_policy,
)
from harness.model_provenance import qwen_prime_runtime_identity
from harness.dflash_runtime import initialize_mlx_streams

MODEL_ID = "qwen3.8-27b"

class MLXWorkerThread:
    def __init__(self):
        self.task_queue = queue.Queue()
        self.engine = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.ready_event = threading.Event()
        self.warmup_complete = False
        self.thread.start()

    def _run(self):
        mx.set_default_device(mx.gpu)
        initialize_mlx_streams()
        print("[UnifiedServer] Initializing local DFlash MLX engine into GPU RAM...", flush=True)
        self.engine = DFlashEngine(draft_quant="w8")
        self.engine._ensure_loaded()
        print("[UnifiedServer] Compiling Metal generation paths...", flush=True)
        self.engine.warmup()
        self.warmup_complete = True
        print("[UnifiedServer] 🚀 MLX Engine warm and ready for generation!", flush=True)
        self.ready_event.set()

        while True:
            func, args, cancel_event, out_queue, loop = self.task_queue.get()
            try:
                for item in func(self.engine, *args, cancel_event):
                    if cancel_event.is_set():
                        break
                    loop.call_soon_threadsafe(out_queue.put_nowait, item)
            except Exception as e:
                print(f"[MLXWorker] Execution error: {e}", flush=True)
            finally:
                loop.call_soon_threadsafe(out_queue.put_nowait, ("done", None))

worker = MLXWorkerThread()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[UnifiedServer] Waiting for MLX worker thread to warm up...", flush=True)
    while not worker.ready_event.is_set():
        await asyncio.sleep(0.1)
    print("[UnifiedServer] 🚀 Server is ready for OpenAI-compatible requests!", flush=True)
    yield
    print("[UnifiedServer] Shutting down...", flush=True)

app = FastAPI(title="Local MLX Daemon OpenAI API Bridge", lifespan=lifespan)

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local-mlx",
                "permission": [],
                "root": MODEL_ID,
                "parent": None,
            }
        ]
    }


@app.get("/v1/engine")
async def engine_identity():
    if not worker.ready_event.is_set() or worker.engine is None:
        raise HTTPException(status_code=503, detail="Model engine warming up")
    identity = qwen_prime_runtime_identity(
        worker.engine.target_path,
        worker.engine.draft_ref,
        block_tokens=worker.engine.block_tokens,
    )
    draft_model = getattr(getattr(worker.engine, "_bundle", None), "draft_model", None)
    fused_mtp_enabled = getattr(draft_model, "fused_mtp", False)

    identity["prefix_cache_enabled"] = bool(
        worker.engine._runtime_context.runtime.prefix_cache
    )
    identity["verify_mode"] = str(
        worker.engine._runtime_context.runtime.verify_mode
    )
    identity["warmup_complete"] = worker.warmup_complete
    identity["capabilities"] = [STRUCTURED_TOOL_CALLS_V1]
    identity["runtime_features"] = {
        "verify_mode": identity["verify_mode"],
        "fused_mtp": bool(fused_mtp_enabled),
        "fp8_kv_cache": False,
    }
    return identity

def format_messages_to_qwen_chat(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    enable_thinking: bool = False,
    max_reasoning_tokens: int = 96,
) -> str:
    """Format standard OpenAI messages and tools to Qwen ChatML prompt."""
    prompt = ""
    system_rendered = False

    tools_section = ""
    mode_policy = runtime_reasoning_policy(enable_thinking, max_reasoning_tokens)
    if tools:
        tools_json = json.dumps(tools, indent=2)
        tools_section = (
            "\n\n# Tools\n"
            "You may call one or more functions to assist with the user query.\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": "<function-name>", "arguments": <args-json-object>}\n'
            "</tool_call>"
        )

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = "\n".join(text_parts) if text_parts else str(content)

        if role in ("system", "developer"):
            full_sys = (
                f"{content}{tools_section}\n\n{mode_policy}"
                if not system_rendered else content
            )
            prompt += f"<|im_start|>system\n{full_sys}<|im_end|>\n"
            system_rendered = True
        elif role == "user":
            if not system_rendered and tools_section:
                prompt += f"<|im_start|>system\n{tools_section.strip()}<|im_end|>\n"
                system_rendered = True
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            content = format_assistant_turn(
                str(content),
                msg.get("reasoning_content"),
            )
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                tc_str = ""
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "")
                    fn_args = fn.get("arguments", "{}")
                    try:
                        args_obj = json.loads(fn_args) if isinstance(fn_args, str) else fn_args
                    except Exception:
                        args_obj = {"code": str(fn_args)}
                    tc_str += f"\n<tool_call>\n{json.dumps({'name': fn_name, 'arguments': args_obj})}\n</tool_call>"
                prompt += f"<|im_start|>assistant\n{content}{tc_str}<|im_end|>\n"
            else:
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        elif role == "tool":
            prompt += f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
        else:
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    if not system_rendered and tools_section:
        prompt = (
            f"<|im_start|>system\n{tools_section.strip()}\n\n{mode_policy}<|im_end|>\n"
            + prompt
        )
        system_rendered = True

    if not system_rendered:
        prompt = f"<|im_start|>system\n{mode_policy}<|im_end|>\n" + prompt

    if enable_thinking:
        prompt += "<|im_start|>assistant\n<think>\n"
    else:
        prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return prompt

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    if not worker.ready_event.is_set():
        raise HTTPException(status_code=503, detail="Model engine warming up")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model = body.get("model", MODEL_ID)
    messages = body.get("messages", [])
    tools = body.get("tools")
    use_prefix_cache = should_use_prefix_cache(tools)
    temperature = body.get("temperature", 0.1)
    max_tokens, max_reasoning_tokens = generation_limits(body)
    stream = body.get("stream", False)

    # Robust thinking mode extraction from request body
    thinking_param = body.get("thinking")
    if isinstance(thinking_param, dict):
        enable_thinking = thinking_param.get("type", "enabled") == "enabled"
    elif isinstance(thinking_param, bool):
        enable_thinking = thinking_param
    elif "enable_thinking" in body:
        enable_thinking = bool(body.get("enable_thinking"))
    else:
        enable_thinking = False

    # Ensure high-speed coding system context if not provided
    if not any(m.get("role") == "system" for m in messages):
        sys_content = (
            "You are Qwen Prime, a high-performance production code assistant. Write clean, idiomatic, robust code directly."
            if not enable_thinking else
            "You are Qwen Prime. Perform a brief micro-reasoning verification (<80 words) inside <think>, then provide clean production code."
        )
        messages = [{
            "role": "system",
            "content": sys_content
        }] + messages

    prompt = format_messages_to_qwen_chat(
        messages,
        tools=tools,
        enable_thinking=enable_thinking,
        max_reasoning_tokens=max_reasoning_tokens,
    )
    mode_name = "reasoning" if enable_thinking else "direct"
    print(
        f"[UnifiedServer] Request mode={mode_name}, "
        f"max_completion_tokens={max_tokens}, "
        f"max_reasoning_tokens={max_reasoning_tokens}, messages={len(messages)}",
        flush=True,
    )

    if stream:
        async def sse_stream():
            req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            # 1. Immediately emit initial role chunk so client connection is never starved
            first_chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(first_chunk)}\n\n"

            loop = asyncio.get_running_loop()
            event_queue = asyncio.Queue()
            cancel_event = threading.Event()

            def stream_gen(
                engine,
                prompt,
                max_tokens,
                max_reasoning_tokens,
                enable_thinking,
                messages,
                use_prefix_cache,
                cancel_event,
            ):
                phase_usage = []
                content_filter = ControlTokenFilter()
                reasoning_filter = ControlTokenFilter()
                tool_filter = ToolMarkupFilter() if tools else None
                content_stopped = False
                reasoning_stopped = False
                if enable_thinking:
                    # Phase 1: Reasoning phase with hard </think> stop token (248069)
                    thought_text = ""
                    sanitized_thought_text = ""
                    reasoning_complete = False
                    stop_ids_phase1 = [248046, 248044, 248069]
                    for ev in engine.stream_generate_tokens(
                        prompt=prompt,
                        max_tokens=max_reasoning_tokens,
                        cancel_event=cancel_event,
                        use_speculative=True,
                        stop_token_ids=stop_ids_phase1,
                        messages=messages,
                        use_prefix_cache=use_prefix_cache,
                    ):
                        if cancel_event.is_set():
                            return
                        ev_type, payload = ev
                        if ev_type == "token":
                            txt = str(payload)
                            if reasoning_complete:
                                continue
                            thought_text += txt
                            if "</think>" in txt or "</think>" in thought_text:
                                clean_txt = txt.replace("</think>", "")
                                clean_txt, reasoning_stopped = reasoning_filter.push(clean_txt)
                                if clean_txt:
                                    sanitized_thought_text += clean_txt
                                    yield ("reasoning_token", clean_txt)
                                reasoning_complete = True
                            else:
                                clean_txt = txt.replace("<think>", "")
                                clean_txt, reasoning_stopped = reasoning_filter.push(clean_txt)
                                if clean_txt:
                                    sanitized_thought_text += clean_txt
                                    yield ("reasoning_token", clean_txt)
                            if reasoning_stopped:
                                break
                        elif ev_type == "usage":
                            phase_usage.append(payload)

                    if not reasoning_stopped:
                        trailing_reasoning = reasoning_filter.finish()
                        if trailing_reasoning:
                            sanitized_thought_text += trailing_reasoning
                            yield ("reasoning_token", trailing_reasoning)

                    # Phase 2: Direct Structured Code generation after </think>
                    code_prompt = prompt + sanitized_thought_text
                    if not code_prompt.rstrip().endswith("</think>"):
                        code_prompt = code_prompt.rstrip() + "\n</think>\n\n"
                    else:
                        code_prompt = code_prompt.rstrip() + "\n\n"

                    for ev in engine.stream_generate_tokens(
                        prompt=code_prompt,
                        max_tokens=max_tokens,
                        cancel_event=cancel_event,
                        use_speculative=True,
                        stop_token_ids=[248046, 248044],
                        messages=messages,
                        use_prefix_cache=use_prefix_cache,
                    ):
                        if cancel_event.is_set():
                            return
                        if ev[0] == "usage":
                            phase_usage.append(ev[1])
                        elif ev[0] == "token" and not content_stopped:
                            clean_txt = str(ev[1]).replace("<think>", "").replace("</think>", "")
                            clean_txt, content_stopped = content_filter.push(clean_txt)
                            if tool_filter is not None:
                                clean_txt = tool_filter.push(clean_txt)
                            if clean_txt:
                                yield ("token", clean_txt)
                else:
                    # Direct mode skips the separate reasoning generation pass.
                    for ev in engine.stream_generate_tokens(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        cancel_event=cancel_event,
                        use_speculative=True,
                        stop_token_ids=[248046, 248044],
                        messages=messages,
                        use_prefix_cache=use_prefix_cache,
                    ):
                        if cancel_event.is_set():
                            return
                        if ev[0] == "usage":
                            phase_usage.append(ev[1])
                        elif ev[0] == "token" and not content_stopped:
                            clean_txt = str(ev[1]).replace("<think>", "").replace("</think>", "")
                            clean_txt, content_stopped = content_filter.push(clean_txt)
                            if tool_filter is not None:
                                clean_txt = tool_filter.push(clean_txt)
                            if clean_txt:
                                yield ("token", clean_txt)

                if not content_stopped:
                    trailing_content = content_filter.finish()
                    if tool_filter is not None:
                        trailing_content = tool_filter.push(trailing_content)
                    if trailing_content:
                        yield ("token", trailing_content)
                if tool_filter is not None:
                    trailing_content = tool_filter.finish()
                    if trailing_content:
                        yield ("token", trailing_content)
                    if tool_filter.captured_text:
                        yield ("tool_markup", tool_filter.captured_text)

                if phase_usage:
                    yield (
                        "usage",
                        combine_generation_usage(
                            phase_usage,
                            reasoning_phase_count=1 if enable_thinking else 0,
                        ),
                    )

            worker.task_queue.put((
                stream_gen,
                (prompt, max_tokens, max_reasoning_tokens, enable_thinking, messages, use_prefix_cache),
                cancel_event,
                event_queue,
                loop,
            ))

            accumulated_text = ""
            tool_markup_text = ""
            stats = None

            try:
                while True:
                    try:
                        ev_type, payload = await asyncio.wait_for(event_queue.get(), timeout=20.0)
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
                        continue

                    if ev_type == "done":
                        break

                    if ev_type == "usage":
                        stats = payload
                        continue

                    if ev_type == "tool_markup":
                        tool_markup_text += str(payload)
                        continue

                    if ev_type == "reasoning_token":
                        txt = str(payload)
                        if txt:
                            delta_data = {
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"reasoning_content": txt},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(delta_data)}\n\n"

                    elif ev_type == "token":
                        txt = str(payload)
                        clean_txt = txt.replace("<think>", "").replace("</think>", "")
                        accumulated_text += clean_txt
                        if clean_txt:
                            delta_data = {
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": clean_txt},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(delta_data)}\n\n"

                # Check for tool calls in accumulated output
                _, tool_calls = extract_tool_calls(tool_markup_text)
                if tool_calls:
                    for idx, tc in enumerate(tool_calls):
                        tc_chunk = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "index": idx,
                                        "id": tc["id"],
                                        "type": "function",
                                        "function": {
                                            "name": tc["function"]["name"],
                                            "arguments": tc["function"]["arguments"]
                                        }
                                    }]
                                },
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(tc_chunk)}\n\n"

                # Final finish reason chunk with usage
                finish_reason = "tool_calls" if tool_calls else "stop"
                final_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason
                    }],
                }
                if stats:
                    final_chunk["usage"] = {
                        "prompt_tokens": stats.get("prompt_tokens", 0),
                        "completion_tokens": stats.get("completion_tokens", 0),
                        "total_tokens": stats.get("prompt_tokens", 0) + stats.get("completion_tokens", 0),
                        "tokens_per_second": stats.get("tokens_per_second", 0.0),
                        "generation_seconds": stats.get("generation_seconds", 0.0),
                        "reasoning_tokens": stats.get("reasoning_tokens", 0),
                        "reasoning_seconds": stats.get("reasoning_seconds", 0.0),
                        "accepted_from_draft": stats.get("accepted_from_draft", 0),
                        "acceptance_ratio": stats.get("acceptance_ratio", 0.0),
                        "cycles_completed": stats.get("cycles_completed", 0),
                        "phase_count": stats.get("phase_count", 1),
                        "fallback_ar": stats.get("fallback_ar", False),
                        "adaptive_block_reductions": stats.get("adaptive_block_reductions", 0),
                        "adaptive_block_min": stats.get("adaptive_block_min"),
                        "prefill_seconds": stats.get("prefill_seconds", 0.0),
                        "prefill_tokens_per_second": stats.get("prefill_tokens_per_second", 0.0),
                        "physical_prefill_tokens": stats.get("physical_prefill_tokens", 0),
                        "prefill_tokens_restored": stats.get("prefill_tokens_restored", 0),
                        "prefill_tokens_computed": stats.get("prefill_tokens_computed", 0),
                        "prefix_cache_hit_tokens": stats.get("prefix_cache_hit_tokens", 0),
                        "prefix_cache_lookup_ms": stats.get("prefix_cache_lookup_ms", 0.0),
                        "reasoning_enabled": enable_thinking,
                        "max_completion_tokens": max_tokens,
                        "max_reasoning_tokens": max_reasoning_tokens,
                    }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            finally:
                cancel_event.set()

        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        loop = asyncio.get_running_loop()
        event_queue = asyncio.Queue()
        cancel_event = threading.Event()

        def run_sync(engine, prompt, max_tokens, temperature, cancel_event):
            res = engine.run_inference(
                prompt=prompt,
                enable_thinking=False,
                max_tokens=max_tokens,
                temperature=temperature,
                direct=True,
            )
            yield ("result", res)

        worker.task_queue.put((run_sync, (prompt, max_tokens, temperature), cancel_event, event_queue, loop))

        ev_type, out = await event_queue.get()

        thinking = (out.thinking_content or "").strip()
        raw_content = (out.code_content or out.raw_response or "").replace("<|im_end|>", "").strip()
        if raw_content.startswith("<think>"):
            end_think = raw_content.find("</think>")
            if end_think != -1:
                raw_content = raw_content[end_think + len("</think>"):].strip()

        clean_text, tool_calls = extract_tool_calls(raw_content)

        message_dict: Dict[str, Any] = {
            "role": "assistant",
            "content": clean_text if clean_text else None,
        }
        if thinking:
            message_dict["reasoning_content"] = thinking
        if tool_calls:
            message_dict["tool_calls"] = tool_calls

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message_dict,
                    "finish_reason": "tool_calls" if tool_calls else "stop"
                }
            ],
            "usage": {
                "prompt_tokens": out.prompt_tokens,
                "completion_tokens": out.completion_tokens,
                "total_tokens": out.prompt_tokens + out.completion_tokens
            }
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
