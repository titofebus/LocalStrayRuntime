"""OpenAI-Compatible FastAPI Server Bridging Prime Agent to Local Resident MLX Daemon."""
import os
import json
import time
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from harness.daemon.client import DaemonClient

app = FastAPI(title="Local MLX Daemon OpenAI API Bridge")

MODEL_ID = "qwen3.8-27b"

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

def format_messages_to_qwen_chat(messages: List[Dict[str, Any]]) -> str:
    """Format standard OpenAI messages to Qwen3.8 ChatML prompt."""
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Parse multi-part content if present
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = "\n".join(text_parts) if text_parts else str(content)

        if role in ("system", "developer"):
            prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        elif role == "tool":
            prompt += f"<|im_start|>user\n[Tool Result]:\n{content}<|im_end|>\n"
        else:
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    prompt += "<|im_start|>assistant\n"
    return prompt

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model = body.get("model", MODEL_ID)
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.1)
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 4096
    stream = body.get("stream", False)
    enable_thinking = body.get("thinking", {}).get("type") != "disabled"

    prompt = format_messages_to_qwen_chat(messages)

    if stream:
        async def event_generator():
            req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            # Execute inference via resident daemon client in threadpool
            loop = asyncio.get_event_loop()
            out = await loop.run_in_executor(
                None,
                lambda: DaemonClient.run_inference(
                    prompt=prompt,
                    enable_thinking=enable_thinking,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            )

            # Stream out response text
            full_text = out.raw_response or out.code_content or ""

            # Send initial role delta
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

            # Send text chunks
            chunk_size = 32
            for i in range(0, len(full_text), chunk_size):
                text_slice = full_text[i:i+chunk_size]
                chunk_data = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text_slice},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.005)

            # Send finish reason
            final_chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        loop = asyncio.get_event_loop()
        out = await loop.run_in_executor(
            None,
            lambda: DaemonClient.run_inference(
                prompt=prompt,
                enable_thinking=enable_thinking,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )

        content = out.raw_response or out.code_content or ""
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        return {
            "id": req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": out.completion_tokens,
                "total_tokens": len(prompt.split()) + out.completion_tokens
            }
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
