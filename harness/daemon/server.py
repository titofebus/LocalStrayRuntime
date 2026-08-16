"""Lightweight Unix Domain Socket Resident Daemon for Instant DFlash Inference."""
import os
import sys
import json
import time
import socket
import signal
from pathlib import Path
from harness.executors.dflash_engine import DFlashEngine
from harness.core.models import ModelOutput

SOCKET_PATH = "/tmp/local_eval_daemon.sock"
PID_FILE = "/tmp/local_eval_daemon.pid"

os.environ["DFLASH_DAEMON_WORKER"] = "1"

class DaemonServer:
    def __init__(self):
        self.engine = DFlashEngine(draft_quant="w8")

    def warmup(self):
        """Pre-warm Metal GPU shader graph."""
        print("[DaemonServer] Pre-loading models into GPU RAM...", flush=True)
        self.engine._ensure_loaded()
        print("[DaemonServer] Running warmup forward pass...", flush=True)
        warmup_prompt = "<|im_start|>user\nfn main() {}\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        _ = self.engine.run_inference(prompt=warmup_prompt, max_tokens=10, direct=True)
        print("[DaemonServer] Metal GPU graph is 100% warm. Ready for instantaneous inference!", flush=True)

    def handle_request(self, raw_data: bytes) -> bytes:
        try:
            req = json.loads(raw_data.decode("utf-8"))
            prompt = req.get("prompt", "")
            language = req.get("language")
            max_tokens = req.get("max_tokens", 4096)
            temperature = req.get("temperature", 0.1)
            enable_thinking = req.get("enable_thinking", True)

            start = time.perf_counter()
            out = self.engine.run_inference(
                prompt=prompt,
                language=language,
                max_tokens=max_tokens,
                temperature=temperature,
                enable_thinking=enable_thinking,
                direct=True,
            )
            elapsed = time.perf_counter() - start

            resp = {
                "success": True,
                "model_name": out.model_name,
                "raw_response": out.raw_response,
                "code_content": out.code_content,
                "thinking_content": out.thinking_content,
                "completion_tokens": out.completion_tokens,
                "tokens_per_sec": out.tokens_per_sec,
                "elapsed_seconds": round(elapsed, 3),
                "peak_memory_gb": out.peak_memory_gb,
            }
            return json.dumps(resp).encode("utf-8")
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}).encode("utf-8")

    def run(self):
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)

        # 1. Warm up model & Metal GPU graph FIRST
        self.warmup()

        # 2. Write PID file and bind socket when 100% READY
        Path(PID_FILE).write_text(str(os.getpid()))
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(SOCKET_PATH)
        server_sock.listen(16)
        print(f"[DaemonServer] Listening on {SOCKET_PATH} (PID: {os.getpid()})", flush=True)

        def sig_handler(signum, frame):
            print("[DaemonServer] Shutting down cleanly...", flush=True)
            if os.path.exists(SOCKET_PATH):
                os.remove(SOCKET_PATH)
            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)
            sys.exit(0)

        signal.signal(signal.SIGTERM, sig_handler)
        signal.signal(signal.SIGINT, sig_handler)

        while True:
            try:
                conn, _ = server_sock.accept()
                with conn:
                    data = b""
                    while True:
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        data += chunk
                        if b"__END_OF_REQUEST__" in data:
                            data = data.replace(b"__END_OF_REQUEST__", b"")
                            break
                    if data.strip():
                        resp = self.handle_request(data)
                        try:
                            conn.sendall(resp)
                        except Exception as e:
                            print(f"[DaemonServer] Client disconnected before response: {e}", flush=True)
            except Exception as e:
                print(f"[DaemonServer] Connection error: {e}", flush=True)

if __name__ == "__main__":
    server = DaemonServer()
    server.run()
