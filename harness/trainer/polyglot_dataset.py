"""Polyglot & Multi-Domain dataset curator for DFlash 5,000-prompt training (Go, TypeScript/JS, Python, Rust, Swift, C++, SQL, Reasoning)."""
import json
from pathlib import Path
from typing import List, Dict
from harness.config import TRAINING_DIR

DATASET_DIR = TRAINING_DIR / "dataset"

# 1. Polyglot Systems Code Seeds (50% -> 2,500 prompts)
CODE_SEEDS = [
    # Go (Golang)
    {"lang": "go", "seed": "Write a high-throughput worker pool in Go using buffered channels, sync.WaitGroup, context cancellation, and exponential backoff retry."},
    {"lang": "go", "seed": "Implement an HTTP middleware pipeline in Go with request logging, JWT validation, rate limiting with token bucket, and recover panic handler."},
    {"lang": "go", "seed": "Build a thread-safe in-memory cache in Go using sync.RWMutex, generic type parameters [K comparable, V any], TTL eviction, and background cleanup goroutine."},
    {"lang": "go", "seed": "Write a streaming JSON NDJSON parser in Go with io.Reader, bufio.Scanner, and custom error unmarshaling into strongly-typed domain structs."},
    {"lang": "go", "seed": "Implement a distributed lock manager client in Go using Redis SETNX/Redlock algorithm with TTL heartbeat renewal and lease expiration context."},

    # TypeScript & JavaScript
    {"lang": "typescript", "seed": "Implement a strictly-typed state machine in TypeScript with exhaustive discriminating union transitions, event payload validation, and middleware interceptors."},
    {"lang": "typescript", "seed": "Write a Next.js 15 Server Action handler with Zod input validation, optimistic UI mutation rollback, and cookie-based session management."},
    {"lang": "typescript", "seed": "Build a type-safe API client in TypeScript with automatic query parameter serialization, exponential retry, and typed error responses using Fetch API."},
    {"lang": "typescript", "seed": "Create an async stream multiplexer in TypeScript using AsyncIterableIterator, WebStreams API, and backpressure buffer management."},
    {"lang": "typescript", "seed": "Implement a React 19 custom hook for real-time WebSocket state syncing with auto-reconnection, offline queueing, and useSyncExternalStore."},

    # Python
    {"lang": "python", "seed": "Write an asynchronous SSE streaming endpoint in FastAPI with Python async generators, Redis pub/sub listener, and client disconnect cancellation."},
    {"lang": "python", "seed": "Implement a vectorized 2D spatial collision detector in Python using pure NumPy broadcasting, bounding box intersection, and quadtree spatial partitioning."},
    {"lang": "python", "seed": "Build a thread-safe connection pool in Python with asyncio.Queue, async context managers (__aenter__/__aexit__), and healthcheck keepalives."},
    {"lang": "python", "seed": "Write a high-performance batch pipeline in Python using multiprocessing, shared memory (multiprocessing.shared_memory), and zero-copy NumPy array views."},

    # Rust
    {"lang": "rust", "seed": "Implement a lock-free multi-producer single-consumer ring buffer in Rust using atomic CAS operations (AtomicUsize), crossbeam epoch, and UnsafeCell."},
    {"lang": "rust", "seed": "Write a Tokio-based streaming WebSocket frame parser in Rust with zero-copy BytesMut slicing, Ping/Pong heartbeat, and graceful shutdown signal."},
    {"lang": "rust", "seed": "Create an async LRU cache in Rust with tokio::sync::RwLock, TTL expiration binary heap, and O(1) key eviction using std::collections::HashMap and LinkedList."},
    {"lang": "rust", "seed": "Implement a custom memory allocator in Rust using std::alloc::GlobalAlloc with jemalloc alignment, arena chunking, and memory telemetry counters."},

    # Swift
    {"lang": "swift", "seed": "Write a Swift 6 actor-isolated network cache with task coalescing, async/await cancellation, OSAllocatedUnfairLock, and Sendable error propagation."},
    {"lang": "swift", "seed": "Implement a high-performance SwiftUI View with custom Layout protocol, preference keys, matchedGeometryEffect, and fluid spring animations."},
    {"lang": "swift", "seed": "Build a thread-safe ring buffer in Swift using Noncopyable types (~Copyable), raw byte pointers, and generic element storage."},

    # C / C++
    {"lang": "cpp", "seed": "Write a high-performance thread-safe memory pool in C++20 with std::pmr::memory_resource, cache-line aligned chunks, and lock-free freelist."},
    {"lang": "cpp", "seed": "Implement an asynchronous ring-buffer event loop in C++20 using coroutines (co_await, co_yield), std::span, and POSIX io_uring support."}
]

# 2. Reasoning & Algorithmic Seeds (25% -> 1,250 prompts)
REASONING_SEEDS = [
    "Analyze the asymptotic time and space complexity of hierarchical speculative tree verification versus standard speculative sampling on Unified Memory.",
    "Formally prove the correctness of a lock-free work-stealing deque using TSO memory model invariants, acquire-release fences, and sequential consistency.",
    "Derive the exact closed-form gradient update for group-relative policy optimization (GRPO) with normalized advantage estimation and KL penalty.",
    "Provide a detailed step-by-step mathematical proof of why Rotary Position Embeddings (RoPE) preserve relative token distances across attention heads.",
    "Walk through the memory layout, TLB translation, and cache line contention mechanisms in Apple Silicon Unified Memory during concurrent 4-bit GEMM matrix multiplication.",
    "Step through the algorithmic resolution of cyclic graph dependencies in topological build ordering with Tarjan's strongly connected components algorithm."
]

# 3. Multi-Turn Dialogue & Architecture (15% -> 750 prompts)
DIALOGUE_SEEDS = [
    "Refactor this legacy monolithic controller into clean domain-driven service layers with clear interface boundaries, DTOs, and dependency injection.",
    "Debug this concurrency deadlock where two worker threads acquire mutex locks in opposing order and propose a deterministic lock-hierarchy solution.",
    "Optimize this slow database query by introducing compound B-tree indexing, covering indexes, and rewriting recursive subqueries into PostgreSQL CTEs.",
    "Convert this synchronous blocking file I/O pipeline into an asynchronous non-blocking stream with reactive backpressure and flow control."
]

# 4. SQL, Schema & Tool Calling (10% -> 500 prompts)
SCHEMA_SEEDS = [
    "Design a distributed event-sourcing database schema in PostgreSQL with immutable ledger tables, JSONB payload indexing, and materialized aggregate views.",
    "Write a complete OpenAPI 3.1 specification in YAML for an asynchronous LLM generation service with streaming SSE endpoints and usage telemetry schemas.",
    "Execute a python script in the sandbox to verify the statistical acceptance rate and latency percentiles (p50, p95, p99) of 10,000 speculative token trials."
]

class PolyglotDatasetCurator:
    @staticmethod
    def prepare_dataset(total_prompts: int = 5000) -> Path:
        """Prepare and cache 5,000 multi-domain prompts into JSONL format."""
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        output_file = DATASET_DIR / "polyglot_corpus.jsonl"

        print(f"[PolyglotCurator] Generating {total_prompts} multi-domain training prompts (Go, TS, Python, Rust, Swift, C++, SQL)...")
        samples = []

        # 1. Systems Code (50% -> 2,500 prompts)
        n_code = int(total_prompts * 0.50)
        for i in range(n_code):
            item = CODE_SEEDS[i % len(CODE_SEEDS)]
            samples.append({
                "id": f"code_{item['lang']}_{i:04d}",
                "domain": f"code_{item['lang']}",
                "prompt": f"<|im_start|>system\nYou are a Principal {item['lang'].capitalize()} Engineer. Write clean, idiomatic, high-performance production code.<|im_end|>\n<|im_start|>user\n{item['seed']}<|im_end|>\n<|im_start|>assistant\n```\n"
            })

        # 2. Reasoning Traces (25% -> 1,250 prompts)
        n_reasoning = int(total_prompts * 0.25)
        for i in range(n_reasoning):
            seed = REASONING_SEEDS[i % len(REASONING_SEEDS)]
            samples.append({
                "id": f"reasoning_{i:04d}",
                "domain": "reasoning",
                "prompt": f"<|im_start|>user\n{seed} Think through this carefully and break down each step in detail.<|im_end|>\n<|im_start|>assistant\n<think>\nLet's analyze the problem step-by-step."
            })

        # 3. Multi-Turn Dialogue (15% -> 750 prompts)
        n_dialogue = int(total_prompts * 0.15)
        for i in range(n_dialogue):
            seed = DIALOGUE_SEEDS[i % len(DIALOGUE_SEEDS)]
            samples.append({
                "id": f"dialogue_{i:04d}",
                "domain": "dialogue",
                "prompt": f"<|im_start|>system\nYou are an expert systems software architect.<|im_end|>\n<|im_start|>user\n{seed}<|im_end|>\n<|im_start|>assistant\n"
            })

        # 4. SQL, Schemas & Tools (10% -> 500 prompts)
        n_schema = total_prompts - len(samples)
        for i in range(n_schema):
            seed = SCHEMA_SEEDS[i % len(SCHEMA_SEEDS)]
            samples.append({
                "id": f"schema_{i:04d}",
                "domain": "schema",
                "prompt": f"<|im_start|>user\n{seed}<|im_end|>\n<|im_start|>assistant\n"
            })

        with open(output_file, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        print(f"[PolyglotCurator] Successfully saved {len(samples)} multi-domain prompts to {output_file}")
        return output_file

    @staticmethod
    def load_prompts(total_prompts: int = 5000) -> List[str]:
        output_file = DATASET_DIR / "polyglot_corpus.jsonl"
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                lines = [l for l in f if l.strip()]
            if len(lines) == total_prompts:
                return [json.loads(l)["prompt"] for l in lines]

        PolyglotDatasetCurator.prepare_dataset(total_prompts)
        prompts = []
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    prompts.append(json.loads(line)["prompt"])
        return prompts
