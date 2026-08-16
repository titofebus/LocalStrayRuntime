"""Run Rounds 9-12 of Polyglot Battlegrounds Arena (Go, C++, Bash, Distributed Architecture)."""
import time
from harness.core.suites import SuiteLoader
from harness.executors.dflash_engine import DFlashEngine
from harness.core.sandbox import CodeSandbox
from harness.core.recorder import ResultsRecorder
from harness.core.models import ModelOutput, ComparisonRecord

CHALLENGES = [
    ("go_01_worker_pool_backpressure", "Round 9 (Go Worker Pool)"),
    ("cpp_01_cacheline_ring_buffer", "Round 10 (C++20 Atomic Ring Buffer)"),
    ("bash_01_streaming_log_aggregator", "Round 11 (Bash Streaming Aggregator)"),
    ("arch_01_distributed_rate_limiter", "Round 12 (Distributed Rate Limiter)"),
]

# Opus reference solutions
OPUS_SOLUTIONS = {
    "go_01_worker_pool_backpressure": """
package main

import (
    "context"
    "sync"
)

type Task func(ctx context.Context) error

type Pool struct {
    workers   int
    queue     chan Task
    wg        sync.WaitGroup
    cancelCtx context.CancelFunc
}

func NewPool(workers int, queueSize int) *Pool {
    return &Pool{
        workers: workers,
        queue:   make(chan Task, queueSize),
    }
}

func (p *Pool) Submit(task Task) bool {
    select {
    case p.queue <- task:
        return true
    default:
        return false
    }
}

func (p *Pool) Start(ctx context.Context) {
    wCtx, cancel := context.WithCancel(ctx)
    p.cancelCtx = cancel

    for i := 0; i < p.workers; i++ {
        p.wg.Add(1)
        go func() {
            defer p.wg.Done()
            for {
                select {
                case <-wCtx.Done():
                    return
                case task, ok := <-p.queue:
                    if !ok {
                        return
                    }
                    _ = task(wCtx)
                }
            }
        }()
    }
}

func (p *Pool) Stop() {
    close(p.queue)
    if p.cancelCtx != nil {
        p.cancelCtx()
    }
    p.wg.Wait()
}
""",
    "cpp_01_cacheline_ring_buffer": """
#include <atomic>
#include <cstddef>
#include <array>
#include <optional>

template<typename T, size_t Capacity>
class SpscRingBuffer {
private:
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
    std::array<T, Capacity> buffer_{};

public:
    SpscRingBuffer() = default;

    bool push(const T& item) {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        const size_t current_tail = tail_.load(std::memory_order_acquire);

        if (current_head - current_tail >= Capacity) {
            return false; // Full
        }

        buffer_[current_head % Capacity] = item;
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t current_head = head_.load(std::memory_order_acquire);

        if (current_tail == current_head) {
            return false; // Empty
        }

        item = buffer_[current_tail % Capacity];
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }
};
""",
    "bash_01_streaming_log_aggregator": """
aggregate_logs() {
    awk '
    BEGIN {
        count_5xx = 0
        max_lat = 0
    }
    {
        status = $2 + 0
        latency = $4 + 0

        if (status >= 500) {
            count_5xx++
        }
        if (latency > max_lat) {
            max_lat = latency
        }
    }
    END {
        print "5xx_COUNT=" count_5xx
        print "MAX_LATENCY=" max_lat
    }'
}
""",
    "arch_01_distributed_rate_limiter": """
import threading
from collections import deque
from typing import Tuple, Dict

class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.lock = threading.Lock()
        self.user_logs: Dict[str, deque] = {}

    def allow_request(self, key: str, timestamp: float) -> Tuple[bool, int]:
        with self.lock:
            if key not in self.user_logs:
                self.user_logs[key] = deque()

            log = self.user_logs[key]
            boundary = timestamp - self.window_seconds

            # Prune timestamps outside window
            while log and log[0] <= boundary:
                log.popleft()

            if len(log) < self.max_requests:
                log.append(timestamp)
                remaining = self.max_requests - len(log)
                return True, remaining
            else:
                return False, 0
"""
}

def main():
    engine = DFlashEngine()
    print("=" * 70)
    print("POLYGLOT BATTLEGROUNDS: ROUNDS 9 TO 12 (GO, C++, BASH, ARCHITECTURE)")
    print("=" * 70)

    for challenge_id, round_title in CHALLENGES:
        challenge = SuiteLoader.get_challenge(challenge_id)
        print(f"\n► Executing {round_title}: {challenge.title} [{challenge.language.upper()}]")

        start = time.perf_counter()
        qwen_out, qwen_verif = engine.run_with_compiler_feedback(challenge, max_correction_attempts=1)
        elapsed = time.perf_counter() - start

        opus_code = OPUS_SOLUTIONS[challenge_id]
        opus_out = ModelOutput(
            model_name="Claude-4.6-Opus",
            raw_response=opus_code,
            code_content=opus_code,
            completion_tokens=300,
            tokens_per_sec=0.0,
            ttft_seconds=0.0,
        )
        opus_verif = CodeSandbox.run_verification("Claude-4.6-Opus", opus_code, challenge)

        winner = "tie" if (qwen_verif.test_passed and opus_verif.test_passed) else ("opus" if opus_verif.test_passed else "qwen")
        print(f"  • Qwen Generated: {qwen_out.completion_tokens} tok @ {qwen_out.tokens_per_sec:.2f} tok/s ({elapsed:.2f}s)")
        print(f"  • Qwen Sandbox Result: {'PASS' if qwen_verif.test_passed else 'FAIL'}")
        if not qwen_verif.test_passed:
            print(f"    Error: {qwen_verif.error_message}")
        print(f"  • Opus Sandbox Result: {'PASS' if opus_verif.test_passed else 'FAIL'}")
        print(f"  • Outcome: {winner.upper()}")

        rec = ComparisonRecord(
            run_id=f"polyglot_{challenge_id}",
            challenge_id=challenge.id,
            challenge_title=challenge.title,
            qwen_output=qwen_out,
            qwen_verification=qwen_verif,
            opus_output=opus_out,
            opus_verification=opus_verif,
            winner=winner,
        )
        ResultsRecorder.save_record(rec)

    print("\n" + "=" * 70)
    print("EXTENDED TOURNAMENT (ROUNDS 9-12) COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    main()
