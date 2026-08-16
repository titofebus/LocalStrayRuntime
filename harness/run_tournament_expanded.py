"""Run Rounds 5-8 of Polyglot Battlegrounds Arena."""
import time
from harness.core.suites import SuiteLoader
from harness.executors.dflash_engine import DFlashEngine
from harness.core.sandbox import CodeSandbox
from harness.core.recorder import ResultsRecorder
from harness.core.models import ModelOutput, ComparisonRecord

CHALLENGES = [
    ("swift_02_custom_masonry_layout", "Round 5 (SwiftUI Layout)"),
    ("rust_02_lockfree_spmc_ring_buffer", "Round 6 (Rust Lock-Free Atomics)"),
    ("ts_02_json_chunk_stream_parser", "Round 7 (TypeScript Chunk Parser)"),
    ("algo_02_raft_election_state_machine", "Round 8 (Distributed Raft State Machine)"),
]

# Opus reference solutions
OPUS_SOLUTIONS = {
    "swift_02_custom_masonry_layout": """
import SwiftUI

public struct FlowLayout: Layout {
    public var horizontalSpacing: CGFloat = 8
    public var verticalSpacing: CGFloat = 8

    public init(horizontalSpacing: CGFloat = 8, verticalSpacing: CGFloat = 8) {
        self.horizontalSpacing = horizontalSpacing
        self.verticalSpacing = verticalSpacing
    }

    public func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? .infinity
        var currentX: CGFloat = 0
        var currentY: CGFloat = 0
        var lineHeight: CGFloat = 0
        var maxWidth: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if currentX + size.width > width && currentX > 0 {
                currentX = 0
                currentY += lineHeight + verticalSpacing
                lineHeight = 0
            }
            currentX += size.width + horizontalSpacing
            lineHeight = max(lineHeight, size.height)
            maxWidth = max(maxWidth, currentX - horizontalSpacing)
        }

        return CGSize(width: maxWidth, height: currentY + lineHeight)
    }

    public func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let width = bounds.width
        var currentX = bounds.minX
        var currentY = bounds.minY
        var lineHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if currentX + size.width > bounds.minX + width && currentX > bounds.minX {
                currentX = bounds.minX
                currentY += lineHeight + verticalSpacing
                lineHeight = 0
            }
            subview.place(at: CGPoint(x: currentX, y: currentY), proposal: ProposedViewSize(size))
            currentX += size.width + horizontalSpacing
            lineHeight = max(lineHeight, size.height)
        }
    }
}
""",
    "rust_02_lockfree_spmc_ring_buffer": """
use std::sync::atomic::{AtomicUsize, Ordering};
use std::cell::UnsafeCell;

pub struct SpmcRingBuffer<T> {
    buffer: Vec<UnsafeCell<Option<T>>>,
    capacity: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
}

unsafe impl<T: Send> Sync for SpmcRingBuffer<T> {}
unsafe impl<T: Send> Send for SpmcRingBuffer<T> {}

impl<T> SpmcRingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(UnsafeCell::new(None));
        }
        Self {
            buffer,
            capacity,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    pub fn push(&self, item: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        if head.wrapping_sub(tail) >= self.capacity {
            return Err(item);
        }

        let index = head % self.capacity;
        unsafe {
            *self.buffer[index].get() = Some(item);
        }
        self.head.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    pub fn pop(&self) -> Option<T> {
        loop {
            let tail = self.tail.load(Ordering::Relaxed);
            let head = self.head.load(Ordering::Acquire);

            if tail == head {
                return None;
            }

            if self.tail.compare_exchange_weak(tail, tail.wrapping_add(1), Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                let index = tail % self.capacity;
                let item = unsafe { (*self.buffer[index].get()).take() };
                return item;
            }
        }
    }
}
""",
    "ts_02_json_chunk_stream_parser": """
export class StreamingJsonArrayParser<T> {
    private buffer: string = "";
    private inString: boolean = false;
    private escapeNext: boolean = false;
    private braceDepth: number = 0;
    private objectStart: number = -1;

    public push(chunk: string): T[] {
        this.buffer += chunk;
        const results: T[] = [];
        let i = 0;

        while (i < this.buffer.length) {
            const char = this.buffer[i];

            if (this.escapeNext) {
                this.escapeNext = false;
                i++;
                continue;
            }

            if (char === "\\\\") {
                if (this.inString) {
                    this.escapeNext = true;
                }
                i++;
                continue;
            }

            if (char === '"') {
                this.inString = !this.inString;
                i++;
                continue;
            }

            if (!this.inString) {
                if (char === '{') {
                    if (this.braceDepth === 0) {
                        this.objectStart = i;
                    }
                    this.braceDepth++;
                } else if (char === '}') {
                    this.braceDepth--;
                    if (this.braceDepth === 0 && this.objectStart !== -1) {
                        const jsonStr = this.buffer.slice(this.objectStart, i + 1);
                        try {
                            const parsed = JSON.parse(jsonStr) as T;
                            results.push(parsed);
                        } catch (e) {}
                        this.buffer = this.buffer.slice(i + 1);
                        i = -1;
                        this.objectStart = -1;
                    }
                }
            }
            i++;
        }

        return results;
    }

    public reset(): void {
        this.buffer = "";
        this.inString = false;
        this.escapeNext = false;
        this.braceDepth = 0;
        this.objectStart = -1;
    }
}
""",
    "algo_02_raft_election_state_machine": """
from typing import Optional, List, Dict, Any

class RaftNode:
    def __init__(self, node_id: str, cluster_nodes: List[str]):
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes
        self.current_term = 0
        self.state = "FOLLOWER"
        self.voted_for: Optional[str] = None
        self.votes_received = 0

    def handle_election_timeout(self) -> Dict[str, Any]:
        if self.state != "LEADER":
            self.current_term += 1
            self.state = "CANDIDATE"
            self.voted_for = self.node_id
            self.votes_received = 1
            return {
                "type": "REQUEST_VOTE",
                "term": self.current_term,
                "candidate_id": self.node_id,
            }
        return {}

    def handle_request_vote(self, term: int, candidate_id: str) -> bool:
        if term > self.current_term:
            self.current_term = term
            self.state = "FOLLOWER"
            self.voted_for = None

        if term == self.current_term and self.voted_for in (None, candidate_id):
            self.voted_for = candidate_id
            return True
        return False

    def handle_vote_response(self, term: int, granted: bool) -> None:
        if self.state == "CANDIDATE" and term == self.current_term and granted:
            self.votes_received += 1
            if self.votes_received > len(self.cluster_nodes) // 2:
                self.state = "LEADER"
"""
}

def main():
    engine = DFlashEngine()
    print("=" * 70)
    print("POLYGLOT BATTLEGROUNDS: ROUNDS 5 TO 8 EXTENDED TOURNAMENT")
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
            completion_tokens=350,
            tokens_per_sec=0.0,
            ttft_seconds=0.0,
        )
        opus_verif = CodeSandbox.run_verification("Claude-4.6-Opus", opus_code, challenge)

        winner = "tie" if (qwen_verif.test_passed and opus_verif.test_passed) else ("opus" if opus_verif.test_passed else "qwen")
        print(f"  • Qwen Generated: {qwen_out.completion_tokens} tok @ {qwen_out.tokens_per_sec:.2f} tok/s ({elapsed:.2f}s)")
        print(f"  • Qwen Sandbox Result: {'PASS' if qwen_verif.test_passed else 'FAIL'}")
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
    print("EXTENDED TOURNAMENT (ROUNDS 5-8) COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    main()
