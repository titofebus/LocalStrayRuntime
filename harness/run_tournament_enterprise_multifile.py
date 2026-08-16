"""Run Arena 2.0: Multi-Module Polyglot Enterprise Tournament (Rounds 13 to 16)."""
import time
from harness.core.suites import SuiteLoader
from harness.executors.dflash_engine import DFlashEngine
from harness.core.sandbox import CodeSandbox
from harness.core.recorder import ResultsRecorder
from harness.core.models import ModelOutput, ComparisonRecord

CHALLENGES = [
    ("rust_03_multifile_event_pipeline", "Round 13 (Rust Multi-Module Pipeline)"),
    ("go_02_multifile_raft_wal", "Round 14 (Go Multi-Module Raft WAL)"),
    ("swift_03_multifile_actor_rpc", "Round 15 (Swift 6 Multi-Module RPC)"),
    ("ts_03_multifile_event_broker", "Round 16 (TS Multi-Module Broker)"),
]

OPUS_SOLUTIONS = {
    "rust_03_multifile_event_pipeline": """
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Debug, Clone, PartialEq)]
pub enum EventPayload {
    Text(String),
    Binary(Vec<u8>),
}

#[derive(Debug, Clone)]
pub struct Event {
    pub id: u64,
    pub topic: String,
    pub payload: EventPayload,
}

#[derive(Default)]
pub struct PipelineMetrics {
    pub processed_count: AtomicU64,
    pub error_count: AtomicU64,
}

pub struct EventPipeline {
    workers: usize,
    sender: mpsc::Sender<Event>,
    receiver: tokio::sync::Mutex<Option<mpsc::Receiver<Event>>>,
    metrics: Arc<PipelineMetrics>,
}

impl EventPipeline {
    pub fn new(workers: usize, buffer_size: usize) -> Self {
        let (tx, rx) = mpsc::channel(buffer_size);
        Self {
            workers,
            sender: tx,
            receiver: tokio::sync::Mutex::new(Some(rx)),
            metrics: Arc::new(PipelineMetrics::default()),
        }
    }

    pub async fn submit(&self, event: Event) -> Result<(), &'static str> {
        self.sender.send(event).await.map_err(|_| "Pipeline channel closed")
    }

    pub async fn start<F>(&self, handler: F)
    where
        F: Fn(Event) -> Result<(), &'static str> + Send + Sync + 'static,
    {
        let mut rx_guard = self.receiver.lock().await;
        if let Some(mut rx) = rx_guard.take() {
            let handler = Arc::new(handler);
            let metrics = Arc::clone(&self.metrics);

            tokio::spawn(async move {
                while let Some(event) = rx.recv().await {
                    match handler(event) {
                        Ok(_) => {
                            metrics.processed_count.fetch_add(1, Ordering::SeqCst);
                        }
                        Err(_) => {
                            metrics.error_count.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                }
            });
        }
    }

    pub fn metrics(&self) -> &PipelineMetrics {
        &self.metrics
    }
}
""",
    "go_02_multifile_raft_wal": """
package main

import (
    "bytes"
    "fmt"
    "hash/crc32"
    "sync"
)

type LogEntry struct {
    Index    uint64
    Term     uint64
    Data     []byte
    Checksum uint32
}

type Snapshot struct {
    LastIndex uint64
    LastTerm  uint64
    State     []byte
}

type StorageEngine struct {
    mu       sync.RWMutex
    logs     map[uint64]LogEntry
    snapshot *Snapshot
}

func NewStorageEngine() *StorageEngine {
    return &StorageEngine{
        logs: make(map[uint64]LogEntry),
    }
}

func (s *StorageEngine) AppendLog(entry LogEntry) error {
    s.mu.Lock()
    defer s.mu.Unlock()

    entry.Checksum = crc32.ChecksumIEEE(entry.Data)
    s.logs[entry.Index] = entry
    return nil
}

func (s *StorageEngine) ReadLog(index uint64) (*LogEntry, error) {
    s.mu.RLock()
    defer s.mu.RUnlock()

    entry, ok := s.logs[index]
    if !ok {
        return nil, fmt.Errorf("log entry not found")
    }

    calc := crc32.ChecksumIEEE(entry.Data)
    if entry.Checksum != calc {
        return nil, fmt.Errorf("crc32 checksum mismatch")
    }
    return &entry, nil
}

func (s *StorageEngine) ApplySnapshot(snap Snapshot) error {
    s.mu.Lock()
    defer s.mu.Unlock()

    s.snapshot = &snap
    for idx := range s.logs {
        if idx <= snap.LastIndex {
            delete(s.logs, idx)
        }
    }
    return nil
}

func (s *StorageEngine) CompactLogs(uptoIndex uint64) error {
    s.mu.Lock()
    defer s.mu.Unlock()

    for idx := range s.logs {
        if idx <= uptoIndex {
            delete(s.logs, idx)
        }
    }
    return nil
}
""",
    "swift_03_multifile_actor_rpc": """
import Foundation

struct RPCRequest: Sendable, Hashable {
    let id: String
    let method: String
    let payload: String
}

struct RPCResponse: Sendable {
    let id: String
    let result: String
    let isError: Bool
}

actor RPCClient {
    private let maxConcurrent: Int
    private var inFlight: [String: Task<RPCResponse, any Error>] = [:]

    init(maxConcurrent: Int) {
        self.maxConcurrent = maxConcurrent
    }

    func send(
        request: RPCRequest,
        transport: @Sendable @escaping (RPCRequest) async throws -> RPCResponse
    ) async throws -> RPCResponse {
        if let existing = inFlight[request.id] {
            return try await existing.value
        }

        let task = Task {
            try await transport(request)
        }
        inFlight[request.id] = task

        do {
            let res = try await task.value
            inFlight.removeValue(forKey: request.id)
            return res
        } catch {
            inFlight.removeValue(forKey: request.id)
            throw error
        }
    }
}
""",
    "ts_03_multifile_event_broker": """
interface EventEnvelope<T = unknown> {
    id: string;
    topic: string;
    timestamp: number;
    payload: T;
}

type EventHandler<T = unknown> = (event: EventEnvelope<T>) => void | Promise<void>;

class EventBroker {
    private subscriptions: Array<{ pattern: RegExp; handler: EventHandler }> = [];

    subscribe<T = unknown>(topicPattern: string, handler: EventHandler<T>): void {
        const regexStr = "^" + topicPattern.replace(/\./g, "\\.").replace(/\*/g, "[^.]+") + "$";
        this.subscriptions.push({ pattern: new RegExp(regexStr), handler: handler as EventHandler });
    }

    publish<T = unknown>(event: EventEnvelope<T>): void {
        for (const sub of this.subscriptions) {
            if (sub.pattern.test(event.topic)) {
                sub.handler(event);
            }
        }
    }
}

class ReplayBuffer {
    private events: EventEnvelope[] = [];

    record(event: EventEnvelope): void {
        this.events.push(event);
    }

    replay(topicPattern: string): EventEnvelope[] {
        const regexStr = "^" + topicPattern.replace(/\./g, "\\.").replace(/\*/g, "[^.]+") + "$";
        const regex = new RegExp(regexStr);
        return this.events.filter(e => regex.test(e.topic));
    }
}
"""
}

def main():
    engine = DFlashEngine()
    print("=" * 75)
    print("ARENA 2.0: MULTI-MODULE POLYGLOT ENTERPRISE TOURNAMENT (ROUNDS 13-16)")
    print("=" * 75)

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
        if not qwen_verif.test_passed:
            print(f"    Compiler Stderr: {qwen_verif.error_message}")
        print(f"  • Opus Sandbox Result: {'PASS' if opus_verif.test_passed else 'FAIL'}")
        print(f"  • Outcome: {winner.upper()}")

        rec = ComparisonRecord(
            run_id=f"enterprise_{challenge_id}",
            challenge_id=challenge.id,
            challenge_title=challenge.title,
            qwen_output=qwen_out,
            qwen_verification=qwen_verif,
            opus_output=opus_out,
            opus_verification=opus_verif,
            winner=winner,
        )
        ResultsRecorder.save_record(rec)

    print("\n" + "=" * 75)
    print("ARENA 2.0 ENTERPRISE TOURNAMENT COMPLETED")
    print("=" * 75)

if __name__ == "__main__":
    main()
