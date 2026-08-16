"""Step 1 Benchmark Test: Go Raft WAL and Rust Event Pipeline."""
import sys
import time
from harness.core.suites import SuiteLoader
from harness.executors.dflash_engine import DFlashEngine
from harness.core.sandbox import CodeSandbox

def main():
    print("=" * 60, flush=True)
    print("STEP 1 BENCHMARK: ZERO-DEP & BINARY PACKING INVARIANTS", flush=True)
    print("=" * 60, flush=True)

    engine = DFlashEngine()

    # Test 1: Go Raft WAL
    ch_go = SuiteLoader.get_challenge("go_02_multifile_raft_wal")
    print(f"\n[1/2] Running {ch_go.title} [GO]...", flush=True)

    start_go = time.perf_counter()
    out_go, verif_go = engine.run_with_compiler_feedback(ch_go, max_correction_attempts=1)
    el_go = time.perf_counter() - start_go

    status_go = "PASS" if verif_go.test_passed else "FAIL"
    print(f"  • Go Result: {out_go.completion_tokens} tokens @ {out_go.tokens_per_sec:.2f} tok/s ({el_go:.2f}s) -> {status_go}", flush=True)
    if not verif_go.test_passed:
        print(f"    Error: {verif_go.error_message}", flush=True)

    # Test 2: Rust Multi-Module Pipeline
    ch_rust = SuiteLoader.get_challenge("rust_03_multifile_event_pipeline")
    print(f"\n[2/2] Running {ch_rust.title} [RUST]...", flush=True)

    start_rust = time.perf_counter()
    out_rust, verif_rust = engine.run_with_compiler_feedback(ch_rust, max_correction_attempts=1)
    el_rust = time.perf_counter() - start_rust

    status_rust = "PASS" if verif_rust.test_passed else "FAIL"
    print(f"  • Rust Result: {out_rust.completion_tokens} tokens @ {out_rust.tokens_per_sec:.2f} tok/s ({el_rust:.2f}s) -> {status_rust}", flush=True)
    if not verif_rust.test_passed:
        print(f"    Error: {verif_rust.error_message}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"STEP 1 COMPLETE: Go={status_go}, Rust={status_rust}", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
