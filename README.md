# Qwen Prime Runtime

Qwen Prime Runtime is a local, OpenAI-compatible inference server for the
official Qwen3.8-27B model on Apple Silicon. It pairs a 6-bit MLX target with
the matching 6-bit native MTP head and uses DFlash verification with a default
four-token speculative block.

The runtime, macOS client, Prime Agent integration, and model weights are
separate release units. Model weights are not bundled with this repository.

## Requirements

- Apple Silicon Mac
- macOS 14 or newer
- Python 3.12 recommended
- Approximately 24 GB of available unified memory for the current target and
  draft pair
- A 6-bit Qwen3.8-27B MLX target and its matching native-MTP artifact

## Install from source

```bash
git clone https://github.com/adriancmurray/qwen-prime-runtime.git
cd qwen-prime-runtime
./scripts/install_qwen_prime_runtime.command
```

Configure model locations without editing source files:

```bash
qwen-prime-runtime configure \
  --target "/path/to/Qwen3.8-27B-MLX-6bit" \
  --draft "/path/to/Qwen3.8-27B-MTP-MLX-6bit"

qwen-prime-runtime doctor
```

The configuration is stored at
`~/Library/Application Support/QwenPrime/runtime.json`.

## Run

```bash
qwen-prime-runtime serve
```

The server binds to `127.0.0.1:8000` by default. A non-loopback bind is refused
unless `--allow-remote` is supplied explicitly. The API does not implement
authentication and should not be exposed to an untrusted network.

Endpoints:

- `GET /v1/models`
- `GET /v1/engine`
- `POST /v1/chat/completions`

`/v1/engine` reports the target and draft identities, quantization, MTP hash,
block size, prefix-cache state, and warmup state so clients can reject a stale
or mismatched process.

## Prime Agent

Install Prime Agent from its official release, then merge the local provider
entry into the existing configuration:

```bash
qwen-prime-runtime configure-prime-agent
prime-agent --provider local-mlx --model qwen3.8-27b
```

The configuration command preserves other providers and credentials. It does
not require a fork of Prime Agent.

## Native MTP export

The companion draft is the official checkpoint's native MTP head exported as a
standalone MLX artifact. It is not a separately trained DFlash diffusion model.

```bash
uv run python -m harness.trainer.export_qwen38_mtp \
  --source /path/to/Qwen3.8-27B \
  --output /path/to/Qwen3.8-27B-MTP-MLX-6bit \
  --source-revision <full-source-commit>
```

The exporter records the source revision and SHA-256 of the generated weights.
Redistributed derivatives must include the original Qwen Apache-2.0 license,
attribution, model card, and a notice that the weights were modified and
quantized.

## Performance

Performance depends on hardware, prompt length, cache state, generation length,
and draft acceptance. On the development M4 Max, a warm 256-token coding test
measured approximately 26 server tokens/second with 53.9% draft acceptance. A
direct block-size sweep measured approximately 28 tokens/second at block size
four. These are measurements, not guaranteed minimums.

Time to first model token must be measured from the first non-empty generation
delta. The initial empty SSE role event is connection metadata and is not a
model-token measurement.

## Security

The inference server loads local models and does not execute generated code.
The evaluation and autonomous-agent utilities elsewhere in the source tree can
compile or execute model-generated programs with the current user's
permissions. A working directory is not a security sandbox. Run untrusted code
in a separate OS-level sandbox or virtual machine.

The experimental training utilities are excluded from the runtime wheel. They
contain research prototypes and synthetic calibration stages; they are not a
reproducible, validated training release and their projected loss, acceptance,
or throughput values must not be presented as measured results.

## Development

```bash
uv sync --extra dev
uv run pytest
```

Benchmark scripts read the same runtime configuration and do not require local
paths to be embedded in source.

## License

Runtime and harness source is MIT-licensed. DFlash, Qwen weights, MLX, and other
dependencies retain their respective licenses. See `THIRD_PARTY_NOTICES.md`.

Qwen Prime is an independent project and is not affiliated with or endorsed by
Alibaba Cloud, the Qwen team, Prime Intellect, or Apple.
