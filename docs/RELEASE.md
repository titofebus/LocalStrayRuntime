# Release checklist

## Runtime source and wheel

- Run `uv lock` with the declared DFlash, MLX, and MLX LM versions.
- Run the full test suite from a clean environment.
- Build the wheel with `uv build` and inspect its contents.
- Confirm `harness/trainer`, `harness/agent`, local results, caches, and model
  weights are absent from the wheel.
- Run `qwen-prime-runtime doctor` against the published target/draft pair.
- Publish source and wheel checksums with the release.

## Embedded macOS runtime

- Build the relocatable payload with
  `scripts/build_embedded_runtime.command /path/to/QwenPrimeRuntime`.
- The builder copies a uv-managed, relocatable CPython 3.12 distribution and
  installs exactly the non-development packages locked by `uv.lock` into a
  bundle-relative `site-packages` directory.
- Move the completed payload to a different absolute path and run
  `bin/qwen-prime-runtime --help` and `doctor` before packaging it. The launcher
  must resolve both Python and packages relative to itself.
- Confirm the payload contains no absolute symlinks, source checkout paths,
  caches, tests, or model-weight files. Model weights remain user-managed.
- Run `tests/test_embedded_runtime.py` whenever the builder or launcher changes.

## Model repositories

- Publish target and native-MTP artifacts separately from the application.
- Include `LICENSE`, `NOTICE`, a model card, source repository and revision,
  quantization/export commands, dependency versions, and SHA-256 checksums.
- Mark every derivative as modified from the Alibaba Cloud source.
- Re-run the benchmark suite from the published artifacts and report hardware,
  prompt, cache state, token count, acceptance, server t/s, and wall time.

## macOS application

- Build from a clean checkout against the published runtime interface.
- Set a release version and build number.
- Sign with a Developer ID Application identity and hardened runtime.
- Verify the signature with `codesign --verify --deep --strict`.
- Submit the archive for Apple notarization and staple the ticket.
- Test the stapled archive on a clean Apple Silicon account.
- Publish the ZIP and SHA-256 checksum.

## Prime Agent

- Test against the current official Prime Agent release.
- Use `qwen-prime-runtime configure-prime-agent`; do not distribute a modified
  Prime Agent monorepo unless a separately reviewed fork is actually required.
