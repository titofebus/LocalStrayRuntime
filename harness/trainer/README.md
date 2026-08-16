# Experimental training prototypes

This directory is not part of the installable `qwen-prime-runtime` wheel. Its
older orchestration paths use synthetic metrics and generated activations for UI
and pipeline experiments; they do not constitute a validated DFlash training
recipe and must not be represented as producing the bundled native MTP draft.

The local web controls are disabled unless
`QWEN_PRIME_ENABLE_EXPERIMENTAL_TRAINING=1` is set explicitly. Treat all output
as experimental until a real corpus, held-out evaluation, reproducible training
configuration, and independently verified model artifact are supplied.
