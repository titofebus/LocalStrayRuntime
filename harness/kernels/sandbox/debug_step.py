"""Step-by-step equivalence debugger for MTP operations."""
import mlx.core as mx
import mlx.nn as nn
from harness.kernels.sandbox.fused_mtp_ops import fused_dual_rmsnorm_concat, fast_vocab_argmax

D = 5120
V = 248320
eps = 1e-6

emb = mx.random.normal((1, D)).astype(mx.float16)
hid = mx.random.normal((1, D)).astype(mx.float16)
w_emb = mx.random.normal((D,)).astype(mx.float16)
w_hid = mx.random.normal((D,)).astype(mx.float16)

# 1. Dual RMSNorm
norm1 = mx.fast.rms_norm(emb, w_emb, eps)
norm2 = mx.fast.rms_norm(hid, w_hid, eps)
std_fused = mx.concatenate([norm1, norm2], axis=-1)

custom_fused = fused_dual_rmsnorm_concat(emb, hid, w_emb, w_hid, eps)
mx.eval(std_fused, custom_fused)

diff_norm = mx.max(mx.abs(std_fused - custom_fused)).item()
print(f"Norm Max Diff: {diff_norm:.6e}")
assert mx.allclose(std_fused, custom_fused, atol=1e-2, rtol=1e-2), "Norm mismatch!"

# 2. Argmax
logits = mx.random.normal((1, V)).astype(mx.float16)
std_argmax = mx.argmax(logits, axis=-1)
custom_argmax = fast_vocab_argmax(logits)
mx.eval(std_argmax, custom_argmax)

print(f"Standard Argmax: {std_argmax.item()}, Custom Argmax: {custom_argmax.item()}")
assert std_argmax.item() == custom_argmax.item(), f"Argmax mismatch: {std_argmax.item()} vs {custom_argmax.item()}"

print("[SUCCESS] Step-by-step debugger passed!")
