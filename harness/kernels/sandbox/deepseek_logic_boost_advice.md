# Ranked ROI Solutions for High-Entropy Speculative Decoding on Apple Silicon

## Executive Summary

The core problem is **draft acceptance collapse** during high-entropy regions. Your current 1-layer MTP achieves 85%+ acceptance on boilerplate but drops to ~50% on complex logic. This is fundamentally a **draft quality** problem, not a verification throughput problem. The highest ROI solutions target draft quality first, then verification efficiency.

---

## Ranked ROI List

| Rank | Solution | Expected Gain | Implementation Cost | Risk |
|------|----------|---------------|-------------------|------|
| 1 | **Target Hidden-State Conditioning** | +15-25% acceptance | Medium | Low |
| 2 | **Dynamic Entropy-Gated Drafting** | +10-15% throughput | Low | Low |
| 3 | **Speculative Tree Decoding** | +20-30% acceptance | High | Medium |
| 4 | **Drafter Capacity Upgrade** | +10-20% acceptance | High | Medium |
| 5 | **KV-Cache Compression** | +5-10% throughput | Medium | Medium |

---

## Solution #1: Target Hidden-State Conditioning & Residual Feature Injection

### Why This Wins

Your 1-layer MTP is **starved for context**. During boilerplate, the next token is predictable from surface patterns. During complex logic, the drafter needs to understand:
- Variable dependencies across long ranges
- Control flow state (which branch are we in?)
- Arithmetic/type constraints

A 1-layer drafter cannot extract these from embeddings alone. By injecting the **target model's hidden states** (specifically from mid-to-late layers), you give the drafter access to deeply contextualized features without adding drafter depth.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    TARGET MODEL (27B)                   │
│                                                         │
│  Layer L-4 ──→ h_{L-4} ──┐                             │
│  Layer L-2 ──→ h_{L-2} ──┼──→ Feature Projector ──→ f  │
│  Layer L   ──→ h_L    ──┘        (MLP 2-layer)         │
│                                                         │
│  Output: logits_target, h_L (for next step)             │
└─────────────────────────────────────────────────────────┘
                          │
                          │ f (residual features)
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    DRAFTER (1-layer MTP)                │
│                                                         │
│  Input: x_t, KV_cache_draft, f                         │
│                                                         │
│  h_draft = LayerNorm(x_t + f)                          │
│  h_draft = SelfAttn(h_draft, KV_cache_draft)           │
│  h_draft = FFN(h_draft)                                │
│  logits_draft = LM_Head(h_draft + f)  ← residual       │
│                                                         │
│  Output: logits_draft, KV_cache_draft                  │
└─────────────────────────────────────────────────────────┘
```

### MLX Implementation

```python
import mlx.core as mx
import mlx.nn as nn

class FeatureProjector(nn.Module):
    """Projects target hidden states to drafter dimension."""
    def __init__(self, target_dim: int, draft_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(target_dim, draft_dim * 2),
            nn.GELU(),
            nn.Linear(draft_dim * 2, draft_dim)
        )
    
    def __call__(self, h_target: mx.array) -> mx.array:
        return self.proj(h_target)

class ConditionalDrafter(nn.Module):
    """1-layer MTP with target hidden-state conditioning."""
    def __init__(self, config, target_dim: int):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.draft_dim)
        self.ln1 = nn.LayerNorm(config.draft_dim)
        self.attn = nn.MultiHeadAttention(
            dims=config.draft_dim,
            num_heads=config.num_heads,
            bias=False
        )
        self.ln2 = nn.LayerNorm(config.draft_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.draft_dim, config.draft_ffn_dim),
            nn.GELU(),
            nn.Linear(config.draft_ffn_dim, config.draft_dim)
        )
        self.ln3 = nn.LayerNorm(config.draft_dim)
        self.lm_head = nn.Linear(config.draft_dim, config.vocab_size)
        
        # Feature conditioning
        self.feature_proj = FeatureProjector(target_dim, config.draft_dim)
        self.feature_gate = nn.Linear(config.draft_dim, config.draft_dim)
        
    def __call__(
        self, 
        x: mx.array, 
        kv_cache: dict,
        h_target: mx.array  # from target model layer L-2
    ) -> tuple[mx.array, dict]:
        # Project and gate target features
        f = self.feature_proj(h_target)
        gate = mx.sigmoid(self.feature_gate(f))
        f = f * gate
        
        # Inject into drafter
        h = self.embed(x) + f
        h = self.ln1(h)
        h = self.attn(h, kv_cache)
        h = self.ln2(h + f)  # residual injection
        h = self.ffn(h)
        h = self.ln3(h + f)  # second residual injection
        logits = self.lm_head(h + f)  # final injection
        
        return logits, kv_cache
```

### Key Design Decisions

1. **Which target layers to use**: Extract from layers `L-4` and `L-2` (where L is the last layer). Concatenate and project. This captures both mid-level syntax and high-level semantics.

2. **Gated injection**: The gate `σ(W·f)` allows the drafter to learn *when* to trust target features. During boilerplate, the gate can close (rely on drafter's own patterns). During complex logic, it opens.

3. **Residual connections**: Add `f` at 3 points (input, after attention, after FFN). This creates multiple pathways for target knowledge to influence draft predictions.

4. **Training**: Freeze target model. Train drafter + projector with:
   - Standard MTP loss: `L = CE(logits_draft, target_tokens)`
   - Add auxiliary loss: `L_aux = ||gate||_1` to encourage sparsity (gate should be near 0 for easy tokens, near 1 for hard tokens)

### Performance Expectations

- **Acceptance rate on complex logic**: 50% → 65-75%
- **Overall throughput**: 28-29 → 33-36 tok/s
- **Memory overhead**: ~200-400 MB (projector + activations)

---

## Solution #2: Dynamic Entropy-Gated Drafting

### Why It Works

When the target model's logits have high entropy (uncertain), the drafter is *also* uncertain. Drafting K tokens when the first token has 60% entropy wastes verification compute. Instead:

1. Compute target logits for current position
2. Calculate entropy: `H = -Σ p_i log(p_i)`
3. If `H > threshold`: draft only 1-2 tokens (or skip drafting entirely)
4. If `H < threshold`: draft full K tokens

### MLX Implementation

```python
def entropy_gated_draft_length(
    target_logits: mx.array,
    base_k: int = 5,
    entropy_threshold: float = 2.5,
    min_k: int = 1
) -> int:
    """Determine draft length based on target logit entropy."""
    probs = mx.softmax(target_logits, axis=-1)
    log_probs = mx.log(probs + 1e-10)
    entropy = -mx.sum(probs * log_probs, axis=-1)
    
    # Normalize entropy to [0, 1] relative to vocab size
    norm_entropy = entropy / mx.log(mx.array(target_logits.shape[-1]))
    
    # Linear interpolation between min_k and base_k
    k = min_k + (base_k - min_k) * (1.0 - norm_entropy)
    return int(mx.round(k).item())
```

### Integration with Speculative Decoding Loop

```python
def speculative_decode_with_entropy_gate(
    target_model, drafter, tokens, kv_cache, base_k=5
):
    # Get target logits for current position
    target_logits, kv_cache = target_model(tokens[-1:], kv_cache)
    
    # Determine draft length
    k = entropy_gated_draft_length(target_logits, base_k)
    
    if k <= 1:
        # Skip drafting, just use target prediction
        next_token = mx.argmax(target_logits, axis=-1)
        return next_token, kv_cache, 1
    
    # Draft k tokens
    draft_tokens = []
    draft_logits = []
    draft_kv = kv_cache.copy()
    x = tokens[-1:]
    
    for _ in range(k):
        logits, draft_kv = drafter(x, draft_kv)
        draft_tokens.append(mx.argmax(logits, axis=-1))
        draft_logits.append(logits)
        x = draft_tokens[-1]
    
    # Verify all at once
    verified, kv_cache, accepted = verify_draft(
        target_model, tokens, draft_tokens, draft_logits, kv_cache
    )
    
    return verified, kv_cache, accepted
```

### Performance Expectations

- **Throughput gain**: 10-15% (less wasted verification on high-entropy tokens)
- **Latency reduction**: Lower variance in per-token latency
- **Memory**: Negligible overhead

---

## Solution #3: Speculative Tree Decoding

### Why It's Ranked #3

Tree decoding gives the biggest theoretical acceptance boost (20-30%) but has the highest implementation complexity. On Apple Silicon, the memory bandwidth advantage of MLX makes tree verification efficient, but you need careful kernel design.

### Architecture

```
Draft step 1: [A] (top-1)
Draft step 2: [B1, B2, B3] (top-3 from A)
Draft step 3: [C1, C2] from B1, [C3] from B2, [C4, C5] from B3

Tree structure:
        A
       /|\
     B1 B2 B3
     /|  |  /\
   C1 C2 C3 C4 C5
```

### MLX Implementation

```python
class TreeVerifier:
    """Verifies a draft tree in one target forward pass."""
    
    def __init__(self, target_model, max_branch=3, max_depth=3):
        self.target_model = target_model
        self.max_branch = max_branch
        self.max_depth = max_depth
    
    def build_tree(self, drafter, tokens, kv_cache):
        """Build draft tree using beam search."""
        tree = {
            'tokens': [tokens[-1]],
            'children': [],
            'logits': [],
            'kv_cache': kv_cache
        }
        
        # BFS expansion
        frontier = [tree]
        for depth in range(self.max_depth):
            new_frontier = []
            for node in frontier:
                # Get draft logits for this node
                logits, node_kv = drafter(node['tokens'][-1:], node['kv_cache'])
                node['logits'] = logits
                
                # Top-k expansion
                top_k = mx.topk(logits, k=self.max_branch, axis=-1)
                for token in top_k:
                    child = {
                        'tokens': node['tokens'] + [token],
                        'children': [],
                        'logits': [],
                        'kv_cache': node_kv
                    }
                    node['children'].append(child)
                    new_frontier.append(child)
            frontier = new_frontier
        
        return tree
    
    def verify_tree(self, tree):
        """Verify all tree paths in one forward pass using attention masking."""
        # Flatten tree into sequence with attention mask
        seq_tokens = []
        attention_mask = []
        
        def flatten(node, path_mask):
            seq_tokens.append(node['tokens'][-1])
            attention_mask.append(path_mask)
            for child in node['children']:
                flatten(child, path_mask + [1])
        
        flatten(tree, [1])
        
        # Single forward pass with custom attention mask
        logits = self.target_model(
            mx.array(seq_tokens), 
            attention_mask=mx.array(attention_mask)
        )
        
        # Find longest accepted path
        return self.find_best_path(tree, logits)
```

### Critical MLX Optimization

The key insight is that tree verification requires **causal attention masking** where each token only attends to its ancestors. In MLX:

```python
def create_tree_attention_mask(tree_structure: list[list[int]]) -> mx.array:
    """Create attention mask for tree verification.
    
    tree_structure[i][j] = 1 if token j is an ancestor of token i.
    """
    n = len(tree_structure)
    mask = mx.zeros((n, n))
    for i, ancestors in enumerate(tree_structure):
        for j in ancestors:
            mask[i, j] = 1.0
    return mask
```

### Performance Expectations

- **Acceptance rate**: 50% → 70-80% on complex logic
- **Throughput**: 28-29 → 34-38 tok/s
- **Memory**: +1-2 GB for tree states

---

## Solution #4: Drafter Capacity Upgrade

### Analysis

| Drafter | Size | Acceptance (Complex) | Throughput | Memory |
|---------|------|---------------------|------------|--------|
| 1-layer MTP | 345 MB | 50% | 28-29 | Low |
| 3-layer MTP | ~1 GB | 60-65% | 31-33 | Medium |
| 5-layer DFlash | 3.46 GB | 65-70% | 32-34 | High |
| Multi-head MTP | ~700 MB | 62-68% | 32-35 | Medium |

**Recommendation**: Multi-head MTP (2-3 heads) offers the best ROI. It's smaller than DFlash but captures multiple plausible continuations, which is exactly what complex logic needs.

### Multi-Head MTP Architecture

```python
class MultiHeadMTP(nn.Module):
    """Multi-head MTP drafter with shared backbone."""
    def __init__(self, config, num_heads=3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(config.draft_dim, config.draft_ffn_dim),
            nn.GELU(),
            nn.Linear(config.draft_ffn_dim, config.draft_dim)
        )
        self.heads = [
            nn.Linear(config.draft_dim, config.vocab_size) 
            for _ in range(num_heads)
        ]
        self.head_weights = nn.Parameter(mx.ones(num_heads) / num_heads)
    
    def __call__(self, x, kv_cache):
        h = self.backbone(x)
        logits = mx.stack([head(h) for head in self.heads])
        # Weighted combination
        logits = mx.sum(logits * self.head_weights[:, None, None], axis=0)
        return logits, kv_cache
```

---

## Solution #5: KV-Cache Compression

### Why It's Last

KV-cache compression helps with *long context* but doesn't directly address the *acceptance collapse* problem. It's a complementary optimization.

### MLX Implementation

```python
def compress_kv_cache(kv_cache: dict, compression_ratio: float = 0.5):
    """Compress KV cache using low-rank approximation."""
    compressed = {}
    for layer, (k, v) in kv_cache.items():
        # SVD-based compression
        u_k, s_k, v_k = mx.linalg.svd(k)
        u_v, s_v, v_v = mx.linalg.svd(v)
        
        # Keep top singular values
        k_dim = int(k.shape[-1] * compression_ratio)
        v_dim = int(v.shape[-1] * compression_ratio)
        
        compressed[layer] = (
            u_k[:, :, :k_dim] @ mx.diag(s_k[:k_dim]),
            u_v[:, :, :v_dim] @ mx.diag(s_v[:v_dim])
        )
    return compressed
```

---

## Recommended Implementation Roadmap

### Phase 1 (Week 1-2): Quick Wins
1. Implement **entropy-gated drafting** (Solution #2)
   - Expected: +10-15% throughput immediately
   - Low risk, minimal code changes

2. Add **target hidden-state conditioning** to existing 1-layer MTP (Solution #1)
   - Expected: +15-25% acceptance on complex logic
   - Moderate code changes, requires retraining drafter

### Phase 2 (Week 3-4): Structural Improvements
3. Upgrade to **multi-head MTP** (2-3 heads) with hidden-state conditioning
   - Expected: +20-30% acceptance on complex logic
   - Requires drafter retraining

4. Implement **tree verification** with attention masking
   - Expected: +30-40% acceptance on complex logic
   - Most complex, but highest ceiling

### Phase 3 (Week 5-6): Optimization
5. Add **KV-cache compression** for long-context scenarios
6. Profile and optimize Metal kernels for tree attention

---

## Critical Implementation Details for MLX

### 1. Memory Bandwidth Optimization
Apple Silicon's unified memory is a bottleneck. Use:
- `mx.float16` for all intermediate activations
- `mx.metal` kernel fusion to reduce memory round-trips
- Batch verification of multiple draft tokens in single kernel

### 2. Kernel Fusion Example

```python
@mx.compile
def fused_draft_verify(draft_tokens, target_kv, drafter_weights):
    """Fuse draft generation and verification into single Metal kernel."""
    # Single kernel that:
    # 1. Runs drafter forward for all draft positions
    # 2. Runs target model verification
