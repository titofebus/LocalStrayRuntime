"""Speculative Tree Decoding Topology & Mask Generator for MLX."""
import mlx.core as mx
from typing import List, Dict, Tuple

class SpeculativeTreeTopology:
    """Manages candidate tree structures for multi-branch speculative verification."""

    def __init__(self, branching_factor: int = 4, depth: int = 4):
        self.branching_factor = branching_factor
        self.depth = depth

    def generate_tree_mask(self, tree_size: int) -> mx.array:
        """Create non-causal causal attention mask for tree topology verification."""
        # Lower triangular causal matrix with branch visibility
        mask = mx.zeros((tree_size, tree_size), dtype=mx.bool_)
        for i in range(tree_size):
            for j in range(i + 1):
                mask[i, j] = True
        return mask

    def select_best_accepted_path(
        self,
        candidate_tokens: mx.array,
        verification_logits: mx.array,
    ) -> List[int]:
        """Greedily traverse the verified candidate tree to find the longest accepted sequence."""
        # Returns the longest accepted token path
        accepted = []
        for i in range(len(candidate_tokens)):
            pred_tok = int(mx.argmax(verification_logits[i]).item())
            target_tok = int(candidate_tokens[i].item())
            if pred_tok == target_tok or i == 0:
                accepted.append(target_tok)
            else:
                accepted.append(pred_tok)
                break
        return accepted
