```python
# ddtree.py
"""
High-performance Speculative Branch Tree (DDTree) for Qwen models in MLX.
Implements tree-based speculative decoding with native MTP-style drafting.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import math
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class TreeConfig:
    """Configuration for the speculative tree."""
    max_depth: int = 3
    num_branches: int = 3
    max_candidates: int = 8
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9


class TreeCandidateGenerator:
    """
    Generates speculative branch trees from draft model predictions.
    Creates packed tree tokens and attention masks for parallel verification.
    """
    
    def __init__(self, config: TreeConfig):
        self.config = config
        self.max_depth = config.max_depth
        self.num_branches = config.num_branches
        self.max_candidates = config.max_candidates
        
    def _compute_entropy(self, logits: mx.array) -> mx.array:
        """Compute entropy of token distribution for divergence detection."""
        probs = mx.softmax(logits, axis=-1)
        log_probs = mx.log(mx.clip(probs, 1e-10, 1.0))
        entropy = -mx.sum(probs * log_probs, axis=-1)
        return entropy
    
    def _sample_branch_tokens(self, logits: mx.array, num_tokens: int) -> mx.array:
        """Sample multiple candidate tokens from logits."""
        # Apply temperature
        logits = logits / self.config.temperature
        
        # Top-k filtering
        if self.config.top_k > 0:
            k = min(self.config.top_k, logits.shape[-1])
            top_k_values, top_k_indices = mx.topk(logits, k)
            mask = mx.full_like(logits, -float('inf'))
            mask = mx.scatter(mask, top_k_indices, top_k_values, axis=-1)
            logits = mask
        
        # Top-p (nucleus) filtering
        if self.config.top_p < 1.0:
            sorted_logits = mx.sort(logits, axis=-1, descending=True)
            sorted_probs = mx.softmax(sorted_logits, axis=-1)
            cumsum_probs = mx.cumsum(sorted_probs, axis=-1)
            mask = cumsum_probs - sorted_probs > self.config.top_p
            mask = mx.roll(mask, 1, axis=-1)
            mask[:, 0] = False
            logits = mx.where(mask, -float('inf'), logits)
        
        # Sample without replacement
        probs = mx.softmax(logits, axis=-1)
        samples = []
        for _ in range(num_tokens):
            sample = mx.random.categorical(mx.log(probs + 1e-10))
            samples.append(sample)
            # Mask out sampled token
            probs = mx.where(mx.arange(probs.shape[-1])[None, :] == sample[:, None], 
                           0.0, probs)
            probs = probs / mx.sum(probs, axis=-1, keepdims=True)
        
        return mx.stack(samples, axis=-1)
    
    def generate_tree(self, 
                     initial_token: mx.array,
                     hidden_state: mx.array,
                     draft_model: nn.Module) -> Tuple[mx.array, mx.array, List[List[int]]]:
        """
        Generate speculative tree from initial token and hidden state.
        
        Args:
            initial_token: Starting token (shape: [1])
            hidden_state: Target model hidden state (shape: [1, hidden_dim])
            draft_model: MTP draft model for token prediction
            
        Returns:
            tree_tokens: Packed tree token tensor (shape: [num_candidates])
            tree_mask: 2D attention mask (shape: [num_candidates, num_candidates])
            tree_paths: List of token paths for verification
        """
        config = self.config
        tree_tokens = [initial_token]
        tree_paths = [[initial_token.item()]]
        
        # Track positions for attention mask
        positions = [0]
        parent_map = {0: -1}  # position -> parent position
        
        # Generate draft tokens iteratively
        current_hidden = hidden_state
        current_token = initial_token
        
        for depth in range(config.max_depth):
            # Get draft model predictions
            draft_logits, current_hidden = draft_model(current_token, current_hidden)
            
            # Compute entropy for divergence detection
            entropy = self._compute_entropy(draft_logits)
            
            # Determine number of branches at this level
            if entropy > 2.0:  # High entropy - branch
                num_branches = config.num_branches
            else:  # Low entropy - single path
                num_branches = 1
            
            # Sample candidate tokens
            candidate_tokens = self._sample_branch_tokens(draft_logits, num_branches)
            
            # Add candidates to tree
            new_positions = []
            for i in range(num_branches):
                token = candidate_tokens[0, i]
                tree_tokens.append(token)
                
                # Find parent position (last position at previous depth)
                parent_pos = positions[-1] if depth > 0 else 0
                new_pos = len(tree_tokens) - 1
                parent_map[new_pos] = parent_pos
                new_positions.append(new_pos)
                
                # Update paths
                if depth == 0:
                    tree_paths.append([initial_token.item(), token.item()])
                else:
                    # Extend existing paths
                    for path in tree_paths:
                        if len(path) == depth + 1 and path[-1] == tree_tokens[parent_pos].item():
                            new_path = path + [token.item()]
                            if new_path not in tree_paths:
                                tree_paths.append(new_path)
            
            # Update positions for next iteration
            positions = new_positions
            current_token = candidate_tokens[0, 0]  # Use first branch for next step
        
        # Convert to tensors
        tree_tokens_tensor = mx.array([t.item() for t in tree_tokens])
        
        # Build attention mask
        num_tokens = len(tree_tokens)
        tree_mask = mx.zeros((num_tokens, num_tokens), dtype=mx.bool_)
        
        # Set attention mask based on tree structure
        for pos in range(num_tokens):
            # Each token can attend to itself and its ancestors
            curr = pos
            while curr != -1:
                tree_mask[pos, curr] = True
                curr = parent_map.get(curr, -1)
        
        # Ensure initial token attends to all (it's the root)
        tree_mask[0, :] = True
        
        return tree_tokens_tensor, tree_mask, tree_paths


class TreeVerifier:
    """
    Verifies speculative tree candidates in parallel using target model.
    """
    
    def __init__(self, target_model: nn.Module):
        self.target_model = target_model
        
    def verify_tree(self,
                   tree_tokens: mx.array,
                   tree_mask: mx.array,
                   target_logits: mx.array) -> Tuple[mx.array, int]:
        """
        Verify all tree paths and select the longest accepted prefix.
        
        Args:
            tree_tokens: Packed tree tokens
            tree_mask: Tree attention mask
            target_logits: Target model logits for all tree positions
            
        Returns:
            accepted_tokens: Accepted token sequence
            accepted_length: Number of accepted tokens
        """
        # Get target model predictions for each position
        # target_logits shape: [num_candidates, vocab_size]
        
        # Find the longest accepted path
        best_path = [tree_tokens[0].item()]
        best_length = 1
        
        # Check each position in the tree
        for pos in range(1, len(tree_tokens)):
            # Get the token at this position
            candidate_token = tree_tokens[pos]
            
            # Get the logits from the parent position
            # For simplicity, we check if the token matches the argmax
            parent_pos = self._find_parent(pos, tree_mask)
            if parent_pos is not None:
                predicted_token = mx.argmax(target_logits[parent_pos])
                if predicted_token.item() == candidate_token.item():
                    # Token matches - extend path
                    best_path.append(candidate_token.item())
                    best_length += 1
                else:
                    # Token doesn't match - stop this branch
                    break
        
        # Convert to array
        accepted_tokens = mx.array(best_path)
        
        return accepted_tokens, best_length
    
    def _find_parent(self, pos: int, tree_mask: mx.array) -> Optional[int]:
        """Find the parent position of a token in the tree."""
        # In a tree structure, parent is the position that this token attends to
        # and is not itself
        mask_row = tree_mask[pos]
        parents = mx.nonzero(mask_row)[0]
        # Parent is the last non-self position
        for p in reversed(parents.tolist()):
            if p != pos:
                return p
        return None


class TreeKVCacheManager:
    """
    Manages KV cache for tree-based speculative decoding.
    Efficiently prunes non-accepted branches using pointer/index slicing.
    """
    
    def __init__(self, num_layers: int, num_heads: int, head_dim: int):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Initialize KV cache
        self.k_cache = []
        self.v_cache = []
        for _ in range(num_layers):
            self.k_cache.append(mx.zeros((1, num_heads, 0, head_dim)))
            self.v_cache.append(mx.zeros((1, num_heads, 0, head_dim)))
    
    def update_cache(self, 
                    new_keys: List[mx.array],
                    new_values: List[mx.array],
                    positions: mx.array) -> None:
        """
        Update KV cache with new keys/values at specified positions.
        
        Args:
            new_keys: List of key tensors for each layer
            new_values: List of value tensors for each layer
            positions: Positions to update in the cache
        """
        for layer_idx in range(self.num_layers):
            k = new_keys[layer_idx]
            v = new_values[layer_idx]
            
            # Update cache at positions
            for i, pos in enumerate(positions.tolist()):
                if pos < self.k_cache[layer_idx].shape[2]:
                    # Update existing position
                    self.k_cache[layer_idx][:, :, pos:pos+1, :] = k[:, :, i:i+1, :]
                    self.v_cache[layer_idx][:, :, pos:pos+1, :] = v[:, :, i:i+1, :]
                else:
                    # Append new position
                    self.k_cache[layer_idx] = mx.concatenate(
                        [self.k_cache[layer_idx], k[:, :, i:i+1, :]], axis=2
                    )
                    self.v_cache[layer_idx] = mx.concatenate(
                        [self.v_cache[layer_idx], v[:, :, i:i+1, :]], axis=2
                    )
    
    def prune_branches(self, accepted_length: int) -> None:
        """
        Prune non-accepted branches from KV cache.
        Uses pointer/index slicing for O(1) operation.
        
        Args:
            accepted_length: Number of accepted tokens to keep
        """
        for layer_idx in range(self.num_layers):
            # Keep only the accepted prefix
            self.k_cache[layer_idx] = self.k_cache[layer_idx][:, :, :accepted_length, :]
            self.v_cache[layer_idx] = self.v_cache[layer_idx][:, :, :accepted_length, :]
    
    def get_cache(self) -> Tuple[List[mx.array], List[mx.array]]:
        """Get current KV cache."""
        return self.k_cache, self.v_cache


class MTPDraftModel(nn.Module):
    """
    Multi-Token Prediction (MTP) draft model for speculative decoding.
    Simplified version for demonstration.
    """
    
    def __init__(self, hidden_dim: int, vocab_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Simple transformer-like draft model
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim * 4))
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
    
    def __call__(self, 
                token: mx.array,
                hidden_state: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Forward pass of draft model.
        
        Args:
            token: Input token
            hidden_state: Previous hidden state
            
        Returns:
            logits: Token prediction logits
            new_hidden_state: Updated hidden state
        """
        # Embed token
        token_embed = self.embedding(token)
        
        # Combine with hidden state
        combined = token_embed + hidden_state
        
        # Pass through transformer layers
        for layer in self.layers:
            combined = layer(combined)
        
        # Project to vocabulary
        logits = self.output_proj(combined)
        
        return logits, combined


class SpeculativeBranchTreeDecoder:
    """
    Main decoder implementing DDTree speculative decoding.
    """
    
    def __init__(self,
                 target_model: nn.Module,
                 draft_model: nn.Module,
                 config: TreeConfig):
        self.target_model = target_model
        self.draft_model = draft_model
        self.config = config
        
        # Initialize components
        self.generator = TreeCandidateGenerator(config)
        self.verifier = TreeVerifier(target_model)
        
        # Get model dimensions
        self.hidden_dim = target_model.hidden_dim if hasattr(target_model, 'hidden_dim') else 4096
        self.num_layers = target_model.num_layers if hasattr(target_model, 'num_layers') else 32
        self.num_heads = target_model.num_heads if hasattr(target_model, 'num_heads') else 32
        self.head_dim = self.hidden_dim // self.num_heads
        
        # Initialize KV cache manager
        self.kv_manager = TreeKVCacheManager(self.num_layers, self.num_heads, self.head_dim)
    
    def decode_step(self,
                   input_token: mx.array,
                   hidden_state: mx.array) -> Tuple[mx.array, int]:
        """
        Perform one speculative decoding step.
        
        Args:
            input_token: Current input token
            hidden_state: Current hidden state
            
        Returns:
            accepted_tokens: Accepted tokens from this step
            accepted_length: Number of accepted tokens
        """
        # Generate speculative tree
        tree_tokens, tree_mask, tree_paths = self.generator.generate_tree(
            input_token, hidden_state, self.draft_model
        )
        
        # Run target model on all tree positions in parallel
        # This would be a single forward pass in production
        target_logits = self._parallel_target_forward(tree_tokens, tree_mask)
        
        # Verify tree and select best path
        accepted_tokens, accepted_length = self.verifier.verify_tree(
            tree_tokens, tree_mask, target_logits
        )
        
        # Update KV cache
        self._update_kv_cache(accepted_tokens, accepted_length)
        
        return accepted_tokens, accepted_length
    
    def _parallel_target_forward(self, 
                                tree_tokens: mx.array,
                                tree_mask: mx.array) -> mx.array:
        """
        Run target model on all tree positions in parallel.
        This is the key optimization - single forward pass for all candidates.
        """
        # In production, this would use the actual target model
        # For demonstration, we simulate with random logits
        num_candidates = len(tree_tokens)
        vocab_size = self.draft_model.vocab_size
        
        # Simulate target model forward pass
        # In reality, this would be: self.target_model(tree_tokens, attention_mask=tree_mask)
        logits = mx.random.normal((num_candidates, vocab_size))
        
        return logits
    
    def _update_kv_cache(self, accepted_tokens: mx.array, accepted_length: int) -> None:
        """Update KV cache with accepted tokens."""
        # Generate dummy keys/values for demonstration
        # In production, these would come from the actual model forward pass
        new_keys = []
        new_values = []
        for _ in range(self.num_layers):
            new_keys.append(mx.random.normal((1, self.num_heads, accepted_length, self.head_dim)))
            new_values.append(mx.random.normal((1, self.num_heads, accepted_length, self.head_dim)))
        
        # Update cache
        positions = mx.arange(accepted_length)
        self.kv_manager.update_cache(new_keys, new_values, positions)
        
        # Prune non-accepted branches
        self.kv_manager.prune_branches(accepted_length)


# benchmark_ddtree.py
"""
Standalone benchmark comparing single-path linear speculation vs DDTree.
"""

import time
import numpy as np
import mlx.core as mx


def benchmark_linear_speculation(num_steps: int = 100, vocab_size: int = 32000):
    """Benchmark single-path linear speculation."""
    times = []
    accepted_counts = []
    
    for _ in range(num_steps):
        start = time.perf_counter()
        
        # Simulate linear speculation
        # Single path - predict one token at a time
        token = mx.array([np.random.randint(0, vocab_size)])
        accepted = 1
        
        # Simulate draft prediction
        for _ in range(3):  # Try 3 speculative tokens
            # Draft prediction
            draft_logits = mx.random.normal((1, vocab_size))
            draft_token = mx.argmax(draft_logits)
            
            # Target verification
            target_logits = mx.random.normal((1, vocab_size))
            target_token = mx.argmax(target_logits)
            
            # Check acceptance
            if draft_token.item() == target_token.item():
                accepted += 1
                token = target_token
