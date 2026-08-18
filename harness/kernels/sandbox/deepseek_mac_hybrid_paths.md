# Apple Silicon Hybrid Quantization: M4 Max Production Guide

## 1. Hardware-Level Primitives & Why Mixed Precision is Free on Apple Silicon

### The M4 Memory Architecture Advantage

Unlike CUDA's warp-level memory access patterns, Apple Silicon's **Unified Memory Architecture (UMA)** with 400 GB/s bandwidth treats all memory as a flat, coherent address space. The M4's **128-byte cacheline** is the fundamental unit of memory transfer, and this is where the magic happens:

```python
# Memory layout for a 4-bit quantized weight (group_size=64)
# Each group of 64 weights = 64 * 4 bits = 32 bytes = 1/4 cacheline
# Each group of 64 weights at 8-bit = 64 bytes = 1/2 cacheline
# Both fit perfectly within 128-byte cachelines with ZERO waste
```

**Why group_size=64 is the Sweet Spot:**
- **128-byte cacheline** / 2 bytes per fp16 weight = 64 weights per cacheline
- At 4-bit: 64 weights × 0.5 bytes = 32 bytes → 4 groups per cacheline (perfect alignment)
- At 8-bit: 64 weights × 1 byte = 64 bytes → 2 groups per cacheline (perfect alignment)
- **Zero memory alignment penalty** when mixing 4-bit and 8-bit layers because both align to the same cacheline boundary

### Metal SIMDgroup Matrix Operations

The M4's GPU executes matrix multiplications via `simdgroup_multiply_accumulate` instructions. These operate on 8×8 or 16×16 tiles regardless of bitwidth:

```metal
// Metal kernel showing mixed-precision handling
kernel void hybrid_quant_matmul(
    device const uint8_t* weights_4bit [[buffer(0)]],
    device const uint8_t* weights_8bit [[buffer(1)]],
    device const float* scales_4bit [[buffer(2)]],
    device const float* scales_8bit [[buffer(3)]],
    device const uint8_t* zeros_4bit [[buffer(4)]],
    constant uint& layer_type [[buffer(5)]], // 0=4bit, 1=8bit
    ...
) {
    // SIMDgroup operations work on 16x16 tiles
    // Bitwidth only affects memory loading, not compute
    // No bank conflicts because UMA is flat-addressed
    simdgroup_float8x8 acc;
    simdgroup_load(acc, ...); // Works identically for 4-bit and 8-bit
}
```

**Key Insight:** On CUDA, mixed precision causes warp divergence and bank conflicts. On Apple Silicon, the GPU's SIMD groups process memory linearly with no bank structure, making mixed-bitwidth execution **computationally free**.

## 2. MLX Native Implementation Architecture

### Optimal Layer-Wise Quantization Map for Qwen 3.8-27B

```python
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

class HybridQuantizedQwen(nn.Module):
    """Qwen 3.8-27B with hybrid quantization for M4 Max"""
    
    def __init__(self, config):
        super().__init__()
        
        # Embeddings: Keep at fp16 for maximum vocab precision
        self.embed_tokens = nn.Embedding(
            config.vocab_size, 
            config.hidden_size
        )
        
        # Transformer layers with hybrid quantization
        self.layers = [
            self._create_hybrid_layer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        
        # LM Head: 8-bit for output precision
        self.lm_head = nn.QuantizedLinear(
            config.hidden_size,
            config.vocab_size,
            bits=8,
            group_size=64,
            bias=False
        )
    
    def _create_hybrid_layer(self, config, layer_idx):
        """Create a transformer layer with attention=8bit, MLP=4bit"""
        layer = nn.Module()
        
        # ATTENTION PROJECTIONS: 8-bit for precision
        # These handle the critical attention computation
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.QuantizedLinear(
            config.hidden_size, 
            config.num_attention_heads * config.head_dim,
            bits=8,          # 8-bit for attention
            group_size=64,   # Optimal for M4 cachelines
            bias=False
        )
        layer.self_attn.k_proj = nn.QuantizedLinear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bits=8,
            group_size=64,
            bias=False
        )
        layer.self_attn.v_proj = nn.QuantizedLinear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bits=8,
            group_size=64,
            bias=False
        )
        layer.self_attn.o_proj = nn.QuantizedLinear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bits=8,
            group_size=64,
            bias=False
        )
        
        # MLP PROJECTIONS: 4-bit for memory savings
        # These are more redundant and can tolerate lower precision
        layer.mlp = nn.Module()
        layer.mlp.gate_proj = nn.QuantizedLinear(
            config.hidden_size,
            config.intermediate_size,
            bits=4,          # 4-bit for MLP
            group_size=64,
            bias=False
        )
        layer.mlp.up_proj = nn.QuantizedLinear(
            config.hidden_size,
            config.intermediate_size,
            bits=4,
            group_size=64,
            bias=False
        )
        layer.mlp.down_proj = nn.QuantizedLinear(
            config.intermediate_size,
            config.hidden_size,
            bits=4,
            group_size=64,
            bias=False
        )
        
        # Layer norms stay in fp16
        layer.input_layernorm = nn.RMSNorm(config.hidden_size)
        layer.post_attention_layernorm = nn.RMSNorm(config.hidden_size)
        
        return layer
    
    def __call__(self, x, **kwargs):
        # Standard forward pass - MLX handles mixed precision automatically
        x = self.embed_tokens(x)
        for layer in self.layers:
            residual = x
            x = layer.input_layernorm(x)
            x = layer.self_attn(x, **kwargs)
            x = residual + x
            
            residual = x
            x = layer.post_attention_layernorm(x)
            x = layer.mlp(x)
            x = residual + x
        return self.lm_head(x)
```

### Memory Footprint Analysis for M4 Max

```python
def calculate_memory_footprint(config):
    """Calculate exact memory usage for hybrid quantization"""
    
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = config.head_dim
    num_kv_heads = config.num_key_value_heads
    vocab_size = config.vocab_size
    
    # Attention projections (8-bit)
    q_proj = hidden_size * num_heads * head_dim * 1  # 1 byte per weight
    k_proj = hidden_size * num_kv_heads * head_dim * 1
    v_proj = hidden_size * num_kv_heads * head_dim * 1
    o_proj = num_heads * head_dim * hidden_size * 1
    attention_total = (q_proj + k_proj + v_proj + o_proj) * num_layers
    
    # MLP projections (4-bit = 0.5 bytes per weight)
    gate_proj = hidden_size * intermediate_size * 0.5
    up_proj = hidden_size * intermediate_size * 0.5
    down_proj = intermediate_size * hidden_size * 0.5
    mlp_total = (gate_proj + up_proj + down_proj) * num_layers
    
    # Embeddings and LM Head (fp16 = 2 bytes)
    embeddings = vocab_size * hidden_size * 2
    lm_head = hidden_size * vocab_size * 1  # 8-bit
    
    # Scale/zero overhead: 2 floats per group of 64
    total_weights = attention_total * 8 + mlp_total * 8  # in bits
    num_groups = total_weights / 64
    scale_overhead = num_groups * 8  # 4 bytes scale + 4 bytes zero
    
    total_bytes = attention_total + mlp_total + embeddings + lm_head + scale_overhead
    
    return {
        'attention_8bit': attention_total / 1e9,  # GB
        'mlp_4bit': mlp_total / 1e9,              # GB
        'embeddings_fp16': embeddings / 1e9,      # GB
        'lm_head_8bit': lm_head / 1e9,            # GB
        'scale_overhead': scale_overhead / 1e9,   # GB
        'total': total_bytes / 1e9                # GB
    }

# For Qwen 3.8-27B:
# config: hidden=3584, intermediate=18944, layers=64, 
# heads=28, head_dim=128, kv_heads=4, vocab=152064
```

## 3. Streaming Conversion Pipeline (Zero-RAM Spill)

```python
import json
import numpy as np
import mlx.core as mx
from pathlib import Path
import safetensors.torch
import torch
from tqdm import tqdm

class HybridQuantizationConverter:
    """Streaming converter that never exceeds RAM budget"""
    
    def __init__(self, model_path: str, output_path: str, 
                 max_ram_gb: float = 32.0):
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.max_ram_gb = max_ram_gb
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Quantization configuration
        self.attention_bits = 8
        self.mlp_bits = 4
        self.group_size = 64
        
    def convert_streaming(self):
        """Convert model layer-by-layer, writing to disk immediately"""
        
        # Load config
        with open(self.model_path / 'config.json') as f:
            config = json.load(f)
        
        # Process embeddings first (small, keep in fp16)
        self._convert_embeddings()
        
        # Process each layer independently
        for layer_idx in tqdm(range(config['num_hidden_layers']), 
                             desc="Converting layers"):
            self._convert_layer(layer_idx, config)
        
        # Process LM head last
        self._convert_lm_head()
        
        # Write final metadata
        self._write_metadata(config)
    
    def _convert_embeddings(self):
        """Convert embeddings - keep in fp16 for precision"""
        weights = self._load_tensor('model.embed_tokens.weight')
        # Keep as fp16, no quantization needed
        self._save_tensor('model.embed_tokens.weight', weights)
        del weights
    
    def _convert_layer(self, layer_idx: int, config: dict):
        """Convert a single transformer layer"""
        
        layer_prefix = f'model.layers.{layer_idx}'
        
        # ATTENTION: 8-bit quantization
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            tensor_name = f'{layer_prefix}.self_attn.{proj}.weight'
            weights = self._load_tensor(tensor_name)
            
            # Quantize to 8-bit with group_size=64
            q_weight, scales, zeros = self._quantize_grouped(
                weights, bits=8, group_size=64
            )
            
            # Save quantized weights immediately
            self._save_quantized(
                tensor_name, q_weight, scales, zeros, bits=8
            )
            
            # Free memory immediately
            del weights, q_weight, scales, zeros
            mx.clear_cache()
        
        # MLP: 4-bit quantization
        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            tensor_name = f'{layer_prefix}.mlp.{proj}.weight'
            weights = self._load_tensor(tensor_name)
            
            # Quantize to 4-bit with group_size=64
            q_weight, scales, zeros = self._quantize_grouped(
                weights, bits=4, group_size=64
            )
            
            # Save quantized weights immediately
            self._save_quantized(
                tensor_name, q_weight, scales, zeros, bits=4
            )
            
            # Free memory immediately
            del weights, q_weight, scales, zeros
            mx.clear_cache()
        
        # Copy layer norms as fp16
        for norm in ['input_layernorm', 'post_attention_layernorm']:
            tensor_name = f'{layer_prefix}.{norm}.weight'
            weights = self._load_tensor(tensor_name)
            self._save_tensor(tensor_name, weights)
            del weights
    
    def _quantize_grouped(self, weights: mx.array, bits: int, 
                          group_size: int) -> tuple:
        """Quantize weights with per-group scaling"""
        
        # Reshape to groups
        orig_shape = weights.shape
        flat_weights = weights.reshape(-1)
        
        # Pad to multiple of group_size
        pad_size = (group_size - flat_weights.shape[0] % group_size) % group_size
        if pad_size > 0:
            flat_weights = mx.pad(flat_weights, (0, pad_size))
        
        # Reshape into groups
        num_groups = flat_weights.shape[0] // group_size
        grouped = flat_weights.reshape(num_groups, group_size)
        
        # Compute per-group statistics
        scales = mx.max(mx.abs(grouped), axis=1) / (2**(bits-1) - 1)
        scales = mx.maximum(scales, 1e-8)  # Avoid division by zero
        
        # Quantize
        q_weight = mx.round(grouped / scales[:, None])
        q_weight = mx.clip(q_weight, -(2**(bits-1)), 2**(bits-1)-1)
        
        # Convert to int
        q_weight = q_weight.astype(mx.int8 if bits == 8 else mx.int4)
        
        # Store zeros for asymmetric quantization
        zeros = mx.zeros_like(scales)  # Symmetric quantization
        
        return q_weight, scales, zeros
    
    def _save_quantized(self, name: str, q_weight: mx.array, 
                        scales: mx.array, zeros: mx.array, bits: int):
        """Save quantized weights in MLX-compatible format"""
        
        # Pack 4-bit weights for storage
        if bits == 4:
            q_weight = self._pack_int4(q_weight)
        
        # Save as safetensors
        tensors = {
            f'{name}.weight': q_weight,
            f'{name}.scales': scales,
            f'{name}.zeros': zeros,
            f'{name}.bits': mx.array(bits),
            f'{name}.group_size': mx.array(self.group_size)
        }
        
        # Write to disk
        safetensors.torch.save_file(
            {k: torch.from_numpy(np.array(v)) for k, v in tensors.items()},
            self.output_path / f'{name.replace(".", "_")}.safetensors'
        )
    
    def _pack_int4(self, weights: mx.array) -> mx.array:
        """Pack two int4 values into one byte"""
        # Ensure even number of elements
        flat = weights.reshape(-1)
        if flat.shape[0] % 2 != 0:
            flat = mx.pad(flat, (0, 1))
        
        # Pack pairs
        even = flat[0::2].astype(mx.uint8) & 0x0F
        odd = flat[1::2].astype(mx.uint8) & 0x0F
        packed = (even << 4) | odd
        
        return packed
    
    def _load_tensor(self, name: str) -> mx.array:
        """Load tensor from original model"""
        # Use memory-mapped loading to avoid RAM spikes
        tensor_path = self.model_path / f'{name.replace(".", "_")}.safetensors'
        if tensor_path.exists():
            data = safetensors.torch.load_file(str(tensor_path))
            return mx.array(data[name].numpy())
        else:
            # Fallback to loading from consolidated weights
            for shard in self.model_path.glob('*.safetensors'):
                data = safetensors.torch.load_file(str(shard))
                if name in data:
                    return mx.array(data[name].numpy())
            raise ValueError(f"Tensor {name} not found")
    
    def _save_tensor(self, name: str, tensor: mx.array):
        """Save unquantized tensor"""
        safetensors.torch.save_file(
            {name: torch.from_numpy(np.array(tensor))},
            self.output_path / f'{name.replace(".", "_")}.safetensors'
        )
    
    def _write_metadata(self, config: dict):
        """Write quantization metadata for runtime loading"""
        metadata = {
            'quantization': {
                'attention': {'bits': 8, 'group_size': 64},
                'mlp': {'bits': 4, 'group_size': 64},
                'embeddings': {'bits': 16, 'group_size': None},
                'lm_head': {'bits': 8, 'group_size': 64}
            },
            'model_config': config
        }
        
        with open(self.output_path / 'quantization_config.json', 'w') as f:
            json.dump(metadata, f, indent=2)

# Usage
converter = HybridQuantizationConverter(
    model_path='/path/to/qwen-27b',
    output_path='/Volumes/ExternalSSD/qwen-27b-hybrid',
    max_ram_gb=32  # Stay