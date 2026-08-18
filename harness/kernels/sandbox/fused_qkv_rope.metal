#include <metal_stdlib>
using namespace metal;

// Fused QKV GEMV + in-register Rotary Position Embedding (RoPE)
// Computes QKV projections in threadgroup SRAM and applies RoPE without extra memory roundtrips.

kernel void fused_qkv_gemv_rope_kernel(
    device const half* input           [[buffer(0)]], // (hidden_size)
    device const half* w_q             [[buffer(1)]], // (q_dim, hidden_size)
    device const half* w_k             [[buffer(2)]], // (kv_dim, hidden_size)
    device const half* w_v             [[buffer(3)]], // (kv_dim, hidden_size)
    device half* out_q                 [[buffer(4)]], // (q_dim)
    device half* out_k                 [[buffer(5)]], // (kv_dim)
    device half* out_v                 [[buffer(6)]], // (kv_dim)
    constant uint& hidden_size         [[buffer(7)]],
    constant uint& num_q_heads         [[buffer(8)]],
    constant uint& num_kv_heads        [[buffer(9)]],
    constant uint& head_dim            [[buffer(10)]],
    constant uint& position            [[buffer(11)]],
    constant float& rope_theta         [[buffer(12)]],
    uint tid                           [[thread_position_in_grid]]
) {
    // Each thread computes one head channel
    const uint total_q = num_q_heads * head_dim;
    if (tid >= total_q) return;

    // Compute Q GEMV row
    float sum_q = 0.0f;
    const uint row_offset = tid * hidden_size;
    for (uint i = 0; i < hidden_size; ++i) {
        sum_q += float(w_q[row_offset + i]) * float(input[i]);
    }

    // Apply Rotary Position Embedding in-register
    const uint head_idx = tid / head_dim;
    const uint dim_idx = tid % head_dim;
    const uint half_dim = head_dim / 2;

    if (dim_idx < half_dim) {
        // Rotate pair (dim_idx, dim_idx + half_dim)
        float freq = float(position) / pow(rope_theta, float(2 * dim_idx) / float(head_dim));
        float cos_val = cos(freq);
        float sin_val = sin(freq);

        float sum_q_partner = 0.0f;
        uint partner_offset = (head_idx * head_dim + dim_idx + half_dim) * hidden_size;
        for (uint i = 0; i < hidden_size; ++i) {
            sum_q_partner += float(w_q[partner_offset + i]) * float(input[i]);
        }

        float q_rot = sum_q * cos_val - sum_q_partner * sin_val;
        out_q[tid] = half(q_rot);
    } else {
        uint orig_dim = dim_idx - half_dim;
        float freq = float(position) / pow(rope_theta, float(2 * orig_dim) / float(head_dim));
        float cos_val = cos(freq);
        float sin_val = sin(freq);

        float sum_q_partner = 0.0f;
        uint partner_offset = (head_idx * head_dim + orig_dim) * hidden_size;
        for (uint i = 0; i < hidden_size; ++i) {
            sum_q_partner += float(w_q[partner_offset + i]) * float(input[i]);
        }

        float q_rot = sum_q_partner * sin_val + sum_q * cos_val;
        out_q[tid] = half(q_rot);
    }

    // Compute K and V if thread index is within KV bounds
    const uint total_kv = num_kv_heads * head_dim;
    if (tid < total_kv) {
        float sum_v = 0.0f;
        uint v_offset = tid * hidden_size;
        for (uint i = 0; i < hidden_size; ++i) {
            sum_v += float(w_v[v_offset + i]) * float(input[i]);
        }
        out_v[tid] = half(sum_v);

        // K with RoPE
        const uint kv_head_idx = tid / head_dim;
        const uint kv_dim_idx = tid % head_dim;
        float sum_k = 0.0f;
        uint k_offset = tid * hidden_size;
        for (uint i = 0; i < hidden_size; ++i) {
            sum_k += float(w_k[k_offset + i]) * float(input[i]);
        }

        if (kv_dim_idx < half_dim) {
            float freq = float(position) / pow(rope_theta, float(2 * kv_dim_idx) / float(head_dim));
            float cos_val = cos(freq);
            float sin_val = sin(freq);

            float sum_k_partner = 0.0f;
            uint partner_k_offset = (kv_head_idx * head_dim + kv_dim_idx + half_dim) * hidden_size;
            for (uint i = 0; i < hidden_size; ++i) {
                sum_k_partner += float(w_k[partner_k_offset + i]) * float(input[i]);
            }
            float k_rot = sum_k * cos_val - sum_k_partner * sin_val;
            out_k[tid] = half(k_rot);
        } else {
            uint orig_k_dim = kv_dim_idx - half_dim;
            float freq = float(position) / pow(rope_theta, float(2 * orig_k_dim) / float(head_dim));
            float cos_val = cos(freq);
            float sin_val = sin(freq);

            float sum_k_partner = 0.0f;
            uint partner_k_offset = (kv_head_idx * head_dim + orig_k_dim) * hidden_size;
            for (uint i = 0; i < hidden_size; ++i) {
                sum_k_partner += float(w_k[partner_k_offset + i]) * float(input[i]);
            }
            float k_rot = sum_k_partner * sin_val + sum_k * cos_val;
            out_k[tid] = half(k_rot);
        }
    }
}
