#pragma once

#include "common.cuh"

// GP100 (sm_60) fp16 matrix-vector path for batch-size-1 K-quant matmuls. See mmvq-gp100.cu.
// fusion: nullptr, or a plain gate/up/SWIGLU fusion (src0 = ffn_up, fusion->gate = ffn_gate, dst = the GLU output)
bool ggml_cuda_gp100_mmvq_supported(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                                    const ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion);

void ggml_cuda_gp100_mmvq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                          const ggml_cuda_mm_fusion_args_host * fusion);

// Fused RMS_NORM * weight for a single row that also fills the GP100 activation cache for the matmuls that
// consume it. Returns false (nothing launched) when not applicable.
// multi_row: several rows (a batched decode's sequences) are allowed; the caller sets it only when a matmul reads mul
bool ggml_cuda_gp100_rms_norm_mul(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul, bool multi_row);

// Gated per-head RMS norm, out = silu(z) * (rms_norm(x) * w) over rows of D = x->ne[0] (64/128/256), writing out
// and the activation cache keyed to `key` (the tensor the consuming matmul takes as src1). Only for shapes
// ggml_cuda_gp100_gated_norm_supported(D, nelements) accepts.
bool ggml_cuda_gp100_gated_norm_supported(int64_t D, int64_t K);
void ggml_cuda_gp100_gated_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w, const ggml_tensor * z,
                                ggml_tensor * out, const ggml_tensor * key, float eps);

// types with a GP100 kernel, so MMVQ gate/up/GLU fusion can be allowed on GP100 for them
static inline bool ggml_cuda_gp100_mmvq_fusable_type(const ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q3_K;
}
