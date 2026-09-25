#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
// GP100 in-place state read: most sequences one batched-decode step may carry
#define GDN_GATHER_MAX_SEQS 8

struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
    bool    l2_qk  = false; // GP100: src[0]/src[1] are l2_norm nodes the kernel applies itself (skipped in the graph)
    float   l2_eps = 0.0f;
    // GP100, with l2_qk: the raw beta projection; the kernel applies sigmoid(beta) itself (SIGMOID node skipped)
    const float * ab_beta = nullptr;
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// GP100: when the state input (src[5]) is a one-row GET_ROWS gather of the cache, returns that GET_ROWS; the op
// then reads the cache row in place and the GET_ROWS need not run (see ggml_cuda_gp100_skip_state_gather)
const ggml_tensor * ggml_cuda_gdn_state_gather(const ggml_tensor * gdn);

// kill switch for the GP100 in-place recurrent-state paths (GDN gather, conv window): GGML_CUDA_GP100_STATE_DISABLE=1
bool ggml_cuda_gp100_state_fusion_disabled();

// GP100: dst = softplus(x + b) * a over n contiguous elements (ADD -> SOFTPLUS -> MUL in one launch)
void ggml_cuda_gp100_softplus_bias_mul(ggml_backend_cuda_context & ctx, const float * x, const float * b, const float * a,
                                       float * dst, int n);
