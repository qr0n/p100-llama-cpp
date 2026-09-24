#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// GP100 decode: the conv reads its cached window in place and writes the shifted window back itself
// (the GET_ROWS / CONCAT / CPY around it are skipped; see ggml_cuda_gp100_conv_state_match)
struct ggml_cuda_ssm_conv_state {
    const float *   state;     // conv cache base
    const int32_t * ids;       // device row index of the source window
    int64_t         state_row; // cache row stride, floats
    const float *   x;         // new input, channel c at x[c * x_stride]
    int64_t         x_stride;
    float *         state_out; // destination row of the shifted window
};

void ggml_cuda_op_ssm_conv_gp100_state(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node,
                                       ggml_tensor * silu_dst, const ggml_cuda_ssm_conv_state & st);
