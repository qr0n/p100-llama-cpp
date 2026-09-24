#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

template <int S_v, bool KDA, bool keep_rs_t, bool gather_t, bool l2_qk_t, bool ab_t>
static __device__ __forceinline__ void gated_delta_net_body(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const int32_t * s_ids,
                                     int64_t       s_row,
                                     float         l2_eps) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    if constexpr (gather_t) {
        // GP100: curr_state is the cache itself; the row is picked here instead of by a GET_ROWS copy
        curr_state += (int64_t) s_ids[sequence] * s_row + h_idx * S_v * S_v + col * S_v;
        GGML_UNUSED(state_in_offset);
    } else {
        curr_state += state_in_offset + col * S_v;
        GGML_UNUSED_VARS(s_ids, s_row);
    }
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        // GP100 (ab_t): beta arrives as the raw projection; apply the graph's sigmoid here (same op as unary.cu)
        float beta_val;
        if constexpr (ab_t) {
            beta_val = 1.0f / (1.0f + expf(-*beta_t));
        } else {
            beta_val = *beta_t;
        }

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (l2_qk_t) {
            // GP100: the graph's l2_norm(q) / l2_norm(k), done here. Same per-lane partial sums in the same
            // order and the same warp reduction as l2_norm_f32<32>, so the result is bit-identical.
            float kk = 0.0f;
            float qq = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kk += k_reg[r] * k_reg[r];
                qq += q_reg[r] * q_reg[r];
            }
            const float k_scale = rsqrtf(fmaxf(warp_reduce_sum<warp_size>(kk), l2_eps * l2_eps));
            const float q_scale = rsqrtf(fmaxf(warp_reduce_sum<warp_size>(qq), l2_eps * l2_eps));
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                k_reg[r] = k_scale * k_reg[r];
                q_reg[r] = q_scale * q_reg[r];
            }
        } else {
            GGML_UNUSED(l2_eps);
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

#define GDN_LAUNCH_BOUNDS __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void GDN_LAUNCH_BOUNDS gated_delta_net_cuda(
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        const float * curr_state, float * dst, float * state, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, const uint3 neqk1_magic, const uint3 rq3_magic,
        float scale, int64_t state_slot_stride, int K) {
    gated_delta_net_body<S_v, KDA, keep_rs_t, false, false, false>(q, k, v, g, beta, curr_state, dst, state, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, nullptr, 0, 0.0f);
}

// GP100 only: the state input is read in place from the cache row s_ids[seq] (see ggml_cuda_gdn_state_gather)
template <int S_v, bool KDA, bool keep_rs_t, bool l2_qk_t, bool ab_t>
__global__ void GDN_LAUNCH_BOUNDS gated_delta_net_gather_cuda(
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        const float * curr_state, float * dst, float * state, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, const uint3 neqk1_magic, const uint3 rq3_magic,
        float scale, int64_t state_slot_stride, int K, const int32_t * s_ids, int64_t s_row, float l2_eps) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    gated_delta_net_body<S_v, KDA, keep_rs_t, true, l2_qk_t, ab_t>(q, k, v, g, beta, curr_state, dst, state, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row, l2_eps);
#else
    GGML_UNUSED_VARS(q, k, v, g, beta, curr_state, dst, state, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row, l2_eps);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}

template <int S_v, bool KDA, bool keep_rs_t>
static void launch_gated_delta_net_sv(
        const ggml_cuda_kernel_launch_params & launch_params,
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        const uint3 neqk1_magic, const uint3 rq3_magic,
        float scale, int64_t state_slot_stride, int K, const int32_t * s_ids, int64_t s_row, bool l2_qk, float l2_eps,
        bool ab) {
    if (s_ids != nullptr && l2_qk && ab) {
        ggml_cuda_kernel_launch(gated_delta_net_gather_cuda<S_v, KDA, keep_rs_t, true, true>, launch_params,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row, l2_eps);
    } else if (s_ids != nullptr && l2_qk) {
        ggml_cuda_kernel_launch(gated_delta_net_gather_cuda<S_v, KDA, keep_rs_t, true, false>, launch_params,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row, l2_eps);
    } else if (s_ids != nullptr) {
        ggml_cuda_kernel_launch(gated_delta_net_gather_cuda<S_v, KDA, keep_rs_t, false, false>, launch_params,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row, 0.0f);
    } else {
        ggml_cuda_kernel_launch(gated_delta_net_cuda<S_v, KDA, keep_rs_t>, launch_params,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const int32_t * s_ids, int64_t s_row, bool l2_qk, float l2_eps,
        bool ab, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
#define GDN_LAUNCH(SV) launch_gated_delta_net_sv<SV, KDA, keep_rs_t>(launch_params, q_d, k_d, v_d, g_d, b_d, s_d, \
        dst_d, state_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, \
        scale, state_slot_stride, K, s_ids, s_row, l2_qk, l2_eps, ab)
    switch (S_v) {
        case 16:  GDN_LAUNCH(16);  break;
        case 32:  GDN_LAUNCH(32);  break;
        case 64:  GDN_LAUNCH(64);  break;
        case 128: GDN_LAUNCH(128); break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
#undef GDN_LAUNCH
}

bool ggml_cuda_gp100_state_fusion_disabled() {
    static const bool disabled = getenv("GGML_CUDA_GP100_STATE_DISABLE") != nullptr && std::atoi(getenv("GGML_CUDA_GP100_STATE_DISABLE"));
    return disabled;
}

const ggml_tensor * ggml_cuda_gdn_state_gather(const ggml_tensor * gdn) {
    if (gdn->op != GGML_OP_GATED_DELTA_NET || ggml_cuda_gp100_state_fusion_disabled()) {
        return nullptr;
    }
    const ggml_tensor * src_v = gdn->src[2];
    const ggml_tensor * s     = gdn->src[5];
    const int64_t S_v = src_v->ne[0], H = src_v->ne[1], n_seqs = src_v->ne[3];
    // one sequence: every thread then reads exactly the state elements it later writes, and no other
    // block reads them, so the cache row may double as the kernel's input even when it is also the output
    if (n_seqs != 1 || s->type != GGML_TYPE_F32 || !ggml_is_contiguous(s)) {
        return nullptr;
    }
    const ggml_tensor * rows = s->view_src ? s->view_src : s;
    if (s->view_src && s->view_offs != 0) {
        return nullptr;
    }
    const ggml_tensor * states = rows->src[0];
    const ggml_tensor * ids    = rows->src[1];
    if (rows->op != GGML_OP_GET_ROWS || rows->type != GGML_TYPE_F32 || !ggml_is_contiguous(rows) ||
        rows->ne[0] != S_v * S_v * H || rows->ne[1] != 1 || rows->ne[2] != 1 || rows->ne[3] != 1 ||
        states->type != GGML_TYPE_F32 || states->nb[0] != sizeof(float) || states->ne[0] != rows->ne[0] ||
        states->nb[1] % sizeof(float) != 0 || states->ne[2] != 1 || states->ne[3] != 1 ||
        ids->type != GGML_TYPE_I32 || ids->ne[0] != 1 || states->data == nullptr || ids->data == nullptr) {
        return nullptr;
    }
    return rows;
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    // GP100: read the recurrent state in place from the cache row (see ggml_cuda_gdn_state_gather)
    const ggml_tensor * gather = nullptr;
    if (ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_PASCAL) {
        gather = ggml_cuda_gdn_state_gather(dst);
    }
    const int32_t * s_ids = gather ? (const int32_t *) gather->src[1]->data : nullptr;
    const int64_t   s_row = gather ? gather->src[0]->nb[1] / sizeof(float) : 0;
    // GP100: q and k arrive un-normalised and the gather kernel applies the graph's l2_norm itself
    const bool  l2_qk  = gather != nullptr && cache != nullptr && cache->l2_qk;
    const float l2_eps = l2_qk ? cache->l2_eps : 0.0f;
    if (l2_qk) {
        GGML_ASSERT(src_q->op == GGML_OP_L2_NORM && src_k->op == GGML_OP_L2_NORM);
        src_q = src_q->src[0];
        src_k = src_k->src[0];
    }
    // GP100: beta as the raw projection; the kernel applies the sigmoid itself
    const bool ab = l2_qk && cache->ab_beta != nullptr;

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = ab ? cache->ab_beta  : (const float *) src_beta->data;

    const float * s_d   = gather ? (const float *) gather->src[0]->data : (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row, l2_qk, l2_eps, ab, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row, l2_qk, l2_eps, ab, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row, l2_qk, l2_eps, ab, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row, l2_qk, l2_eps, ab, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}

// GP100: ADD(alpha, dt) -> SOFTPLUS -> MUL(., A), one value per head, in one launch (same ops and order as
// binbcast.cu and unary.cu's unary_gated, so the result is bit-identical)
static __global__ void gp100_softplus_bias_mul(const float * __restrict__ x, const float * __restrict__ b,
                                               const float * __restrict__ a, float * __restrict__ dst, const int n) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float v = x[i] + b[i];
    dst[i] = ((v > 20.0f) ? v : logf(1.0f + expf(v))) * a[i];
#else
    GGML_UNUSED_VARS(x, b, a, dst, n);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}

void ggml_cuda_gp100_softplus_bias_mul(ggml_backend_cuda_context & ctx, const float * x, const float * b, const float * a,
                                       float * dst, int n) {
    gp100_softplus_bias_mul<<<(n + 255) / 256, 256, 0, ctx.stream()>>>(x, b, a, dst, n);
}
