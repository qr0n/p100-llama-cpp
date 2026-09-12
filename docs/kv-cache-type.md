# Use `f16` KV, not `q8_0`, on GP100

**+24.1% generation at 75k context, no prompt-processing cost, and it needs no
patch — only a flag change.** This is specific to GP100 (sm_60) and is the
opposite of the right answer on almost every other card, including the other
Pascals.

| | q8_0 KV | f16 KV | |
|---|---:|---:|---|
| tg @ depth 0 | 24.45 | 25.04 | +2.4% |
| tg @ depth 49152 | 19.18 | 22.53 | +17.5% |
| **tg @ real 74,919-token request** | **16.85** | **20.92** | **+24.1%** |
| prompt @ same request | 271.59 | 272.40 | +0.3% (noise) |
| KV size | 23.7 KiB/tok/card | 36.0 KiB/tok/card | +52% |

Qwen3.8-27B UD-Q4_K_XL, 2x P100, `-sm tensor`, `-c 131072`. The real-request
arms were run mirrored (f16, q8_0, q8_0, f16) with a cooldown gate before each;
repeats agree to 0.02 t/s.

## Why

At batch-1 decode, attention runs `flash_attn_ext_vec`. With a **quantized K**
its K·Q dot goes:

    vec_dot_fattn_vec_KQ_q8_0      fattn-common.cuh
    -> vec_dot_q8_0_q8_1_impl      vecdotq.cuh
    -> ggml_cuda_dp4a              common.cuh

        #if __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A    // 610
            return __dp4a(a, b, c);
        #else                                     // GP100 is 600 -> HERE
            return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];

With an **F16 K** it takes `ggml_cuda_mad(half2)` instead, and
`FAST_FP16_AVAILABLE` *is* defined for sm_60 — `common.cuh` excludes only 610.

GP100 is the only Pascal with full-rate fp16 and no `__dp4a`. Measured on this
card with `bench/sweep.cu`:

| unit | throughput |
|---|---:|
| fp16 packed half2 | **15.80 TOP/s** |
| int8 4-way dot, emulated | **3.95 TOP/s** |

So `-ctk q8_0` sends the kernel that dominates decode-at-depth through the
card's slowest arithmetic. On a P40 (sm_61) the same flag is correct — it *has*
`__dp4a` and *slow* fp16. GP100 inverts the trade and nothing in the config
knows the difference.

**The confirming detail:** f16 KV reads 52% *more* memory and wins anyway. That
is independent evidence the kernel was arithmetic-bound, not bandwidth-bound.

## The win scales with depth

+2.4% at d=0, +17.5% at 49k, +24.1% at 75k. Expected: the `dp4a` penalty is paid
per KV element attended to, so it grows with context.

## Where it does NOT apply

**Speculative decoding (MTP).** Measured +5.1%, not +24%. MTP verifies
`spec-draft-n-max + 1` tokens per pass, so `Q->ne[1] > 2` and
`ggml_cuda_get_best_fattn_kernel` falls through to `flash_attn_tile` — which sets
`need_f16_K = need_f16_V = true` and converts to f16 **regardless** of cache
type. The `dp4a` path is never taken, so there is almost nothing to win, and the
extra VRAM is not worth it.

**Anything not using the vector kernel.** The dispatcher picks it only for
`Q->ne[1] <= 2` with quantized KV. Batched serving will not see this.

## Cost

KV grows 23.7 -> 36.0 KiB/token/card (measured by diffing steady-state VRAM at
`-c 16384` vs `-c 98304`). On a 27B at `-c 131072` with `--mmproj` and `-b 8192`
that is 14591 MiB used and **1680 MiB free** on the tight card, falling to
1588 MiB during a 75k prefill. It fits, with less margin than you might assume —
measure on your own entry rather than trusting a bare-server number.

## Blocked: the variant that should be better

Only **K** drives the dot product; V is merely dequantized. So `-ctk f16
-ctv q8_0` ought to buy the whole arithmetic win at half the memory cost. It
crashes under `-sm tensor`:

    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN)   ggml-backend-meta.cpp:537

via `ggml_gallocr_alloc_graph`. Uniform types are fine; `-sm layer` is fine. An
unhandled split-axis case, not a fundamental limit. Unfixed.
