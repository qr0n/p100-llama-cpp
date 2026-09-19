# Patch 0007: an fp16 matrix-vector path for GP100

Measured on 2x Tesla P100-PCIE-16GB (dual socket, 250 W cap), Qwen3.8-27B UD-Q4_K_XL, `-sm tensor`,
f16 KV, llama.cpp `b10660` + `0001`-`0005`. Every number, including the dead ends, is logged in the
host's `~/BENCH.log`; this page is the summary.

## Result

| | before | after |
|---|---|---|
| `llama-bench` tg128 | 25.04 tok/s | **33.22** |
| `llama-bench` pp512 | 240.5 | 240.6 (MMVQ is batch-1 only; prefill untouched) |
| MTP speculative decoding, real-use prompt set (below) | 26.15 tok/s | **40.91** (draft length 3) |
| decode-path KLD vs a Q8_0 reference, `-ub 1` | 0.011857 | **0.011263** |
| decode-path KLD vs a Q8_0 reference, `-ub 5` | 0.011880 | **0.011456** |

**It is more accurate than the path it replaces**, not less: the old path quantised activations to
8 bits (q8_1). On a real Q5_K tensor the fp16 kernel has 1.1e-3 relative RMS error against an fp64
reference; the q8_1 path has 1.2e-2.

## Why

sm_60 is below `GGML_CUDA_CC_DP4A` (610), so every int8 dot in `mul_mat_vec_q` is four scalar
multiplies. After `0001`/`0002` the K-quant kernels still ran at only 246-298 GB/s — 40-49% of the
608 GB/s this card streams (re-measured: 128-bit coalesced reads; the older "499" figure is the
read+write pattern). GP100, unlike the P40, has full-rate half2 FMA.

## How (`ggml/src/ggml-cuda/mmvq-gp100.cu`)

1. **Activations** are converted once per matmul input to fp16, pre-scaled per 32-block by a power of
   two so `max|y| < 2^15` (no overflow is possible), with exact fp32 block sums kept for the min/offset
   terms of each quant type.
2. **Weights** are unpacked with `PRMT` straight into *subnormal* half bit patterns — the 4-6-bit value
   `q` becomes the half `q * 2^-24`, exactly. `HFMA2` runs at full rate on subnormal operands (measured:
   14.98 vs 14.96 TFLOP/s), so there is no magic-number subtraction.
3. **Accumulation** is `HFMA2` in fp16 within one sub-block (16 terms per lane, bounded), then fp32.
   Offsets (`-32` for Q6_K, `-4` for Q3_K, the `+128` codebook shift for IQ4_XS/IQ4_NL/Q8_0) are removed
   in fp32 with the block sums.

Coverage: Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL, Q8_0 at batch 1; fused ffn gate/up/SWIGLU; 2-8 columns
for Q4_K/Q5_K/IQ4_XS (speculative verification — the multi-column kernels unpack each weight once and
reuse it for every column). Plus:

- an activation-conversion cache: matmuls that share `src1` within one graph evaluation convert it once;
- a fused single-row `RMS_NORM * weight` that writes the cache directly (the stock kernel ran a 5120-float
  row in one latency-bound block).

Device code is compiled for sm_60 only; everything else sees `NO_DEVICE_CODE` stubs and the existing path.
Kill switch: `GGML_CUDA_GP100_MMVQ_DISABLE=1`. Also `GGML_CUDA_GP100_NO_ACT_CACHE`,
`GGML_CUDA_GP100_NO_NORM_FUSION`, and `GP100_RMAP="nrows:R,..."` to override rows per warp for tuning.

## Speculative decoding

With the int8 path, verifying 5 tokens cost ~3x a single-token step, so MTP lost to plain decoding on
realistic prompts. The multi-column kernels cut a 5-token batch from ~88 to ~60 ms. Real-use set: 6
prompts x 2 seeds, temp 0.6 / top_p 0.95 / top_k 20, 600 tokens (agentic ~11k-token context, code edit on
a real CUDA file, code generation, factual QA, ~8k-token summarisation, arithmetic):

| config | tok/s (aggregate) |
|---|---|
| stock, MTP draft 4 | 26.15 |
| 0007, no speculation | 32.25 |
| 0007, MTP draft 4 / 6 / 2 | 36.56 / 33.14 / 40.19 |
| **0007, MTP draft 3** | **40.91** |

The Q6_K multi-column kernel exists but lost to the generic kernel at 5 columns in situ (7.7 vs 6.2 ms per
verify batch), so Q6_K batches stay on the generic path.

## Lessons (read before changing the kernels)

- **Rank kernel variants in situ, not in a standalone harness.** A Q5_K body 18% faster standalone was 18%
  slower inside the model (ptxas load placement). `nvprof` per-kernel durations in situ are reliable (they
  matched an ablation build to 0.5%); its token span is not.
- **Plain `llama-perplexity` does not test these kernels** — batched prefill never runs MMVQ. Gate with
  `-ub 1` (and `-ub 5` for the multi-column path) and KLD against a higher-precision reference.
- On GP100 `PRMT` and shifts are **half rate**; `LOP3`/`HFMA2` are full rate. Count them double.
- Several standalone kernels turned out load-structure bound rather than issue bound: a loads-only
  variant ran as fast as the real kernel. Profile that before trimming instructions.
- CUDA graphs are still a small loss here (31.35 vs 31.82 tok/s), even though the CPU launch floor is
  ~17.8 ms/token; they will matter once GPU time approaches that.
