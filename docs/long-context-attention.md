# Patch 0010: long-context flash attention for pre-Volta (by Kmic-68)

**This patch is [Kmic-68](https://github.com/Kmic-68)'s work, from
[github.com/Kmic-68/llama.cpp](https://github.com/Kmic-68/llama.cpp).** It is
included here unchanged as `0010` (commit author: Kmic-68). The explanations below
summarise his code comments; the measurements are ours, taken on this repo's box.

Pascal has no tensor cores, so long-context attention runs on llama.cpp's
`flash_attn_tile` kernel, far below what the card's fp16 units can do. `0010`
does three things:

- **cuBLAS-GEMM flash attention for long prefill** (`fattn-gemm.cu`). This is a
  standard online-softmax flash attention whose two matmuls are cuBLAS calls,
  which reach 13-15 TFLOPS at attention shapes on a P100. It is used when a batch
  has >= 128 query tokens and the KV cache holds >= 4096. K/V are dequantised
  one chunk at a time, so it also skips the whole-cache f16 staging buffer (512
  MiB per GPU at 262144 context). `GGML_CUDA_FA_GEMM=0` turns it off.
- **GQA-ratio-6 tile kernels** for Qwen3.8-27B under `-sm tensor` (12 query heads
  over 2 KV heads per GPU). The stock ladder folds only 2 of the 6 heads and so
  reads the KV cache 3 times per pass.
- **q4_0 K/V dequantised straight into the tile kernel's shared memory**, with no
  whole-cache conversion first.

## Measured here

2x P100, `-sm tensor`, Qwen3.8-27B UD-Q4_K_XL, **f16 KV** (this box's production
setting; the q4_0-cache parts of the patch are not exercised by these numbers):

| test | before (`0001`-`0009`) | with `0010` | |
|---|---|---|---|
| pp2048 at depth 32768 | 285.6 tok/s | **352.9** | **+23.6%** |
| pp2048 at depth 0 | 446.5 | 445.4 | unchanged |
| tg64 at depth 0 / 32768 | 35.95 / 32.65 | 35.99 / 32.94 | unchanged |
| 4 sequences, 24k context each: prompt (both builds with `0011`) | 251.2 | **318.9** | **+27%** |

Quality:

- **Long-context perplexity**, 6 x 16384 tokens with `-ub 2048`, so the GEMM path
  runs in the scored half of every chunk: before 2.5031, `0010` 2.5016,
  `0010` with fp32 accumulation (`GGML_CUDA_FA_GEMM_PREC=32`) 2.5018. The error
  bar is +/- 0.021. The default accumulates QK^T in fp16, and that lands within
  0.0002 of the fp32 reference.
- **Decode-path KLD** against a Q8_0 reference (`-ub 1`, 20 x 512): identical to
  before in every digit (0.011261, top-1 94.686%).
- `test-backend-ops -o FLASH_ATTN_EXT`: 2936/2936 pass.

One interaction, handled in `0011`: for 2..4-token batches (several sequences
decoding one token each) the 6-head fold is slower than the 2-head ladder until
the KV cache is large. 4 sequences, decode tok/s: ~1.5k KV 82.9 vs 86.4, 32k KV
74.3 vs 76.0, 98k KV 61.9 vs 60.5. So `0011` sends those batches to the ladder
below 65536 KV and leaves everything else as `0010` has it.
