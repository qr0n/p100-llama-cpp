# Faster llama.cpp decode on the Tesla P100 (GP100, sm_60)

Two small patches to llama.cpp's CUDA backend that make token generation
**33% faster on one P100 and 23% faster across two**, plus the measurements that
led to them.

| model | config | before | after | |
|---|---|---|---|---|
| Llama-3.1-8B Q4_K_M | 1 card | 39.28 tok/s | **52.22** | **+32.9%** |
| Llama-3.1-8B Q4_K_M | 2 cards, `-sm tensor` | 56.80 tok/s | **70.00** | **+23.2%** |
| Qwen3.8-27B Q4_K_XL | 2 cards, `-sm tensor` | 19.40 tok/s | **22.70** | **+17.0%** |
| any | prompt processing | — | — | unchanged |

A third patch, added later, targets **multi-GPU** rather than the quantized
kernels, and moves prompt processing as well:

| model | config | before | after | |
|---|---|---|---|---|
| Qwen3.8-27B Q4_K_XL | 2 cards, `-sm tensor`, pp8192 | 323.6 tok/s | **378.7** | **+17.0%** |
| Qwen3.8-27B Q4_K_XL | 2 cards, `-sm tensor`, tg128 | 22.3 tok/s | **24.5** | **+9.6%** |
| Qwen3.8-27B Q4_K_XL | real 7,655-token request | 326.9 tok/s | **360.0** | **+10.1%** |

Both patches are 61 added lines across two files. One is architecture-neutral;
the other is guarded to GP100 and is byte-identical SASS on every other card.

## Why the P100 is a special case

GP100 is the only Pascal with **full-rate fp16 and no `__dp4a`**. `__dp4a` — the
4-way byte dot product that every quantized CUDA kernel in llama.cpp is built
around — is sm_61 and up. The P100 is sm_60, so it emulates it with four scalar
int8 multiply-adds, each an `XMAD`.

Measured on this card with `bench/sweep.cu`:

| workload | throughput |
|---|---|
| fp16 (packed `half2` FMA) | **15.80 TOP/s** |
| fp32 | 8.68 TOP/s |
| **int8 (4-way byte dot, emulated)** | **3.95 TOP/s** |
| int32 (XMAD) | 2.81 TOP/s |

**fp16 runs at exactly 4.0x the emulated byte-dot rate.** llama.cpp's quantized
kernels assume the opposite — that int8 is cheap and fp16 is not — because the
Pascal population that matters commercially is the P40 (sm_61), which has `__dp4a`
and *crippled* fp16. Every Pascal-gated decision in the CUDA backend inherits that
assumption, and on GP100 several of them are wrong.

## The actual bottleneck

Not bandwidth. Profiling `mul_mat_vec_q` (92.4% of decode time) against a measured
607 GB/s HBM2 read ceiling:

| | achieved | ceiling | |
|---|---|---|---|
| memory bandwidth | 165.8 GB/s | 607 GB/s | 27% |
| int8 MAC rate | 0.274 T MAC/s | 1.98 T MAC/s | 14% |
| **warp-instruction issue** | 1.13-1.47e11/s | 1.487e11/s | **76-99%** |

The kernel is **instruction-issue bound**, costing **9.25 SASS instructions per
weight**, with the GPU busy 97% of decode wall time. So the lever is instruction
*count*, and cuts convert to speedup nearly 1:1 with ~3.6x of bandwidth headroom
still in reserve.

## The two changes

**`patches/0001` — stop recomputing the q8_1 block sum.**
`vec_dot_q4_K_q8_1_impl_vmmq` recomputed the sum of the activation bytes for every
output row via `dp4a(0x01010101, u, ...)`. That sum does not depend on the weight
row, and `block_q8_1` already stores it in `ds.y` — the sibling `..._impl_mmq`
reads it. Four threads cover one q8_1 block, so exactly one now contributes the
min term. 148 → 138 instructions per 16 weights. **+4%.**

Architecture-neutral: it removes 4 `dp4a` per call everywhere, just worth most
where `dp4a` is emulated. Not bit-identical — `ds.y` is the sum of the original
floats rather than `d * sum(quants)` — but that is the approximation `..._impl_mmq`
already ships, and it is arguably the more faithful of the two since it skips a
quantisation round-trip.

**`patches/0002` — a GP100 MMVQ parameter table.**
`calc_rows_per_block` returned 1 for `ncols_dst == 1`, so every output row
re-loaded and re-unpacked the same activations. A `MMVQ_PARAMETERS_PASCAL` table
(guarded `__CUDA_ARCH__ == GGML_CUDA_CC_PASCAL`, i.e. GP100 only — sm_61 keeps the
generic path) gives 4 rows per block for K-quants, sharing that work across rows.
`calc_rows_per_block` gains a `ggml_type` parameter, mirroring `calc_nwarps`.
**This is the large one.**

Rows-per-block sweep, 1 card, tg128, with 0001 applied:

| rows | 1 | 2 | **4** | 8 |
|---|---|---|---|---|
| tok/s | 40.82 | 48.31 | **52.23** | 51.30 |

K-quants use 4; legacy quants stay at 2, because at 4 exactly one case
(`MUL_MAT type_a=q4_1, n=1`) drifts to NMSE 6.93e-4 against `test-backend-ops`'
5e-4 bound. Everything passes at the shipped setting.

**`patches/0003` — enable the internal AllReduce on Pascal.**
`ggml_cuda_ar_pipeline_init` rejected every device below Volta, with the comment
"the chunked kernel uses `__nanosleep`, which is sm70+". That is the only
Volta-specific construct in `allreduce.cu`: the transport is mapped pinned host
memory, the arrival flags are plain `volatile int` paired with
`__threadfence_system()` — chosen, per the file's own comment, for
"PCIe-attached GPUs without NVLink" — and `ggml_cuda_get_max_cpy_bytes()` already
returns 8 instead of 16 on pre-Volta, so the vector width adapts on its own.

Failing that gate meant `-sm tensor` fell through to `try_allreduce_butterfly`,
a stub that returns `false`, and every layer boundary was serviced by the ggml
scheduler's generic inter-backend copy: **f32, through host RAM, plus a separate
add kernel.** On a 2048-token prefill of a *fully GPU-resident* 27B, an nvprof
trace of the timed window shows **256 device-to-host and 258 host-to-device
transfers of exactly 40.0 MB each — 20.5 GB of PCIe traffic, 23.1% of GPU time.**
(40 MB = 2048 tokens x 5120 embed x 4 bytes; four round trips per layer.)

The patch replaces `__nanosleep(100)` on pre-Volta with a `clock64()` delay loop
of the same duration and lowers the gate to `GGML_CUDA_CC_PASCAL`. 20 added
lines. **Tested on GP100; sm_61 shares the code path but was not available.**

Note the two wins have different mechanisms. Prefill tensors are above the 1 MB
`copy_threshold` and take the copy-engine route, so their gain is the bf16 wire
format halving the bytes. Generation tensors are below it and take the fused
chunked kernel, so their gain is fusion — and it survives with the bf16 wire
disabled, i.e. bit-exact.

**A warning that came out of the same investigation:** llama.cpp only enables
peer access when `GGML_CUDA_P2P` is set. On this dual-socket P100 box, setting
it is a **10.7x regression** (pp2048 357.34 -> 33.45), even though
`cudaDeviceCanAccessPeer` returns true both ways. Peer access across the UPI hop
is pathological here.

## Validation

- `test-backend-ops test -b CUDA0 -o MUL_MAT,MUL_MAT_ID` — **1855 passed, 0 failed**
- perplexity **identical to four decimals** before and after (5.8938 ± 0.06590)
- patch 0002 produces **byte-identical SASS on sm_61 and sm_75** — a strict no-op
  off GP100, verified by compiling both trees and diffing `cuobjdump -sass`
- benchmarks run from a matched start temperature; these cards thermally throttle
  after ~60 s, which will silently corrupt an A/B otherwise

For `0003`, which changes a transport rather than a kernel: perplexity over an
identical corpus is **3.5372 ± 0.06192 before, 3.5330 ± 0.06180 after**, and
**3.5372 ± 0.06192 — exactly the baseline — with `GGML_CUDA_AR_BF16_THRESHOLD=0`**.
That last row is the proof the transport change is numerically exact; the delta
in the middle row is upstream's default bf16 wire format alone, about 7% of one
standard error. The generated SASS of the new spin loop was checked to confirm
the `clock64` backoff was not elided and that warp reconvergence uses `@!P2 SYNC`.

## Applying

Against llama.cpp `b10660` (commit `6c84c7d`):

    git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
    git checkout b10660
    git am /path/to/patches/*.patch

    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 \
          -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
          -DCMAKE_CUDA_HOST_COMPILER=g++-13
    cmake --build build -j$(nproc)

CUDA 13 dropped Pascal, so build with **CUDA 12.x**. 12.4's `nvcc` rejects gcc-15,
hence the explicit gcc-13.

## What else is in here

| path | |
|---|---|
| `bench/sweep.cu` | self-verifying arithmetic and bandwidth sweep for GP100 — the source of the fp16-vs-int8 numbers above |
| `tools/ggufinfo.py` | dependency-free gguf parser; prints exact bytes-read-per-token for roofline work (not the file size — subtract `token_embd`) |
| `data/q4k_one.sass` | extracted baseline SASS of the Q4_K MMVQ kernel |
| `docs/` | the full investigation, method, and the power/thermal findings |

## Method note

The step that mattered was an **attribution experiment**: compile variants of the
kernel with each sub-expression removed and diff the SASS. Three 100-second builds
overturned the working hypothesis — the redundant sum was assumed to cost ~16 XMAD
and actually costs 6, because nvcc folds the multiply-by-one. Without it the effort
would have gone into a 4% change instead of a 33% one.

Single-TU iteration made that cheap: lift the exact `nvcc` line out of
`build/compile_commands.json` and one translation unit rebuilds in ~100 s against
~20 minutes for the library.

## Scope and caveats

- Measured on two Tesla P100-PCIE-16GB, CUDA 12.4, driver 580, Ubuntu, dual Xeon
  Gold 6148. Numbers elsewhere will differ; the *analysis* should hold for any GP100.
- Patches `0001` and `0002` are decode only. Dense prompt processing on sm_60 goes
  to cuBLAS (`ggml_cuda_should_use_mmq` gates MMQ to MoE), and cuBLAS `hgemm`
  already runs at ~86% of this card's fp16 peak, so for those two treat any `pp`
  change as a regression signal. Patch `0003` is the exception: it is a multi-GPU
  transport fix and moves `pp` and `tg` together. It does nothing on one GPU, and
  nothing under `-sm layer`, which does not allreduce.
- **This helps GPU-resident models. It does not help an MoE run with `-ncmoe`.**
  Profiled on a 176B-A3B whose experts live in system RAM: 75% of GPU time is the
  one-time model upload, generation is dominated by the CPU computing 30 of 48
  layers' experts in place, and the end-to-end effect is below that configuration's
  ~13% run-to-run noise. Nothing here changes the host-side path.
- Rows-per-block is tuned per quant *type*, and the right value is type-dependent:
  K-quants want 4, everything else 2, and 1 for both is what upstream does. A
  "Q4_K_XL" model is not all K-quants — Qwen3.8-27B spends ~7% of its decode in
  `iq4_xs` alone, which is why the non-K value matters and is worth ~4% tg there.
- These have **not** been submitted upstream. They are arch-guarded and validated,
  but upstreaming is a separate conversation with the maintainers.

## Credit

llama.cpp is MIT, © the ggml authors. Patches here are against it and carry the
same licence. AI (Claude) was used substantially in the investigation and to write
the code and these notes; the `Co-authored-by` trailers in the patches record that.
