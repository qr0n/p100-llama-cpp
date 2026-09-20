# Faster llama.cpp decode on the Tesla P100 (GP100, sm_60)

## Quick start

This repo ships a ready-to-build llama.cpp with every patch already applied:

    git clone https://github.com/qr0n/p100-llama-cpp && cd p100-llama-cpp
    ./build.sh                       # needs CUDA 12.x + cmake; binaries land in llama.cpp/build/bin/

    # one P100
    llama.cpp/build/bin/llama-server -m model.gguf -ngl 99

    # two P100s: tensor split is where patches 0003/0004 apply
    llama.cpp/build/bin/llama-server -m model.gguf -ngl 99 -sm tensor

`llama.cpp/` is upstream **`b10660`** plus `patches/0001`-`0008`, nothing else;
`tools/verify-source.sh` re-derives it from upstream and diffs to prove that. It is
the exact source running on the machine these numbers came from.

What you get depends on your setup:

| your setup | what applies | expect |
|---|---|---|
| any P100 (sm_60), quantized model | `0007` + `0008` | fp16 matrix-vector path: **Llama-3.1-8B Q4_K_M tg 39.3 -> 79.6, i.e. 2.03x**; qwen3.8-27b tg 25.0 -> 33.7 on two cards, MTP 26.2 -> 40.9 tok/s on real prompts, *more* accurate than stock — see [`docs/fp16-mmvq.md`](docs/fp16-mmvq.md) |
| one P100, K-quant model (Q4_K_M etc.) | `0001` + `0002` | the big decode win, e.g. +33% tg on Llama-3.1-8B |
| any P100 | `0005`, and the default f16 KV cache (do **not** pass `-ctk q8_0`) | ~+5% prompt processing; f16 KV grows to +24% tg at long context |
| two P100s with `-sm tensor` | all of the above + `0003` + `0004` | a further ~+15% prompt processing, +10% tg |
| three or more P100s with `-sm tensor` | all of the above + `0006` | **untested for speed** — see "Three or more cards" below |
| P40 / GTX 10xx (sm_61) | `0001`, `0005` | small; `0002` is GP100-only by design. `./build.sh` detects the card; pass `CUDA_ARCH="60;61"` for a binary that serves both |

**Three or more cards.** Upstream llama.cpp's internal AllReduce
(`ggml/src/ggml-cuda/allreduce.cu`) only handles exactly two devices; with more it
logs `internal AllReduce init failed (n_devices != 2?)` and falls back to the
generic f32-through-host-RAM exchange that `0003` exists to avoid. `0006` extends
it to N devices. It is **correctness-tested only**: on 2x P100 split into 3 and 4
virtual devices with `GGML_CUDA_DEVICES`, perplexity matches the stock fallback
(identical at 4, within 0.02% at 3) and is identical across the chunked,
copy-engine and mixed paths; two-card output is bit-identical to `0005` and tg is
unchanged. Virtual devices share a GPU and a PCIe link, so there is **no speed
data for a real 3+ card box** — if you have one, compare `-sm tensor` with and
without `GGML_CUDA_ALLREDUCE=none` and please report back. `-sm layer` never uses
the AllReduce and works on any number of cards.

**Do not set `GGML_CUDA_P2P`** — on a dual-socket P100 box it is a 10.7x
regression (see below). Numbers everywhere in this README were measured on one
machine (2x P100-PCIE-16GB on different CPU sockets, 250 W cap); yours will
differ, the mechanisms should not.

llama.cpp is MIT-licensed, © the ggml authors — see `llama.cpp/LICENSE`. This
fork is not affiliated with or endorsed by the llama.cpp project.

---

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

A fourth patch overlaps the two directions of that multi-GPU exchange:

| model | config | before | after | |
|---|---|---|---|---|
| Qwen3.8-27B Q4_K_XL | 2 cards, `-sm tensor`, pp2048 | 401.3 tok/s | **424.6** | **+5.7%** |
| Qwen3.8-27B Q4_K_XL | real 7,655-token request | 361.1 tok/s | **379.7** | **+5.1%** |
| Qwen3.8-27B Q4_K_XL | 35-min soak, 1.7k-17k-token prompts | 334.7 tok/s | **350.4** | **+4.7%** |

A fifth patch vectorises the f32<->f16 conversions that feed the cuBLAS prefill
path:

| model | config | before | after | |
|---|---|---|---|---|
| Qwen3.8-27B Q4_K_XL | 2 cards, `-sm tensor`, pp2048 | 427.0 tok/s | **448.7** | **+5.1%** |
| Qwen3.8-27B Q4_K_XL | 2 cards, `-sm tensor`, pp16384 | 384.0 tok/s | **401.4** | **+4.5%** |
| Qwen3.8-27B Q4_K_XL | real 7,655-token request | 381.1 tok/s | **398.5** | **+4.6%** |

Patches `0007` and `0008` replace the batch-1 matrix-vector path entirely. sm_60
has no `__dp4a`, so stock llama.cpp emulates every int8 dot with four scalar
multiplies; GP100 does have a full-rate packed fp16 unit. `0007` converts the
activations to fp16 once and unpacks the weights straight into **subnormal** half
bit patterns, which is exact and needs no magic-number subtraction. `0008` splits
short matrices' walk over K across the block's four warps, so a 24-row matmul
stops being a 160-block serial chain. Measured on one card, same flags, cooled
start, r=3:

| model | config | stock `b10660` | `0001`-`0005` | `+0007`/`0008` | |
|---|---|---|---|---|---|
| Llama-3.1-8B Q4_K_M | 1 card, tg128 | 39.26 tok/s | 52.11 | **79.57** | **2.03x** |
| Llama-3.1-8B Q4_K_M | 1 card, pp512 | 692.9 tok/s | 720.6 | 720.0 | unchanged |

The 8B gains more than Qwen3.8-27B does (+34%) for a reason worth knowing before
you predict your own model: Q4_K_M is dominated by **Q4_K, the fastest of these
kernels** (517 GB/s in situ, 95% of what this access shape can reach), while
Qwen3.8-27B is 45% Q5_K at 427 GB/s and 18% IQ4_XS at 346, and pays a cross-card
AllReduce that a single-card model does not. **The closer your mix is to plain
Q4_K, the bigger the win.** It is also *more* accurate than stock, not less — the
path it replaces quantised the activations to 8 bits.

`0001` and `0002` are 61 added lines across two files: one architecture-neutral,
the other guarded to GP100 and byte-identical SASS on every other card. `0003` and
`0004` are both in `allreduce.cu` and only affect `-sm tensor` across two GPUs.

**The largest single win needs no patch at all — just a flag.** Use `f16` KV
instead of `q8_0`:

| model | config | before | after | |
|---|---|---|---|---|
| Qwen3.8-27B Q4_K_XL | real 74,919-token request, tg | 16.85 tok/s | **20.92** | **+24.1%** |
| Qwen3.8-27B Q4_K_XL | same request, prompt | 271.59 tok/s | 272.40 | +0.3% |

A quantized K sends attention's K·Q dot through `ggml_cuda_dp4a`, which sm_60
emulates as four scalar int8 multiply-adds; an F16 K uses the packed fp16 unit,
which GP100 has and the other Pascals do not. It reads 52% *more* memory and wins
anyway. See [`docs/kv-cache-type.md`](docs/kv-cache-type.md) — including where it
does **not** apply (speculative decoding) and the variant that would be better but
crashes.

Two plausible optimisations were also tried and **lost**: CUDA graphs (−1.4%) and
MMVQ fusion (−2.1%), both of which reduce kernel launches. This used to be stated
here as "launch-count reductions buy nothing and instruction-count reductions are
everything", and that blanket form is **wrong** — a later fusion that removed 128
launches per token moved wall-clock time about 4x more than it moved kernel time,
because the saving is the gap *between* kernels, which a profiler's kernel
durations do not show. The accurate rule is narrower: a launch-count reduction
pays only if it does not add work to a hot loop. Graphs add replay and update
cost; MMVQ fusion adds predicates to an issue-bound loop. Both lost for that
reason, not because launches are free. See
[`docs/negative-results.md`](docs/negative-results.md).

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

## The changes

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
(40 MB = 2048 tokens x 5120 embed x 4 bytes. The counts are both cards summed:
128 per card = 64 layers x **2 exchanges per layer**, the tensor-parallel minimum —
one after the attention/linear-attention output projection, one after the FFN.)

The patch replaces `__nanosleep(100)` on pre-Volta with a `clock64()` delay loop
of the same duration and lowers the gate to `GGML_CUDA_CC_PASCAL`. 20 added
lines. **Tested on GP100; sm_61 shares the code path but was not available.**

Note the two wins have different mechanisms. Prefill tensors are above the 1 MB
`copy_threshold` and take the copy-engine route, so their gain is the bf16 wire
format halving the bytes. Generation tensors are below it and take the fused
chunked kernel, so their gain is fusion — and it survives with the bf16 wire
disabled, i.e. bit-exact.

**`patches/0004` — run the copy-engine H2D on its own stream.**
With `0003` in place, prefill-sized exchanges take the copy-engine path: each card
copies its partial sum to pinned host memory (D2H), then pulls the peer's back
(H2D). Both were issued on one stream, so **each card copied one direction at a
time** although the P100 has two copy engines (`asyncEngineCount = 2`). A trace of
a 2048-token prefill showed 642 ms per card of copies with zero overlap, 12.2% of
wall time. A standalone test of 2 MB pinned copies — the chunk size the AllReduce
uses — moves both directions in 55 ms on two streams against 100-111 ms serially,
cross-socket included.

The patch gives H2D its own stream and keeps every ordering the single stream gave
implicitly with an explicit event. One of those orderings is easy to miss: the
compute stream must wait for the card's *own* D2H to finish, not just the peer's
data to arrive, because the add kernel writes the buffer the D2H is reading (and
on the bf16 path that buffer is a pool allocation freed on return). **A build
without that edge was exactly as fast and produced perplexity 7.68 instead of
3.53.** llama-bench cannot see that; perplexity can. Architecture-neutral — any
multi-GPU `-sm tensor` setup on the copy-engine path should benefit — but only
measured here.

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

`0003` was later **audited for the pre-Volta deadlock hazard** (no independent
thread scheduling below sm_70): only lane 0 of each block spins, the flag it polls
is written by the *peer GPU*, and both barriers are reached converged in SASS. Its
small-tensor path was also checked for exactness by forcing every reduction
through it — perplexity 3.5372 / 3.5330, identical to the copy-engine path. A
35-minute soak with three concurrent clients ran clean.

For `0004`: perplexity identical to `0003` alone in both wire modes (3.5330 /
3.5372); 256-token temp-0 output byte-identical; a second 35-minute concurrent soak
clean (68 requests, 0 hangs, output byte-identical to the pre-`0004` build). The
full series `0001..0006` applied with `git am` onto `b10660` reproduces the tested
tree byte for byte (re-checked 2026-09-18; that tree is `llama.cpp/` in this repo).

## Applying the patches yourself

The bundled `llama.cpp/` plus `./build.sh` is the easy path. To apply the patches
to your own checkout instead, against llama.cpp `b10660` (commit `6c84c7d`):

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
| `llama.cpp/` | upstream `b10660` + all six patches, ready to build |
| `build.sh` | CUDA 12 / sm_60 build of `llama.cpp/`; `CUDA_ARCH`, `JOBS` overridable |
| `tools/verify-source.sh` | re-derives `llama.cpp/` from upstream + `patches/` and diffs it |
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
  already runs at ~80% of this card's true fp16 peak (19.0 TFLOP/s at 1325 MHz — the
  15.80 TOP/s above is a microbenchmark that ran throttled), so for those two treat
  any `pp` change as a regression signal. Patches `0003` and `0004` are the
  exception: they are multi-GPU transport fixes (`0004` moves `pp` only). It does nothing on one GPU, and
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
- These are **not submitted upstream, by decision**. They live here as patches
  against a pinned llama.cpp release.

## Credit

llama.cpp is MIT, © the ggml authors. Patches here are against it and carry the
same licence. AI (Claude) was used substantially in the investigation and to write
the code and these notes; the `Co-authored-by` trailers in the patches record that.
