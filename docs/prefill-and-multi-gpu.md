# GP100 prefill / TTFT — findings, 2026-09-10

Goal for the session: **time-to-first-token hurts.** Find where prefill time
actually goes on 2x P100 and fix what is fixable. Raw data: `RAW.md`.

---

## One-line result

`-sm tensor` on this box was moving **20.5 GB of f32 activations per 2048-token
prefill through host RAM** because llama.cpp's internal AllReduce is gated to
Volta+ by a single `__nanosleep` call. Porting that one line to Pascal is worth
**+10.1% prompt processing and +9.6% generation**, verified on a real request.

## How it was found

1. **Roofline first, as the plan demanded.** Qwen3.8-27B has 26.05 B matmul
   params => 5.21e10 FLOP/token => a two-card ceiling of **522 tok/s** at the
   86% of fp16 peak that cuBLAS achieves here. Measured pp2048 was 356 = 68% of
   that. Within 2x, so **the GEMM is not the opportunity** — the plan's own stop
   rule fires for dense prefill, and it was right to.
2. **The workload said which lengths matter.** 461 logged prompt-evals: **73% of
   all TTFT time is spent on prompts of 1k-16k tokens.** That is the band to
   optimise, not 512 and not 128k.
3. **No depth cliff.** pp512 218, pp2048 356, pp8192 324, pp16384 318. Track C
   (attention at depth) does not qualify at the lengths that matter. The one
   apparent cliff in the first sweep (pp8192 280 +/- 30.78) was thermal drift;
   with a cooldown gate it measures 323.6 +/- small.
4. **So the 32% that is not GEMM had to be attributed.** `nvprof --print-gpu-trace`
   on a 2048-token prefill, restricted to the timed window (model already
   resident), found **256 device-to-host copies of exactly 40.0 MB and 258
   host-to-device of 40.0 MB — 2,664 ms, 23.1% of GPU time.**
   40 MB = 2048 tokens x 5120 embed x 4 bytes. 4 round trips per layer x 64 layers.

## Root cause

`ggml/src/ggml-cuda/allreduce.cu`, `ggml_cuda_ar_pipeline_init`:

    // The chunked kernel uses __nanosleep, which is sm70+ (Volta+).
    if (cc < GGML_CUDA_CC_VOLTA) { ... return nullptr; }

sm_60 fails the gate, so the dispatcher installs `try_allreduce_butterfly` —
a stub that returns `false` — and every layer boundary falls back to the ggml
scheduler's generic inter-backend copy: **f32, through host RAM, with a separate
add kernel.**

`__nanosleep` is the *only* Volta-specific construct in the file. The transport
is mapped pinned host memory (fine on Pascal); the arrival flags are plain
`volatile int` + `__threadfence_system()`, which the file's own comment says
were chosen deliberately for "PCIe-attached GPUs without NVLink"; and
`ggml_cuda_get_max_cpy_bytes()` already returns 8 on pre-Volta instead of 16, so
the vector width adapts by itself.

This is the **same class of finding as patch 0002**: a capability gate that
excludes GP100 for a reason that turns out to be incidental.

## The patch

Branch `gp100-allreduce` (8f28aa5), 20 insertions / 4 deletions, one file:

1. On pre-Volta, replace `__nanosleep(100)` with a `clock64()` delay loop of
   equivalent duration (128 SM cycles ~ 100 ns at 1328 MHz).
2. Lower the init gate from `GGML_CUDA_CC_VOLTA` to `GGML_CUDA_CC_PASCAL`.

Exported as `~/p100-llama-cpp/patches/0003-*.patch`.
Built to `/mnt/llm-cache/tools/llama-cuda-8f28aa5`, wrappers
`llama-{server,bench,cli,perplexity}-cuda-ar`.

## Results

llama-bench, matched start temperature, r=3, production flags:

| test | q4kopt | +allreduce | |
|---|---|---|---|
| pp512 | 218.9 | 226.9 | +3.7% |
| pp2048 | 356.3 | 401.9 | +12.8% |
| pp8192 | 323.6 | 378.7 | **+17.0%** |
| pp16384 | 318.1 | 345.3 | +8.5% |
| tg128 | 22.3 | 24.5 | **+9.6%** |

Real request through llama-swap, 7,655-token prompt:

| entry | prompt eval | |
|---|---|---|
| `qwen3.8-27b` | 326.94 -> **360.04 t/s** | **+10.1%** |
| `qwen3.8-27b-mtp` | 197.07 -> **203.79 t/s** | +3.4% |

TTFT on that prompt: **23.41 s -> 21.26 s.**

## Numerics

The internal AllReduce sends cross-card partials as **bf16** by default (this is
upstream's default on Volta+, not something introduced here).

| build | PPL, identical 60-chunk corpus |
|---|---|
| q4kopt (production) | 3.5372 +/- 0.06192 |
| +allreduce, bf16 wire (default) | 3.5330 +/- 0.06180 |
| +allreduce, `GGML_CUDA_AR_BF16_THRESHOLD=0` | **3.5372 +/- 0.06192 (identical)** |

The bf16-off run reproducing production exactly proves the transport change is
numerically exact. The 0.0042 delta with bf16 on is the wire format alone, ~7%
of one standard error.

**And the two wins separate cleanly** (RAW.md §8): prefill's win IS the bf16
wire (40 MB tensors take the copy-engine route), generation's win is the fused
chunked kernel and survives bf16-off intact. Anyone needing bit-exact output
sets that env var and keeps +9.5% tg while giving up the pp gain.

`test-backend-ops test -b CUDA0 -o MUL_MAT`: 3/3 backends passed, OK.

## NEGATIVE RESULT — do not set GGML_CUDA_P2P

llama.cpp only calls `cudaDeviceEnablePeerAccess` when `GGML_CUDA_P2P` is set.
It looks like an obvious free win, and CLAUDE.md notes that
`torch.cuda.can_device_access_peer` returns True both ways on this box.

| config | pp2048 |
|---|---|
| baseline | **357.34 +/- 0.57** |
| `GGML_CUDA_P2P=1` | **33.45 +/- 0.03** |

**A 10.7x regression.** Same family as `-sm row` failing with "device CUDA0 does
not support split buffers": peer access across the UPI cross-socket hop is
pathological on this platform. Do not retry it.

## What this does NOT touch

- The `-ncmoe` entries run `-sm layer`, which does not allreduce. No effect.
- `llama-3.1-8b` is deliberately single-card (`-sm none -mg 0`). No effect.

## Still open, ranked

1. **The remaining memcpy.** Even after the patch, cross-card exchange is still
   a host round trip — it is just half the bytes and fused. A real P2P or NVLink
   path is impossible here (see the negative result), but the *count* is 4 round
   trips per layer, and whether all four are necessary was not investigated.
2. **`convert_unary` f32<->f16, 6.3% of prefill**, 3968 + 3968 calls on an 8k
   prefill (~15.5 conversions per layer). This is the "format shuffling" item 7
   from the decode handoff and it is still unexamined.
3. **`gated_delta_net_cuda`, 4.7-5.2% of prefill.** qwen35 is a hybrid: only 16
   of 64 layers are full attention (128 `flash_attn_tile` calls vs 384
   `gated_delta_net_cuda`). Nobody has looked at the linear-attention kernel on
   sm_60 at all.
4. **NCCL.** The startup log says "NCCL not compiled in; falling back to internal
   AllReduce. Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance."
   Untested here, and given the P2P result it may well be worse — but it is the
   path upstream considers primary.
5. `-ub` sweep on flash-next (throughput half still not done — see the 2026-09-08
   handoff, and mind the ~13% noise floor on `-ncmoe` entries).
