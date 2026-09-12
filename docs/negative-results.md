# What does not work on GP100, and the rule that explains it

Two plausible optimisations were tried and both lost. They are recorded because
the reason they lost is a general property of this card, and it predicts which
future changes are worth attempting.

## The rule

**Launch-count reductions buy nothing here. Instruction-count reductions are
everything.**

| change | mechanism | result |
|---|---|---|
| CUDA graphs | 4572 -> 254 kernel launches per token | **-1.4%** |
| MMVQ fusion | -4992 launches, +instructions in the hot loop | **-2.1%** |
| patch 0002 (GP100 MMVQ table) | fewer instructions per weight | **+17 to +33%** |
| f16 KV cache | fp16 unit instead of emulated `dp4a` | **+24%** |

Every measurement below is one binary with the behaviour toggled by an
environment variable, so there is no cross-build confound; mirrored ordering;
cooldown gate before each arm.

---

## 1. CUDA graphs — enabled, works, costs 1.4%

`ggml_cuda_graph_set_enabled` disables CUDA graphs for `cc < GGML_CUDA_CC_VOLTA`.
That is not a capability test — `cudaGraph*` is CUDA 10+ and
architecture-independent, and nothing in the capture path uses a Volta-only
construct. Lowering the gate to PASCAL does enable them.

    graphs ON    tg128  24.690 / 24.706
    graphs OFF   tg128  25.043 / 25.035

**The gate is right; its stated reason is wrong.**

### Why it does nothing — four hypotheses, all refuted

The premise looked strong: batch-1 decode issues **4572 kernel launches per
token** (994 `quantize_q8_1` at 2.3 us each, 418 `rms_norm_f32`, 352
`k_bin_bcast`, 352 `unary_gated_op_kernel`), and `nvidia-smi` on an unprofiled
run reports only ~69% GPU utilization.

1. **Architecture** — no. Graphs run once the gate is lowered.
2. **Launch overhead is the bottleneck** — no. Utilization is 69.5% with graphs
   and 69.0% without.
3. **Cross-card sync under tensor split** — no. A single-card control with no
   AllReduce at all measures **64.0%**, i.e. it idles *more*, not less.
4. **Graph thrash** (KV pointers changing each token would trip
   `ggml_cuda_graph_update_required`, resetting warmup forever) — no.
   `llama-bench -v` over 64 tokens: **0** warmup resets, **16254** graph reuses,
   0 disables.

### The decisive test

Two cards fragment the work: the ggml scheduler splits at every device boundary
and each split becomes its own graph.

| config | distinct graphs | launches/token | kernels per graph |
|---|---:|---:|---:|
| 2 cards, `-sm tensor` | 516 | 254 | 18.0 |
| **1 card, llama-3.1-8b** | **1** | **1.0** | **~1000** |

Single card is the ideal case — the whole token is ONE graph:

    1 card, graphs ON    tg256  52.222 / 52.218
    1 card, graphs OFF   tg256  52.211 / 52.207     -> +0.02%

**Replacing ~1000 kernel launches with one changes nothing.** The CPU enqueues
faster than the GPU drains. The 2-card regression is then just per-graph
bookkeeping across 254 graphs with no launch saving to offset it.

---

## 2. MMVQ fusion — enabled, works, costs 2.1%

`ggml_cuda_should_fuse_mul_mat_vec_q` refuses fusion for `cc <=
GGML_CUDA_CC_PASCAL` with the comment *"fusion is not universally faster on
Pascal"*. Note `GGML_CUDA_CC_PASCAL` is **600**, so that test excludes GP100 and
Maxwell while **sm_61 (P40/P4, 610) is allowed through** — it singles GP100 out
rather than lumping it in.

    fusion ON    tg128  24.533 / 24.517
    fusion OFF   tg128  25.060 / 25.053

### Mechanism, confirmed by profile

`nvprof --print-gpu-summary`, 64 tokens, same binary both arms:

| kernel | OFF ms | OFF calls | ON ms | ON calls |
|---|---:|---:|---:|---:|
| `mul_mat_vec_q` | 3894.5 | 63616 | **3995.8** | 58624 |
| `unary_gated_op_kernel` | 60.5 | 22528 | 49.1 | 17536 |
| `k_bin_bcast` | 69.3 | 22528 | 69.4 | 22528 |

Fusion does exactly what it claims: 4992 `unary_gated` launches absorbed,
**saving 11.4 ms**. The matmul then costs **+101.3 ms**. Net ~90 ms worse.

The fused instantiation carries `has_fusion = true`, adding predicates and
register pressure to an inner loop that on GP100 is instruction-issue bound (no
`__dp4a`, emulated as four scalar int8 MADs). Trading hot-loop instructions for
kernel launches is the wrong trade on this card.

---

## Still open: where the ~31% idle goes

GPU utilization during decode is ~64-69% on one card and two, with graphs and
without. Four explanations are dead (above). The leading remaining hypothesis is
**inter-token CPU time** — sampling, batch bookkeeping, the scheduler walking the
graph — which graphs structurally cannot help, because nothing has been issued
yet. At 52 tok/s a token is 19.2 ms; ~6 ms of CPU work between forward passes
would produce this.

Untested predictions that would confirm or kill it:

1. Utilization should **rise with batch size**. Speculative decoding already
   verifies ~5 tokens per pass and is faster per token.
2. A CPU profile of the decode loop should show the gap outside any CUDA call.

**Rule out the instrument first.** `nvidia-smi utilization.gpu` reports the
fraction of a ~100 ms window in which any kernel ran, and its resolution for
microsecond kernels is not documented well enough to trust to 5 pp.
`nvprof --print-gpu-trace` is no help — it inflated the same measurement 22x
(883 ms/token traced against ~41 ms real), and all of the inflation lands in the
gaps it is supposed to be measuring.
