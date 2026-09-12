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

## The "~31% idle" was a measurement error — CORRECTED

Everything above about a ~31% GPU idle was wrong, and the error was in the
analysis, not the hardware.

`nvidia-smi utilization.gpu` was sampled across a whole `llama-bench` run and
averaged over every sample where utilization was non-zero. That run contains two
completely different phases, and the distribution is bimodal:

    203 samples @ 98.7%   <- steady-state decode
    104 samples @ 10.3%   <- model load, which is memcpy-bound
    -----------------------
    mean of both = 68.8%  <- what was reported as "utilization while active"

**Steady-state decode runs at 98.7%.** Measured independently through llama-swap
on a real 700-token completion: **95.8%** for `qwen3.8-27b` and 92.8% for the MTP
entry. The GPUs are saturated; there is no idle worth chasing.

This makes the CUDA-graph result ordinary rather than mysterious. With ~1-4% idle
there is no launch gap to recover, so graphs can only add their own per-graph
bookkeeping — which is exactly the -1.4% measured. The premise that motivated
trying them ("4572 launches per token and a third of the time idle") was half
right: the launch count is real, the idle was not.

The single-card control stands on its own and is the durable result: collapsing
~1000 launches into ONE graph moved throughput 0.02%.

**Lesson for the next person.** `nvidia-smi utilization.gpu` is usable for this
question, but only on a steady-state window. Averaging across model load silently
mixes a memcpy phase into the number. And `nvprof --print-gpu-trace` cannot
substitute: it inflated the same workload 22x in wall time, and its reported
GPU-busy time alone is 2.23x the entire unprofiled wall time.
