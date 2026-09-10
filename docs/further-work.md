# Remaining headroom

Ranked by expected value, from the profiles in `investigation.md`. Nothing here has
been implemented.

**1. Speculative decoding with a low-bit self-quant as the draft.** Batch-1 decode
uses ~40% of bandwidth and a fraction of compute, so verifying several draft tokens
against one weight read is close to free. A Q1_0/Q2_0 quant of the *same* model
makes an ideal draft: same architecture, same tokenizer, a quarter of the size. No
code required.

**2. Concurrency.** On an issue-bound kernel, concurrent requests scale close to
linearly. `--parallel 1` leaves most of that on the table.

**3. Two more Pascal gates that were tuned for the P40.** Both lump GP100 in with
sm_61:
- MMVQ fusion is disabled for `cc <= GGML_CUDA_CC_PASCAL`, commented "not
  universally faster on Pascal" — on *which* Pascal?
- CUDA graphs are disabled for all `cc < GGML_CUDA_CC_VOLTA`. Launch overhead is
  proportionally larger now that decode is 33% faster; re-measure the GPU-busy
  fraction before dismissing it.

Patch 0002 came from exactly this class of assumption, so the prior is good.

**4. Prefill format-shuffling.** cuBLAS `hgemm` is 73.9% of prefill and already at
~86% of fp16 peak — nothing to win in the GEMM. But `dequantize_block_q4_K/q6_K` is
12.8% and `convert_unary` f32↔f16 round-trips are 6.3%. That ~19% is where an fp16
magic-number conversion actually belongs: dequant is a genuine per-value
conversion, which the dot product was not.

> **Partly overtaken by `patches/0003` (see `prefill-and-multi-gpu.md`).** That
> profile was single-GPU. Under `-sm tensor` the largest single prefill cost was
> not in this list at all: 23.1% of GPU time was the cross-card AllReduce being
> staged through host RAM in f32, because the internal AllReduce was gated to
> Volta+. `convert_unary` at ~6.3% is still unexamined and is still the next
> format-shuffling item.

**5. fp16 MMVQ, revisited.** Rejected on instruction count (see
`investigation.md` §7), but its budget only closes if the activation-side
conversion is amortised across rows — which patch 0002 now does, 4x. The
precondition is newly satisfied.

**6. Long context.** Pascal has no MMA, so attention runs the `flash_attn_tile`
fallback. It is only 3.9% of prefill at 512 tokens but grows quadratically, and
there is an open upstream report of a crash past ~24K context on Pascal. Largest
ceiling, least explored.

**7. Thermal.** Sustained decode settles at ~1220 of 1328 MHz against a 79 °C
setpoint even at 100% fans. See `power-and-thermal.md`; the remaining ~8% is
ducting, not software.
