# Remaining headroom

Ranked by expected value, from the profiles in `investigation.md`.

**Updated 2026-09-12.** Items 3 and 7 are now closed, item 1 is dead, item 5 is
promoted, and item 6 named the wrong kernel. See `negative-results.md` and
`kv-cache-type.md` for the work behind those changes.

---

**1. ~~Speculative decoding with a low-bit self-quant as the draft.~~ DEAD.**
The original argument — batch-1 decode uses ~40% of bandwidth, so verifying
several draft tokens against one weight read is nearly free, and a Q1_0/Q2_0
quant of the same model is an ideal draft needing no code — is still sound in
the abstract. It cannot be done here:

- There is **no higher-precision source** on the box. Only the Q4_K_XL exists;
  an F16 of a 27B is ~54 GB, a download rather than a local operation.
- It **does not fit**. Production MTP already sits at 13671/12535 MiB of 16269.
  Dropping the MTP head to make room leaves ~7.3 GiB; a Q2_K of ~29B params is
  ~9.8 GiB.
- The one size that *would* fit (IQ1_S, ~5.9 GiB) must be **requantized from the
  Q4**, so it would agree with the target *less* than the existing MTP head does
  — and draft acceptance is the entire point.

Do not retry without an F16/BF16 source and more VRAM.

**2. Concurrency.** On an issue-bound kernel, concurrent requests scale close to
linearly. `--parallel 1` leaves most of that on the table. **Still untested**, and
now the most promising untouched item. Note the cost is specific: the SSM
recurrent-state buffer scales with `n_seq_max` times `spec-draft-n-max + 1`
(~78 MiB per sequence), which is exactly why `--parallel 1` is what makes the
MTP config fit at all. This trades VRAM for throughput.

**3. ~~Two more Pascal gates that were tuned for the P40.~~ CLOSED — both
negative.** Full write-up in `negative-results.md`.

- **MMVQ fusion**, disabled for `cc <= GGML_CUDA_CC_PASCAL`: **gate is correct.**
  Forcing it on measures **-2.1%**. Profile shows why: 4992 `unary_gated`
  launches absorbed (saving 11.4 ms) while the matmul costs +101.3 ms, because
  `has_fusion = true` adds predicates to an instruction-issue-bound inner loop.
- **CUDA graphs**, disabled for `cc < GGML_CUDA_CC_VOLTA`: **gate is correct**,
  though its stated reason is not. Graphs run fine on Pascal once enabled; they
  cost **-1.4%**.

> **This item's own text was wrong.** It said both gates "lump GP100 in with
> sm_61". `GGML_CUDA_CC_PASCAL` is 600, so the fusion gate `cc <= PASCAL`
> excludes GP100 and **lets sm_61 (610) through** — it singles GP100 out. Only
> the graphs gate lumps. The prior ("patch 0002 came from exactly this class of
> assumption") was reasonable and still is; it just did not pay here.

**4. Prefill format-shuffling.** cuBLAS `hgemm` is 73.9% of prefill and already at
~86% of fp16 peak — nothing to win in the GEMM. `dequantize_block_q4_K/q6_K` is
12.8% and `convert_unary` f32<->f16 round-trips are 6.3%. That ~19% is where an
fp16 magic-number conversion actually belongs.

> Partly overtaken by `patches/0003` (see `prefill-and-multi-gpu.md`): under
> `-sm tensor` the largest single prefill cost was the cross-card AllReduce
> staged through host RAM in f32, not anything in this list. `convert_unary` at
> ~6.3% is still unexamined and is still the next format-shuffling item.

**5. fp16 MMVQ, revisited. PROMOTED — now the best-supported idea here.**
Rejected on instruction count (`investigation.md` §7); its budget only closes if
the activation-side conversion is amortised across rows, which patch 0002 now
does 4x. That precondition was already satisfied. What is new is direct evidence
that the trade works on this card:

**Switching the KV cache from `q8_0` to `f16` — which replaces an emulated
`dp4a` dot with packed fp16 arithmetic — is worth +24.1% generation, while
reading 52% MORE memory** (`kv-cache-type.md`). That is the same trade item 5
proposes, in attention rather than the matmul. And `mul_mat_vec_q` is **67% of
decode time** against attention's 1.4% at batch 1.

Note this does **not** require f16 weights in VRAM (27.32 B params x 2 B =
54.6 GB, impossible on 32 GB). It converts each quantized block to f16 in
registers at the dot product. Zero extra VRAM.

**6. Long context.** Pascal has no MMA, so attention falls back — **but to
`flash_attn_ext_vec` at batch-1 decode, not `flash_attn_tile`.** The tile kernel
is the *prefill* path, where it is real and large (87.3 s of a ~332 s profiled
prefill at d=49152, 26%). The two are different kernels needing different work;
this item previously conflated them.

Measured 2026-09-12 on Qwen3.8-27B, 2 cards, `-sm tensor`:

- Decode falls **-37.4%** from d=0 to d=98304 (24.49 -> 15.32 tg128).
- **96.1%** of that depth cost is `flash_attn_ext_vec` alone — every matmul is
  flat to 1.00x. It grows 18.08x, 39.8 -> 719.4 us per call, 32 calls/token
  (2 per layer x 16 full-attention layers of 64).
- Growth is **linear** in depth (~13.8 ns per call per token of context), not
  quadratic. Quadratic is prefill behaviour.
- The "open upstream report of a crash past ~24K context on Pascal" **did not
  reproduce**: a 74,919-token request completed clean.

Partly addressed by moving to f16 KV (item 5 / `kv-cache-type.md`), which removes
the emulated-`dp4a` path from this kernel. Still the largest ceiling.

**7. ~~Thermal.~~ RESOLVED — by cardboard.** Sustained decode used to settle at
~1220 of 1328 MHz against a 79 °C setpoint even at 100% fans, and this item said
"the remaining ~8% is ducting, not software." That was right.

The fix was a folded cardboard baffle from a GPU box, blocking the open slot left
by a missing third card. The empty opening was a low-resistance bypass: air took
the easy path out of the hole instead of through the card shrouds. Six fans at
100% could not fix it because the fans were never the problem — the air was going
somewhere else.

| | before | after |
|---|---:|---:|
| `sw_thermal_slowdown` | 32% of busy samples | **0%** (0 of 249) |
| average clock | 1237 MHz | **1317 / 1323 MHz** |
| peak temperature | 80 °C | **72 °C** |

Two caveats. The baffle is **not fastened** and will not survive anyone opening
the chassis — put it back. And room AC was switched on in the same change, so the
split between baffle and ambient is unmeasured; only the baffle is permanent.
