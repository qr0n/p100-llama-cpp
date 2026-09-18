# Patches

Against llama.cpp **`b10660`** (commit `6c84c7d`).

    git checkout b10660
    git am *.patch

The result is already in [`../llama.cpp/`](../llama.cpp) — all five applied,
checked 2026-09-18 by `../tools/verify-source.sh`.

| patch | scope | effect |
|---|---|---|
| `0001` reuse the stored q8_1 block sum | all architectures | 148 → 138 instructions per 16 weights, +4% tg on GP100 |
| `0002` GP100 MMVQ parameter table | `__CUDA_ARCH__ == 600` only | 4 rows per block for K-quants, the bulk of the win |
| `0003` internal AllReduce on Pascal | `-sm tensor`, cc >= 600 | +10.1% pp / +9.6% tg on qwen3.8-27b; applies on top of `0002` |
| `0004` AllReduce H2D on its own stream | `-sm tensor`, copy-engine path, all architectures | +5.1% real-request pp on qwen3.8-27b, tg unchanged; needs `0003` on Pascal |
| `0005` vectorise contiguous `convert_unary` | all architectures; contiguous + aligned only | +4.6% real-request pp on qwen3.8-27b, tg unchanged; independent of `0001`-`0004` |

`0005` targets the f32<->f16 conversions that feed the cuBLAS prefill path.
`convert_unary` handles arbitrary strides one element per thread, and the
contiguous entry point always hits the degenerate `y[i] = cast(x[i])` case — a
4-byte load and a 2-byte store per thread. On GP100 that measured **164 GB/s**
against ~500 GB/s achievable for mixed read+write traffic, while accounting for
**6.1% of prefill**. Processing 4 elements per thread makes it a 16-byte load and
an 8-byte store. Guarded on element count and pointer alignment, with a fallback
to the generic kernel, so it is a no-op wherever those do not hold. Perplexity is
bit-identical: 3.5330 / 3.5372 on the 60-chunk corpus, matching the baseline to
four decimals in both wire formats.

`0004` lets the two copy directions of each cross-card exchange overlap on the
second copy engine. It adds a stream, so it needs a new ordering edge: the compute
stream must wait for its *own* D2H too. A build without that edge was just as fast
and gave perplexity 7.68 instead of 3.53 — the full series is gated on perplexity
for exactly this reason. Verified 2026-09-11: `git am 0001..0004` onto `b10660`
reproduces the tested tree byte for byte; perplexity identical to `0003` alone;
35-min concurrent soak clean.

`0003` spin-loop safety (pre-Volta has no independent thread scheduling) was
audited 2026-09-11: the only spinner is lane 0 of each block, the flag it polls
is written by the *peer GPU*, and every `BAR.SYNC` is reached converged. See
`/mnt/llm-cache/tools/src/gp100-phase0-2026-09-11/`.

`0002` is a strict no-op elsewhere — verified by compiling the tree with and
without it for sm_61 and sm_75 and diffing `cuobjdump -sass`; byte-identical.

`0001` deliberately is not: it removes 4 `dp4a` per call on every backend. It is
also not bit-identical, since `ds.y` holds the sum of the original floats rather
than `d * sum(quants)`. That is the approximation `vec_dot_q4_K_q8_1_impl_mmq`
already uses. Perplexity is unchanged to four decimals.

They apply independently, but the sweep that picked 4 rows for `0002` was run with
`0001` in place.

## Rebasing onto a newer llama.cpp

Both touch small, stable areas — `vec_dot_q4_K_q8_1{,_impl_vmmq}` in `vecdotq.cuh`
and the parameter tables in `mmvq.cu`. If `calc_rows_per_block` has since gained a
`ggml_type` parameter upstream, `0002`'s signature change will conflict; take
upstream's and keep the `MMVQ_PARAMETERS_PASCAL` branch.
