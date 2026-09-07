# Patches

Against llama.cpp **`b10660`** (commit `6c84c7d`).

    git checkout b10660
    git am *.patch

| patch | scope | effect |
|---|---|---|
| `0001` reuse the stored q8_1 block sum | all architectures | 148 → 138 instructions per 16 weights, +4% tg on GP100 |
| `0002` GP100 MMVQ parameter table | `__CUDA_ARCH__ == 600` only | 4 rows per block for K-quants, the bulk of the win |

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
