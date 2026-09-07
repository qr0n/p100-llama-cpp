# How the bottleneck was found

The method, in the order it was run, with the numbers. Reproducible on any GP100.

## 1. Baseline

    llama-bench -m Llama-3.1-8B-Instruct-Q4_K_M.gguf -p 512 -n 128 -ngl 99 -r 5

Model must fit **entirely** on one card; partial CPU offload moves the bottleneck
and makes everything downstream meaningless. Pin to the card's own NUMA node.

On a dual-socket box this matters more than expected. Same binary, same model,
tg128:

| pinning | tok/s |
|---|---|
| `numactl --cpunodebind=0 --membind=0` (card's node) | **36.05 ± 0.32** |
| none | 32.84 ± 3.64 |
| `numactl --interleave=all` | 21.99 ± 2.82 |

Interleaving costs 39% and wrecks run-to-run stability on a GPU-resident model.
Interleave is advice for CPU-bound work; it is actively wrong here.

## 2. Roofline — use a measured bandwidth number

Paper bandwidth for the 16 GB P100 is 732 GB/s. A minimal `float4` read loop
reaches **607 GB/s (83%)**, flat from 1792 to 14336 blocks. That is the real
ceiling; judge kernels against 607.

| access pattern (`float4`, 8 GB buffer) | GB/s |
|---|---|
| pure read | 607.1 |
| pure write | 527.8 |
| read + write, same buffer | 498.7 |

`bytes_read_per_token` is **not** the file size. Parse the gguf, sum the tensor
bytes, and subtract `token_embd` — only one row of it is read per token. For
Llama-3.1-8B-Q4_K_M that is **4.617 GB**, not the 4.92 GB the file weighs.
`tools/ggufinfo.py` does this.

    ceiling = 607e9 / 4.617e9 = 131.5 tok/s
    measured 36.05  ->  27.4% of roofline

Under 30% means something is wrong beyond "it's bandwidth bound". It was.

## 3. Which kernel

    nvprof --print-gpu-summary llama-bench -m <model> -p 0 -n 64 -ngl 99 -r 1

Excluding the model-load `HtoD` memcpy:

| kernel | share of decode |
|---|---|
| `mul_mat_vec_q<Q4_K>` | 67.4% |
| `mul_mat_vec_q<Q6_K>` | 25.0% |
| everything else | 7.6% |

MMVQ, not MMQ, not a cuBLAS fallback. On sm_60 it takes the scalar path in
`ggml_cuda_dp4a` (`common.cuh`), four int8 multiply-adds per call.

Note hardware counters (`nvprof --metrics`) fail as a normal user with
`Internal profiling error 4211` — the driver restricts profiling to admins. Kernel
*timing* works fine without root, which is enough.

## 4. Neither ceiling is the wall

| metric | achieved | ceiling | |
|---|---|---|---|
| memory bandwidth | 165.8 GB/s | 607 GB/s | 27% |
| int8 MAC rate (emulated dp4a) | 0.274 T MAC/s | 1.98 T MAC/s | 14% |

Saturating neither, while the GPU is busy 96.9% of decode wall time (1.754 s of
kernels in 1.81 s). That combination means latency- or issue-bound.

## 5. Instruction issue — the real ceiling

    cuobjdump -sass -arch sm_60 libggml-cuda.so
    # symbol: _Z13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb0ELb0E...
    # inner loop = the backward branch, 0xa30 -> 0x410

288 instructions total, **148 in the inner loop**. One iteration is one
`vec_dot_q4_K_q8_1` call covering **16 weights** — verified twice:
`blocks_per_iter = vdr * nwarps * warp_size / qi = 2*4*32/32 = 8` superblocks over
128 threads, and the 0x620-byte body is 147 instructions once Pascal's
one-control-word-per-three-instructions encoding is accounted for.

**9.25 instructions per weight.** Opcode mix:

| op | n | |
|---|---|---|
| XMAD | 36 | emulated int8 multiplies + address arithmetic |
| BFE | 26 | 4-bit unpack / sign extend |
| LOP32I | 14 | masking |
| IADD | 14 | |
| LDG | 12 | |
| SHR | 9 | nibble select |
| FFMA | 5 | scale application |
| I2F | 4 | int→float, **only 4 per 16 weights** |

Per output row the kernel issues `140 + iters*148` instructions. For a
4096-wide row that is 436, and against the P100's single-dispatch issue rate
(56 SMs x 2 schedulers x 1328 MHz = 1.487e11 warp-instructions/s) the large
matmuls land at **76-99% of peak issue**. That is the wall.

Occupancy is not the problem: 128 threads/block, 48 registers, ~62%, with grids of
1024-14336 blocks over 56 SMs.

## 6. Attribute before optimising

The step that changed the outcome. Compile variants with each sub-expression
removed, diff the inner loop:

| variant | loop instr | XMAD |
|---|---|---|
| baseline | 148 | 36 |
| `dot2` (the redundant sum) forced to 0 | 128 (**-20**) | 30 (**-6**) |
| `dot1` (the real products) forced to 0 | 91 (-57) | 14 (-22) |

The working hypothesis had been that the redundant sum cost ~16 XMAD. It costs
**6** — nvcc largely folds the multiply-by-one. By subtraction, **8 of the 36
XMADs are pure address arithmetic**, in neither dot product.

That capped the first optimisation at ~13% and sent the effort at the per-row
repetition instead, which was worth 33%.

Iterate on one translation unit, not the library: lift the `nvcc` line from
`build/compile_commands.json` and rebuild in ~100 s instead of ~20 minutes.

## 7. On the fp16 rewrite that was *not* done

The obvious idea for a card with full-rate fp16 is to redo the dot product in fp16
with a magic-number int→half conversion. It was rejected on the numbers:

- The usual argument is that it removes an `I2F` per value. **There is no such
  cost here** — the current kernel does the whole dot in integer and converts 4
  times per 16 weights, at the end.
- So fp16 must beat ~1 XMAD/weight for the multiply *and* fund fp32 accumulation.
  `HFMA2` accumulates in fp16, which is unusable: products reach 1905 and any sum
  of two exceeds fp16's 2048 exact-integer limit. That forces `HMUL2` + convert +
  add.
- Rough budget: ~20 instructions to convert the weights, ~24 the activations,
  8 `HMUL2`, ~24 to accumulate — at or above `dot1`'s measured 57.

Its natural home is **dequantisation**, not the dot product. In prefill,
`dequantize_block_q4_K<half>` is 12.8% of GPU time and *is* a genuine per-value
int→fp16 conversion. That is where the trick pays.

Worth revisiting for decode now that patch 0002 amortises the activation-side
conversion across 4 rows — that was precisely the missing precondition.

## 8. Where prefill time goes

For completeness, `-p 512`, excluding model load:

| | share |
|---|---|
| `maxwell_hgemm` (cuBLAS) | 73.9% |
| `dequantize_block_q4_K/q6_K` → fp16 | 12.8% |
| `convert_unary` f32↔f16 round-trips | 6.3% |
| `flash_attn_tile` | 3.9% |

Backing out the FLOPs, cuBLAS runs at **~86% of this card's measured fp16 peak** —
near optimal. But ~19% of prefill is format-shuffling that exists only because the
quantized path bails to cuBLAS. On sm_60, `ggml_cuda_should_use_mmq` gates MMQ to
MoE only, so dense prompt processing never uses the quantized kernels at all.
