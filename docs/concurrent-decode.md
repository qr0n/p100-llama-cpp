# Patch 0011: batched decode for concurrent agents

`llama-server --parallel N` decodes one token for each of N sequences per step.
On GP100 most of what `0007`-`0009` did for a single sequence declined that shape
and fell back to stock kernels. Profiling one 4-sequence step on card 1
(`llama-batched-bench`, qwen3.8-27b, `-sm tensor`) showed 56.8 ms of kernel time
against 26.2 ms for one sequence. `0011` goes after the parts that did not need to
be there. It is sm_60-only, and every change leaves single-sequence output
identical in every digit.

| change | what it fixes | 4-seq aggregate tok/s |
|---|---|---|
| (published `0001`-`0009`) | | 68.0 |
| MMVQ takes `[K, 1, n_seqs]` activations as n columns | qwen35's `ssm_out` is shaped per sequence, so 48 matmuls per step went to the emulated-int8 kernel at one column per sequence | 73.1 |
| Q6_K multi-column kernel for 2..4 columns | Q6_K batches stayed on the generic int8 path | 74.2 |
| gated_delta_net reads every sequence's state in place | the 6 MB/card state gather, l2_norm x2 and beta sigmoid per layer | 76.7 |
| fused RMS_NORM x weight + fp16 prep for up to 8 rows | the stock norm and a separate activation conversion | 78.0 |
| ssm_conv reads/writes every sequence's window in place | GET_ROWS + CONCAT + CPY per layer | 79.8 |
| Q4_K/Q5_K multi-column epilogue: 2 FMUL + 4 FFMA | | 80.2 |
| Q6_K multi-column: unconditional activation loads | `act ? __ldg(p) : 0` stopped the compiler from sharing a column's activations across the 4 rows of a warp, so it loaded them 4 times (104 -> 60 loads per iteration) | **86.4** |

The last row is the largest single step. The Q6_K multi-column kernel ran at ~4x its
one-column cost where the Q4_K/Q5_K one runs at ~1.7x, and the SASS showed why:
the loads were guarded by the "is this super-block in range" predicate, and a
predicated load cannot be merged with an identical one in another row's code.
Reading a clamped, always-valid super-block and gating only the accumulation made
them identical and the compiler emitted them once.

## Results

2x P100, `-sm tensor`, Qwen3.8-27B UD-Q4_K_XL, f16 KV.

`llama-batched-bench` decode, 256-token prompts, 128 generated per sequence:

| sequences | `0001`-`0009` | `+0010`, `0011` | |
|---|---|---|---|
| 1 | 35.2 | 35.8 | |
| 2 | 48.6 | 59.4 | +22% |
| 4 | 68.0 | 86.8 | **+28%** |
| 8 | 83.8 | 96.6 | +15% |

Real use: `llama-server --parallel 4 --kv-unified`, four distinct 400-token
completions at once, thinking off, temp 0.6, 2 rounds:

| | 1 agent | 4 agents, per agent | 4 agents, aggregate |
|---|---|---|---|
| `0001`-`0009` | 35.6 tok/s | 16.0 | 58.7 |
| `+0010`, `0011` | 35.5 tok/s | **20.3** | **72.9** |

Quality, KLD against a Q8_0 reference over 20 x 512 tokens:

| path | `0001`-`0009` | `+0010`, `0011` |
|---|---|---|
| one token (`-ub 1`) | 0.011261, top-1 94.686% | 0.011261, top-1 94.686% (identical in every digit) |
| four sequences x one token (`-b 2048 -ub 4`) | 0.011445, top-1 94.686% | 0.011340, top-1 94.745% |

The four-sequence path is slightly *more* accurate than before, for the same
reason `0007` was: its matmuls now take fp16 activations instead of 8-bit ones.
The state fusions are bit-identical to the stock kernels they replace (KLD equal in
every digit with `GGML_CUDA_GP100_STATE_DISABLE=1`); the norm fusion reduces in a
different order than the stock norm, which is the only numeric change.

## The race that one sequence never hits

Reading the recurrent state straight from the cache is safe for one sequence
because each thread reads exactly the elements it later writes. With several
sequences it is not automatically safe: llama.cpp's recurrent memory can reorder
cells, so one sequence's source row can be another sequence's destination row. If
different blocks handled different sequences, one could overwrite a row before
another read it.

`0011` makes that impossible by construction. A gated_delta_net block covers a
head's columns in *every* sequence (one warp per column and sequence) and
barriers between the state loads and the state stores. An ssm_conv thread owns
one channel in every sequence and loads all of them before storing any. No other
block or thread touches those elements.

There is a second writer. `llm_graph_context::build_rs` copies "extra" states
(cells outside the batch that must move) into the same cache right after the
gather that the fusion skips. Every in-place read, including the one-sequence
paths from `0009`, now proves that copy is empty (it is zero-sized whenever the
batch's cells are contiguous, the usual case). If it isn't, the stock gather runs.

## What did not work

Measured in situ; all reverted.

| attempt | result |
|---|---|
| Q4_K/Q5_K multi-column: loop-carried activation pointers (fewer address multiplies) | **-4.4%** |
| multi-column rows per warp R=8 (R=4 kept; R=2 and R=1 are worse still) | -10% |
| gated_delta_net, 8 warps per block instead of 4 | flat |

## Scope

- sm_60 only. New kernels are `NO_DEVICE_CODE` elsewhere; on sm_61 and sm_75
  every pre-existing kernel in the touched files compiles to byte-identical SASS
  (the only differences are line-number constants inside sm_60-only stubs).
- The state paths match Qwen3.8 / Qwen3-Next style gated delta-net graphs; the
  matmul and norm changes apply to any model decoding 2..8 sequences at once.
- Q6_K takes the multi-column kernel for 2..4 columns only. At 5 (an MTP verify
  batch) it measured slower than the generic path before the load fix; that has
  not been re-measured.
- Kill switches: `GGML_CUDA_GP100_STATE_DISABLE=1` (in-place state paths),
  `GGML_CUDA_GP100_MMVQ_DISABLE=1` (the whole fp16 matrix-vector path).
