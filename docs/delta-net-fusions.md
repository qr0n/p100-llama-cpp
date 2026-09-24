# Patch 0009: folding the gated delta-net state plumbing into its kernels

Qwen3.8-27B (arch `qwen35`) is a hybrid: 48 of its 64 layers are gated delta-net
(linear attention) layers with a recurrent state and a short 1-D convolution. At
decode, each of those layers ran a string of tiny kernels whose only job was to
move state around or apply one elementwise op:

| per linear-attention layer (stock) | what it does | card-1 cost |
|---|---|---|
| `k_get_rows_float_vec` | copies the whole recurrent state (~1.5 MB/card) out of the cache | 9.4 us |
| `k_get_rows_float` + `concat_cont` + `cpy_scalar` | builds the conv window, writes the shifted window back | ~14 us |
| `l2_norm_f32` x2 | normalises q and k | ~5 us |
| `k_bin_bcast`, `unary_gated`, `unary_op` | `softplus(alpha + dt) * A`, `sigmoid(beta)` | ~7 us |
| `rms_norm_f32`, `unary_gated`, `gp100_prep_act` | gated head norm, then fp16 conversion for the output projection | ~9 us |

On GP100 each of these is mostly launch latency: a norm over one 5120-float row
moves 30 KB and takes 3-6 us. `0009` folds them into the kernels that already read
the same data (sm_60 only, one sequence per decode step):

- **gated_delta_net reads the recurrent state in place** from the cache row. Every
  thread loads the state elements it later writes into registers before writing
  any, and no other block touches them, so the row can be input and output.
- **ssm_conv reads its cached window in place and writes the shifted window back**;
  the GET_ROWS / CONCAT / CPY are skipped.
- **gated_delta_net applies the q/k l2_norm and the beta sigmoid itself.**
- **`ADD(alpha, dt) -> SOFTPLUS -> MUL(A)` runs as one kernel.**
- **The gated head norm** `silu(z) * rms_norm(x) * w` runs as one kernel that also
  fills the fp16 activation cache for the output projection.

Result on 2x P100, `-sm tensor`, UD-Q4_K_XL: **1786 -> 1306 kernel launches per
token**, card-1 kernel time 28.68 -> 26.69 ms/token.

| measurement | before | after | |
|---|---|---|---|
| tg128 (lab proxy, r=3, cooled, alternating) | 33.69 | **36.31** | +7.8% |
| real-use aggregate, 6 prompts x 2 seeds, temp 0.6, via llama-server | 32.30 | **35.25** | **+9.1%** |
| same, 11,367-token agentic prompt | 31.60 | 34.45 | +9.0% |
| pp512 | 240.6 | 239.3 | within noise |
| decode-path KLD vs a Q8_0 reference (`-ub 1`, 20 x 512) | 0.011261 | **0.011261** | identical in every digit |

"before" for the real-use row is the previously deployed build (0001-0007 plus
earlier fp16 kernels); for tg128 it is the same tree plus `0008`.

## Every fused result is bit-identical

Each fusion performs the same float operations in the same order as the kernels it
replaces: the l2 norm keeps `l2_norm_f32<32>`'s per-lane partial sums and warp
reduction; the gated norm pads its cross-warp reduction exactly as the stock
256-thread `block_reduce` does (the stock kernel's extra warps contribute exact
zeros); softplus/sigmoid use the same expressions as `unary.cu`. The KLD against
the Q8_0 reference therefore did not move in any digit at any of the five steps.

## The trap: skipping a node changes the allocator's view of lifetimes

A fused kernel that reads a tensor *later* than that tensor's last graph consumer
is reading memory `ggml-alloc` considers free. Any node executed in between may
have been placed on it. The first version of the alpha/beta fusion deferred the
raw alpha read to gated_delta_net; the beta projection, same size, reused alpha's
buffer. It benchmarked **+1.7% and was wrong: KLD 0.0855 vs 0.0113, top-1 88.0%**.
The benchmark cannot see this; only the quality gate did.

`ggml_cuda_gp100_not_clobbered()` now proves, at match time, that no node which
actually runs between the last graph consumer and the fused read overlaps the
source, and fused outputs must be disjoint from deferred sources or alias them
exactly where the kernel is element-wise in place. When a check fails the stock
nodes run. Matching is stateless and use-count exact: the eval loop skips a node
only if the same match holds for the kernel that replaces it.

## Scope

- **One sequence per decode step.** With several sequences in a step the matches
  decline and stock kernels run. Measured with `--parallel 4 --kv-unified`, 400-token
  completions, thinking off:

  | | 1 agent | 4 agents, per agent | 4 agents, aggregate |
  |---|---|---|---|
  | before (0001-0007) | 32.6 tok/s | 16.3 | 59.2 |
  | with 0008 + 0009 | **35.5** | 16.1 | 59.0 |

  A single active agent gets the +9%; four concurrent ones run the stock kernels.
- **sm_60 only.** New kernels are `NO_DEVICE_CODE` elsewhere; pre-existing kernels in
  the touched files compile to byte-identical SASS on sm_61 and sm_75 (the only
  difference in `mmvq-gp100.cu`'s off-arch stubs is the constant-bank offset of
  their name strings).
- **Only `qwen35`-style gated delta-net graphs** (Qwen3.8, Qwen3-Next family) match.
  Other models are unaffected.
- Kill switch: `GGML_CUDA_GP100_STATE_DISABLE=1` turns off the in-place state paths;
  `GGML_CUDA_DISABLE_FUSION=1` turns off everything that relies on graph fusion.
- `0009` also carries a Q6_K row mapping for the output head (8 rows per warp for
  nrows >= 32768). It measured null (-0.2%, within noise); it is included because it
  is part of the tree that was tested.
