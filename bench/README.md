# sweep.cu — GP100 arithmetic and bandwidth sweep

Self-verifying throughput microbenchmarks. Every workload checks its own results,
so a number that prints is a number that computed the right answer.

    nvcc -O3 -arch=sm_60 -o p100_sweep sweep.cu
    ./p100_sweep              # all workloads
    ./p100_sweep -p mem       # one workload
    ./p100_sweep -p int8 -t 8 -g 0

Workloads: `bit1 tern1 tern8 int8 fp16 int32 fp32 fp64 sfu mem`.

Ops are counted consistently — every MAC is 2 ops (`kOpsPerStep`) — so the
workloads are directly comparable. That matters for the headline claim: the fp16
figure is `half2`, 2 MACs per instruction, and the int8 figure is a 4-way byte dot,
4 MACs per step, and after normalising, fp16 delivers exactly 4.0x the MAC rate.

## Measured, Tesla P100-PCIE-16GB, 175 W cap, card 0

| workload | throughput |
|---|---|
| fp16 (packed half2 FMA) | 15.80 TOP/s |
| fp32 | 8.68 TOP/s |
| int8 (4-way byte dot, emulated on sm_60) | 3.95 TOP/s |
| int32 (XMAD) | 2.81 TOP/s |
| mem (HBM2 read+write) | 498.8 GB/s |

Paper bandwidth is 732 GB/s; a minimal `float4` read loop reaches 607 GB/s (83%),
which is the real ceiling. The `mem` figure exercises the hardest pattern
(read-modify-write on one buffer) and 498.8 is healthy at 82% of the read ceiling —
below both pure read and pure write because of HBM bus turnaround on mixed traffic.

**The 175 W cap does not bind on memory-bound work.** Through the sweep the card
drew 120-133 W with the SM clock pinned at its 1328 MHz maximum. If a bandwidth
number looks low, check `power.draw` and `clocks.sm` before blaming the cap.

## Known issue

`identical()` takes `const T &`, so the verification compare walks *global* memory
one byte at a time (8 `LDG.E.U8` per element) and reports ~183 GB/s for what should
be a pure read. Passing by value gets 319 GB/s; by value with `float4` loads and the
reference hoisted to a register gets 606. Not yet fixed — the clean fix is a
`mask == 0` fast path with 128-bit loads, keeping a by-value byte compare for the
chain-layout call site.
