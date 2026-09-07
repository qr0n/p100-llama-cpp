// p100_sweep.cu - self-verifying throughput sweep across every arithmetic
// width GP100 can actually execute, from single-bit logic up to FP64.
//
//   nvcc -O3 -std=c++17 -arch=sm_60 -o p100_sweep p100_sweep.cu
//
//   ./p100_sweep                 every workload, 15s each
//   ./p100_sweep -t 120          15 minutes total, 2 min per workload
//   ./p100_sweep -p fp16 -t 600  hammer one workload
//
// Each workload is a dependent chain held in registers, so the number that
// comes out is the sustained issue rate of that pipeline rather than a memory
// benchmark. Results are checked bit-for-bit every round: exactly reproducible
// workloads are checked against a CPU reference, the rest against the card's
// own first round. Ctrl-C stops after the current round and still reports.
//
// Pascal has no tensor cores, no BF16, no TF32 and no FP8, so the sweep stops
// where the silicon does. DP4A is sm_61+, absent on GP100, so the INT8
// workload falls back to an unpacked emulation there - that is a real result,
// not a workaround: it is what INT8 costs on this chip.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <csignal>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t status_ = (call);                                         \
        if (status_ != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,\
                         cudaGetErrorString(status_));                        \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kChains = 16;          // independent chains per thread, hides latency
constexpr int kFmasPerElement = 4;   // arithmetic per element per memory pass
constexpr double kTargetRoundSeconds = 0.25;

volatile std::sig_atomic_t g_interrupted = 0;

void request_stop(int) { g_interrupted = 1; }

// ---------------------------------------------------------------------------
// Primitives that exist on both sides of the PCIe bus, so a workload's step
// function is written once and replayed on the CPU to check the GPU's answer.
// ---------------------------------------------------------------------------

__host__ __device__ inline float fused(float a, float b, float c) {
#ifdef __CUDA_ARCH__
    return __fmaf_rn(a, b, c);
#else
    return std::fmaf(a, b, c);
#endif
}

__host__ __device__ inline double fused(double a, double b, double c) {
#ifdef __CUDA_ARCH__
    return __fma_rn(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

__host__ __device__ inline unsigned population_count(unsigned bits) {
#ifdef __CUDA_ARCH__
    return __popc(bits);
#else
    return __builtin_popcount(bits);
#endif
}

// Four uint8 lanes multiplied pairwise and summed into an accumulator: one
// instruction from sm_61, a handful of shifts and multiplies before that.
__host__ __device__ inline unsigned dot4(unsigned a, unsigned b, unsigned accumulator) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, accumulator);
#else
    unsigned sum = accumulator;
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        const int shift = lane * 8;
        sum += ((a >> shift) & 0xFFu) * ((b >> shift) & 0xFFu);
    }
    return sum;
#endif
}

// ---------------------------------------------------------------------------
// Workload policies. Each names its state, one step of its recurrence, and how
// many arithmetic ops that step is worth. Recurrences are chosen to be
// non-degenerate: no fixed point, no overflow, no denormals, no matter how
// long the chain runs.
// ---------------------------------------------------------------------------

// x <- x*0.999 + 0.001 converges on 1.0 and stays there, well inside normal range.
struct Fp32Workload {
    using State = float;
    static constexpr double kOpsPerStep = 2.0;
    static constexpr bool kReplayableOnHost = true;
    __device__ static State seed(int chain) { return 0.5f + chain * 0.05f; }
    __host__ __device__ static State step(State x) { return fused(x, 0.999f, 0.001f); }
};

struct Fp64Workload {
    using State = double;
    static constexpr double kOpsPerStep = 2.0;
    static constexpr bool kReplayableOnHost = true;
    __device__ static State seed(int chain) { return 0.5 + chain * 0.05; }
    __host__ __device__ static State step(State x) { return fused(x, 0.999, 0.001); }
};

// Packed half2: GP100 issues these two-lanes-at-a-time, the one Pascal chip
// that does. Host arithmetic on __half2 does not exist, so this one is checked
// against the GPU's own first round.
struct Fp16Workload {
    using State = __half2;
    static constexpr double kOpsPerStep = 4.0;  // two lanes, multiply and add each
    static constexpr bool kReplayableOnHost = false;
    __device__ static State seed(int chain) {
        return __floats2half2_rn(0.5f + chain * 0.05f, 0.6f + chain * 0.05f);
    }
    __device__ static State step(State x) {
        return __hfma2(x, __float2half2_rn(0.999f), __float2half2_rn(0.001f));
    }
};

// Integer arithmetic mod 2^32 is associative, which the compiler exploits: an
// affine chain (x = x*a + c) has consecutive steps that compose into a single
// multiply-add with folded constants, so an LCG here measures the compiler
// rather than the multiplier. Squaring one half and feeding it into the other
// keeps both operands runtime values, so every step has to be issued.
struct Int32Workload {
    struct State { unsigned mixed, squared; };
    static constexpr double kOpsPerStep = 4.0;  // two multiplies, two adds
    static constexpr bool kReplayableOnHost = true;
    __device__ static State seed(int chain) {
        return State{0x9E3779B9u * (chain + 1), 0x85EBCA6Bu * (chain + 1) + 1u};
    }
    __host__ __device__ static State step(State x) {
        return State{x.mixed * x.squared + 1013904223u, x.squared * x.squared + 1664525u};
    }
};

// Feeding the accumulator back in as the chain state keeps the byte lanes
// churning; the trailing increment removes zero as an absorbing state.
struct Int8Workload {
    using State = unsigned;
    static constexpr double kOpsPerStep = 8.0;  // four products, four adds
    static constexpr bool kReplayableOnHost = true;
    __device__ static State seed(int chain) { return 0x51ED2701u * (chain + 1); }
    __host__ __device__ static State step(State x) { return dot4(x, 0x03050709u, x) + 1u; }
};

// XNOR-popcount, the binarised-network primitive: 32 lanes of logic per word.
// The rotate keeps bits moving through the shifter instead of settling.
struct Bit1Workload {
    using State = unsigned;
    static constexpr double kOpsPerStep = 64.0;  // 32 lanes XNOR + 32 lanes accumulate
    static constexpr bool kReplayableOnHost = true;
    __device__ static State seed(int chain) { return 0xC2B2AE35u * (chain + 1) + 1u; }
    __host__ __device__ static State step(State x) {
        return ((x << 1) | (x >> 31)) ^ population_count(x ^ 0xA5A5A5A5u);
    }
};

// 1.58-bit (BitNet-style) weights: every weight is -1, 0 or +1, so the matmul
// contains no multiplies at all - only adds, subtracts and skips. Both
// variants derive their weight planes from the running accumulator rather than
// from constants, because constant planes would let the compiler specialise the
// skips away and report a rate no streaming kernel could reach.
//
// Binary activations decompose the dot product into two popcounts: the
// positive plane minus the negative plane, 32 ternary MACs per word.
struct Tern1Workload {
    struct State { unsigned activations, accumulator; };
    static constexpr double kOpsPerStep = 64.0;  // 32 lanes, multiply and add each
    static constexpr bool kReplayableOnHost = true;
    __device__ static State seed(int chain) {
        return State{0xB5297A4Du * (chain + 1) + 1u, 0x68E31DA4u * (chain + 1) + 1u};
    }
    __host__ __device__ static State step(State x) {
        constexpr unsigned kNonzero = 0xF3C5A96Bu;  // which lanes hold a non-zero weight
        const unsigned positive = x.accumulator & kNonzero;
        const unsigned negative = ~x.accumulator & kNonzero;
        const unsigned activations = x.activations;
        const unsigned accumulator = x.accumulator
                                   + population_count(activations & positive)
                                   - population_count(activations & negative) + 1u;
        return State{((activations << 1) | (activations >> 31)) ^ accumulator, accumulator};
    }
};

// Int8 activations instead: no popcount shortcut, so each lane is a branchless
// conditional negate masked by the zero plane. Still multiply-free, which is
// the point on a chip whose 32-bit multiplier is built out of XMAD pieces.
struct Tern8Workload {
    struct State { unsigned activations, accumulator; };
    static constexpr double kOpsPerStep = 8.0;  // four lanes, multiply and add each
    static constexpr bool kReplayableOnHost = true;
    __device__ static State seed(int chain) {
        return State{0x9E3779B9u * (chain + 1) + 1u, 0xC2B2AE35u * (chain + 1) + 1u};
    }
    __host__ __device__ static State step(State x) {
        const unsigned signPlane = x.accumulator;
        const unsigned zeroPlane = x.accumulator >> 8;
        unsigned accumulator = x.accumulator + 1u;
#pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
            const unsigned value = (x.activations >> (lane * 8)) & 0xFFu;
            const unsigned negate = 0u - ((signPlane >> lane) & 1u);
            const unsigned keep = 0u - ((zeroPlane >> lane) & 1u);
            accumulator += ((value ^ negate) - negate) & keep;
        }
        const unsigned activations = x.activations;
        return State{((activations << 1) | (activations >> 31)) ^ accumulator, accumulator};
    }
};


struct SfuWorkload {
    using State = float;
    static constexpr double kOpsPerStep = 1.0;
    static constexpr bool kReplayableOnHost = false;
    __device__ static State seed(int chain) { return 0.5f + chain * 0.05f; }
    __device__ static State step(State x) { return __sinf(x) + 0.5f; }
};

// ---------------------------------------------------------------------------
// Kernels, written once against the policy interface.
// ---------------------------------------------------------------------------

template <typename Workload>
__global__ void write_seeds(typename Workload::State *seeds) {
    seeds[threadIdx.x] = Workload::seed(threadIdx.x);
}

template <typename Workload>
__global__ void burn(typename Workload::State *results,
                     const typename Workload::State *seeds, long long iterations) {
    using State = typename Workload::State;
    State chain[kChains];
#pragma unroll
    for (int i = 0; i < kChains; ++i) chain[i] = seeds[i];

    for (long long step = 0; step < iterations; ++step) {
#pragma unroll
        for (int i = 0; i < kChains; ++i) chain[i] = Workload::step(chain[i]);
    }

    const size_t thread = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
#pragma unroll
    for (int i = 0; i < kChains; ++i) results[thread * kChains + i] = chain[i];
}

// One read and one write per element with a little arithmetic between:
// bandwidth-bound, and it walks the whole allocation so bad cells surface.
// Accesses are 16 bytes wide because a warp issuing 4-byte loads cannot keep
// enough requests in flight to fill HBM2.
__global__ void memory_burn(float4 *buffer, size_t vectorCount) {
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < vectorCount; i += stride) {
        float4 lanes = buffer[i];
#pragma unroll
        for (int f = 0; f < kFmasPerElement; ++f) {
            lanes.x = Fp32Workload::step(lanes.x);
            lanes.y = Fp32Workload::step(lanes.y);
            lanes.z = Fp32Workload::step(lanes.z);
            lanes.w = Fp32Workload::step(lanes.w);
        }
        buffer[i] = lanes;
    }
}

__global__ void fill(float *buffer, size_t count, float value) {
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride)
        buffer[i] = value;
}

template <typename T>
__device__ inline bool identical(const T &left, const T &right) {
    const unsigned char *a = reinterpret_cast<const unsigned char *>(&left);
    const unsigned char *b = reinterpret_cast<const unsigned char *>(&right);
#pragma unroll
    for (size_t i = 0; i < sizeof(T); ++i)
        if (a[i] != b[i]) return false;
    return true;
}

// referenceMask picks which reference entry element i must equal: kChains-1
// for chain layouts, 0 when every element shares one expected value.
template <typename T>
__global__ void count_mismatches(const T *data, const T *reference, size_t count,
                                 size_t referenceMask, unsigned long long *mismatches) {
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride) {
        if (!identical(data[i], reference[i & referenceMask]))
            atomicAdd(mismatches, 1ULL);
    }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct Report {
    double rate;      // tera-ops or GB/s
    const char *unit;
    unsigned long long mismatches;
    double elapsed;
};

template <typename KernelPtr>
int resident_blocks(KernelPtr kernel, int multiprocessors) {
    int blocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel,
                                                             kThreadsPerBlock, 0));
    return blocksPerSM * multiprocessors;
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// Times a round at a small size and scales to kTargetRoundSeconds, so the
// progress line ticks at the same rate whatever the pipeline's throughput is.
template <typename Round>
long long calibrate(Round runRound, long long probeUnits) {
    const auto start = std::chrono::steady_clock::now();
    runRound(probeUnits);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = seconds_since(start);
    const double scaled = probeUnits * (kTargetRoundSeconds / (elapsed > 0 ? elapsed : 1e-3));
    return scaled < 1 ? 1 : (long long)scaled;
}

template <typename Workload>
void replay_on_host(typename Workload::State *values, int count, long long iterations) {
    for (int i = 0; i < count; ++i) {
        auto x = values[i];
        for (long long step = 0; step < iterations; ++step) x = Workload::step(x);
        values[i] = x;
    }
}

void print_progress(const Report &report, double duration) {
    std::printf("\r    %5.0fs / %.0fs   %9.2f %-8s mismatches: %llu    ",
                report.elapsed, duration, report.rate, report.unit, report.mismatches);
    std::fflush(stdout);
}

template <typename Workload>
Report run_compute(int multiprocessors, double duration) {
    using State = typename Workload::State;

    const int blocks = resident_blocks(burn<Workload>, multiprocessors);
    const size_t threads = size_t(blocks) * kThreadsPerBlock;
    const size_t elements = threads * kChains;

    State *seeds = nullptr, *results = nullptr, *reference = nullptr;
    unsigned long long *mismatches = nullptr;
    CUDA_CHECK(cudaMalloc(&seeds, kChains * sizeof(State)));
    CUDA_CHECK(cudaMalloc(&reference, kChains * sizeof(State)));
    CUDA_CHECK(cudaMalloc(&results, elements * sizeof(State)));
    CUDA_CHECK(cudaMalloc(&mismatches, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(mismatches, 0, sizeof(unsigned long long)));
    write_seeds<Workload><<<1, kChains>>>(seeds);
    CUDA_CHECK(cudaDeviceSynchronize());

    const long long iterations = calibrate(
        [&](long long n) { burn<Workload><<<blocks, kThreadsPerBlock>>>(results, seeds, n); },
        20000);

    std::printf("    %zu threads x %d chains, %lld iterations/round, checked against %s\n",
                threads, kChains, iterations,
                Workload::kReplayableOnHost ? "CPU replay" : "first round");

    if constexpr (Workload::kReplayableOnHost) {
        State expected[kChains];
        CUDA_CHECK(cudaMemcpy(expected, seeds, sizeof(expected), cudaMemcpyDeviceToHost));
        replay_on_host<Workload>(expected, kChains, iterations);
        CUDA_CHECK(cudaMemcpy(reference, expected, sizeof(expected), cudaMemcpyHostToDevice));
    } else {
        burn<Workload><<<blocks, kThreadsPerBlock>>>(results, seeds, iterations);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(reference, results, kChains * sizeof(State),
                              cudaMemcpyDeviceToDevice));
    }

    const double opsPerRound = double(elements) * iterations * Workload::kOpsPerStep;
    const int checkBlocks = resident_blocks(count_mismatches<State>, multiprocessors);
    const auto start = std::chrono::steady_clock::now();
    Report report{0, "TOP/s", 0, 0};
    double completedOps = 0;

    while (report.elapsed < duration && !g_interrupted) {
        burn<Workload><<<blocks, kThreadsPerBlock>>>(results, seeds, iterations);
        count_mismatches<State><<<checkBlocks, kThreadsPerBlock>>>(
            results, reference, elements, kChains - 1, mismatches);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&report.mismatches, mismatches, sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));

        completedOps += opsPerRound;
        report.elapsed = seconds_since(start);
        report.rate = completedOps / report.elapsed / 1e12;
        print_progress(report, duration);
    }
    std::printf("\n");

    CUDA_CHECK(cudaFree(seeds));
    CUDA_CHECK(cudaFree(reference));
    CUDA_CHECK(cudaFree(results));
    CUDA_CHECK(cudaFree(mismatches));
    return report;
}

Report run_memory(int multiprocessors, double duration) {
    size_t freeBytes = 0, totalBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    const size_t bytes = size_t(freeBytes * 0.9) & ~size_t(4095);
    const size_t count = bytes / sizeof(float);

    float *buffer = nullptr, *reference = nullptr;
    unsigned long long *mismatches = nullptr;
    CUDA_CHECK(cudaMalloc(&buffer, bytes));
    CUDA_CHECK(cudaMalloc(&reference, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mismatches, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(mismatches, 0, sizeof(unsigned long long)));

    float4 *vectors = reinterpret_cast<float4 *>(buffer);
    const size_t vectorCount = count / 4;

    const int blocks = resident_blocks(memory_burn, multiprocessors);
    float expected = 0.5f;
    fill<<<blocks, kThreadsPerBlock>>>(buffer, count, expected);
    CUDA_CHECK(cudaDeviceSynchronize());

    const long long passes = calibrate(
        [&](long long n) {
            for (long long p = 0; p < n; ++p)
                memory_burn<<<blocks, kThreadsPerBlock>>>(vectors, vectorCount);
        },
        4);

    // Calibration already mutated the buffer, so the reference absorbs those
    // passes too: buffer state carries forward across the entire run.
    replay_on_host<Fp32Workload>(&expected, 1, 4 * kFmasPerElement);
    std::printf("    %.2f GB allocated, %zu elements, %lld passes/round\n",
                bytes / 1073741824.0, count, passes);

    const double bytesPerRound = double(bytes) * 2.0 * passes;
    const int checkBlocks = resident_blocks(count_mismatches<float>, multiprocessors);

    // The verification sweep is a full read of the allocation. Timing it into
    // the same figure as the burn would report a blend of the two rates, so
    // each is measured on its own events.
    cudaEvent_t burnStart, burnStop, checkStart, checkStop;
    for (cudaEvent_t *event : {&burnStart, &burnStop, &checkStart, &checkStop})
        CUDA_CHECK(cudaEventCreate(event));

    const auto start = std::chrono::steady_clock::now();
    Report report{0, "GB/s", 0, 0};
    double movedBytes = 0, burnSeconds = 0, readBytes = 0, checkSeconds = 0;

    while (report.elapsed < duration && !g_interrupted) {
        CUDA_CHECK(cudaEventRecord(burnStart));
        for (long long p = 0; p < passes; ++p)
            memory_burn<<<blocks, kThreadsPerBlock>>>(vectors, vectorCount);
        CUDA_CHECK(cudaEventRecord(burnStop));

        replay_on_host<Fp32Workload>(&expected, 1, passes * kFmasPerElement);
        CUDA_CHECK(cudaMemcpy(reference, &expected, sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(checkStart));
        count_mismatches<float><<<checkBlocks, kThreadsPerBlock>>>(buffer, reference, count, 0,
                                                                   mismatches);
        CUDA_CHECK(cudaEventRecord(checkStop));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&report.mismatches, mismatches, sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));

        float burnMilliseconds = 0, checkMilliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&burnMilliseconds, burnStart, burnStop));
        CUDA_CHECK(cudaEventElapsedTime(&checkMilliseconds, checkStart, checkStop));
        burnSeconds += burnMilliseconds / 1000.0;
        checkSeconds += checkMilliseconds / 1000.0;
        movedBytes += bytesPerRound;
        readBytes += double(bytes);

        report.elapsed = seconds_since(start);
        report.rate = movedBytes / burnSeconds / 1e9;
        print_progress(report, duration);
    }
    std::printf("\n    verification sweep (pure read): %.2f GB/s\n",
                readBytes / checkSeconds / 1e9);

    for (cudaEvent_t event : {burnStart, burnStop, checkStart, checkStop})
        CUDA_CHECK(cudaEventDestroy(event));
    CUDA_CHECK(cudaFree(buffer));
    CUDA_CHECK(cudaFree(reference));
    CUDA_CHECK(cudaFree(mismatches));
    return report;
}

struct Entry {
    const char *name;
    const char *description;
    Report (*run)(int multiprocessors, double duration);
};

// Ordered narrowest to widest, so the table reads as a precision ladder.
const Entry kSweep[] = {
    {"bit1",  "XNOR + popcount, 32 lanes/word", run_compute<Bit1Workload>},
    {"tern1", "1.58-bit weights, binary acts",  run_compute<Tern1Workload>},
    {"tern8", "1.58-bit weights, int8 acts",    run_compute<Tern8Workload>},
    {"int8",  "4-way byte dot product",         run_compute<Int8Workload>},
    {"fp16",  "packed half2 FMA",               run_compute<Fp16Workload>},
    {"int32", "32-bit multiply-add",            run_compute<Int32Workload>},
    {"fp32",  "single-precision FMA",           run_compute<Fp32Workload>},
    {"fp64",  "double-precision FMA",           run_compute<Fp64Workload>},
    {"sfu",   "transcendental (sin)",           run_compute<SfuWorkload>},
    {"mem",   "HBM2 bandwidth + integrity",     run_memory},
};
constexpr int kSweepCount = sizeof(kSweep) / sizeof(kSweep[0]);

void print_usage(const char *program) {
    std::printf("usage: %s [-t seconds] [-g device] [-p workload]\n"
                "  -t  seconds per workload (default 15)\n"
                "  -g  device ordinal (default 0)\n"
                "  -p  run one workload only\n\n"
                "workloads:\n", program);
    for (const Entry &entry : kSweep)
        std::printf("  %-6s %s\n", entry.name, entry.description);
}

}  // namespace

int main(int argc, char **argv) {
    double duration = 15.0;
    int device = 0;
    const char *selected = nullptr;

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (!std::strcmp(arg, "-t") && i + 1 < argc) duration = std::atof(argv[++i]);
        else if (!std::strcmp(arg, "-g") && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (!std::strcmp(arg, "-p") && i + 1 < argc) selected = argv[++i];
        else { print_usage(argv[0]); return std::strcmp(arg, "-h") == 0 ? 0 : 1; }
    }

    std::signal(SIGINT, request_stop);
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    std::printf("%s  (sm_%d%d, %d SMs, %.1f GB, %.0f MHz)\n\n", properties.name,
                properties.major, properties.minor, properties.multiProcessorCount,
                properties.totalGlobalMem / 1073741824.0, properties.clockRate / 1000.0);

    Report results[kSweepCount];
    bool executed[kSweepCount] = {false};
    bool matched = false;

    for (int i = 0; i < kSweepCount && !g_interrupted; ++i) {
        if (selected && std::strcmp(selected, kSweep[i].name) != 0) continue;
        matched = true;
        std::printf("%s - %s\n", kSweep[i].name, kSweep[i].description);
        results[i] = kSweep[i].run(properties.multiProcessorCount, duration);
        executed[i] = true;
    }

    if (!matched) {
        std::fprintf(stderr, "unknown workload '%s'\n", selected);
        return 1;
    }

    unsigned long long totalMismatches = 0;
    std::printf("\n  workload   throughput            status\n");
    for (int i = 0; i < kSweepCount; ++i) {
        if (!executed[i]) continue;
        totalMismatches += results[i].mismatches;
        std::printf("  %-10s %9.2f %-8s     %s\n", kSweep[i].name, results[i].rate,
                    results[i].unit, results[i].mismatches ? "FAILED" : "ok");
    }

    if (totalMismatches)
        std::printf("\n%llu incorrect values - this card computes wrong answers under load\n",
                    totalMismatches);
    return totalMismatches ? 1 : 0;
}
