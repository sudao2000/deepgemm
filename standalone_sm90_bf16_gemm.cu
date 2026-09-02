// standalone_sm90_bf16_gemm.cu
//
// Self-contained SM90 (Hopper) BF16 GEMM using DeepGEMM-style WGMMA + TMA.
// This is a simplified version of deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh
// that supports only normal GEMM: D[M,N] = A[M,K] @ B[N,K]^T, with K-major A/B
// and N-major D, BF16 output, and no accumulation.
//
// Usage: ./standalone_sm90_bf16_gemm <m> <n> <k>
// Compile: bash compile_standalone.sh
//
// NOTE: this binary must run on a Hopper (SM90) GPU. The launch grid is sized
// for 132 SMs, which matches H100/H800/H200. Change kNumSMs below if your GPU
// has a different SM count.

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cutlass/detail/helper_macros.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/bfloat16.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_gmma_ext.hpp>
#include <cute/arch/mma_sm100_desc.hpp>

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

#ifndef DG_DEVICE_ASSERT
#define DG_DEVICE_ASSERT(cond) \
    do { if (not (cond)) { printf("Assert failed: %s:%d: %s\n", __FILE__, __LINE__, #cond); asm("trap;"); } } while (0)
#endif

// ---------------------------------------------------------------------------
// Minimal deep_gemm helper namespaces
// ---------------------------------------------------------------------------
namespace deep_gemm {

enum class GemmType { Normal = 0 };

namespace math {

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
CUTLASS_HOST_DEVICE T ceil_div(T a, T b) { return (a + b - 1) / b; }

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_align(T a, T b) {
    return constexpr_ceil_div(a, b) * b;
}

CUTLASS_DEVICE void swap(uint32_t& a, uint32_t& b) {
    uint32_t t = a; a = b; b = t;
}

} // namespace math

namespace utils {

template <typename FuncT>
struct PatternVisitor {
    FuncT func;
    CUTLASS_HOST_DEVICE explicit PatternVisitor(FuncT&& f) : func(static_cast<FuncT&&>(f)) {}
    CUTLASS_HOST_DEVICE auto operator[](const uint32_t& i) const { return func(i); }
};

} // namespace utils

} // namespace deep_gemm

namespace deep_gemm {
namespace ptx {

CUTLASS_DEVICE uint32_t get_lane_idx() {
    uint32_t lane_id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
}

CUTLASS_DEVICE void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

CUTLASS_DEVICE void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

CUTLASS_DEVICE void warpgroup_fence_operand(float& reg) {
    asm volatile("" : "+f"(reg) :: "memory");
}

template <int N>
CUTLASS_DEVICE void warpgroup_wait() {
    static_assert(N >= 0 and N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory");
}

template <typename dtype_t>
struct SM90_U32x2_STSM_N {
    CUTLASS_DEVICE static void copy(dtype_t src_0, dtype_t src_1, void* smem_dst) {
        static_assert(sizeof(dtype_t) == sizeof(uint32_t), "Invalid dtype");
        const uint32_t src[2] = { *reinterpret_cast<const uint32_t*>(&src_0),
                                  *reinterpret_cast<const uint32_t*>(&src_1) };
        asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};"
                     :: "l"(__cvta_generic_to_shared(smem_dst)), "r"(src[0]), "r"(src[1]));
    }
};

} // namespace ptx
} // namespace deep_gemm

namespace deep_gemm {
namespace comm {

CUTLASS_DEVICE void cluster_sync_with_relaxed_arrive() {
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
}

} // namespace comm
} // namespace deep_gemm

namespace deep_gemm {
namespace mma {
namespace sm90 {

using namespace cute;

template <int N_, typename MMA>
struct BF16MMA {
    template <size_t ...Idx>
    CUTLASS_DEVICE static void call_fma_impl(uint64_t const& desc_a, uint64_t const& desc_b,
                                               float* d, bool scale_d, index_sequence<Idx...>) {
        using namespace cute::SM90::GMMA;
        MMA::fma(desc_a, desc_b, d[Idx]..., (scale_d ? ScaleOut::One : ScaleOut::Zero));
    }

    CUTLASS_DEVICE static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        call_fma_impl(desc_a, desc_b, d, scale_d, make_index_sequence<N_ / 2>{});
    }

    static constexpr int M = 64;
    static constexpr int N = N_;
    static constexpr int K = 16;
    static constexpr int kNumAccum = M * N / 128;
};

template <cute::UMMA::Major kMajor>
constexpr cute::SM90::GMMA::Major to_sm90_major() {
    return kMajor == cute::UMMA::Major::K ? cute::SM90::GMMA::Major::K : cute::SM90::GMMA::Major::MN;
}

template <int N, cute::UMMA::Major kMajorA = cute::UMMA::Major::K,
                 cute::UMMA::Major kMajorB = cute::UMMA::Major::K>
struct BF16MMASelector {
    static constexpr auto select_mma() {
        using namespace cute::SM90::GMMA;
        constexpr auto kGMMAMajorA = to_sm90_major<kMajorA>();
        constexpr auto kGMMAMajorB = to_sm90_major<kMajorB>();
        if constexpr (N == 128) return MMA_64x128x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        DG_STATIC_ASSERT(N == 128, "Only N=128 WGMMA is supported in this standalone file");
    }
    static constexpr auto select_type() { return BF16MMA<N, decltype(select_mma())>(); }
    using type = decltype(select_type());
};

template <class PointerType>
CUTLASS_DEVICE cute::GmmaDescriptor make_smem_desc(PointerType smem_ptr, const int layout_type,
                                                   const uint32_t leading_byte_offset = 0,
                                                   const uint32_t stride_byte_offset = 1024) {
    cute::GmmaDescriptor desc;
    const auto uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    desc.bitfield.start_address_ = uint_ptr >> 4;
    desc.bitfield.layout_type_ = layout_type;
    desc.bitfield.leading_byte_offset_ = leading_byte_offset >> 4;
    desc.bitfield.stride_byte_offset_ = stride_byte_offset >> 4;
    desc.bitfield.base_offset_ = 0;
    return desc;
}

template <cute::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t kSwizzleMode, typename dtype_t>
CUTLASS_DEVICE constexpr uint32_t get_gmma_desc_stride_k() {
    constexpr uint32_t atom = (kSwizzleMode == 0) ? BLOCK_MN : (kSwizzleMode / sizeof(dtype_t));
    return kMajorMode == cute::UMMA::Major::K ? 1 : atom;
}

template <cute::UMMA::Major kMajorMode, uint32_t kSwizzleMode, typename dtype_t>
CUTLASS_HOST_DEVICE constexpr static cute::SM90::GMMA::LayoutType to_gmma_layout_type() {
    DG_STATIC_ASSERT(kSwizzleMode == 0 or kSwizzleMode == 16 or
                     kSwizzleMode == 32 or kSwizzleMode == 64 or
                     kSwizzleMode == 128, "Invalid swizzling mode");
    if constexpr (kSwizzleMode == 0)   return cute::SM90::GMMA::LayoutType::INTERLEAVE;
    if constexpr (kSwizzleMode == 16)  return cute::SM90::GMMA::LayoutType::INTERLEAVE;
    if constexpr (kSwizzleMode == 32)  return cute::SM90::GMMA::LayoutType::B32;
    if constexpr (kSwizzleMode == 64)  return cute::SM90::GMMA::LayoutType::B64;
    if constexpr (kSwizzleMode == 128) return cute::SM90::GMMA::LayoutType::B128;
}

template <cute::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode, typename dtype_t>
CUTLASS_DEVICE uint32_t advance_gmma_desc_lo(const uint32_t& base, const uint32_t& mn_idx,
                                             const uint32_t& k_idx, const uint32_t& offset = 0) {
    const uint32_t stride_k = get_gmma_desc_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, dtype_t>();
    return base + (((offset + mn_idx * BLOCK_K + k_idx * stride_k) * static_cast<uint32_t>(sizeof(dtype_t))) >> 4u);
}

template <cute::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode, typename dtype_t>
CUTLASS_DEVICE cute::GmmaDescriptor make_gmma_desc(dtype_t* base_smem_ptr, uint32_t mn_idx, uint32_t k_idx) {
    const uint32_t stride_k = get_gmma_desc_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, dtype_t>();
    const auto layout_type = to_gmma_layout_type<kMajorMode, kSwizzleMode, dtype_t>();
    constexpr uint32_t num_non_contiguous = 128 / 16;
    if constexpr (kMajorMode == cute::UMMA::Major::K) {
        DG_STATIC_ASSERT(kSwizzleMode == BLOCK_K * sizeof(dtype_t), "Unexpected value");
        const uint32_t stride_byte_offset = num_non_contiguous * BLOCK_K * sizeof(dtype_t);
        const uint32_t leading_byte_offset = 0;
        return make_smem_desc(base_smem_ptr + mn_idx * BLOCK_K + k_idx * stride_k,
                              static_cast<uint32_t>(layout_type), leading_byte_offset, stride_byte_offset);
    } else {
        constexpr uint32_t BLOCK_MN_ATOM = (kSwizzleMode == 0) ? BLOCK_MN : (kSwizzleMode / sizeof(dtype_t));
        DG_DEVICE_ASSERT(mn_idx % BLOCK_MN_ATOM == 0);
        DG_STATIC_ASSERT(kSwizzleMode > 0, "Invalid swizzling");
        uint32_t stride_byte_offset = num_non_contiguous * BLOCK_MN_ATOM * sizeof(dtype_t);
        uint32_t leading_byte_offset = BLOCK_K * BLOCK_MN_ATOM * sizeof(dtype_t);
        if constexpr (kSwizzleMode == 16)
            math::swap(stride_byte_offset, leading_byte_offset);
        return make_smem_desc(base_smem_ptr + mn_idx * BLOCK_K + k_idx * stride_k,
                              static_cast<uint32_t>(layout_type), leading_byte_offset, stride_byte_offset);
    }
}

} // namespace sm90
} // namespace mma
} // namespace deep_gemm

namespace deep_gemm {
namespace tma {

template <uint32_t BLOCK_INNER, uint32_t kSwizzleMode, typename dtype_t>
CUTLASS_HOST_DEVICE constexpr uint32_t get_inner_block_atom_size() {
    return kSwizzleMode == 0 ? BLOCK_INNER : kSwizzleMode / sizeof(dtype_t);
}

template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER, uint32_t kSwizzleMode, typename dtype_t>
CUTLASS_DEVICE void copy(void const* desc_ptr, cutlass::arch::ClusterTransactionBarrier* barrier_ptr,
                         dtype_t* smem_ptr, const uint32_t& inner_idx, const uint32_t& outer_idx) {
    constexpr uint32_t BLOCK_INNER_ATOM = get_inner_block_atom_size<BLOCK_INNER, kSwizzleMode, dtype_t>();
    #pragma unroll
    for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
        cute::SM90_TMA_LOAD_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                       static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                       smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                       inner_idx + i * BLOCK_INNER_ATOM, outer_idx);
    }
}

} // namespace tma
} // namespace deep_gemm

namespace deep_gemm {
namespace sched {

// Minimal scheduler supporting only GemmType::Normal, no multicast, fixed kNumSMs.
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumSMs>
struct Scheduler {
    int current_iter = -1;
    uint32_t num_m_blocks = 0;
    uint32_t num_n_blocks = 0;
    uint32_t num_blocks = 0;
    uint32_t current_shape_k = 0;

    CUTLASS_DEVICE Scheduler(const uint32_t& shape_m, const uint32_t& shape_n,
                             const uint32_t& shape_k, int* /*grouped_layout*/ = nullptr) {
        num_m_blocks = math::ceil_div(shape_m, BLOCK_M);
        num_n_blocks = math::ceil_div(shape_n, BLOCK_N);
        num_blocks = num_m_blocks * num_n_blocks;
        current_shape_k = shape_k;
    }

    CUTLASS_DEVICE bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const auto next_block_idx = static_cast<uint32_t>((++current_iter) * static_cast<int>(kNumSMs) + blockIdx.x);
        if (next_block_idx >= num_blocks)
            return false;
        m_block_idx = next_block_idx % num_m_blocks;
        n_block_idx = next_block_idx / num_m_blocks;
        return true;
    }
};

} // namespace sched
} // namespace deep_gemm

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
namespace deep_gemm {

template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K_,
          uint32_t kNumStages_,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleDMode,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumSMs>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
void sm90_bf16_gemm_impl(int* grouped_layout,
                         uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr uint32_t kNumStagesPerMerge = 1;
    constexpr uint32_t BLOCK_K = BLOCK_K_;
    constexpr uint32_t kNumStages = kNumStages_;
    using WGMMA = typename mma::sm90::BF16MMASelector<BLOCK_N, kMajorA, kMajorB>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }
    __syncwarp();

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    constexpr uint32_t SMEM_D_SIZE = math::constexpr_align(
        BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(cutlass::bfloat16_t)), 1024u);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_bfloat16);

    auto smem_d = reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer);
    auto smem_a = [&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    };
    auto smem_b = [&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    };

    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_D_SIZE +
                                                         kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers = [&](const uint32_t& i) { return barrier_start_ptr + i; };
    auto empty_barriers = [&](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; };

    // Initialize barriers
    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers(i)->init(1);
            empty_barriers(i)->init(kNumMathThreads / 32);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    constexpr uint32_t kNumTMARegisters = 48;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 224;

    cudaGridDependencySynchronize();

    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<BLOCK_M, BLOCK_N, kNumSMs>(shape_m, shape_n, shape_k, grouped_layout);

    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = (stage_idx == kNumStages - 1) ? 0 : stage_idx + 1;
        phase ^= (stage_idx == 0);
    };

    if (warp_idx >= kNumMathThreads / 32) {
        // TMA warp-group
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        if (warp_idx == kNumMathThreads / 32 + 2 and cute::elect_one_sync()) {
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    empty_barriers(stage_idx)->wait(phase ^ 1);
                    auto& full_barrier = *full_barriers(stage_idx);

                    const auto m_idx = m_block_idx * BLOCK_M;
                    const auto n_idx = n_block_idx * BLOCK_N;
                    const auto k_a_idx = k_block_idx * BLOCK_K;
                    const auto k_b_idx = k_block_idx * BLOCK_K;

                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t>(
                        &tensor_map_a, &full_barrier, smem_a(stage_idx), k_a_idx, m_idx);
                    tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t>(
                        &tensor_map_b, &full_barrier, smem_b(stage_idx), k_b_idx, n_idx);
                    full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
                }
            }
        }
    } else {
        // Math warp-groups
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);

        constexpr uint32_t BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge;
        auto a_desc = mma::sm90::make_gmma_desc<kMajorA, BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode, cutlass::bfloat16_t>(
            smem_a(0), static_cast<uint32_t>(math_wg_idx * WGMMA::M), 0u);
        auto b_desc = mma::sm90::make_gmma_desc<kMajorB, BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode, cutlass::bfloat16_t>(
            smem_b(0), 0u, 0u);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            constexpr uint32_t WAVE_BLOCK_M = (BLOCK_M <= WGMMA::M) ? BLOCK_M : WGMMA::M * 2;
            float accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};

            constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA::M);

            auto empty_barrier_arrive = [&](uint32_t s) {
                if (lane_idx == 0)
                    empty_barriers(s)->arrive();
            };

            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                const auto b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);

                full_barriers(stage_idx)->wait(phase);

                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M); ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_arrive();

                #pragma unroll
                for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                    auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                        const uint32_t atom_k_idx = k * WGMMA::K / BLOCK_ATOM_K;
                        a_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<kMajorA, BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode, cutlass::bfloat16_t>(
                            a_desc_base_lo, local_idx * WAVE_BLOCK_M, (k * WGMMA::K) % BLOCK_ATOM_K,
                            atom_k_idx * BLOCK_M * BLOCK_ATOM_K);
                        b_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<kMajorB, BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode, cutlass::bfloat16_t>(
                            b_desc_base_lo, 0, (k * WGMMA::K) % BLOCK_ATOM_K,
                            atom_k_idx * BLOCK_N * BLOCK_ATOM_K);
                        WGMMA::wgmma(a_desc, b_desc, shifted_accum, 1);
                    }
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M); ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_wait<0>();

                empty_barrier_arrive(stage_idx);
            }

            // Epilogue: store accumulator to shared memory and TMA-store to global D
            constexpr uint32_t kNumElemBytes = sizeof(cutlass::bfloat16_t);
            constexpr uint32_t TMA_D_BLOCK_N = (kSwizzleDMode == 0) ? BLOCK_N : (kSwizzleDMode / kNumElemBytes);
            constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;

            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 0);

            #pragma unroll
            for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                auto m_offset = local_idx * WAVE_BLOCK_M;
                auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
                #pragma unroll
                for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    uint8_t* smem_ptr = nullptr;
                    constexpr uint32_t kNumBankGroupBytes = 16;
                    auto atom_offset = i / (TMA_D_BLOCK_N / 8);
                    auto in_atom_offset = i % (TMA_D_BLOCK_N / 8);
                    auto bank_group_index = in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);
                    constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;
                    auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
                    auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
                    col ^= row % (kSwizzleDMode / 16);
                    smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +
                               warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +
                               m_offset * kSwizzleDMode +
                               atom_offset * BLOCK_M * kSwizzleDMode +
                               row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
                    ptx::SM90_U32x2_STSM_N<__nv_bfloat162>::copy(
                        __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}),
                        __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                        smem_ptr);
                }
            }

            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 0);

            const auto m_idx = m_block_idx * BLOCK_M;
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_cd, smem_ptr,
                                              n_block_idx * BLOCK_N + in_block_n_offset, m_idx);
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0) {
        printf("This kernel requires sm_90a\n");
        asm("trap;");
    }
#endif
}

} // namespace deep_gemm

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
using bf16_t = cutlass::bfloat16_t;

static CUtensorMap make_tma_desc_2d(void* gmem_ptr, uint32_t dim0, uint32_t dim1,
                                      uint32_t box0, uint32_t box1,
                                      uint64_t outer_stride_bytes,
                                      CUtensorMapSwizzle swizzle) {
    CUtensorMap tensor_map;
    cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(dim0), static_cast<cuuint64_t>(dim1)};
    cuuint32_t smem_dims[2] = {static_cast<cuuint32_t>(box0), static_cast<cuuint32_t>(box1)};
    cuuint64_t gmem_strides[1] = {outer_stride_bytes};
    cuuint32_t elem_strides[2] = {1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gmem_ptr, gmem_dims,
        gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "cuTensorMapEncodeTiled failed: %d\n", static_cast<int>(result));
        exit(1);
    }
    return tensor_map;
}

static inline float bf16_to_float(bf16_t x) {
    return static_cast<float>(x);
}

static inline bf16_t float_to_bf16(float x) {
    return bf16_t(x);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <m> <n> <k>\n", argv[0]);
        return 1;
    }

    int m_in = std::atoi(argv[1]);
    int n_in = std::atoi(argv[2]);
    int k_in = std::atoi(argv[3]);
    if (m_in <= 0 || n_in <= 0 || k_in <= 0) {
        fprintf(stderr, "m, n, k must be positive\n");
        return 1;
    }

    // Device check
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device: %s, SM count: %d, compute capability: %d.%d\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    if (prop.major != 9) {
        fprintf(stderr, "Warning: this kernel targets SM90 (Hopper). Current GPU is SM%d.%d.\n",
                prop.major, prop.minor);
    }

    // Tile configuration (BF16 WGMMA descriptors support up to 128B swizzle,
    // so for K-major BF16 we use BLOCK_K = 64 bytes / 2 = 64 elements.)
    constexpr uint32_t BLOCK_M = 128;
    constexpr uint32_t BLOCK_N = 128;
    constexpr uint32_t BLOCK_K = 64;
    constexpr uint32_t kNumStages = 6;
    constexpr uint32_t kNumTMAThreads = 128;
    constexpr uint32_t kNumMathThreads = 256;
    constexpr uint32_t kNumSMs = 132;  // H100/H800/H200 have 132 SMs
    constexpr uint32_t kSwizzleAMode = 128;
    constexpr uint32_t kSwizzleBMode = 128;
    constexpr uint32_t kSwizzleDMode = 128;

    // The persistent scheduler assumes grid size == kNumSMs.
    if (static_cast<uint32_t>(prop.multiProcessorCount) != kNumSMs) {
        fprintf(stderr,
                "Warning: GPU has %d SMs but kNumSMs is fixed at %u. "
                "Edit the constexpr kNumSMs to match your GPU for best results.\n",
                prop.multiProcessorCount, kNumSMs);
    }

    // Round dimensions up to tile multiples so TMA boxes always stay in bounds
    auto round_up = [](int x, int tile) { return (x + tile - 1) / tile * tile; };
    int M = round_up(m_in, BLOCK_M);
    int N = round_up(n_in, BLOCK_N);
    int K = round_up(k_in, BLOCK_K);

    printf("Running standalone BF16 GEMM: m=%d n=%d k=%d (rounded to M=%d N=%d K=%d)\n",
           m_in, n_in, k_in, M, N, K);

    // Allocate host / device memory
    size_t a_size = static_cast<size_t>(M) * K;
    size_t b_size = static_cast<size_t>(K) * N;
    size_t d_size = static_cast<size_t>(M) * N;

    std::vector<bf16_t> h_A(a_size), h_B(b_size), h_D(d_size), h_ref(d_size);

    // Initialize with simple reproducible patterns; zero-pad the rounded-up regions
    std::fill(h_A.begin(), h_A.end(), float_to_bf16(0.0f));
    std::fill(h_B.begin(), h_B.end(), float_to_bf16(0.0f));
    for (int i = 0; i < m_in; ++i)
        for (int j = 0; j < k_in; ++j)
            h_A[i * K + j] = float_to_bf16(0.01f * static_cast<float>(((i * K + j) % 13) - 6));
    for (int i = 0; i < k_in; ++i)
        for (int j = 0; j < n_in; ++j)
            h_B[i * N + j] = float_to_bf16(0.01f * static_cast<float>(((i * N + j) % 11) - 5));

    bf16_t *d_A, *d_B, *d_D;
    cudaMalloc(&d_A, a_size * sizeof(bf16_t));
    cudaMalloc(&d_B, b_size * sizeof(bf16_t));
    cudaMalloc(&d_D, d_size * sizeof(bf16_t));

    cudaMemcpy(d_A, h_A.data(), a_size * sizeof(bf16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), b_size * sizeof(bf16_t), cudaMemcpyHostToDevice);
    cudaMemset(d_D, 0, d_size * sizeof(bf16_t));

    // CPU reference (compute in float, then round to BF16)
    for (int i = 0; i < m_in; ++i) {
        for (int j = 0; j < n_in; ++j) {
            float acc = 0.0f;
            for (int kk = 0; kk < k_in; ++kk)
                acc += bf16_to_float(h_A[i * K + kk]) * bf16_to_float(h_B[kk * N + j]);
            h_ref[i * N + j] = float_to_bf16(acc);
        }
    }

    // Build TMA descriptors. Layouts:
    //   A: K-major => inner=K, outer=M, outer stride = K elements
    //   B: K-major => inner=K, outer=N, outer stride = K elements
    //   D: N-major => inner=N, outer=M, outer stride = N elements
    // For 128B swizzle, the TMA box inner dimension is 128B / elem_size = 64 elements.
    constexpr uint32_t TMA_INNER_A = kSwizzleAMode / sizeof(bf16_t);
    constexpr uint32_t TMA_INNER_B = kSwizzleBMode / sizeof(bf16_t);
    constexpr uint32_t TMA_INNER_D = kSwizzleDMode / sizeof(bf16_t);

    CUtensorMap tma_a = make_tma_desc_2d(d_A, K, M, TMA_INNER_A, BLOCK_M,
                                          K * sizeof(bf16_t),
                                          CU_TENSOR_MAP_SWIZZLE_128B);
    CUtensorMap tma_b = make_tma_desc_2d(d_B, K, N, TMA_INNER_B, BLOCK_N,
                                          K * sizeof(bf16_t),
                                          CU_TENSOR_MAP_SWIZZLE_128B);
    CUtensorMap tma_d = make_tma_desc_2d(d_D, N, M, TMA_INNER_D, BLOCK_M,
                                          N * sizeof(bf16_t),
                                          CU_TENSOR_MAP_SWIZZLE_128B);

    // Compute shared memory requirements
    constexpr uint32_t SMEM_D_SIZE = deep_gemm::math::constexpr_align(
        static_cast<uint32_t>(BLOCK_M * BLOCK_N * sizeof(bf16_t)), 1024u);
    constexpr uint32_t SMEM_A_SIZE = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
    constexpr uint32_t SMEM_B_SIZE = BLOCK_N * BLOCK_K * sizeof(__nv_bfloat16);
    constexpr size_t BARRIER_SIZE = sizeof(cutlass::arch::ClusterTransactionBarrier);
    size_t smem_total = SMEM_D_SIZE +
                        kNumStages * (SMEM_A_SIZE + SMEM_B_SIZE) +
                        2 * kNumStages * BARRIER_SIZE;
    smem_total = ((smem_total + 1023) / 1024) * 1024;

    dim3 block(kNumTMAThreads + kNumMathThreads);
    dim3 grid(kNumSMs);

    // Launch kernel
    auto kernel = &deep_gemm::sm90_bf16_gemm_impl<
        cute::UMMA::Major::K, cute::UMMA::Major::K,
        BLOCK_M, BLOCK_N, BLOCK_K,
        kNumStages,
        kSwizzleAMode, kSwizzleBMode, kSwizzleDMode,
        kNumTMAThreads, kNumMathThreads, kNumSMs>;

    cudaFuncSetAttribute((const void*)kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_total));

    kernel<<<grid, block, smem_total>>>(nullptr, m_in, n_in, k_in, tma_a, tma_b, tma_d);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_D.data(), d_D, d_size * sizeof(bf16_t), cudaMemcpyDeviceToHost);

    // Verify top-left m_in x n_in region
    double max_rel_err = 0.0;
    double max_abs_err = 0.0;
    int fail_count = 0;
    for (int i = 0; i < m_in; ++i) {
        for (int j = 0; j < n_in; ++j) {
            float got = bf16_to_float(h_D[i * N + j]);
            float ref = bf16_to_float(h_ref[i * N + j]);
            float abs_err = std::fabs(got - ref);
            float rel_err = (std::fabs(ref) > 1e-6f) ? (abs_err / std::fabs(ref)) : abs_err;
            max_rel_err = std::max(max_rel_err, static_cast<double>(rel_err));
            max_abs_err = std::max(max_abs_err, static_cast<double>(abs_err));
            if (rel_err > 1e-2f || abs_err > 1e-2f) {
                if (fail_count < 10)
                    printf("MISMATCH at (%d,%d): got=%f ref=%f rel=%e abs=%e\n",
                           i, j, got, ref, rel_err, abs_err);
                ++fail_count;
            }
        }
    }

    if (fail_count == 0) {
        printf("PASSED  max_rel_err=%e  max_abs_err=%e\n", max_rel_err, max_abs_err);
    } else {
        printf("FAILED  mismatches=%d  max_rel_err=%e  max_abs_err=%e\n",
               fail_count, max_rel_err, max_abs_err);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);
    return (fail_count == 0) ? 0 : 1;
}
