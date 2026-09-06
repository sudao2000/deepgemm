// standalone_sm90_fp8_gemm_1d1d.cu
//
// Self-contained SM90 (Hopper) FP8 (e4m3) GEMM with 1D1D per-token-per-128-channel
// scaling factors, based on deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh.
//
// Supported case (only): D[M,N](fp32) = A[M,K](fp8, K-major) @ B[N,K]^T(fp8, K-major)
// with per-(row, 128-K-block) scaling factors SFA[ceil(K/128), M], SFB[ceil(K/128), N].
// D is accumulated with TMA REDUCE_ADD, so it must be zero-initialized.
//
// Usage: ./standalone_sm90_fp8_gemm_1d1d <m> <n> <k>
// Compile: bash compile_standalone_fp8.sh
//
// NOTE: must run on a Hopper (SM90) GPU. The launch grid is sized for 132 SMs
// (H100/H800/H200); change kNumSMs below if your GPU differs.

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include <cutlass/detail/helper_macros.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/int_tuple.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm100_tma.hpp>
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
// cute_tie (from deep_gemm/common/cute_tie.cuh)
// ---------------------------------------------------------------------------
namespace cute {

struct ignore_t {
    template <typename T>
    constexpr const ignore_t& operator=(T&&) const noexcept {
        return *this;
    }
};

inline constexpr ignore_t ignore{};

} // namespace cute

#define CUTE_TIE_CONCAT_IMPL(A, B) A##B
#define CUTE_TIE_CONCAT(A, B) CUTE_TIE_CONCAT_IMPL(A, B)

#define CUTE_TIE_GET_NTH_ARG(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#define CUTE_TIE_COUNT_ARGS(...) \
    CUTE_TIE_GET_NTH_ARG(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)

#define CUTE_TIE_OP_DECL(I, TUPLE, VAR) auto VAR = ::cute::get<I>(TUPLE)

#define CUTE_TIE_APPLY_OP_1(OP, T, V1) OP(0, T, V1);
#define CUTE_TIE_APPLY_OP_2(OP, T, V1, V2) OP(0, T, V1); OP(1, T, V2);
#define CUTE_TIE_APPLY_OP_3(OP, T, V1, V2, V3) OP(0, T, V1); OP(1, T, V2); OP(2, T, V3);
#define CUTE_TIE_APPLY_OP_4(OP, T, V1, V2, V3, V4) OP(0, T, V1); OP(1, T, V2); OP(2, T, V3); OP(3, T, V4);

#define CUTE_TIE_DECL(TUPLE_EXPR, ...) \
    auto&& CUTE_TIE_CONCAT(cute_tie__temp_tuple_, __LINE__) = (TUPLE_EXPR); \
    CUTE_TIE_CONCAT(CUTE_TIE_APPLY_OP_, CUTE_TIE_COUNT_ARGS(__VA_ARGS__)) ( \
        CUTE_TIE_OP_DECL, \
        CUTE_TIE_CONCAT(cute_tie__temp_tuple_, __LINE__), \
        __VA_ARGS__ \
    )

// ---------------------------------------------------------------------------
// Minimal deep_gemm helper namespaces
// ---------------------------------------------------------------------------
namespace deep_gemm {

enum class GemmType {
    Normal                              = 0,
    MGroupedContiguous                  = 1,
    MGroupedMasked                      = 2,
    KGroupedContiguous                  = 3,
    Batched                             = 4,
    MGroupedContiguousWithPsumLayout    = 5,
    KGroupedContiguousWithPsumLayout    = 6,
};

constexpr CUTLASS_HOST_DEVICE bool is_k_grouped_contiguous(const GemmType& gemm_type) {
    switch (gemm_type) {
        case GemmType::KGroupedContiguous:                  return true;
        case GemmType::KGroupedContiguousWithPsumLayout:    return true;
        default: return false;
    }
}

namespace math {

template <typename T>
CUTLASS_HOST_DEVICE T ceil_div(T a, T b) { return (a + b - 1) / b; }

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_align(T a, T b) {
    return constexpr_ceil_div(a, b) * b;
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

/// Shared memory loads
CUTLASS_DEVICE float ld_shared(const float* ptr) {
    float ret;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

CUTLASS_DEVICE float2 ld_shared(const float2* ptr) {
    float2 ret;
    asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(ret.x), "=f"(ret.y) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

/// Shared memory stores
CUTLASS_DEVICE void st_shared(const float2* ptr, float2 val) {
    asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "l"(__cvta_generic_to_shared(ptr)), "f"(val.x), "f"(val.y));
}

/// Tensor-map instructions (only used by the K-grouped path, kept for fidelity)
CUTLASS_DEVICE void tensor_map_release_gpu() {
    asm volatile ("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
}

CUTLASS_DEVICE void tensor_map_acquire_gpu(const cute::TmaDescriptor* gmem_desc_ptr) {
    auto gmem_int_desc = reinterpret_cast<uint64_t>(gmem_desc_ptr);
    asm volatile ("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"(gmem_int_desc) : "memory");
}

CUTLASS_DEVICE void tensor_map_replace_global_addr_in_smem(cute::TmaDescriptor* smem_desc, const void* new_addr) {
    auto smem_int_desc = static_cast<uint32_t>(__cvta_generic_to_shared(smem_desc));
    const auto new_int64_addr = reinterpret_cast<uint64_t>(new_addr);
    asm volatile ("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;" :: "r"(smem_int_desc), "l"(new_int64_addr));
}

CUTLASS_DEVICE void tensor_map_replace_global_inner_dim_stride_in_smem(cute::TmaDescriptor* smem_desc,
                                                                       const uint32_t& new_dim,
                                                                       const uint64_t& new_stride) {
    auto smem_int_desc = __cvta_generic_to_shared(smem_desc);
    asm volatile ("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0, %1;" :: "l"(smem_int_desc), "r"(new_dim));
#if ((__CUDACC_VER_MAJOR__ > 12) or ((__CUDACC_VER_MAJOR__ == 12) and (__CUDACC_VER_MINOR__ >= 3)))
    asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0, %1;" :: "l"(smem_int_desc), "l"(new_stride));
#else
    DG_STATIC_ASSERT(false, "Invalid CUDA version");
#endif
}

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

/// FP8 WGMMA (from deep_gemm/mma/sm90.cuh)
template <int N_, typename MMA>
struct FP8MMA {
    template <size_t ...Idx>
    CUTLASS_DEVICE static void call_fma_impl(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d, cute::index_sequence<Idx...>) {
        using namespace cute::SM90::GMMA;
        MMA::fma(desc_a, desc_b, d[Idx]..., (scale_d ? ScaleOut::One : ScaleOut::Zero));
    }

    CUTLASS_DEVICE static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        call_fma_impl(desc_a, desc_b, d, scale_d, cute::make_index_sequence<N_ / 2>{});
    }

    static constexpr int M = 64;
    static constexpr int N = N_;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

template <int N>
struct FP8MMASelector {
    static constexpr auto select_mma() {
        using namespace cute::SM90::GMMA;
        if constexpr (N == 8) return MMA_64x8x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 16) return MMA_64x16x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 24) return MMA_64x24x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 32) return MMA_64x32x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 40) return MMA_64x40x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 48) return MMA_64x48x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 56) return MMA_64x56x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 64) return MMA_64x64x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 72) return MMA_64x72x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 80) return MMA_64x80x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 88) return MMA_64x88x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 96) return MMA_64x96x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 104) return MMA_64x104x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 112) return MMA_64x112x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 120) return MMA_64x120x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 128) return MMA_64x128x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 136) return MMA_64x136x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 144) return MMA_64x144x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 152) return MMA_64x152x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 160) return MMA_64x160x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 168) return MMA_64x168x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 176) return MMA_64x176x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 184) return MMA_64x184x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 192) return MMA_64x192x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 200) return MMA_64x200x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 208) return MMA_64x208x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 216) return MMA_64x216x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 224) return MMA_64x224x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 232) return MMA_64x232x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 240) return MMA_64x240x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 248) return MMA_64x248x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 256) return MMA_64x256x32_F32E4M3E4M3_SS_TN();
    }

    static constexpr auto select_type() {
        return FP8MMA<N, decltype(select_mma())>();
    }

    using type = decltype(select_type());
};

/// Shared memory descriptor (from deep_gemm/mma/sm90.cuh)
template <class PointerType>
CUTLASS_DEVICE cute::GmmaDescriptor
make_smem_desc(PointerType smem_ptr, const int& layout_type,
               const uint32_t& leading_byte_offset = 0,
               const uint32_t& stride_byte_offset = 1024) {
    cute::GmmaDescriptor desc;
    const auto uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    desc.bitfield.start_address_ = uint_ptr >> 4;
    desc.bitfield.layout_type_ = layout_type;
    desc.bitfield.leading_byte_offset_ = leading_byte_offset >> 4;
    desc.bitfield.stride_byte_offset_ = stride_byte_offset >> 4;
    desc.bitfield.base_offset_ = 0;
    return desc;
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

template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER,
          uint32_t kSwizzleMode,
          typename dtype_t, bool kIs3DTMA = false>
CUTLASS_DEVICE void
copy(void const* desc_ptr, cutlass::arch::ClusterTransactionBarrier* barrier_ptr,
     dtype_t* smem_ptr, const uint32_t& inner_idx, const uint32_t& outer_idx,
     const uint32_t& num_tma_multicast = 1, const uint32_t& batch_idx = 0) {
    DG_STATIC_ASSERT(static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL) ==
                     static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL), "Invalid cache hint");
    constexpr uint32_t BLOCK_INNER_ATOM = get_inner_block_atom_size<BLOCK_INNER, kSwizzleMode, dtype_t>();

    if constexpr (not kIs3DTMA) {
        if (num_tma_multicast == 1) {
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                cute::SM90_TMA_LOAD_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                             static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                             smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                             inner_idx + i * BLOCK_INNER_ATOM, outer_idx);
            }
        } else {
            #if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))
                if (cute::block_rank_in_cluster() == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                        cute::SM90_TMA_LOAD_MULTICAST_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                                               (1 << num_tma_multicast) - 1, static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                                               smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                                               inner_idx + i * BLOCK_INNER_ATOM, outer_idx);
                    }
                }
            #endif
        }
    } else {
        if (num_tma_multicast == 1) {
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                cute::SM90_TMA_LOAD_3D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                            static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                            smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                            inner_idx + i * BLOCK_INNER_ATOM, outer_idx, batch_idx);
            }
        } else {
            #if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))
                if (cute::block_rank_in_cluster() == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                        cute::SM90_TMA_LOAD_MULTICAST_3D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                                               (1 << num_tma_multicast) - 1, static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                                               smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                                               inner_idx + i * BLOCK_INNER_ATOM, outer_idx, batch_idx);
                    }
                }
            #endif
        }
    }
}

} // namespace tma
} // namespace deep_gemm

namespace deep_gemm {
namespace sched {

// Minimal scheduler supporting only GemmType::Normal, no multicast, fixed kNumSMs.
template <GemmType kGemmType,
          uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumGroups,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          bool kEnsureZeroPadding = true,
          uint32_t kKAlignment = 128u,
          uint32_t kSFKSpan = 512u>
struct Scheduler {
    int current_iter = -1;

    uint32_t num_blocks;
    uint32_t num_m_blocks;
    uint32_t num_n_blocks;

    // Grouped-GEMM states (unused for Normal, but referenced by runtime branches)
    uint32_t current_group_idx = 0;
    uint32_t current_sf_k_cumsum = 0;
    uint32_t current_k_cumsum = 0;
    uint32_t current_shape_k;
    bool is_peer_cta_alive = true;

    CUTLASS_DEVICE explicit Scheduler(const uint32_t& shape_m, const uint32_t& shape_n,
                                      const uint32_t& shape_k, int* /*grouped_layout*/ = nullptr) {
        static_assert(kGemmType == GemmType::Normal, "This standalone scheduler only supports GemmType::Normal");
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

    CUTLASS_DEVICE bool is_tma_multicast_valid(const uint32_t& /*m_block_idx*/) const {
        return true;
    }
};

} // namespace sched
} // namespace deep_gemm

// ---------------------------------------------------------------------------
// Kernel (adapted from deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh)
// ---------------------------------------------------------------------------
namespace deep_gemm {

template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType, typename cd_dtype_t>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_fp8_gemm_1d1d_impl(__nv_fp8_e4m3* gmem_a_ptr, __nv_fp8_e4m3* gmem_b_ptr,
                        int* grouped_layout,
                        cute::TmaDescriptor* tensor_map_buffer,
                        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_a_base,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_b_base,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // Scaling checks
    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0, "Invalid Threads");
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous, "Invalid GEMM type");

    // C/D type: only FP32 with accumulation
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");

    // Types
    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0, "Invalid block size");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Shared memory
    static constexpr uint32_t SMEM_TENSOR_MAP_SIZE = (kGemmType == GemmType::KGroupedContiguous ? sizeof(cute::TmaDescriptor) * 2 : 0);
    static constexpr uint32_t SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(float);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    static constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = BLOCK_N * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_SFB_SIZE_PER_STAGE = math::constexpr_align(SMEM_SFB_SIZE_PER_STAGE, 128u);
    DG_STATIC_ASSERT(SMEM_SFA_SIZE_PER_STAGE % 128 == 0, "Invalid TMA alignment");

    // Configs
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = threadIdx.x % 32;

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a_base);
        cute::prefetch_tma_descriptor(&tensor_map_b_base);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Tensor maps on shared and global memory
    auto smem_tensor_map_a = reinterpret_cast<cute::TmaDescriptor*>(smem_buffer);
    auto smem_tensor_map_b = smem_tensor_map_a + 1;
    auto gmem_tensor_map_a = tensor_map_buffer + blockIdx.x * 2;
    auto gmem_tensor_map_b = gmem_tensor_map_a + 1;

    // Data on shared memory
    auto smem_d = reinterpret_cast<float*>(smem_buffer + SMEM_TENSOR_MAP_SIZE);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE));
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE));
    });
    constexpr auto SMEM_SF_OFFSET = SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + (SMEM_SF_OFFSET + i * SMEM_SFA_SIZE_PER_STAGE));
    });
    auto smem_sfb = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + (SMEM_SF_OFFSET + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * ALIGNED_SMEM_SFB_SIZE_PER_STAGE));
    });

    // Barriers on shared memory
    constexpr auto SMEM_BARRIER_OFFSET = SMEM_SF_OFFSET + kNumStages * (SMEM_SFA_SIZE_PER_STAGE + ALIGNED_SMEM_SFB_SIZE_PER_STAGE);
    auto full_barriers = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<Barrier*>(smem_buffer + (SMEM_BARRIER_OFFSET + i * static_cast<uint32_t>(sizeof(Barrier))));
    });
    auto empty_barriers = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<Barrier*>(smem_buffer + (SMEM_BARRIER_OFFSET + (kNumStages + i) * static_cast<uint32_t>(sizeof(Barrier))));
    });

    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        // Load tensormap A/B to shared memory
        if constexpr (kGemmType == GemmType::KGroupedContiguous) {
            *smem_tensor_map_a = tensor_map_a_base;
            *smem_tensor_map_b = tensor_map_b_base;
        }

        // Initialize barriers
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

    // Pipeline unroll control
    constexpr uint32_t kNumPipelineUnrolls = (kGemmType == GemmType::KGroupedContiguous ? 0 : kNumStages);

    // Register reconfigurations (more math registers are needed with unrolling)
    constexpr uint32_t kNumTMARegisters = (kNumPipelineUnrolls == 0 ? 40 : 24);
    constexpr uint32_t kNumMathRegisters = (kNumPipelineUnrolls == 0 ? 232 : 240);

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs, true, 128u, 128u>(shape_m, shape_n, shape_k, grouped_layout);

    // TMA and MMA pipeline
    const auto get_pipeline = [=](const uint32_t& iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {iter_idx % kNumStages, (iter_idx / kNumStages) & 1}; // Pipeline stage and phase
    };
    uint32_t iter_idx = 0;

    if (warp_idx >= kNumMathThreads / 32) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
            uint32_t last_group_idx = kNumGroups;

            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                // Assign TMA multicast number into A and B
                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");

                const uint32_t num_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
                const uint32_t m_idx = m_block_idx * BLOCK_M;
                const uint32_t n_idx = n_block_idx * BLOCK_N;

                if (kGemmType == GemmType::KGroupedContiguous and last_group_idx != scheduler.current_group_idx) {
                    last_group_idx = scheduler.current_group_idx;

                    // Directly update current tensor map
                    const uint64_t current_k_offset = scheduler.current_k_cumsum;
                    ptx::tensor_map_replace_global_addr_in_smem(smem_tensor_map_a, gmem_a_ptr + current_k_offset * shape_m);
                    ptx::tensor_map_replace_global_addr_in_smem(smem_tensor_map_b, gmem_b_ptr + current_k_offset * shape_n);
                    ptx::tensor_map_replace_global_inner_dim_stride_in_smem(smem_tensor_map_a, scheduler.current_shape_k, scheduler.current_shape_k);
                    ptx::tensor_map_replace_global_inner_dim_stride_in_smem(smem_tensor_map_b, scheduler.current_shape_k, scheduler.current_shape_k);

                    cute::tma_desc_commit_group();
                    cute::tma_desc_wait_group();
                    __syncwarp(1u << lane_idx);

                    *(gmem_tensor_map_a) = *(smem_tensor_map_a);
                    *(gmem_tensor_map_b) = *(smem_tensor_map_b);
                    ptx::tensor_map_release_gpu();

                    ptx::tensor_map_acquire_gpu(gmem_tensor_map_a);
                    ptx::tensor_map_acquire_gpu(gmem_tensor_map_b);
                }

                #pragma unroll kNumPipelineUnrolls
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++ k_block_idx) {
                    // Wait consumer release
                    CUTE_TIE_DECL(get_pipeline(iter_idx ++), stage_idx, phase);
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    // Issue TMA
                    auto& full_barrier = *full_barriers[stage_idx];
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    const uint32_t sf_k_idx = scheduler.current_sf_k_cumsum + k_block_idx;
                    const auto tensor_map_a_ptr = (kGemmType == GemmType::KGroupedContiguous ? gmem_tensor_map_a : &tensor_map_a_base);
                    const auto tensor_map_b_ptr = (kGemmType == GemmType::KGroupedContiguous ? gmem_tensor_map_b : &tensor_map_b_base);
                    tma::copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, &full_barrier, smem_sfa[stage_idx], m_idx, sf_k_idx, num_tma_multicast_a);
                    tma::copy<BLOCK_N, BLOCK_K, 0>(&tensor_map_sfb, &full_barrier, smem_sfb[stage_idx], n_idx, sf_k_idx, num_tma_multicast_b);
                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode>(tensor_map_a_ptr, &full_barrier, smem_a[stage_idx], k_idx, m_idx, num_tma_multicast_a);
                    tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode>(tensor_map_b_ptr, &full_barrier, smem_b[stage_idx], k_idx, n_idx, num_tma_multicast_b);
                    full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE);
                }
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumStages; ++ s) {
                    CUTE_TIE_DECL(get_pipeline(iter_idx ++), stage_idx, phase);
                    empty_barriers[stage_idx]->wait(phase ^ 1);
                }
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        const auto row_idx = lane_idx / 4, col_idx = lane_idx % 4;
        const auto r_0 = warp_idx * 16 + row_idx, r_1 = r_0 + 8;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Accumulation for WGMMA or CUDA promotion
            DG_STATIC_ASSERT(BLOCK_M == WGMMA::M * (BLOCK_M <= 64 ? 1 : 2), "Invalid block sizes");
            const uint32_t current_shape_k = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_shape_k : shape_k);
            const uint32_t current_group_idx = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_group_idx : 0);
            const uint32_t num_k_blocks = math::ceil_div(current_shape_k, BLOCK_K);
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum] = {0};
            float2 scales_b[WGMMA::kNumAccum / 4];

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](uint32_t s) {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[s]->arrive() : void();
                } else {
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(target_cta) : void();
                }
            };

            #pragma unroll kNumPipelineUnrolls
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++ k_block_idx) {
                // Wait TMA arrivals
                CUTE_TIE_DECL(get_pipeline(iter_idx ++), stage_idx, phase);
                full_barriers[stage_idx]->wait(phase);

                // Read A scales
                auto scale_a_0 = ptx::ld_shared(smem_sfa[stage_idx] + r_0);
                auto scale_a_1 = ptx::ld_shared(smem_sfa[stage_idx] + r_1);

                // Read B scales
                #pragma unroll
                for (int i = 0; i < WGMMA::kNumAccum / 4; ++i)
                    scales_b[i] = ptx::ld_shared(reinterpret_cast<float2*>(smem_sfb[stage_idx] + i * 8 + col_idx * 2));

                // Commit WGMMA instructions
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                    auto desc_a = mma::sm90::make_smem_desc(smem_a[stage_idx] + math_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
                    auto desc_b = mma::sm90::make_smem_desc(smem_b[stage_idx] + k * WGMMA::K, 1);
                    WGMMA::wgmma(desc_a, desc_b, accum, k);
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_wait<0>();

                // Notify barrier arrival
                empty_barrier_arrive(stage_idx);

                // Promote with scales
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    const float &scale_b_0 = scales_b[i].x;
                    const float &scale_b_1 = scales_b[i].y;
                    final_accum[i * 4 + 0] += scale_a_0 * scale_b_0 * accum[i * 4 + 0];
                    final_accum[i * 4 + 1] += scale_a_0 * scale_b_1 * accum[i * 4 + 1];
                    final_accum[i * 4 + 2] += scale_a_1 * scale_b_0 * accum[i * 4 + 2];
                    final_accum[i * 4 + 3] += scale_a_1 * scale_b_1 * accum[i * 4 + 3];
                }
            }

            // Flush previous stores
            if (warp_idx % 4 == 0 and cute::elect_one_sync())
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(128, math_wg_idx);

            // Store to D shared memory
            const auto smem_d_0 = reinterpret_cast<float2*>(smem_d + r_0 * BLOCK_N + col_idx * 2);
            const auto smem_d_1 = reinterpret_cast<float2*>(smem_d + r_1 * BLOCK_N + col_idx * 2);
            #pragma unroll
            for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                ptx::st_shared(smem_d_0 + i * 4, {final_accum[i * 4 + 0], final_accum[i * 4 + 1]});
                ptx::st_shared(smem_d_1 + i * 4, {final_accum[i * 4 + 2], final_accum[i * 4 + 3]});
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(128, math_wg_idx);

            // Use TMA store to write back to global memory
            if (warp_idx % 4 == 0 and cute::elect_one_sync()) {
                cute::SM90_TMA_REDUCE_ADD_2D::copy(
                    &tensor_map_cd, smem_d_0, n_block_idx * BLOCK_N,
                    current_group_idx * shape_m + m_block_idx * BLOCK_M + r_0);
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
static CUtensorMap make_tma_desc_2d(void* gmem_ptr, CUtensorMapDataType dtype,
                                    uint32_t dim0, uint32_t dim1,
                                    uint32_t box0, uint32_t box1,
                                    uint64_t stride_bytes,
                                    CUtensorMapSwizzle swizzle) {
    CUtensorMap tensor_map;
    cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(dim0), static_cast<cuuint64_t>(dim1)};
    cuuint32_t smem_dims[2] = {static_cast<cuuint32_t>(box0), static_cast<cuuint32_t>(box1)};
    cuuint64_t gmem_strides[1] = {stride_bytes};
    cuuint32_t elem_strides[2] = {1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &tensor_map, dtype, 2, gmem_ptr, gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "cuTensorMapEncodeTiled failed: %d\n", static_cast<int>(result));
        exit(1);
    }
    return tensor_map;
}

static float fp8_to_float(uint8_t x) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(x), __NV_E4M3);
    return __half2float(__half(hr));
}

static uint8_t float_to_fp8(float x) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

// UE8M0-style power-of-2 scale for a given amax: 2^ceil(log2(amax / 448))
static float compute_sf(float amax) {
    if (amax <= 0.0f)
        return 1.0f;
    return exp2f(ceilf(log2f(amax / 448.0f)));
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

    // Tile configuration (1D1D FP8 kernel requires BLOCK_K = 128, FP32 output)
    constexpr uint32_t BLOCK_M = 128;
    constexpr uint32_t BLOCK_N = 128;
    constexpr uint32_t BLOCK_K = 128;
    constexpr uint32_t kNumStages = 4;   // fits SM90 shared memory (BLOCK_M = BLOCK_N = 128, FP32 D)
    constexpr uint32_t kNumTMAThreads = 128;
    constexpr uint32_t kNumMathThreads = 256;
    constexpr uint32_t kNumSMs = 132;    // H100/H800/H200 have 132 SMs
    constexpr uint32_t kSwizzleAMode = 128;
    constexpr uint32_t kSwizzleBMode = 128;

    if (static_cast<uint32_t>(prop.multiProcessorCount) != kNumSMs) {
        fprintf(stderr,
                "Warning: GPU has %d SMs but kNumSMs is fixed at %u. "
                "Edit the constexpr kNumSMs to match your GPU for best results.\n",
                prop.multiProcessorCount, kNumSMs);
    }

    // Padding requirements:
    //  - A/B row stride must be a multiple of 16 bytes -> K padded to 128
    //  - D row stride (fp32) must be a multiple of 16 bytes -> N padded to 4
    //  - SF inner dim must be TMA aligned (16 bytes / 4 = 4) -> M/N padded to 4
    auto align_up = [](int x, int a) { return (x + a - 1) / a * a; };
    const int K_pad = align_up(k_in, 128);
    const int N_pad = align_up(n_in, 4);
    const int M_pad = align_up(m_in, 4);
    const int SF_K = (k_in + 127) / 128;

    printf("Running standalone FP8 1D1D GEMM: m=%d n=%d k=%d (K_pad=%d N_pad=%d M_pad=%d SF_K=%d)\n",
           m_in, n_in, k_in, K_pad, N_pad, M_pad, SF_K);

    // Generate source matrices and quantize (per-row, per-128-K-block UE8M0 scales)
    std::vector<float> a_f(static_cast<size_t>(m_in) * K_pad, 0.0f);
    std::vector<float> b_f(static_cast<size_t>(n_in) * K_pad, 0.0f);
    std::vector<uint8_t> h_A(static_cast<size_t>(m_in) * K_pad, 0);
    std::vector<uint8_t> h_B(static_cast<size_t>(n_in) * K_pad, 0);
    std::vector<float> h_sfa(static_cast<size_t>(SF_K) * M_pad, 0.0f);
    std::vector<float> h_sfb(static_cast<size_t>(SF_K) * N_pad, 0.0f);

    auto quantize = [&](int rows, int cols, std::vector<float>& dequant, std::vector<uint8_t>& q,
                        std::vector<float>& sf, int sf_ld) {
        for (int r = 0; r < rows; ++r) {
            for (int kb = 0; kb < SF_K; ++kb) {
                const int kend = std::min((kb + 1) * 128, cols);
                float amax = 0.0f;
                for (int kk = kb * 128; kk < kend; ++kk) {
                    // simple deterministic pattern in [-0.4, 0.4]
                    float v = 0.05f * static_cast<float>((static_cast<int>((static_cast<size_t>(r) * cols + kk) * 7) % 17) - 8);
                    amax = std::max(amax, std::fabs(v));
                }
                const float scale = compute_sf(amax);
                sf[static_cast<size_t>(kb) * sf_ld + r] = scale;
                for (int kk = kb * 128; kk < kend; ++kk) {
                    float v = 0.05f * static_cast<float>((static_cast<int>((static_cast<size_t>(r) * cols + kk) * 7) % 17) - 8);
                    uint8_t qv = float_to_fp8(v / scale);
                    q[static_cast<size_t>(r) * K_pad + kk] = qv;
                    dequant[static_cast<size_t>(r) * K_pad + kk] = fp8_to_float(qv) * scale;
                }
            }
        }
    };

    quantize(m_in, k_in, a_f, h_A, h_sfa, M_pad);
    quantize(n_in, k_in, b_f, h_B, h_sfb, N_pad);

    // CPU reference: same math as the kernel (per-128-block fp32 sum, then scaled accumulate)
    std::vector<float> h_ref(static_cast<size_t>(m_in) * N_pad, 0.0f);
    for (int i = 0; i < m_in; ++i) {
        for (int j = 0; j < n_in; ++j) {
            float total = 0.0f;
            for (int kb = 0; kb < SF_K; ++kb) {
                const int kend = std::min((kb + 1) * 128, k_in);
                float block = 0.0f;
                for (int kk = kb * 128; kk < kend; ++kk)
                    block += a_f[static_cast<size_t>(i) * K_pad + kk] * b_f[static_cast<size_t>(j) * K_pad + kk];
                total += h_sfa[static_cast<size_t>(kb) * M_pad + i] * h_sfb[static_cast<size_t>(kb) * N_pad + j] * block;
            }
            h_ref[static_cast<size_t>(i) * N_pad + j] = total;
        }
    }

    // Allocate device memory
    uint8_t *d_A, *d_B;
    float *d_sfa, *d_sfb, *d_D;
    cudaMalloc(&d_A, static_cast<size_t>(m_in) * K_pad);
    cudaMalloc(&d_B, static_cast<size_t>(n_in) * K_pad);
    cudaMalloc(&d_sfa, static_cast<size_t>(SF_K) * M_pad * sizeof(float));
    cudaMalloc(&d_sfb, static_cast<size_t>(SF_K) * N_pad * sizeof(float));
    cudaMalloc(&d_D, static_cast<size_t>(m_in) * N_pad * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), static_cast<size_t>(m_in) * K_pad, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), static_cast<size_t>(n_in) * K_pad, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sfa, h_sfa.data(), static_cast<size_t>(SF_K) * M_pad * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sfb, h_sfb.data(), static_cast<size_t>(SF_K) * N_pad * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_D, 0, static_cast<size_t>(m_in) * N_pad * sizeof(float));  // kernel uses TMA REDUCE_ADD

    // Build TMA descriptors.
    //   A:    [m, K_pad] fp8, K-major  -> dims (k, m), box (128, BLOCK_M), swizzle 128B
    //   B:    [n, K_pad] fp8, K-major  -> dims (k, n), box (128, BLOCK_N), swizzle 128B
    //   SFA:  [SF_K, M_pad] fp32       -> dims (M_pad, SF_K), box (BLOCK_M, 1), no swizzle
    //   SFB:  [SF_K, N_pad] fp32       -> dims (N_pad, SF_K), box (BLOCK_N, 1), no swizzle
    //   D:    [m, N_pad] fp32, N-major -> dims (n, m), box (BLOCK_N, 64), no swizzle
    CUtensorMap tma_a = make_tma_desc_2d(d_A, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                         static_cast<uint32_t>(k_in), static_cast<uint32_t>(m_in),
                                         kSwizzleAMode, BLOCK_M,
                                         static_cast<uint64_t>(K_pad),
                                         CU_TENSOR_MAP_SWIZZLE_128B);
    CUtensorMap tma_b = make_tma_desc_2d(d_B, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                         static_cast<uint32_t>(k_in), static_cast<uint32_t>(n_in),
                                         kSwizzleBMode, BLOCK_N,
                                         static_cast<uint64_t>(K_pad),
                                         CU_TENSOR_MAP_SWIZZLE_128B);
    CUtensorMap tma_sfa = make_tma_desc_2d(d_sfa, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                           static_cast<uint32_t>(M_pad), static_cast<uint32_t>(SF_K),
                                           BLOCK_M, 1,
                                           static_cast<uint64_t>(M_pad) * sizeof(float),
                                           CU_TENSOR_MAP_SWIZZLE_NONE);
    CUtensorMap tma_sfb = make_tma_desc_2d(d_sfb, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                           static_cast<uint32_t>(N_pad), static_cast<uint32_t>(SF_K),
                                           BLOCK_N, 1,
                                           static_cast<uint64_t>(N_pad) * sizeof(float),
                                           CU_TENSOR_MAP_SWIZZLE_NONE);
    CUtensorMap tma_d = make_tma_desc_2d(d_D, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                         static_cast<uint32_t>(n_in), static_cast<uint32_t>(m_in),
                                         BLOCK_N, 64,
                                         static_cast<uint64_t>(N_pad) * sizeof(float),
                                         CU_TENSOR_MAP_SWIZZLE_NONE);

    // Compute shared memory requirement (must match the kernel's layout)
    constexpr uint32_t SMEM_D_SIZE = deep_gemm::math::constexpr_align(
        static_cast<uint32_t>(BLOCK_M * BLOCK_N * sizeof(float)), 1024u);
    constexpr uint32_t SMEM_A_SIZE = BLOCK_M * BLOCK_K * sizeof(uint8_t);
    constexpr uint32_t SMEM_B_SIZE = BLOCK_N * BLOCK_K * sizeof(uint8_t);
    constexpr uint32_t SMEM_SFA_SIZE = BLOCK_M * sizeof(float);
    constexpr uint32_t SMEM_SFB_SIZE = deep_gemm::math::constexpr_align(
        static_cast<uint32_t>(BLOCK_N * sizeof(float)), 128u);
    constexpr uint32_t SMEM_TOTAL = SMEM_D_SIZE +
        kNumStages * (SMEM_A_SIZE + SMEM_B_SIZE + SMEM_SFA_SIZE + SMEM_SFB_SIZE) +
        2 * kNumStages * static_cast<uint32_t>(sizeof(cutlass::arch::ClusterTransactionBarrier));

    dim3 block(kNumTMAThreads + kNumMathThreads);
    dim3 grid(kNumSMs);

    auto kernel = &deep_gemm::sm90_fp8_gemm_1d1d_impl<
        0, 0, 0,
        1,
        BLOCK_M, BLOCK_N, BLOCK_K,
        kSwizzleAMode, kSwizzleBMode,
        kNumStages,
        kNumTMAThreads, kNumMathThreads,
        1, false,
        kNumSMs,
        deep_gemm::GemmType::Normal, float>;

    cudaFuncSetAttribute((const void*)kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(SMEM_TOTAL));

    // tensor_map_buffer is only dereferenced for KGroupedContiguous; nullptr is fine for Normal
    kernel<<<grid, block, SMEM_TOTAL>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(d_A), reinterpret_cast<__nv_fp8_e4m3*>(d_B),
        nullptr, nullptr,
        static_cast<uint32_t>(m_in), static_cast<uint32_t>(n_in), static_cast<uint32_t>(k_in),
        tma_a, tma_b, tma_sfa, tma_sfb, tma_d);

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

    std::vector<float> h_D(static_cast<size_t>(m_in) * N_pad, 0.0f);
    cudaMemcpy(h_D.data(), d_D, static_cast<size_t>(m_in) * N_pad * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the m_in x n_in region
    double max_rel_err = 0.0;
    double max_abs_err = 0.0;
    int fail_count = 0;
    for (int i = 0; i < m_in; ++i) {
        for (int j = 0; j < n_in; ++j) {
            float got = h_D[static_cast<size_t>(i) * N_pad + j];
            float ref = h_ref[static_cast<size_t>(i) * N_pad + j];
            float abs_err = std::fabs(got - ref);
            float rel_err = (std::fabs(ref) > 1e-6f) ? (abs_err / std::fabs(ref)) : abs_err;
            max_rel_err = std::max(max_rel_err, static_cast<double>(rel_err));
            max_abs_err = std::max(max_abs_err, static_cast<double>(abs_err));
            if (rel_err > 1e-3f && abs_err > 1e-3f) {
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
    cudaFree(d_sfa);
    cudaFree(d_sfb);
    cudaFree(d_D);
    return (fail_count == 0) ? 0 : 1;
}
