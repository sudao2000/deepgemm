#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

namespace deep_gemm {

template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K_,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleDMode,
          uint32_t kNumStages_,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType, bool kWithAccumulation,
          typename cd_dtype_t>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_bf16_gemm_impl(int* grouped_layout,
                    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                    const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                    const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                    const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Enlarge `BLOCK_K` for some cases
    // NOTES: this is for reducing the `warpgroup_wait<0>()` overhead
    constexpr uint32_t kDoMergeStages =
        kNumStages_ >= 10 and
        kGemmType == GemmType::Normal and
        kMajorA == cute::UMMA::Major::K and kMajorB == cute::UMMA::Major::K and
        kNumMathThreads == 128;
    // Ensure there are at least `kNumMinStages` stages after merge
    constexpr uint32_t kNumMinStages = 5;
    constexpr uint32_t kNumStagesPerMerge = kDoMergeStages ? kNumStages_ / kNumMinStages : 1;
    constexpr uint32_t BLOCK_K = BLOCK_K_ * kNumStagesPerMerge;
    constexpr uint32_t kNumStages = kNumStages_ / kNumStagesPerMerge;

    // Types
    using WGMMA = typename mma::sm90::BF16MMASelector<BLOCK_N, kMajorA, kMajorB>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0 or BLOCK_M < WGMMA::M, "Invalid block size");

    // C/D type: BF16 and FP32 are supported, with or without accumulation
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float> or cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Invalid C/D data dtype");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Shared memory
    static constexpr uint32_t SMEM_D_SIZE = math::constexpr_align(BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(cd_dtype_t)), 1024u);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_bfloat16);

    // NOTES: Make sure we have enough shared memory for WGMMA padding
    static constexpr uint32_t WGMMA_A_SIZE_PER_STAGE = WGMMA::M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    DG_STATIC_ASSERT(WGMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory Out of bound for WGMMA");

    // Configs
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0, 
                     "Shared memory of A/B/D must be aligned to 1024 bytes");

    // D/A/B shared memory
    auto smem_d = reinterpret_cast<cd_dtype_t*>(smem_buffer);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });

    // Initialize barriers
    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
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

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 48;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 224;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs>(shape_m, shape_n, shape_k, grouped_layout);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx >= kNumMathThreads / 32) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        // We use the third warp, as warp 0/1 may be doing WGMMA with `BLOCK_M == 32`
        if (warp_idx == kNumMathThreads / 32 + 2 and cute::elect_one_sync()) {
            DG_STATIC_ASSERT(kNumTMAThreads >= 128, "Need at least 128 threads for TMA warp-group");

            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                // Assign TMA multicast number into A and B
                // NOTES: there may be additional odd rows/columns or cases where multicast is not possible.
                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");

                const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait consumer release
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    constexpr bool kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked;
                    auto& full_barrier = *full_barriers[stage_idx];

                    const auto m_idx = scheduler.template get_global_idx<kWithGroupOffsetA, sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
                    const auto n_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::K), sched::IndexType::MN>(shape_n, BLOCK_N, n_block_idx, m_block_idx);

                    DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous or kMajorA == cute::UMMA::Major::K, "Invalid major");
                    uint32_t k_a_idx = scheduler.template get_global_idx<(kMajorA == cute::UMMA::Major::MN), sched::IndexType::K> (
                        shape_k, BLOCK_K, k_block_idx, m_block_idx);
                    uint32_t k_b_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::MN), sched::IndexType::K> (
                        shape_k, BLOCK_K, k_block_idx, m_block_idx);

                    // Issue TMAs
                    constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                    const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);
                    if constexpr (kMajorA == cute::UMMA::Major::K)
                        tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
                            &tensor_map_a, &full_barrier, smem_a[stage_idx], k_a_idx, m_idx, num_tma_multicast_a, batch_idx);
                    if constexpr (kMajorA == cute::UMMA::Major::MN)
                        tma::copy<BLOCK_M, BLOCK_K, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
                            &tensor_map_a, &full_barrier, smem_a[stage_idx], m_idx, k_a_idx, num_tma_multicast_a, batch_idx);
                    if constexpr (kMajorB == cute::UMMA::Major::K)
                        tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                            &tensor_map_b, &full_barrier, smem_b[stage_idx], k_b_idx, n_idx, num_tma_multicast_b, batch_idx);
                    if constexpr (kMajorB == cute::UMMA::Major::MN)
                        tma::copy<BLOCK_N, BLOCK_K, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                            &tensor_map_b, &full_barrier, smem_b[stage_idx], n_idx, k_b_idx, num_tma_multicast_b, batch_idx);
                    full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
                }
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                for (uint32_t i = 0; i < kNumStages; advance_pipeline(i))
                    empty_barriers[stage_idx]->wait(phase ^ 1);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        
        // Merged stages only happens in NT normal GEMM cases
        constexpr uint32_t BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge;
        auto a_desc = mma::sm90::make_gmma_desc<kMajorA, BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode>(smem_a[0], math_wg_idx * WGMMA::M, 0);
        auto b_desc = mma::sm90::make_gmma_desc<kMajorB, BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode>(smem_b[0], 0, 0);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= WGMMA::M ? BLOCK_M : WGMMA::M * 2;
            DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes");
            float accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};

            // Pick threads whose WGMMA results are to be stored in shared memory
            DG_STATIC_ASSERT(BLOCK_M >= 64 or kNumMathThreads == 128, "Only one math warp group for `BLOCK_M < 64`");
            constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA::M);
            const bool do_wgmma_store = BLOCK_M >= 64 or warp_idx < kNumWGMMAStoreThreads / 32;

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](uint32_t s) {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[s]->arrive() : void();
                } else {
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(target_cta) : void();
                }
            };

            // TODO: remove some useless computation for unaligned Ms
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                const auto b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);

                // Wait TMA arrivals
                full_barriers[stage_idx]->wait(phase);

                // Commit WGMMA instructions
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
                        a_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<kMajorA, BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode, nv_bfloat16>(
                            a_desc_base_lo, local_idx * WAVE_BLOCK_M, (k * WGMMA::K) % BLOCK_ATOM_K, atom_k_idx * BLOCK_M * BLOCK_ATOM_K);
                        b_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<kMajorB, BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode, nv_bfloat16>(
                            b_desc_base_lo, 0, (k * WGMMA::K) % BLOCK_ATOM_K, atom_k_idx * BLOCK_N * BLOCK_ATOM_K);
                        WGMMA::wgmma(a_desc, b_desc, shifted_accum, 1);
                    }
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M); ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_wait<0>();

                // Notify barrier arrival
                empty_barrier_arrive(stage_idx);
            }

            // TMA checks
            constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16);
            constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_N : (kSwizzleDMode / kNumElemBytes);
            constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;
            DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
            DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N / TMA_D_BLOCK_N <= 32,
                            "Unaligned TMA store or too many TMA store instructions");
            DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");

            // Skip WGMMA store for the unfilled parts
            if (not do_wgmma_store)
                continue;

            // Wait last TMA store to be finished
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 0);

            if constexpr (cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>) {
                // Write back to shared memory using STSM and issue TMA stores
                DG_STATIC_ASSERT(kSwizzleDMode > 0, "Invalid swizzling type");
                DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
                #pragma unroll
                for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                    auto m_offset = local_idx * WAVE_BLOCK_M;
                    auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
                    #pragma unroll
                    for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                        // Swizzle or padding into the correct address
                        uint8_t* smem_ptr = nullptr;
                        if constexpr (kSwizzleDMode > 0) {
                            // Calculate the swizzling atom offset and in-atom offset
                            constexpr uint32_t kNumBankGroupBytes = 16;
                            auto atom_offset = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);

                            // Calculate the index of the bank group to be written in the atom
                            auto bank_group_index = in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);

                            // Reshape the atom in another view and swizzle
                            //  - original: `(BLOCK_M, kSwizzleDMode / kNumBankGroupBytes)`
                            //  - new: `(BLOCK_M * kSwizzleDMode / kNumBankGroupBytes / 8, 8)`
                            constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;
                            auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
                            auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
                            col ^= row % (kSwizzleDMode / 16);

                            // Add back into the base pointer
                            // NOTES: think twice before modifying this, as changes may affect the number of instructions
                            smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +                // Base pointer
                                warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +            // Warp offset
                                m_offset * kSwizzleDMode +                                 // Wave offset
                                atom_offset * BLOCK_M * kSwizzleDMode +                    // Swizzle atom offset (constants)
                                row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes; // In-atom offset
                        } else {
                            // No swizzling
                            smem_ptr = reinterpret_cast<uint8_t*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx) * BLOCK_N + i * 8);
                        }

                        // NOTES: only 16 lanes' addresses are used
                        ptx::SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                            __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}),
                            __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                            smem_ptr
                        );
                    }
                }
            } else {
                // Use `st.shared` if STSM is not available
                #pragma unroll
                for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                    auto m_offset = local_idx * WAVE_BLOCK_M;
                    auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
                    auto smem_d_0 = reinterpret_cast<float2*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx / 4 + 0) * BLOCK_N + (lane_idx % 4) * 2);
                    auto smem_d_1 = reinterpret_cast<float2*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx / 4 + 8) * BLOCK_N + (lane_idx % 4) * 2);
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                        ptx::st_shared(smem_d_0 + i * 4, make_float2(shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]));
                        ptx::st_shared(smem_d_1 + i * 4, make_float2(shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]));
                    }
                }
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 0);

            // Use TMA store to write back to global memory
            const auto m_idx = scheduler.template get_global_idx<(not is_m_grouped_contiguous(kGemmType)), sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
            DG_STATIC_ASSERT(kNumWGMMAStoreThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
                if constexpr (kGemmType == GemmType::Batched) {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                        cute::SM90_TMA_REDUCE_ADD_3D, cute::SM90_TMA_STORE_3D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_ptr,
                                     n_block_idx * BLOCK_N + in_block_n_offset,
                                     m_idx, scheduler.current_group_idx);
                } else {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                        cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_ptr,
                                     n_block_idx * BLOCK_N + in_block_n_offset, m_idx);
                }
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop

`deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh` 实现了 Hopper（SM90）架构上 BF16 矩阵乘法的核心 CUDA kernel。下面按文件结构逐段、逐行解释。

---

## 1. 文件头与编译器诊断

```cpp
#pragma once
```
标准头文件保护，防止重复包含。

```cpp
#pragma clang diagnostic push
#pragma clangometric ignored "-Wunknown-attributes"
```
保存当前 clang 诊断状态，并忽略“未知属性”警告。CUDA 的 `__grid_constant__` 等属性在 clang 编译时可能会触发此警告，这里选择静默。

```cpp
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
```
- `barrier.h`：CUTLASS 的 mbarrier / 命名屏障实现。
- `reg_reconfig.h`：warp group 级别的寄存器重新分配接口（`warpgroup_reg_alloc` / `warpgroup_reg_dealloc`）。

```cpp
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
```
- `cluster_sm90.hpp`：CTA 集群（cluster）同步原语。
- `copy_sm90_desc.hpp` / `copy_sm90_tma.hpp`：TMA 描述符与 TMA copy 指令封装。
- `mma_sm100_desc.hpp`：WGMMA 描述符相关类型（虽然名字带 sm100，但这里用于描述符构造）。

```cpp
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/scheduler/gemm.cuh>
```
项目内部头文件：
- `comm/barrier.cuh`：集群级同步封装。
- `common/math.cuh` / `utils.cuh` / `types.cuh`：数学工具、类型别名、通用辅助。
- `common/tma_copy.cuh`：项目自己的 TMA copy 封装。
- `mma/sm90.cuh`：SM90 WGMMA 描述符构造与选择器。
- `epilogue/transform.cuh`：输出变换（如 swizzling、类型转换）。
- `ptx/ld_st.cuh` / `utils.cuh` / `wgmma.cuh`：PTX 内联汇编函数（加载/存储、WGMMA fence、操作数 fence）。
- `scheduler/gemm.cuh`：block 调度器，决定当前 CTA 计算哪个输出块。

```cpp
namespace deep_gemm {
```
进入 `deep_gemm` 命名空间。

---

## 2. Kernel 函数签名与模板参数

```cpp
template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K_,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleDMode,
          uint32_t kNumStages_,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType, bool kWithAccumulation,
          typename cd_dtype_t>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_bf16_gemm_impl(int* grouped_layout,
                    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                    const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                    const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                    const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
// ```

// 模板参数解释：
// - `kMajorA`, `kMajorB`：A、B 矩阵的主布局（`K` 主序或 `MN` 主序），决定 TMA copy 时坐标顺序。
// - `SHAPE_M/N/K`：编译时 M/N/K 维度；若为 0 则使用运行时传入值。
// - `kNumGroups`：分组 GEMM 的组数（用于 grouped/batched 变体）。
// - `BLOCK_M/N/K_`：CTA 块大小；`K_` 可能被后面的 stage merge 逻辑放大。
// - `kSwizzleA/B/DMode`：A、B、D 共享内存 swizzle 模式（影响 bank conflict 与 TMA 兼容性）。
// - `kNumStages_`：原始流水线 stage 数；可能被 merge。
// - `kNumTMAThreads` / `kNumMathThreads`：分别用于 TMA 加载和 WGMMA 计算的线程数。
// - `kNumTMAMulticast`：TMA multicast 数（1 或 2）。
// - `kIsTMAMulticastOnA`：multicast 是否作用在 A 上（否则在 B 上）。
// - `kNumSMs`：可用 SM 数量，供调度器使用。
// - `kGemmType`：GEMM 类型（Normal、KGroupedContiguous、MGroupedMasked、Batched 等）。
// - `kWithAccumulation`：输出是否累加到全局内存（使用 TMA reduce-add）。
// - `cd_dtype_t`：C/D 输出数据类型（`float` 或 `cutlass::bfloat16_t`）。

// 函数参数：
// - `grouped_layout`：分组信息数组（例如每组起始偏移）。
// - `shape_m/n/k`：运行时 M/N/K。
// - `tensor_map_a/b/cd`：`__grid_constant__` TMA 描述符，分别对应 A、B、C/D。

// `__launch_bounds__(..., 1)` 表示每个 SM 上最多驻留 1 个 block，以换取更多寄存器。

// ---

// ## 3. 编译时 guard 与 stage merge 逻辑

// ```cpp
// #if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
// ```
// 只让 sm_90a 及以上（或 IDE 解析）编译本段代码，否则后面会报错。

// ```cpp
//     constexpr uint32_t kDoMergeStages =
//         kNumStages_ >= 10 and
//         kGemmType == GemmType::Normal and
//         kMajorA == cute::UMMA::Major::K and kMajorB == cute::UMMA::Major::K and
//         kNumMathThreads == 128;
// ```
// 判断是否需要“合并 stage”。当 stage 数 ≥10、普通 GEMM、A/B 都是 K-major、且数学线程为 128 时启用。

// ```cpp
//     constexpr uint32_t kNumMinStages = 5;
//     constexpr uint32_t kNumStagesPerMerge = kDoMergeStages ? kNumStages_ / kNumMinStages : 1;
//     constexpr uint32_t BLOCK_K = BLOCK_K_ * kNumStagesPerMerge;
//     constexpr uint32_t kNumStages = kNumStages_ / kNumStagesPerMerge;
// ```
// - `kNumStagesPerMerge`：合并因子。例如原始 15 个 stage，合并为 5 个，因子为 3。
// - `BLOCK_K`：实际每个 stage 的 K 块大小被放大。
// - `kNumStages`：合并后逻辑 stage 数。
// - 目的：减少 `warpgroup_wait<0>()` 的同步开销，一次做更多 K 维计算。

// ---

// ## 4. 类型与静态断言

// ```cpp
//     using WGMMA = typename mma::sm90::BF16MMASelector<BLOCK_N, kMajorA, kMajorB>::type;
// ```
// 根据 `BLOCK_N` 和 A/B 主序选择具体的 WGMMA 指令类型（包含 `M`、`K`、`kNumAccum` 等静态成员）。

// ```cpp
//     using Barrier = cutlass::arch::ClusterTransactionBarrier;
// ```
// 定义 barrier 类型，用于 producer（TMA）与 consumer（WGMMA）同步。

// ```cpp
//     DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0 or BLOCK_M < WGMMA::M, "Invalid block size");
// ```
// 确保 `BLOCK_M` 能被 WGMMA 的 M 维度整除，或至少不超出。

// ```cpp
//     DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float> or cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Invalid C/D data dtype");
// ```
// 只支持 `float` 或 `bfloat16` 输出。

// ---

// ## 5. 运行时 shape 与共享内存布局

// ```cpp
//     shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
//     shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
//     shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
// ```
// 如果编译期 shape 不为 0，则覆盖运行时值；否则使用运行时值。

// ```cpp
//     static constexpr uint32_t SMEM_D_SIZE = math::constexpr_align(BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(cd_dtype_t)), 1024u);
// ```
// D（输出）共享内存大小，按 1024 字节对齐。

// ```cpp
//     static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
//     static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_bfloat16);
// ```
// 每个 stage 的 A/B 共享内存大小。

// ```cpp
//     static constexpr uint32_t WGMMA_A_SIZE_PER_STAGE = WGMMA::M * BLOCK_K * sizeof(__nv_fp8_e4m3);
//     DG_STATIC_ASSERT(WGMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory Out of bound for WGMMA");
// ```
// - 计算 WGMMA 对 A 的需求（按 FP8 计，因为底层 WGMMA 描述符可能用 FP8 布局）。
// - 断言共享内存足够。

// ---

// ## 6. 线程索引与 TMA 描述符预取

// ```cpp
//     const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
//     const uint32_t lane_idx = ptx::get_lane_idx();
// ```
// - `warp_idx`：当前 warp 索引，用 `__shfl_sync` 广播到 lane 0 的值（让编译器把结果放到统一寄存器）。
// - `lane_idx`：当前线程在 warp 内的 lane 索引。

// ```cpp
//     if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
//         cute::prefetch_tma_descriptor(&tensor_map_a);
//         cute::prefetch_tma_descriptor(&tensor_map_b);
//         cute::prefetch_tma_descriptor(&tensor_map_cd);
//     }
//     __syncwarp();
// ```
// - 由 TMA warp group 的第一个 elected 线程预取三个 TMA 描述符到 const cache。
// - `cute::elect_one_sync()` 在每个 warp 内选举唯一一个线程。
// - `__syncwarp()` 同步 warp。

// ---

// ## 7. 共享内存指针划分

// ```cpp
//     extern __shared__ __align__(1024) uint8_t smem_buffer[];
// ```
// 声明动态共享内存，按 1024 字节对齐（满足 swizzle-128B 要求）。

// ```cpp
//     DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0, 
//                      "Shared memory of A/B/D must be aligned to 1024 bytes");
// ```

// ```cpp
//     auto smem_d = reinterpret_cast<cd_dtype_t*>(smem_buffer);
//     auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
//         return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
//     });
//     auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
//         return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
//     });
// ```
// - `smem_d`：输出 D 缓冲区起始指针。
// - `smem_a[i]`：第 `i` 个 stage 的 A 缓冲区指针。
// - `smem_b[i]`：第 `i` 个 stage 的 B 缓冲区指针。
// - 布局顺序：D → A[0..kNumStages-1] → B[0..kNumStages-1] → barriers。

// ```cpp
//     auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
//     auto full_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
//     auto empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
// ```
// - `full_barriers[i]`：第 `i` 个 stage 数据已准备好的 barrier。
// - `empty_barriers[i]`：第 `i` 个 stage 已被消费完的 barrier。
// - 共 `2 * kNumStages` 个 barrier 对象。

// ```cpp
//     if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
//         #pragma unroll
//         for (uint32_t i = 0; i < kNumStages; ++ i) {
//             full_barriers[i]->init(1);
//             empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
//         }
//         cutlass::arch::fence_barrier_init();
//     }
// ```
// - 由 TMA warp group 后的某个 warp 内一个 elected 线程初始化 barrier。
// - `full_barriers[i]` 初始计数 1（一个 TMA warp 写入）。
// - `empty_barriers[i]` 初始计数 `kNumTMAMulticast * kNumMathThreads / 32`（所有消费 A/B 的 math warp 都要 arrive）。

// ```cpp
//     (kNumTMAMulticast > 1) ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();
// ```
// - 若使用 TMA multicast，需要集群级同步保证所有 CTA 都看到初始化后的 barrier。
// - 否则普通 `__syncthreads()` 即可。

// ---

// ## 8. 寄存器配置与 grid dependency 同步

// ```cpp
//     constexpr uint32_t kNumTMARegisters = 48;
//     constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 224;
// ```
// - TMA warp group 期望保留 48 个寄存器。
// - Math warp group 保留 248（128 线程模式）或 224 个寄存器。

// ```cpp
//     cudaGridDependencySynchronize();
// ```
// 等待上一个 kernel（或网格级依赖）完成。用于保证输入数据已就绪。

// ---

// ## 9. Block 调度器与流水线状态

// ```cpp
//     uint32_t m_block_idx, n_block_idx;
//     auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs>(shape_m, shape_n, shape_k, grouped_layout);
// ```
// 构造 block 调度器，决定当前 CTA 应该处理哪一组 `(m_block_idx, n_block_idx)`。

// ```cpp
//     uint32_t stage_idx = 0, phase = 0;
//     auto advance_pipeline = [&](uint32_t& k_block_idx) {
//         ++ k_block_idx;
//         stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
//         phase ^= stage_idx == 0;
//     };
// ```
// - `stage_idx`：当前 stage 下标。
// - `phase`：barrier phase 翻转位，配合 `mbarrier.try_wait.parity` 使用。
// - 每次 K 块推进，stage_idx 循环滚动；回到 0 时 phase 翻转。

// ---

// ## 10. TMA warp group（数据加载侧）

// ```cpp
//     if (warp_idx >= kNumMathThreads / 32) {
//         cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
// ```
// - warp_idx 超过 math warp 范围 → 属于 TMA warp group。
// - 调用 `warpgroup_reg_dealloc<48>()` 把该 warp group 的寄存器数降到 48，让渡给 math warp group。

// ```cpp
//         if (warp_idx == kNumMathThreads / 32 + 2 and cute::elect_one_sync()) {
// ```
// 只让 TMA warp group 中的第 3 个 warp（即 `math_threads/32 + 2`）的一个 elected 线程干活。注释说明 warp 0/1 可能在 `BLOCK_M == 32` 时参与 WGMMA。

// ```cpp
//             DG_STATIC_ASSERT(kNumTMAThreads >= 128, "Need at least 128 threads for TMA warp-group");
//             while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
// ```
// 持久化调度：一个 TMA warp 持续为多个输出块加载数据，直到调度器没有更多块。

// ```cpp
//                 const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
//                 const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
//                 const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
//                 DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");
// ```
// 根据调度器判断当前块是否适合 multicast，决定 A/B 各自的 multicast 数。

// ```cpp
//                 const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
//                 for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
// ```
// 对当前 (m,n) 块的 K 维循环，按 `BLOCK_K` 切分。

// ```cpp
//                     empty_barriers[stage_idx]->wait(phase ^ 1);
// ```
// 等待 consumer 释放该 stage（即上一次该 stage 的数据已被 math 侧消费完）。

// ```cpp
//                     constexpr bool kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked;
//                     auto& full_barrier = *full_barriers[stage_idx];
// ```
// - `kWithGroupOffsetA`：在 MGroupedMasked 模式下，A 的索引需要加上组偏移。
// - 获取当前 stage 的 full barrier 引用。

// ```cpp
//                     const auto m_idx = scheduler.template get_global_idx<kWithGroupOffsetA, sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
//                     const auto n_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::K), sched::IndexType::MN>(shape_n, BLOCK_N, n_block_idx, m_block_idx);
// ```
// 计算 A/B 在全局内存中的 M/N 维度起始索引。第二个 `get_global_idx` 多传 `m_block_idx` 是为了 grouped 布局。

// ```cpp
//                     DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous or kMajorA == cute::UMMA::Major::K, "Invalid major");
//                     uint32_t k_a_idx = scheduler.template get_global_idx<(kMajorA == cute::UMMA::Major::MN), sched::IndexType::K> (
//                         shape_k, BLOCK_K, k_block_idx, m_block_idx);
//                     uint32_t k_b_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::MN), sched::IndexType::K> (
//                         shape_k, BLOCK_K, k_block_idx, m_block_idx);
// ```
// 计算 A/B 在 K 维的全局起始索引。模板参数控制是否把 `m_block_idx` 当作组索引传入。

// ```cpp
//                     constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
//                     const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);
// ```
// Batched GEMM 时取当前 group 作为 batch 索引。

// ### 10.1 发起 TMA copy

// ```cpp
//                     if constexpr (kMajorA == cute::UMMA::Major::K)
//                         tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
//                             &tensor_map_a, &full_barrier, smem_a[stage_idx], k_a_idx, m_idx, num_tma_multicast_a, batch_idx);
//                     if constexpr (kMajorA == cute::UMMA::Major::MN)
//                         tma::copy<BLOCK_M, BLOCK_K, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
//                             &tensor_map_a, &full_barrier, smem_a[stage_idx], m_idx, k_a_idx, num_tma_multicast_a, batch_idx);
// ```
// 根据 A 的主序，选择 TMA copy 的坐标顺序，把 A 的 `[m_idx, k_a_idx]` 块加载到 `smem_a[stage_idx]`。

// ```cpp
//                     if constexpr (kMajorB == cute::UMMA::Major::K)
//                         tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
//                             &tensor_map_b, &full_barrier, smem_b[stage_idx], k_b_idx, n_idx, num_tma_multicast_b, batch_idx);
//                     if constexpr (kMajorB == cute::UMMA::Major::MN)
//                         tma::copy<BLOCK_N, BLOCK_K, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
//                             &tensor_map_b, &full_barrier, smem_b[stage_idx], n_idx, k_b_idx, num_tma_multicast_b, batch_idx);
// ```
// 同理加载 B。

// ```cpp
//                     full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
// ```
// TMA warp 在发起所有 copy 后 arrive full barrier，并声明预期事务总字节数，通知 math warp 数据即将到来。

// ```cpp
//             if constexpr (kNumTMAMulticast > 1) {
//                 for (uint32_t i = 0; i < kNumStages; advance_pipeline(i))
//                     empty_barriers[stage_idx]->wait(phase ^ 1);
//             }
// ```
// kernel 结束前额外等待所有 empty barrier，确保不会破坏分布式 shared barrier 的析构语义。

// ---

// ## 11. Math warp group（计算侧）

// ```cpp
//     } else {
//         cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
// ```
// math warp group 分配更多寄存器。

// ```cpp
//         const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
// ```
// 当前线程所属的 math warp group 索引。

// ```cpp
//         constexpr uint32_t BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge;
// ```
// 合并 stage 后，单个 WGMMA 原子操作的 K 维度。

// ```cpp
//         auto a_desc = mma::sm90::make_gmma_desc<kMajorA, BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode>(smem_a[0], math_wg_idx * WGMMA::M, 0);
//         auto b_desc = mma::sm90::make_gmma_desc<kMajorB, BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode>(smem_b[0], 0, 0);
// ```
// 构造 A/B 的 WGMMA 描述符。描述符编码了共享内存地址、swizzle、leading dimension 等信息。
// - A 的描述符按 `math_wg_idx * WGMMA::M` 做 M 方向偏移，让每个 warp group 负责不同 M 行。

// ```cpp
//         const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
//         const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);
// ```
// 把描述符低 32 位从 lane 0 广播出去。后面每次只改低 32 位来推进 K，高 32 位保持不变。

// ```cpp
//         while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
// ```
// 持久化调度循环，获取下一个 (m,n) 输出块。

// ```cpp
//             constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= WGMMA::M ? BLOCK_M : WGMMA::M * 2;
//             DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes");
//             float accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};
// ```
// - `WAVE_BLOCK_M`：一个 WGMMA 波次处理的 M 大小。
// - `accum`：累加器数组，大小按块内 M 切分数量计算，初始化为 0。

// ```cpp
//             DG_STATIC_ASSERT(BLOCK_M >= 64 or kNumMathThreads == 128, "Only one math warp group for `BLOCK_M < 64`");
//             constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA::M);
//             const bool do_wgmma_store = BLOCK_M >= 64 or warp_idx < kNumWGMMAStoreThreads / 32;
// ```
// - 决定哪些线程参与最后的 WGMMA 结果写回。
// - 当 `BLOCK_M < 64` 时，只有部分 warp 需要写回。

// ### 11.1 K 维主循环

// ```cpp
//             auto empty_barrier_arrive = [&](uint32_t s) {
//                 if constexpr (kNumTMAMulticast == 1) {
//                     lane_idx == 0 ? empty_barriers[s]->arrive() : void();
//                 } else {
//                     auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
//                     lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(target_cta) : void();
//                 }
//             };
// ```
// lambda：通知某个 stage 已被消费。
// - 非 multicast：每个 warp 的 lane 0 arrive 一次。
// - multicast：根据 peer CTA 是否存活，向对应 CTA 的 empty barrier arrive。

// ```cpp
//             const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
//             for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
// ```
// 对当前输出块沿 K 维循环。

// ```cpp
//                 const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
//                 const auto b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);
// ```
// 基于 stage 偏移，计算当前 stage 描述符的低 32 位基址。除以 16 是因为描述符中地址按 16 字节对齐编码。

// ```cpp
//                 full_barriers[stage_idx]->wait(phase);
// ```
// 等待 TMA 加载完成。

// ### 11.2 提交 WGMMA 指令

// ```cpp
//                 #pragma unroll
//                 for (uint32_t i = 0; i < WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M); ++ i)
//                     ptx::warpgroup_fence_operand(accum[i]);
//                 ptx::warpgroup_arrive();
// ```
// - 对累加器做 WGMMA 操作数 fence，保证寄存器数据对异步 proxy 可见。
// - `warpgroup_arrive()` 标记 WGMMA 开始。

// ```cpp
//                 #pragma unroll
//                 for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
//                     auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
//                     #pragma unroll
//                     for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
//                         const uint32_t atom_k_idx = k * WGMMA::K / BLOCK_ATOM_K;
//                         a_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<...>(
//                             a_desc_base_lo, local_idx * WAVE_BLOCK_M, (k * WGMMA::K) % BLOCK_ATOM_K, atom_k_idx * BLOCK_M * BLOCK_ATOM_K);
//                         b_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<...>(
//                             b_desc_base_lo, 0, (k * WGMMA::K) % BLOCK_ATOM_K, atom_k_idx * BLOCK_N * BLOCK_ATOM_K);
//                         WGMMA::wgmma(a_desc, b_desc, shifted_accum, 1);
//                     }
//                 }
// ```
// - 外层循环 `local_idx`：沿 M 方向分块，每个块交给一个 warp group 或 wave。
// - 内层循环 `k`：沿 K 方向分步，每次推进 `WGMMA::K`。
// - `advance_gmma_desc_lo`：根据当前 M 偏移、K 偏移和 atom 索引，重新计算描述符低 32 位。
// - `WGMMA::wgmma(a_desc, b_desc, shifted_accum, 1)`：执行一次 WGMMA，累加到 `accum`。

// ```cpp
//                 ptx::warpgroup_commit_batch();
//                 #pragma unroll
//                 for (uint32_t i = 0; i < WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M); ++ i)
//                     ptx::warpgroup_fence_operand(accum[i]);
//                 ptx::warpgroup_wait<0>();
// ```
// - `warpgroup_commit_batch()`：提交本批次 WGMMA。
// - 再次 fence 累加器。
// - `warpgroup_wait<0>()`：等待所有已提交的 WGMMA 完成。

// ```cpp
//                 empty_barrier_arrive(stage_idx);
// ```
// 通知 TMA 侧：当前 stage 已空，可以重新加载下一组数据。

// ---

// ## 12. Epilogue：把结果写回共享内存并 TMA store

// ```cpp
//             constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16);
//             constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_N : (kSwizzleDMode / kNumElemBytes);
//             constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;
//             DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
//             DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N / TMA_D_BLOCK_N <= 32,
//                             "Unaligned TMA store or too many TMA store instructions");
//             DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");
// ```
// - 计算 TMA store 时 N 方向每个 block 的大小。
// - 一系列静态断言保证 swizzle 与 TMA store 对齐合法。

// ```cpp
//             if (not do_wgmma_store)
//                 continue;
// ```
// 不参与写回的线程直接跳到下一个输出块。

// ```cpp
//             if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
//                 cute::tma_store_wait<0>();
//             cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 0);
// ```
// - 需要写回的前几个线程先等待上一次 TMA store 完成。
// - 然后命名屏障同步所有参与写回的线程。

// ### 12.1 BF16 输出路径

// ```cpp
//             if constexpr (cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>) {
//                 DG_STATIC_ASSERT(kSwizzleDMode > 0, "Invalid swizzling type");
//                 DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
//                 #pragma unroll
//                 for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
//                     auto m_offset = local_idx * WAVE_BLOCK_M;
//                     auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
//                     #pragma unroll
//                     for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
//                         uint8_t* smem_ptr = nullptr;
//                         if constexpr (kSwizzleDMode > 0) {
//                             constexpr uint32_t kNumBankGroupBytes = 16;
//                             auto atom_offset = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);
//                             auto bank_group_index = in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);
//                             constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;
//                             auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
//                             auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
//                             col ^= row % (kSwizzleDMode / 16);
//                             smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +
//                                 warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +
//                                 m_offset * kSwizzleDMode +
//                                 atom_offset * BLOCK_M * kSwizzleDMode +
//                                 row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
//                         } else {
//                             smem_ptr = reinterpret_cast<uint8_t*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx) * BLOCK_N + i * 8);
//                         }
//                         ptx::SM90_U32x2_STSM_N<nv_bfloat162>::copy(
//                             __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}),
//                             __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
//                             smem_ptr
//                         );
//                     }
//                 }
//             }
// ```
// - 当输出类型为 `bfloat16` 时，用 `STSM`（store shared-to-shared/multicast 的 PTX）把 `float` 累加器转成 `nv_bfloat162` 对并写入共享内存。
// - 通过 swizzle 公式计算每个线程应写的共享内存地址，避免 bank conflict。
// - 每 4 个 float 组成 2 个 `bfloat162`，用 `SM90_U32x2_STSM_N` 写入。

// ### 12.2 FP32 输出路径

// ```cpp
//             } else {
//                 #pragma unroll
//                 for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
//                     auto m_offset = local_idx * WAVE_BLOCK_M;
//                     auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
//                     auto smem_d_0 = reinterpret_cast<float2*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx / 4 + 0) * BLOCK_N + (lane_idx % 4) * 2);
//                     auto smem_d_1 = reinterpret_cast<float2*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx / 4 + 8) * BLOCK_N + (lane_idx % 4) * 2);
//                     #pragma unroll
//                     for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
//                         ptx::st_shared(smem_d_0 + i * 4, make_float2(shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]));
//                         ptx::st_shared(smem_d_1 + i * 4, make_float2(shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]));
//                     }
//                 }
//             }
// ```
// - 当输出类型为 `float` 时，直接用 `st.shared`（`ptx::st_shared`）把 float2 写入共享内存。
// - 地址按 warp/lane 分布，每个线程负责一部分行与列。

// ```cpp
//             cute::tma_store_fence();
//             cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 0);
// ```
// - `tma_store_fence()`：确保共享内存数据对 TMA store proxy 可见。
// - 再次同步。

// ### 12.3 TMA store 到全局内存

// ```cpp
//             const auto m_idx = scheduler.template get_global_idx<(not is_m_grouped_contiguous(kGemmType)), sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
//             DG_STATIC_ASSERT(kNumWGMMAStoreThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");
//             if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
//                 auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
//                 auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
//                 if constexpr (kGemmType == GemmType::Batched) {
//                     using cute_tma_t = cute::conditional_t<kWithAccumulation,
//                         cute::SM90_TMA_REDUCE_ADD_3D, cute::SM90_TMA_STORE_3D>;
//                     cute_tma_t::copy(&tensor_map_cd, smem_ptr,
//                                      n_block_idx * BLOCK_N + in_block_n_offset,
//                                      m_idx, scheduler.current_group_idx);
//                 } else {
//                     using cute_tma_t = cute::conditional_t<kWithAccumulation,
//                         cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
//                     cute_tma_t::copy(&tensor_map_cd, smem_ptr,
//                                      n_block_idx * BLOCK_N + in_block_n_offset, m_idx);
//                 }
//                 cute::tma_store_arrive();
//             }
//             __syncwarp();
//         }
//     }
// ```
// - 计算全局 M 偏移 `m_idx`。
// - 前 `BLOCK_N / TMA_D_BLOCK_N` 个线程各负责一列 TMA store block。
// - 对 Batched GEMM 使用 3D TMA store/reduce-add；否则使用 2D。
// - 若 `kWithAccumulation` 为真，使用 `SM90_TMA_REDUCE_ADD_*` 把结果累加到全局内存；否则直接 `STORE`。
// - `tma_store_arrive()` 通知 TMA store 事务已发起。
// - `__syncwarp()` 同步 math warp。

// ---

// ## 13. 非 SM90 兜底

// ```cpp
// #else
//     if (blockIdx.x == 0 and threadIdx.x == 0)
//         DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
// #endif
// }
// ```
// 如果不在 SM90 上编译，只有 block 0 的线程 0 触发断言，提示本 kernel 仅支持 sm_90a。

// ---

// ## 14. 命名空间与诊断恢复

// ```cpp
// };  // namespace deep_gemm

// #pragma clang diagnostic pop
// ```
// - 关闭 `deep_gemm` 命名空间。
// - 恢复之前保存的 clang 诊断状态。

// ---

// ## 总结

// 这个 kernel 把 CTA 的线程分成两部分：
// - **TMA warp group**：负责用 TMA 异步加载 A/B 到共享内存，并通过 mbarrier 通知 math 侧。
// - **Math warp group**：负责用 WGMMA 做矩阵乘累加，最后把累加器结果通过共享内存和 TMA store 写回全局内存。

// 核心流水线是经典的 producer-consumer 双缓冲/多缓冲：
// 1. TMA 等待 empty barrier → 加载 A/B → arrive full barrier。
// 2. Math 等待 full barrier → WGMMA 计算 → arrive empty barrier。
// 3. K 维循环结束后做 epilogue（BF16/FP32 分别处理），用 TMA store 写出 D。