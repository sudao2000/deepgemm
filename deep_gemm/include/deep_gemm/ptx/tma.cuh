#pragma once

#include <cutlass/arch/barrier.h>
#include <cute/arch/copy_sm90_desc.hpp>

namespace deep_gemm::ptx {

// Tensor-map instructions
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

CUTLASS_DEVICE void tensor_map_replace_global_inner_dim_stride_in_smem(cute::TmaDescriptor* smem_desc, const uint32_t& new_dim, const uint64_t& new_stride) {
    auto smem_int_desc = __cvta_generic_to_shared(smem_desc);
    asm volatile ("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0, %1;" :: "l"(smem_int_desc), "r"(new_dim));
#if ((__CUDACC_VER_MAJOR__ > 12) or ((__CUDACC_VER_MAJOR__ == 12) and (__CUDACC_VER_MINOR__ >= 3)))
    asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0, %1;" :: "l"(smem_int_desc), "l"(new_stride));
#else
    DG_STATIC_ASSERT(false, "Invalid CUDA version");
#endif
}

/// TMA instructions
CUTLASS_DEVICE void mbarrier_arrive(
    cutlass::arch::ClusterTransactionBarrier* ptr) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0]; \n\t" ::
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));
}

CUTLASS_DEVICE void mbarrier_arrive_and_set_tx(
    cutlass::arch::ClusterTransactionBarrier* ptr, const uint32_t& num_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" ::
                 "r"(num_bytes), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));
}

CUTLASS_DEVICE void mbarrier_wait_and_flip_phase(
    cutlass::arch::ClusterTransactionBarrier* ptr, uint32_t& phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra DONE; \n\t"
        "bra     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}" ::
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))),
        "r"(phase), "r"(0x989680));
    phase ^= 1;
}

CUTLASS_DEVICE void tma_load_1d(
    const void* dst_ptr, const void* src_ptr,
    cutlass::arch::ClusterTransactionBarrier* mbarrier_ptr,
    const uint32_t& num_bytes,
    const cute::TMA::CacheHintSm90& hint = cute::TMA::CacheHintSm90::EVICT_FIRST) {
    // NOTES: normally, the loaded part will be evicted soon
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_ptr))),
        "l"(src_ptr),
        "r"(num_bytes),
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbarrier_ptr))),
        "l"(hint)
        : "memory");
}

CUTLASS_DEVICE void tma_store_1d(
    const void* dst_ptr, const void* src_ptr, const uint32_t& num_bytes,
    const cute::TMA::CacheHintSm90& hint = cute::TMA::CacheHintSm90::EVICT_NORMAL) {
    // NOTES: normally, the stored part will be used soon
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;\n" ::
                 "l"(dst_ptr),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src_ptr))),
                 "r"(num_bytes),
                 "l"(hint)
                 : "memory");
}

template <int kNumRemainingWaits = 0>
__forceinline__ __device__ void tma_store_wait() {
    // NOTES: this function does not have `.read`
    asm volatile("cp.async.bulk.wait_group %0;" ::"n"(kNumRemainingWaits) : "memory");
}

CUTLASS_DEVICE
void tma_gather4(const void* desc_ptr, cutlass::arch::ClusterTransactionBarrier& mbarrier,
                 void* smem_ptr, const uint32_t& col_idx, const int4& row_idxs, const uint64_t& cache_hint) {
    const auto smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    const auto mbarrier_addr = cute::cast_smem_ptr_to_uint(&mbarrier);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
        :
        : "r"(smem_addr), "l"(desc_ptr), "r"(col_idx),
          "r"(row_idxs.x), "r"(row_idxs.y), "r"(row_idxs.z), "r"(row_idxs.w),
          "r"(mbarrier_addr), "l"(cache_hint)
        : "memory"
    );
}

} // namespace deep_gemm::ptx

// `deep_gemm/include/deep_gemm/ptx/tma.cuh` 里封装了 Hopper 架构下 TMA（Tensor Memory Accelerator）和 mbarrier 相关的 PTX 内联汇编函数。它们都位于 `deep_gemm::ptx` 命名空间，主要供 kernel 内部做异步拷贝、同步和 tensor map 修改。

// 按文件顺序逐个函数解释：

// 1. **`tensor_map_release_gpu()`**
//    - 发出 `fence.proxy.tensormap::generic.release.gpu;`。
//    - 这是一个 GPU 侧的 tensormap proxy release fence，确保当前线程对 tensormap 描述符的更新对后续需要 acquire 的线程可见。

// 2. **`tensor_map_acquire_gpu(const cute::TmaDescriptor* gmem_desc_ptr)`**
//    - 执行 `fence.proxy.tensormap::generic.acquire.gpu [%0], 128;`。
//    - 从全局内存（`gmem_desc_ptr`）获取大小为 128 字节的 tensormap 描述符，并保证后续的 tensormap 操作能看到这份描述符。

// 3. **`tensor_map_replace_global_addr_in_smem(cute::TmaDescriptor* smem_desc, const void* new_addr)`**
//    - 在共享内存中的 tensormap 描述符里，把全局地址替换为 `new_addr`。
//    - 使用 PTX 指令 `tensormap.replace.tile.global_address.shared::cta.b1024.b64`。
//    - 常用于动态改变 TMA 拷贝源地址，例如 MoE 中每次选不同 expert 的权重矩阵。

// 4. **`tensor_map_replace_global_inner_dim_stride_in_smem(...)`**
//    - 替换共享内存 tensormap 描述符中“最内层维度”的维度大小（`global_dim`）和步长（`global_stride`）。
//    - 使用了 `tensormap.replace.tile.global_dim.shared::cta.b1024.b32` 和 `global_stride` 的 64-bit 版本。
//    - 包含 CUDA 版本检查：要求 CUDA 12.3 及以上，否则通过 `DG_STATIC_ASSERT` 触发编译错误。

// 5. **`mbarrier_arrive(cutlass::arch::ClusterTransactionBarrier* ptr)`**
//    - 对共享内存中的 mbarrier 执行 `mbarrier.arrive.shared::cta.b64`。
//    - 通知一次到达，不指定预期事务字节数。

// 6. **`mbarrier_arrive_and_set_tx(..., const uint32_t& num_bytes)`**
//    - 执行 `mbarrier.arrive.expect_tx.shared::cta.b64`。
//    - 到达 mbarrier 并声明本次 TMA 事务预期传输的字节数 `num_bytes`，供 mbarrier 等待时追踪完成状态。

// 7. **`mbarrier_wait_and_flip_phase(...)`**
//    - 自旋等待 mbarrier 可用，使用 `mbarrier.try_wait.parity.shared::cta.b64` 检查 phase 奇偶性。
//    - 等待成功后把 `phase` 变量异或 1 翻转，下一次等待相反的 parity。
//    - 循环中的 `0x989680`（十进制 10,000,000）是等待超时阈值参数。

// 8. **`tma_load_1d(...)`**
//    - 一维 TMA 异步加载：从全局内存 `src_ptr` 拷贝 `num_bytes` 字节到共享内存 `dst_ptr`。
//    - 使用 `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint`。
//    - 默认 cache hint 为 `EVICT_FIRST`，提示加载的数据很快会被用完，可优先驱逐。

// 9. **`tma_store_1d(...)`**
//    - 一维 TMA 异步存储：从共享内存 `src_ptr` 拷贝 `num_bytes` 字节到全局内存 `dst_ptr`。
//    - 使用 `cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint`。
//    - 默认 cache hint 为 `EVICT_NORMAL`，提示存储的数据后续可能还会被使用。

// 10. **`tma_store_wait<kNumRemainingWaits = 0>()`**
//     - 等待 TMA store 完成。
//     - 发出 `cp.async.bulk.wait_group %0;`，参数 `kNumRemainingWaits` 表示允许仍有多少个未完成的 bulk group。
//     - 注意这个版本没有 `.read` 后缀，用于等待写回全局内存的操作。

// 11. **`tma_gather4(...)`**
//     - 2D gather4 TMA 加载：根据一个列索引 `col_idx` 和 4 个行索引 `row_idxs`，从全局 tensor 中 gather 4 个元素到共享内存 `smem_ptr`。
//     - 使用 `cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint`。
//     - 通过 `mbarrier` 追踪事务完成，可传入 `cache_hint` 控制 L2 缓存行为。

// 这些函数都直接映射到 Hopper 的 PTX 指令，上层 kernel 调用它们完成高效的异步内存搬运、动态 tensor map 修改以及集群/CTA 级别的同步。