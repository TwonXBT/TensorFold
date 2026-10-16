// L2 prefetch of weight ranges (tensorfold/families/glm_moe_dsa/cuda/forward.py, TF_GLM_PREFETCH): while a decode
// window's all-gather waits on the network (~15 us, the GPU otherwise idle), the side stream pulls the next consumer's
// weights into L2 with cp.async.bulk.prefetch.L2 (sm_90+), so that kernel reads them from L2 instead of DRAM. A hint
// only: no data moves to registers, nothing is written, and the kernel finishes as soon as the prefetches are issued.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

namespace {

constexpr int THREADS = 128;
constexpr long long CHUNK = 32768;       // bytes a prefetch instruction

__global__ void __launch_bounds__(THREADS) prefetch_kernel(const long long* __restrict__ table, int n) {
    // table: [n][2] (device pointer, bytes); chunks are dealt over every thread of every block in order
    long long total = 0;
    for (int t = 0; t < n; ++t) total += (table[2 * t + 1] + CHUNK - 1) / CHUNK;
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < total; c += step) {
        long long rem = c;
        int t = 0;
        while (true) {
            const long long chunks = (table[2 * t + 1] + CHUNK - 1) / CHUNK;
            if (rem < chunks) break;
            rem -= chunks;
            ++t;
        }
        const char* base = reinterpret_cast<const char*>(table[2 * t]);
        const long long bytes = table[2 * t + 1];
        const long long off = rem * CHUNK;
        const unsigned len = (unsigned)(bytes - off < CHUNK ? bytes - off : CHUNK) & ~15u;
        if (len) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(base + off), "r"(len) : "memory");
    }
}

}  // namespace

// table: int64 [n, 2] on the device, (pointer, bytes) a row, bytes a multiple of 16 (the tail is dropped otherwise).
void prefetch(const at::Tensor& table, int64_t n, int64_t blocks) {
    TORCH_CHECK(table.is_cuda() && table.scalar_type() == at::kLong && table.dim() == 2 && table.size(1) == 2 &&
                    table.is_contiguous() && n >= 1 && n <= table.size(0),
                "l2 prefetch: table int64 [n, 2] on the device");
    c10::cuda::CUDAGuard guard(table.device());
    prefetch_kernel<<<(unsigned)std::max<int64_t>(1, blocks), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        table.data_ptr<int64_t>(), (int)n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prefetch", &prefetch, "prefetch (pointer, bytes) ranges into L2");
}
