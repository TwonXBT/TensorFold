// Row top-k for GLM-5.3's DSA token selection (tensorfold/families/glm_moe_dsa/cuda/select.py): one block a row.
//
// Row r's NP fp32 scores as select._ordered's uint32 keys; the K highest keys, ties to the lower index, written in
// ascending index order to TOK[r, :K] (TOK[r, K:width] = -1), CNT[r] = K when the row sees more than topk tokens
// (pos + r + 1 > topk) else 0: the set and order torch.topk + sort produce on select._order_key, and what the
// _split_* / _radix_topk launches produce — in one launch. With PMAP, TOK gets PMAP[r, i] instead of i (the split
// selection's merge: candidates in position order, so the output stays ascending).
//
// Radix select with 11-bit digits: a histogram pass over the row finds the digit of the K-th key; once the K-th
// key's bin holds at most CAP keys they move to shared memory and the remaining digits resolve there (the row is read
// twice in the common case: the first histogram and the compaction), then one pass in index order keeps every key
// above the K-th and the first ``need`` keys equal to it.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

namespace {

constexpr int THREADS = 1024;
constexpr int BINS = 2048;            // 11-bit digits (11 + 11 + 10 = 32)
constexpr int CAP = 4096;             // candidate keys kept in shared memory once the K-th key's bin is this small

__device__ __forceinline__ unsigned ordered(float s) {
    const unsigned b = __float_as_uint(s + 0.0f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// exclusive prefix sum of one int a thread over the block; returns the thread's prefix, *total the block's sum
__device__ __forceinline__ int block_exclusive(int v, int* warp_sums, int* total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_sums[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int s = lane < (THREADS / 32) ? warp_sums[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += y;
        }
        if (lane < (THREADS / 32)) warp_sums[lane] = s;                  // inclusive over warps
    }
    __syncthreads();
    const int before = (warp ? warp_sums[warp - 1] : 0) + x - v;
    if (total) *total = warp_sums[THREADS / 32 - 1];
    __syncthreads();                                                     // warp_sums reused by the next scan
    return before;
}

__global__ void __launch_bounds__(THREADS) topk_rows_kernel(const float* __restrict__ S, long long s_stride, int NP,
                                                             int K, int* __restrict__ TOK, long long t_stride,
                                                             int width, const int* __restrict__ PMAP,
                                                             long long p_stride, int* __restrict__ CNT,
                                                             const int* __restrict__ POS, int topk) {
    __shared__ int hist[BINS];
    __shared__ unsigned cand[CAP];
    __shared__ int warp_sums[THREADS / 32];
    __shared__ int s_pick, s_ncand, s_total, s_eqtotal;

    const int r = blockIdx.x, tid = threadIdx.x;
    const float* row = S + (long long)r * s_stride;
    int* out_row = TOK + (long long)r * t_stride;
    const bool vec4 = (s_stride % 4 == 0) && ((reinterpret_cast<uintptr_t>(S) & 15) == 0) && (NP % 4 == 0);

    unsigned prefix = 0u, fixed = 0u;
    int need = K, shift = 32;
    bool from_shared = false;
    int ncand = 0;

    while (true) {
        const int bits = shift >= 11 ? 11 : shift;
        shift -= bits;
        const unsigned mask = (1u << bits) - 1u;
        for (int b = tid; b < BINS; b += THREADS) hist[b] = 0;
        if (tid == 0) s_pick = BINS;
        __syncthreads();
        if (from_shared) {
            for (int i = tid; i < ncand; i += THREADS) {
                const unsigned key = cand[i];
                if ((key & fixed) == prefix) atomicAdd(&hist[(key >> shift) & mask], 1);
            }
        } else if (vec4) {
            const float4* row4 = reinterpret_cast<const float4*>(row);
            const int n4 = NP / 4;
            for (int i = tid; i < n4; i += THREADS) {
                const float4 v = __ldg(row4 + i);
                const unsigned k0 = ordered(v.x), k1 = ordered(v.y), k2 = ordered(v.z), k3 = ordered(v.w);
                if ((k0 & fixed) == prefix) atomicAdd(&hist[(k0 >> shift) & mask], 1);
                if ((k1 & fixed) == prefix) atomicAdd(&hist[(k1 >> shift) & mask], 1);
                if ((k2 & fixed) == prefix) atomicAdd(&hist[(k2 >> shift) & mask], 1);
                if ((k3 & fixed) == prefix) atomicAdd(&hist[(k3 >> shift) & mask], 1);
            }
        } else {
            for (int i = tid; i < NP; i += THREADS) {
                const unsigned key = ordered(__ldg(row + i));
                if ((key & fixed) == prefix) atomicAdd(&hist[(key >> shift) & mask], 1);
            }
        }
        __syncthreads();
        // the digit d of the K-th key: the largest bin whose suffix count reaches ``need``. Thread t holds the
        // pair of bins (BINS - 1 - 2t, BINS - 2 - 2t), descending; the exclusive scan over t sums the bins above
        const int ba = BINS - 1 - 2 * tid, bb = ba - 1;
        const int ha = hist[ba], hb = hist[bb];
        const int above_pair = block_exclusive(ha + hb, warp_sums, nullptr);
        int pick = BINS;                                        // position in descending order: 2t or 2t + 1
        if (above_pair + ha >= need) pick = 2 * tid;
        else if (above_pair + ha + hb >= need) pick = 2 * tid + 1;
        if (pick != BINS) atomicMin(&s_pick, pick);
        __syncthreads();
        const int pk = s_pick;
        const int d = BINS - 1 - pk;
        const int above_d = (pk == 2 * tid) ? above_pair + ha : (pk == 2 * tid + 1) ? above_pair + ha + hb : 0;
        // every thread needs above_d and hist[d]: the owner publishes
        if (pk == 2 * tid || pk == 2 * tid + 1) s_total = above_d;
        __syncthreads();
        const int at_d = hist[d];
        need -= s_total - at_d;
        prefix |= (unsigned)d << shift;
        fixed |= mask << shift;
        __syncthreads();
        if (shift == 0) break;
        if (!from_shared && at_d <= CAP) {
            // the keys under the prefix so far move to shared memory (order irrelevant)
            if (tid == 0) s_ncand = 0;
            __syncthreads();
            if (vec4) {
                const float4* row4 = reinterpret_cast<const float4*>(row);
                const int n4 = NP / 4;
                for (int i = tid; i < n4; i += THREADS) {
                    const float4 v = __ldg(row4 + i);
                    const unsigned ks[4] = {ordered(v.x), ordered(v.y), ordered(v.z), ordered(v.w)};
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if ((ks[j] & fixed) == prefix) cand[atomicAdd(&s_ncand, 1)] = ks[j];
                }
            } else {
                for (int i = tid; i < NP; i += THREADS) {
                    const unsigned key = ordered(__ldg(row + i));
                    if ((key & fixed) == prefix) cand[atomicAdd(&s_ncand, 1)] = key;
                }
            }
            __syncthreads();
            ncand = s_ncand;
            from_shared = true;
        }
    }
    const unsigned T = prefix;                                  // the K-th key; ``need`` of its ties are kept
    // the pass in index order: every key above T, and the first ``need`` keys equal to it
    int out = 0, ties = 0;
    const int CH = THREADS * 4;
    for (int base = 0; base < NP; base += CH) {
        unsigned ks[4];
        int valid = 0;
        const int i0 = base + 4 * tid;
        if (vec4 && i0 + 4 <= NP) {
            const float4 v = __ldg(reinterpret_cast<const float4*>(row + i0));
            ks[0] = ordered(v.x); ks[1] = ordered(v.y); ks[2] = ordered(v.z); ks[3] = ordered(v.w);
            valid = 4;
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (i0 + j < NP) { ks[j] = ordered(__ldg(row + i0 + j)); valid = j + 1; }
                else ks[j] = 0u;
            }
        }
        int ngt = 0, neq = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (j < valid) {
                ngt += ks[j] > T;
                neq += ks[j] == T;
            }
        }
        int eq_total = 0;
        const int eq_before = block_exclusive(neq, warp_sums, &eq_total);
        // taken among this thread's keys: the ones above T, plus ties while their rank stays under ``need``
        int take_mask = 0, ntake = 0, erank = ties + eq_before;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (j < valid) {
                bool t = ks[j] > T;
                if (ks[j] == T) { t = erank < need; ++erank; }
                if (t) { take_mask |= 1 << j; ++ntake; }
            }
        }
        int take_total = 0;
        int slot = out + block_exclusive(ntake, warp_sums, &take_total);
        if (PMAP) {
            const int* pm = PMAP + (long long)r * p_stride;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if (take_mask & (1 << j)) out_row[slot++] = pm[i0 + j];
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if (take_mask & (1 << j)) out_row[slot++] = i0 + j;
        }
        out += take_total;
        ties += eq_total;
    }
    for (int i = K + tid; i < width; i += THREADS) out_row[i] = -1;
    if (tid == 0 && CNT) CNT[r] = (POS ? (POS[0] + r + 1 > topk) : true) ? K : 0;
}

}  // namespace

// TOK [R, width] int32 <- the K highest keys of each row of S [R, NP] fp32 in ascending index (or PMAP) order,
// -1 past K; CNT [R] (optional) K or 0 by POS (device int32 [1]: the first row's position) and topk.
void topk_rows(const at::Tensor& S, at::Tensor TOK, int64_t K, int64_t NP, const c10::optional<at::Tensor>& PMAP,
               const c10::optional<at::Tensor>& CNT, const c10::optional<at::Tensor>& POS, int64_t topk) {
    TORCH_CHECK(S.is_cuda() && S.scalar_type() == at::kFloat && S.dim() == 2 && S.stride(1) == 1,
                "topk_rows: S [R, NP] fp32 with unit column stride");
    TORCH_CHECK(TOK.is_cuda() && TOK.scalar_type() == at::kInt && TOK.dim() == 2 && TOK.stride(1) == 1,
                "topk_rows: TOK [R, width] int32 with unit column stride");
    const int64_t R = S.size(0), width = TOK.size(1);
    TORCH_CHECK(TOK.size(0) == R && NP >= 1 && NP <= S.size(1) && K >= 1 && K <= NP && K <= width,
                "topk_rows: K within NP and the token width");
    const int* pmap = nullptr;
    int64_t p_stride = 0;
    if (PMAP) {
        TORCH_CHECK(PMAP->is_cuda() && PMAP->scalar_type() == at::kInt && PMAP->dim() == 2 && PMAP->stride(1) == 1 &&
                        PMAP->size(0) == R && PMAP->size(1) >= NP,
                    "topk_rows: PMAP [R, >= NP] int32 with unit column stride");
        pmap = PMAP->data_ptr<int>();
        p_stride = PMAP->stride(0);
    }
    int* cnt = nullptr;
    const int* pos = nullptr;
    if (CNT) {
        TORCH_CHECK(CNT->is_cuda() && CNT->scalar_type() == at::kInt && CNT->numel() >= R, "topk_rows: CNT [R] int32");
        cnt = CNT->data_ptr<int>();
        if (POS) {
            TORCH_CHECK(POS->is_cuda() && POS->scalar_type() == at::kInt && POS->numel() >= 1,
                        "topk_rows: POS int32 [1]");
            pos = POS->data_ptr<int>();
        }
    }
    c10::cuda::CUDAGuard guard(S.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    topk_rows_kernel<<<(unsigned)R, THREADS, 0, stream>>>(S.data_ptr<float>(), S.stride(0), (int)NP, (int)K,
                                                           TOK.data_ptr<int>(), TOK.stride(0), (int)width, pmap,
                                                           p_stride, cnt, pos, (int)topk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_rows", &topk_rows, "row top-k of fp32 scores, ties to the lower index, ascending indices");
}
