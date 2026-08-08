// Tensor-core FlashAttention-style forward kernel (wmma m16n16k16, BF16).
// One block per (query tile, batch*head); BLOCK_M=128 query rows, 8 warps
// (16 rows each); K/V tiles of BLOCK_N=32 staged via cp.async double buffer.
// Supports MHA and GQA (kv_heads <= heads), causal and non-causal, any S.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <math.h>

namespace {

using namespace nvcuda;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 32;
constexpr int HEAD_DIM = 128;
constexpr int NUM_WARPS = 8;
constexpr int NUM_THREADS = NUM_WARPS * 32;
constexpr int PADDED_D = HEAD_DIM + 8;         // pad rows to avoid bank conflicts
constexpr int TILE_ELEMS = BLOCK_N * PADDED_D;
constexpr int SMEM_BYTES = 4 * TILE_ELEMS * 2;  // K0,V0,K1,V1 only: Q/O live in regs

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void load_tile(const __nv_bfloat16* src,
                                          __nv_bfloat16* dst, int row0,
                                          int rows, int seq_len, bool zero_fill) {
    constexpr int CHUNKS_PER_ROW = HEAD_DIM / 8;
    for (int i = threadIdx.x; i < rows * CHUNKS_PER_ROW; i += NUM_THREADS) {
        const int row = i / CHUNKS_PER_ROW, col8 = i % CHUNKS_PER_ROW;
        const int grow = row0 + row;
        __nv_bfloat16* s = dst + row * PADDED_D + col8 * 8;
        if (grow < seq_len) cp_async16(s, src + (long long)grow * HEAD_DIM + col8 * 8);
        else if (zero_fill) *reinterpret_cast<uint4*>(s) = make_uint4(0, 0, 0, 0);
    }
}

__device__ __forceinline__ float warp_max4(float v) {
    for (int off = 1; off < 4; off <<= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}
__device__ __forceinline__ float warp_sum4(float v) {
    for (int off = 1; off < 4; off <<= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

template<bool CAUSAL>
__global__ void __launch_bounds__(NUM_THREADS, 2)
attention_fwd_kernel(const __nv_bfloat16* __restrict__ q,
                     const __nv_bfloat16* __restrict__ k,
                     const __nv_bfloat16* __restrict__ v,
                     __nv_bfloat16* __restrict__ out,
                     int seq_len, int kv_heads, int heads, float scale)
{
    extern __shared__ char smem[];
    __nv_bfloat16* sK0 = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sV0 = sK0 + TILE_ELEMS;
    __nv_bfloat16* sK1 = sV0 + TILE_ELEMS;
    __nv_bfloat16* sV1 = sK1 + TILE_ELEMS;

    const int m_tile = blockIdx.x, bh = blockIdx.y;
    const int b = bh / heads;                 // batch index
    const int qh = bh % heads;                // query head within batch
    const int kv_head = qh / (heads / kv_heads);   // GQA: kv head serving this q head
    const long long qbase = (long long)bh * seq_len * HEAD_DIM;
    const long long kvbase = (long long)(b * kv_heads + kv_head) * seq_len * HEAD_DIM;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int m0 = m_tile * BLOCK_M + warp * 16;
    const int m0b = m_tile * BLOCK_M;

    // Kick off the K/V tile-0 prefetch first so the Q register loads below
    // overlap with the cp.async transfer.
    const int n_tiles_total = (seq_len + BLOCK_N - 1) / BLOCK_N;
    const int n_tiles_block = CAUSAL ? min(n_tiles_total, (m0b + BLOCK_M - 1) / BLOCK_N + 1)
                                     : n_tiles_total;
    if (n_tiles_block > 0) {
        load_tile(k + kvbase, sK0, 0, BLOCK_N, seq_len, true);
        load_tile(v + kvbase, sV0, 0, BLOCK_N, seq_len, true);
        cp_async_commit();
    }

    // Q fragments loaded directly from global memory into registers (never
    // staged in smem). Fragment layout (m16n16k16, A row-major, 8 elems/thread):
    //   rr = (j&2 ? 8 : 0) + lane/4,  cc = 2*(lane%4) + (j&1) + (j&4 ? 8 : 0)
    // Pairs (j=2h, 2h+1) share a row and are adjacent columns -> one 32-bit load.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag[8];
    for (int kt = 0; kt < 8; ++kt) {
        const int kc = kt * 16;
        for (int h = 0; h < 4; ++h) {
            const int rr = (h & 1) * 8 + lane / 4;
            const int cc = 2 * (lane % 4) + ((h >> 1) * 8);
            const int grow = m0 + rr;
            uint32_t val = 0;
            if (grow < seq_len)
                val = *reinterpret_cast<const uint32_t*>(
                    q + qbase + (long long)grow * HEAD_DIM + kc + cc);
            const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&val);
            q_frag[kt].x[2 * h] = p[0];
            q_frag[kt].x[2 * h + 1] = p[1];
        }
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[8];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> v_frag;
    for (int i = 0; i < 8; ++i) wmma::fill_fragment(o_frag[i], 0.f);

    float m_prev[2] = {-INFINITY, -INFINITY};
    float l_prev[2] = {0.f, 0.f};

    for (int nt = 0; nt < n_tiles_block; ++nt) {
        const int buf = nt & 1;
        __nv_bfloat16* sK = buf ? sK1 : sK0;
        __nv_bfloat16* sV = buf ? sV1 : sV0;
        cp_async_wait<0>();
        __syncthreads();   // all threads done with buffer (nt-1)&1 before we overwrite it
        if (nt + 1 < n_tiles_block) {
            // prefetch tile nt+1 into the buffer just consumed; overlaps this tile's mma
            load_tile(k + kvbase, buf ? sK0 : sK1, (nt + 1) * BLOCK_N, BLOCK_N, seq_len, true);
            load_tile(v + kvbase, buf ? sV0 : sV1, (nt + 1) * BLOCK_N, BLOCK_N, seq_len, true);
            cp_async_commit();
        }
        const int kv0 = nt * BLOCK_N;

        // Both 16-col subtiles of this 32-col K tile are computed first, then a
        // single online-softmax update covers the full 32 columns: one row max,
        // one correction pair, one row sum. (Two subtile-local updates would
        // apply a second correction to o_frag/l_prev -- 2 extra exps, 64+ muls.)
        // A subtile entirely above the causal diagonal is skipped: its elements
        // stay -inf so the merged max/sum treats them as zero contribution.
        const bool skip0 = CAUSAL && (kv0 > m0 + 15);
        const bool skip1 = CAUSAL && (kv0 + 16 > m0 + 15);
        if (!(skip0 && skip1)) {
            for (int i = 0; i < 2; ++i) {
                const bool skip = i ? skip1 : skip0;
                for (int j = 0; j < 8; ++j) s_frag[i].x[j] = skip ? -INFINITY : 0.f;
                if (skip) continue;
                const int kvc = kv0 + 16 * i;
                const bool may_mask = CAUSAL && (kvc + 15 >= m0) && (kvc <= m0 + 15);
                const bool kv_oob = (kvc + 15 >= seq_len);   // last tile may extend past seq_len
                for (int kt = 0; kt < 8; ++kt) {
                    wmma::load_matrix_sync(k_frag, sK + (16 * i) * PADDED_D + kt * 16, PADDED_D);
                    wmma::mma_sync(s_frag[i], q_frag[kt], k_frag, s_frag[i]);
                }
                for (int j = 0; j < 8; ++j) s_frag[i].x[j] *= scale;
                if (may_mask || kv_oob) {
                    for (int j = 0; j < 8; ++j) {
                        const int rr = ((j & 2) ? 8 : 0) + lane / 4;
                        const int cc = 2 * (lane % 4) + (j & 1) + ((j & 4) ? 8 : 0);
                        if ((CAUSAL && kvc + cc > m0 + rr) || (kvc + cc >= seq_len))
                            s_frag[i].x[j] = -INFINITY;
                    }
                }
            }
            float tm0 = fmaxf(fmaxf(s_frag[0].x[0], s_frag[0].x[1]),
                              fmaxf(s_frag[0].x[4], s_frag[0].x[5]));
            float tm1 = fmaxf(fmaxf(s_frag[0].x[2], s_frag[0].x[3]),
                              fmaxf(s_frag[0].x[6], s_frag[0].x[7]));
            tm0 = fmaxf(tm0, fmaxf(fmaxf(s_frag[1].x[0], s_frag[1].x[1]),
                                   fmaxf(s_frag[1].x[4], s_frag[1].x[5])));
            tm1 = fmaxf(tm1, fmaxf(fmaxf(s_frag[1].x[2], s_frag[1].x[3]),
                                   fmaxf(s_frag[1].x[6], s_frag[1].x[7])));
            tm0 = warp_max4(tm0); tm1 = warp_max4(tm1);
            const float mn0 = fmaxf(m_prev[0], tm0), mn1 = fmaxf(m_prev[1], tm1);
            const float corr0 = __expf(m_prev[0] - mn0), corr1 = __expf(m_prev[1] - mn1);
            for (int o = 0; o < 8; ++o)
                for (int j = 0; j < 8; ++j)
                    o_frag[o].x[j] *= ((j & 2) ? corr1 : corr0);
            float ps0 = 0.f, ps1 = 0.f;
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 8; ++j) {
                    const float p = __expf(s_frag[i].x[j] - ((j & 2) ? mn1 : mn0));
                    s_frag[i].x[j] = p;
                    if (j & 2) ps1 += p; else ps0 += p;
                }
            ps0 = warp_sum4(ps0); ps1 = warp_sum4(ps1);
            l_prev[0] = l_prev[0] * corr0 + ps0;
            l_prev[1] = l_prev[1] * corr1 + ps1;
            m_prev[0] = mn0; m_prev[1] = mn1;
            // P·V mma: convert P to a bf16 fragment in registers (layout of the
            // accumulator matches matrix_a row_major) and mma directly -- no
            // shared round-trip. Correction factors were applied to o_frag above.
            for (int i = 0; i < 2; ++i) {
                if (i ? skip1 : skip0) continue;   // all -inf: would poison the mma
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> p_frag;
                for (int j = 0; j < 8; ++j) p_frag.x[j] = __float2bfloat16(s_frag[i].x[j]);
                for (int ot8 = 0; ot8 < 8; ++ot8) {
                    wmma::load_matrix_sync(v_frag, sV + (16 * i) * PADDED_D + ot8 * 16, PADDED_D);
                    wmma::mma_sync(o_frag[ot8], p_frag, v_frag, o_frag[ot8]);
                }
            }
        }
    }

    // Write O directly from accumulator registers to global memory (pairs of
    // adjacent columns share a row and rescale factor -> one 32-bit store).
    const float inv0 = 1.f / l_prev[0], inv1 = 1.f / l_prev[1];
    for (int ot8 = 0; ot8 < 8; ++ot8) {
        const int d0 = ot8 * 16;
        for (int h = 0; h < 4; ++h) {
            const int rr = (h & 1) * 8 + lane / 4;
            const int cc = 2 * (lane % 4) + ((h >> 1) * 8);
            const int grow = m0 + rr;
            if (grow < seq_len) {
                const float s0 = o_frag[ot8].x[2 * h] * ((h & 1) ? inv1 : inv0);
                const float s1 = o_frag[ot8].x[2 * h + 1] * ((h & 1) ? inv1 : inv0);
                const uint32_t val = (uint32_t)__bfloat16_as_ushort(__float2bfloat16(s0)) |
                                     ((uint32_t)__bfloat16_as_ushort(__float2bfloat16(s1)) << 16);
                *reinterpret_cast<uint32_t*>(
                    out + qbase + (long long)grow * HEAD_DIM + d0 + cc) = val;
            }
        }
    }
}

}  // namespace

torch::Tensor attention_forward(torch::Tensor q, torch::Tensor k,
                                torch::Tensor v, bool causal)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16 &&
                k.scalar_type() == torch::kBFloat16 &&
                v.scalar_type() == torch::kBFloat16, "q/k/v must be bfloat16");
    TORCH_CHECK(q.dim() == 4, "expected [B, H, S, D]");
    TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "k/v must be [B, kv_heads, S, D]");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(2) == k.size(2) &&
                q.size(3) == k.size(3) && k.size(3) == HEAD_DIM,
                "shape mismatch");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "H must be a multiple of kv_heads");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q/k/v must be contiguous");

    const at::cuda::CUDAGuard guard(q.device());
    const int B = q.size(0), H = q.size(1), S = q.size(2);
    const int kv_heads = k.size(1);
    TORCH_CHECK((long long)B * H <= 65535, "B*H exceeds grid.y limit");

    auto out = torch::empty_like(q);
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    dim3 grid((S + BLOCK_M - 1) / BLOCK_M, B * H);

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(attention_fwd_kernel<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        cudaFuncSetAttribute(attention_fwd_kernel<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        attr_set = true;
    }

    auto launch = [&](auto kernel) {
        kernel<<<grid, NUM_THREADS, SMEM_BYTES, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            S, kv_heads, H, scale);
    };
    if (causal) launch(attention_fwd_kernel<true>);
    else launch(attention_fwd_kernel<false>);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
