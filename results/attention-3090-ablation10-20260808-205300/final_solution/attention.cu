// FlashAttention-2-style BF16 forward kernel for sm_86 (RTX 3090 Ti).
// - wmma 16x16x16 tensor-core matmuls (mma.sync.m16n8k16 under the hood)
// - cp.async single-buffered K/V tiles (39,936 B smem -> 2 blocks/SM on sm_86,
//   which doubles warps-in-flight vs a 96 KB double-buffered layout);
//   padded row stride (136 bf16) makes ldmatrix reads conflict-free
// - online softmax; softmaxed scores are staged as bf16 through a small
//   per-warp shared tile (true row/col layout) before the PV matmul, because
//   the m16n16k16 accumulator layout differs from the matrix_a layout.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <math.h>

namespace {

using namespace nvcuda;

constexpr int BLOCK_M = 64;    // query rows per block
constexpr int BLOCK_N = 32;    // kv rows per tile
constexpr int HEAD_DIM = 128;
constexpr int NUM_WARPS = 4;
constexpr int NUM_THREADS = NUM_WARPS * 32;
constexpr int PADDED_D = HEAD_DIM + 8;        // 16 B/row padding (K/V tiles)
constexpr int PADDED_Q = HEAD_DIM + 8;        // Q/O buffer stride == PADDED_D
constexpr int TILE_ELEMS = BLOCK_N * PADDED_D;  // per K or V tile (bf16)
constexpr int O_ELEMS = BLOCK_M * PADDED_Q;     // O staging (reuses Q buffer)
// Single-buffered K/V (2 tiles) + compact Q/O buffer keeps smem at 25,600 B
// -> 4 blocks/SM on sm_86 (16 warps vs 12) on this latency-bound kernel.
// P is fed to the PV matmul directly from registers (wmma m16n16k16 matrix_a
// row_major layout == accumulator layout), no smem round-trip.
constexpr int SMEM_BYTES = 2 * TILE_ELEMS * 2;  // 17,408 B (no Q/O buffers)

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// Copy an nrows x HEAD_DIM tile from gmem (rows row0..) to padded shared,
// zero-filling rows past seq_len. Used for K/V tiles (nrows = BLOCK_N) and
// to stage Q (nrows = BLOCK_M).
__device__ __forceinline__ void load_tile(
    const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst,
    int row0, int seq_len, int nrows, int pitch)
{
    const int CHUNKS = nrows * HEAD_DIM / 8;  // 16B chunks
#pragma unroll
    for (int i = threadIdx.x; i < CHUNKS; i += NUM_THREADS) {
        const int row = i / (HEAD_DIM / 8);
        const int col8 = i % (HEAD_DIM / 8);
        const int grow = row0 + row;
        __nv_bfloat16* s = dst + row * pitch + col8 * 8;
        if (grow < seq_len) {
            cp_async16(s, src + (long long)grow * HEAD_DIM + col8 * 8);
        } else {
            *reinterpret_cast<uint4*>(s) = make_uint4(0, 0, 0, 0);
        }
    }
}

__device__ __forceinline__ float warp_max4(float v) {
#pragma unroll
    for (int off = 1; off < 4; off <<= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}
__device__ __forceinline__ float warp_sum4(float v) {
#pragma unroll
    for (int off = 1; off < 4; off <<= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__global__ void __launch_bounds__(NUM_THREADS, 4)
attention_fwd_kernel(const __nv_bfloat16* __restrict__ q,
                     const __nv_bfloat16* __restrict__ k,
                     const __nv_bfloat16* __restrict__ v,
                     __nv_bfloat16* __restrict__ out,
                     int seq_len, float scale, bool causal,
                     int num_heads, int kv_heads)
{
    extern __shared__ char smem[];
    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sV = sK + TILE_ELEMS;

    const int m_tile = causal ? (gridDim.x - 1 - blockIdx.x) : blockIdx.x;
    const int bh = blockIdx.y;
    const int b = bh / num_heads;
    const int h = bh % num_heads;
    const int h_per_kv = num_heads / kv_heads;
    const int kvh = h_per_kv > 1 ? h / h_per_kv : h;
    const long long qbase = (long long)bh * seq_len * HEAD_DIM;
    const long long kvbase = ((long long)(b * kv_heads + kvh)) * seq_len * HEAD_DIM;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int m0 = m_tile * BLOCK_M + warp * 16;  // first global row of this warp
    const int m0b = m_tile * BLOCK_M;             // first global row of the block

    // Pull Q directly from gmem into register fragments (once per block).
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag[8];
    const bool q_valid = (m0 + 15 < seq_len);
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int rr = ((j & 2) ? 8 : 0) + lane / 4;
            const int cc = 2 * (lane % 4) + (j & 1) + ((j & 4) ? 8 : 0);
            q_frag[kt].x[j] = q_valid
                ? q[qbase + (long long)(m0 + rr) * HEAD_DIM + kt * 16 + cc]
                : __float2bfloat16(0.f);
        }
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[8];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> v_frag;

#pragma unroll
    for (int i = 0; i < 8; ++i) wmma::fill_fragment(o_frag[i], 0.f);

    float m_prev[2] = {-INFINITY, -INFINITY};
    float l_prev[2] = {0.f, 0.f};

    const int n_tiles_total = (seq_len + BLOCK_N - 1) / BLOCK_N;
    const int n_tiles_block =
        causal ? min(n_tiles_total, (m0b + BLOCK_M - 1) / BLOCK_N + 1) : n_tiles_total;

    if (n_tiles_block > 0) {
        load_tile(k + kvbase, sK, 0, seq_len, BLOCK_N, PADDED_D);
        load_tile(v + kvbase, sV, 0, seq_len, BLOCK_N, PADDED_D);
        cp_async_commit();
    }

    for (int nt = 0; nt < n_tiles_block; ++nt) {
        cp_async_wait<0>();
        __syncthreads();
        const int kv0 = nt * BLOCK_N;

        // ---- per 16-col tile: QK^T, online softmax, stage P, immediate PV ----
        // PV is interleaved with the softmax loop so that the o_frag rescale
        // (exp(m_prev - m_new)) multiplies the *already accumulated* O.
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
#pragma unroll
        for (int i = 0; i < BLOCK_N / 16; ++i) {
            const int kvc = kv0 + 16 * i;
            const bool full_mask = causal ? (kvc > m0 + 15) : (kvc >= seq_len);
            const bool may_mask = !full_mask &&
                                  (causal ? (kvc + 15 >= m0 && kvc <= m0 + 15)
                                          : (kvc + 15 >= seq_len));
#pragma unroll
            for (int j = 0; j < 8; ++j) s_frag.x[j] = 0.f;
            if (full_mask) {
                // Entire 16-col tile is masked (causal future or past seq_len):
                // P = 0, no change to m/l/O.
                continue;
            }
#pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                wmma::load_matrix_sync(k_frag, sK + (16 * i) * PADDED_D + kt * 16, PADDED_D);
                wmma::mma_sync(s_frag, q_frag[kt], k_frag, s_frag);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) s_frag.x[j] *= scale;
            if (may_mask) {
                // Mask causal-future columns and key rows past seq_len (their
                // smem rows were zero-filled by load_tile, which would otherwise
                // contribute spurious logit-0 mass to the softmax).
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int rr = ((j & 2) ? 8 : 0) + lane / 4;
                    const int cc = 2 * (lane % 4) + (j & 1) + ((j & 4) ? 8 : 0);
                    if (kvc + cc >= seq_len || (causal && kvc + cc > m0 + rr))
                        s_frag.x[j] = -INFINITY;
                }
            }
            // ---- online softmax for this 16-col tile ----
            // wmma m16n16k16 accumulator layout: element j of row lane/4 sits at
            // j in {0,1,4,5}; of row lane/4+8 at {2,3,6,7}.
            float tm0 = fmaxf(fmaxf(s_frag.x[0], s_frag.x[1]),
                              fmaxf(s_frag.x[4], s_frag.x[5]));
            float tm1 = fmaxf(fmaxf(s_frag.x[2], s_frag.x[3]),
                              fmaxf(s_frag.x[6], s_frag.x[7]));
            tm0 = warp_max4(tm0);
            tm1 = warp_max4(tm1);
            const float mn0 = fmaxf(m_prev[0], tm0);
            const float mn1 = fmaxf(m_prev[1], tm1);
            const float corr0 = __expf(m_prev[0] - mn0);
            const float corr1 = __expf(m_prev[1] - mn1);
            // Rescale the already-accumulated O (tiles < i are in o_frag now).
            // Skipped when the running max didn't grow (corr == 1): the 64
            // FMULs per sub-tile dominate the FP32 instruction stream, and
            // after the first few tiles a new max is rare.
            if (mn0 > m_prev[0]) {
#pragma unroll
                for (int o = 0; o < 8; ++o) {
                    o_frag[o].x[0] *= corr0; o_frag[o].x[1] *= corr0;
                    o_frag[o].x[4] *= corr0; o_frag[o].x[5] *= corr0;
                }
            }
            if (mn1 > m_prev[1]) {
#pragma unroll
                for (int o = 0; o < 8; ++o) {
                    o_frag[o].x[2] *= corr1; o_frag[o].x[3] *= corr1;
                    o_frag[o].x[6] *= corr1; o_frag[o].x[7] *= corr1;
                }
            }
            float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float p = __expf(s_frag.x[j] - ((j & 2) ? mn1 : mn0));
                s_frag.x[j] = p;
                if (j & 2) ps1 += p; else ps0 += p;
            }
            ps0 = warp_sum4(ps0);
            ps1 = warp_sum4(ps1);
            l_prev[0] = l_prev[0] * corr0 + ps0;
            l_prev[1] = l_prev[1] * corr1 + ps1;
            m_prev[0] = mn0;
            m_prev[1] = mn1;
            // ---- PV: O[16x128] += P[16x16] * V[16x128] for this tile ----
            // wmma m16n16k16 matrix_a row_major fragment layout == accumulator
            // layout, so the softmaxed S fragment converts in-place to bf16 and
            // feeds the PV matmul directly (no shared-memory round-trip).
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> p_frag;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                p_frag.x[j] = __float2bfloat16(s_frag.x[j]);
#pragma unroll
            for (int ot8 = 0; ot8 < 8; ++ot8) {
                wmma::load_matrix_sync(v_frag, sV + (16 * i) * PADDED_D + ot8 * 16, PADDED_D);
                wmma::mma_sync(o_frag[ot8], p_frag, v_frag, o_frag[ot8]);
            }
        }
        __syncthreads();  // all reads of this tile done before prefetch overwrites it
        if (nt + 1 < n_tiles_block) {
            load_tile(k + kvbase, sK, (nt + 1) * BLOCK_N, seq_len, BLOCK_N, PADDED_D);
            load_tile(v + kvbase, sV, (nt + 1) * BLOCK_N, seq_len, BLOCK_N, PADDED_D);
            cp_async_commit();
        }
    }

    // ---- Epilogue: normalize O and store straight to gmem (coalesced) ----
    const float inv0 = 1.f / l_prev[0];
    const float inv1 = 1.f / l_prev[1];
#pragma unroll
    for (int ot8 = 0; ot8 < 8; ++ot8) {
        const int d0 = ot8 * 16;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int rr = ((j & 2) ? 8 : 0) + lane / 4;
            const int cc = 2 * (lane % 4) + (j & 1) + ((j & 4) ? 8 : 0);
            const int grow = m0 + rr;
            if (grow < seq_len) {
                out[qbase + (long long)grow * HEAD_DIM + d0 + cc] =
                    __float2bfloat16(o_frag[ot8].x[j] * ((j & 2) ? inv1 : inv0));
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
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "expected [B, H, S, D]");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(2) == k.size(2) &&
                q.size(3) == k.size(3) && q.size(3) == HEAD_DIM, "shape mismatch");
    TORCH_CHECK(k.size(1) == v.size(1) && q.size(1) % k.size(1) == 0,
                "kv_heads must divide heads");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q/k/v must be contiguous");

    const at::cuda::CUDAGuard guard(q.device());
    const int B = q.size(0), H = q.size(1), S = q.size(2);
    const int kv_heads = k.size(1);
    TORCH_CHECK((long long)B * H <= 65535, "B*H exceeds grid.y limit");
    TORCH_CHECK(S >= BLOCK_M, "seq_len must be >= 64");

    auto out = torch::empty_like(q);
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    dim3 grid((S + BLOCK_M - 1) / BLOCK_M, B * H);

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(attention_fwd_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        attr_set = true;
    }

    attention_fwd_kernel<<<grid, NUM_THREADS, SMEM_BYTES,
                           at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        S, scale, causal, H, kv_heads);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}