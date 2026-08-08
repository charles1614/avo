// Tensor-core FlashAttention forward (sm_86): wmma m16n16k16 bf16, 8 warps,
// BLOCK_M=128 query rows, BLOCK_N=64 keys per tile, cp.async double-buffered
// K/V staging, tile-level (two-pass) softmax in the exp2 domain, GQA support.
// BF16 in/out, FP32 accumulation.
//
// Fragment layout note (verified for sm_86 m16n16k16): the C accumulator
// layout equals the matrix_a layout: lane l (gid=l/4, t=l%4) holds
//   x0=(gid,2t) x1=(gid,2t+1) x2=(gid+8,2t) x3=(gid+8,2t+1)
//   x4=(gid,2t+8) x5=(gid,2t+9) x6=(gid+8,2t+8) x7=(gid+8,2t+9)
// so the softmaxed S accumulator converts to the P matrix_a fragment in
// place (no shuffles). K tiles are consumed as matrix_b col_major (ldm 128),
// V tiles as matrix_b row_major (ldm 128).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <math.h>
#include <cstdint>

namespace {

constexpr int BLOCK_M = 128;  // query rows per block
constexpr int BLOCK_N = 64;   // key/value rows per tile
constexpr int HEAD_DIM = 128;
constexpr int NUM_WARPS = 8;
constexpr int NUM_THREADS = NUM_WARPS * 32;
constexpr int NBUFS = 2;                  // cp.async pipeline stages
constexpr int SMEM_BYTES = NBUFS * 2 * BLOCK_N * HEAD_DIM * 2;  // 64 KiB
constexpr float NEG_INF = -INFINITY;
constexpr float LOG2E = 1.4426950408889634f;  // log2(e): softmax in exp2 domain

using namespace nvcuda;

__device__ __forceinline__ float exp2_approx(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// 16-byte cp.async (Ampere).
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// Stage one K/V tile into buffer `buf` (zero-padding rows >= seq_len).
// sK/sV are raw pointers into dynamic shared memory (NBUFS*64*128 each).
__device__ __forceinline__ void stage_kv(
    __nv_bfloat16* __restrict__ sK, __nv_bfloat16* __restrict__ sV,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    long long base_kv, int seq_len, int nt, int buf)
{
    constexpr int NCHUNK = BLOCK_N * HEAD_DIM / 8;  // uint4 per matrix
    constexpr int TILE = BLOCK_N * HEAD_DIM;
    const uint4 zero = make_uint4(0, 0, 0, 0);
    for (int i = threadIdx.x; i < 2 * NCHUNK; i += NUM_THREADS) {
        const int mat = i / NCHUNK;          // 0 = K, 1 = V
        const int r = (i % NCHUNK) / 16;     // row within tile
        const int u = (i % NCHUNK) % 16;     // 16B chunk within row
        const int grow_kv = nt * BLOCK_N + r;
        __nv_bfloat16* dst = (mat == 0) ? &sK[buf * TILE + r * HEAD_DIM + u * 8]
                                        : &sV[buf * TILE + r * HEAD_DIM + u * 8];
        if (grow_kv < seq_len) {
            const __nv_bfloat16* src = (mat == 0) ? k : v;
            src += base_kv + (long long)grow_kv * HEAD_DIM + u * 8;
            cp_async16(dst, src);
        } else {
            *reinterpret_cast<uint4*>(dst) = zero;
        }
    }
}

__global__ void __launch_bounds__(NUM_THREADS, 1) attention_fwd_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    int seq_len, float scale, bool causal, int q_per_kv)
{
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* sK = smem;                       // NBUFS * 64 * 128
    __nv_bfloat16* sV = smem + NBUFS * BLOCK_N * HEAD_DIM;

    const int m_tile = blockIdx.x;
    const int bh = blockIdx.y;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int gid = lane / 4, tig = lane % 4;

    const long long base = (long long)bh * seq_len * HEAD_DIM;
    const long long base_kv = (long long)(bh / q_per_kv) * seq_len * HEAD_DIM;
    const int grow0 = m_tile * BLOCK_M + warp * 16;   // first query row of warp
    // Scores stay raw (q·k) in the MMA; scale·log2(e) is folded into the
    // exp2 arguments so exp2(s') == exp(s·scale). Softmax is invariant under
    // a common per-row scale factor.
    const float scale_log2e = scale * LOG2E;

    // ---- Q: 8 A-fragments (16 rows x 16 dims each), scale applied ----
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf[8];
    {
        const int rowA = grow0 + gid, rowB = grow0 + gid + 8;
        const bool okA = rowA < seq_len, okB = rowB < seq_len;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const bool isA = (e % 4) < 2;   // e in {0,1,4,5} -> rowA
                const int row = isA ? rowA : rowB;
                const bool ok = isA ? okA : okB;
                const int col = j * 16 + 2 * tig + (e % 2) + ((e >= 4) ? 8 : 0);
                float val = ok ? __bfloat162float(q[base + (long long)row * HEAD_DIM + col]) : 0.f;
                qf[j].x[e] = __float2bfloat16(val);
            }
        }
    }

    // ---- O accumulators (8 head-dim blocks of 16) ----
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) wmma::fill_fragment(acc_o[j], 0.0f);

    // Online softmax state in exp2 domain, per lane per row (rows gid, gid+8).
    float m2[2] = {NEG_INF, NEG_INF}, l2[2] = {0.f, 0.f};

    const int row_hi = min(m_tile * BLOCK_M + BLOCK_M - 1, seq_len - 1);
    int n_tiles = (seq_len + BLOCK_N - 1) / BLOCK_N;
    if (causal) n_tiles = min(n_tiles, row_hi / BLOCK_N + 1);

    // ---- prologue: stage tile 0 ----
    stage_kv(sK, sV, k, v, base_kv, seq_len, 0, 0);
    cp_async_commit();

    for (int nt = 0; nt < n_tiles; ++nt) {
        const int buf = nt % NBUFS;
        if (nt + 1 < n_tiles) {
            stage_kv(sK, sV, k, v, base_kv, seq_len, nt + 1, (nt + 1) % NBUFS);
            cp_async_commit();
            cp_async_wait<1>();   // tile nt arrived
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        // ---- QK: 8x4 mma -> S accumulators (16 rows x 64 cols) ----
        // Loop order keeps at most one K fragment live at a time.
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_s[4];
#pragma unroll
        for (int nb = 0; nb < 4; ++nb) wmma::fill_fragment(acc_s[nb], 0.0f);

        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
        constexpr int TILE = BLOCK_N * HEAD_DIM;
        const __nv_bfloat16* sKb = sK + buf * TILE;
        const __nv_bfloat16* sVb = sV + buf * TILE;
#pragma unroll
        for (int nb = 0; nb < 4; ++nb) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                wmma::load_matrix_sync(kf, sKb + nb * 16 * HEAD_DIM + j * 16, HEAD_DIM);
                wmma::mma_sync(acc_s[nb], qf[j], kf, acc_s[nb]);
            }
        }

        // ---- tile-level (two-pass) softmax on S, exp2 domain ----
        // Pass 1: per-row max over the 4 tiles held by this warp.
        float tile_max[2] = {NEG_INF, NEG_INF};
        const int gcol0 = nt * BLOCK_N;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int rowOff = (((e % 4) >> 1) & 1) * 8;   // 0 or 8
            const int grow = grow0 + rowOff + gid;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = gcol0 + j * 16 + ((e >= 4) ? 8 : 0) + 2 * tig + (e % 2);
                const bool valid = !causal || col <= grow;
                const float s = acc_s[j].x[e];
                tile_max[rowOff / 8] = valid ? fmaxf(tile_max[rowOff / 8], s) : tile_max[rowOff / 8];
            }
        }
#pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            tile_max[0] = fmaxf(tile_max[0], __shfl_xor_sync(0xffffffffu, tile_max[0], off));
            tile_max[1] = fmaxf(tile_max[1], __shfl_xor_sync(0xffffffffu, tile_max[1], off));
        }
        // Rescale running statistics once per tile. A fully-masked tile keeps
        // tile_max at -inf; then m stays put and corr = 1 (p = 0 everywhere).
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const float mtx = (tile_max[r] == NEG_INF) ? m2[r] : tile_max[r];
            const float corr = exp2_approx((m2[r] - mtx) * scale_log2e);  // 0 when m2=-inf
            m2[r] = mtx;
            l2[r] *= corr;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int rowOff = (((e % 4) >> 1) & 1) * 8;
                    if (rowOff / 8 == r) acc_o[j].x[e] *= corr;
                }
            }
        }

        // ---- P = exp2(S - m) in A-fragment layout, accumulate l ----
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pf[4];
        float row_sum[2] = {0.f, 0.f};
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int rowOff = (((e % 4) >> 1) & 1) * 8;
            const int grow = grow0 + rowOff + gid;
            float acc = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = gcol0 + j * 16 + ((e >= 4) ? 8 : 0) + 2 * tig + (e % 2);
                const bool valid = !causal || col <= grow;
                const float p = valid ? exp2_approx((acc_s[j].x[e] - m2[rowOff / 8]) * scale_log2e) : 0.f;
                pf[j].x[e] = __float2bfloat16(p);
                acc += p;
            }
            row_sum[rowOff / 8] += acc;
        }
#pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            row_sum[0] += __shfl_xor_sync(0xffffffffu, row_sum[0], off);
            row_sum[1] += __shfl_xor_sync(0xffffffffu, row_sum[1], off);
        }
        l2[0] += row_sum[0];
        l2[1] += row_sum[1];

        // ---- PV: 4x8 mma into O accumulators (one V fragment live) ----
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vf;
#pragma unroll
        for (int o = 0; o < 8; ++o) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::load_matrix_sync(vf, sVb + j * 16 * HEAD_DIM + o * 16, HEAD_DIM);
                wmma::mma_sync(acc_o[o], pf[j], vf, acc_o[o]);
            }
        }

        __syncthreads();  // cp.async for nt+2 may overwrite buf `nt`
    }

    // ---- epilogue: out[row][col] = acc_o * (1 / l) ----
    const float inv0 = l2[0] > 0.f ? 1.f / l2[0] : 0.f;
    const float inv1 = l2[1] > 0.f ? 1.f / l2[1] : 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int rowOff = (((e % 4) >> 1) & 1) * 8;
            const int colOff = (e >= 4) ? 8 : 0;
            const int grow = grow0 + rowOff + gid;
            if (grow >= seq_len) continue;
            const int col = j * 16 + colOff + 2 * tig + (e % 2);
            const float inv = (rowOff == 0) ? inv0 : inv1;
            out[base + (long long)grow * HEAD_DIM + col] =
                __float2bfloat16(acc_o[j].x[e] * inv);
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
    TORCH_CHECK(k.size(0) == q.size(0) && k.size(2) == q.size(2) &&
                k.size(3) == q.size(3), "k/v shape mismatch");
    TORCH_CHECK(q.size(3) == HEAD_DIM, "head_dim must be ", HEAD_DIM);
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q/k/v must be contiguous");
    const int kv_heads = k.size(1);
    TORCH_CHECK(q.size(1) % kv_heads == 0, "heads must be divisible by kv_heads");

    const at::cuda::CUDAGuard guard(q.device());
    const int B = q.size(0), H = q.size(1), S = q.size(2);
    TORCH_CHECK((long long)B * H <= 65535, "B*H exceeds grid.y limit");

    auto out = torch::empty_like(q);
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    dim3 grid((S + BLOCK_M - 1) / BLOCK_M, B * H);

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(attention_fwd_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM_BYTES);
        attr_set = true;
    }

    attention_fwd_kernel<<<grid, NUM_THREADS, SMEM_BYTES,
                           at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        S, scale, causal, H / kv_heads);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
