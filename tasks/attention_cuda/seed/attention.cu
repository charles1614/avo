// Seed attention forward kernel: FlashAttention-style online softmax, but
// deliberately scalar — no tensor cores, no cp.async, no arch-specific
// intrinsics — so it compiles for both sm_86 and sm_90a and leaves large
// optimization headroom. BF16 in/out, FP32 accumulation.
//
// Work partitioning: one block per (query tile, batch*head). 4 warps per
// block; each warp owns ROWS_PER_WARP query rows; within a row the 32 lanes
// cover the head dimension with D_PER_LANE consecutive elements each.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <math.h>

namespace {

constexpr int BLOCK_M = 32;   // query rows per block
constexpr int BLOCK_N = 64;   // key/value rows per tile
constexpr int HEAD_DIM = 128;
constexpr int NUM_WARPS = 4;
constexpr int NUM_THREADS = NUM_WARPS * 32;
constexpr int ROWS_PER_WARP = BLOCK_M / NUM_WARPS;  // 8
constexpr int D_PER_LANE = HEAD_DIM / 32;           // 4

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

__global__ void attention_fwd_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    int seq_len, float scale, bool causal)
{
    const int m_tile = blockIdx.x;
    const long long bh = blockIdx.y;  // fused batch*head index
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int d0 = lane * D_PER_LANE;

    const long long base = bh * (long long)seq_len * HEAD_DIM;

    __shared__ __nv_bfloat16 sK[BLOCK_N][HEAD_DIM];
    __shared__ __nv_bfloat16 sV[BLOCK_N][HEAD_DIM];

    float q_reg[ROWS_PER_WARP][D_PER_LANE];
    float acc[ROWS_PER_WARP][D_PER_LANE];
    float row_max[ROWS_PER_WARP];
    float row_sum[ROWS_PER_WARP];

#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        row_max[r] = -INFINITY;
        row_sum[r] = 0.f;
        const int grow = m_tile * BLOCK_M + warp * ROWS_PER_WARP + r;
#pragma unroll
        for (int j = 0; j < D_PER_LANE; ++j) {
            acc[r][j] = 0.f;
            q_reg[r][j] = (grow < seq_len)
                ? __bfloat162float(q[base + (long long)grow * HEAD_DIM + d0 + j])
                : 0.f;
        }
    }

    const int row_hi = min(m_tile * BLOCK_M + BLOCK_M - 1, seq_len - 1);
    int n_tiles = (seq_len + BLOCK_N - 1) / BLOCK_N;
    if (causal) n_tiles = min(n_tiles, row_hi / BLOCK_N + 1);

    for (int nt = 0; nt < n_tiles; ++nt) {
        // Cooperative K/V tile load into shared memory.
        for (int idx = threadIdx.x; idx < BLOCK_N * HEAD_DIM; idx += NUM_THREADS) {
            const int row = idx / HEAD_DIM;
            const int col = idx % HEAD_DIM;
            const int grow_kv = nt * BLOCK_N + row;
            if (grow_kv < seq_len) {
                sK[row][col] = k[base + (long long)grow_kv * HEAD_DIM + col];
                sV[row][col] = v[base + (long long)grow_kv * HEAD_DIM + col];
            } else {
                sK[row][col] = __float2bfloat16(0.f);
                sV[row][col] = __float2bfloat16(0.f);
            }
        }
        __syncthreads();

        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const int grow = m_tile * BLOCK_M + warp * ROWS_PER_WARP + r;
            if (grow >= seq_len) continue;  // uniform across the warp
            for (int c = 0; c < BLOCK_N; ++c) {
                const int gcol = nt * BLOCK_N + c;
                if (gcol >= seq_len) break;          // uniform across the warp
                if (causal && gcol > grow) break;    // columns are ascending

                float partial = 0.f;
#pragma unroll
                for (int j = 0; j < D_PER_LANE; ++j)
                    partial += q_reg[r][j] * __bfloat162float(sK[c][d0 + j]);
                const float s = warp_sum(partial) * scale;

                // Online softmax at column granularity (tile size 1).
                const float m_new = fmaxf(row_max[r], s);
                const float corr = expf(row_max[r] - m_new);  // exp(-inf)=0 first col
                const float p = expf(s - m_new);
                row_sum[r] = row_sum[r] * corr + p;
#pragma unroll
                for (int j = 0; j < D_PER_LANE; ++j)
                    acc[r][j] = acc[r][j] * corr
                              + p * __bfloat162float(sV[c][d0 + j]);
                row_max[r] = m_new;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int grow = m_tile * BLOCK_M + warp * ROWS_PER_WARP + r;
        if (grow >= seq_len) continue;
        const float inv = row_sum[r] > 0.f ? 1.f / row_sum[r] : 0.f;
#pragma unroll
        for (int j = 0; j < D_PER_LANE; ++j)
            out[base + (long long)grow * HEAD_DIM + d0 + j]
                = __float2bfloat16(acc[r][j] * inv);
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
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
                "q/k/v shapes must match");
    TORCH_CHECK(q.size(3) == HEAD_DIM, "head_dim must be ", HEAD_DIM);
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q/k/v must be contiguous");

    const at::cuda::CUDAGuard guard(q.device());
    const int B = q.size(0), H = q.size(1), S = q.size(2);
    TORCH_CHECK((long long)B * H <= 65535, "B*H exceeds grid.y limit");

    auto out = torch::empty_like(q);
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    dim3 grid((S + BLOCK_M - 1) / BLOCK_M, B * H);

    attention_fwd_kernel<<<grid, NUM_THREADS, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        S, scale, causal);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
