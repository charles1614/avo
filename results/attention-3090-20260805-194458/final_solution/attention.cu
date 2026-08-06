// Attention forward kernel: FlashAttention-style online softmax, scalar
// FP32 pipeline with vectorized (64/128-bit) global and shared memory
// accesses. BF16 in/out, FP32 accumulation.
//
// Work partitioning: one block per (query tile, batch*head). 8 warps per
// block; each warp owns ROWS_PER_WARP query rows; within a row the 32 lanes
// cover the head dimension with D_PER_LANE consecutive elements each.
// BLOCK_M=64 so each K/V tile is reused across 64 query rows (halves
// K/V global traffic versus BLOCK_M=32).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int BLOCK_M = 64;   // query rows per block
constexpr int BLOCK_N = 64;   // key/value rows per tile
constexpr int HEAD_DIM = 128;
constexpr int NUM_WARPS = 8;
constexpr int NUM_THREADS = NUM_WARPS * 32;
constexpr int ROWS_PER_WARP = BLOCK_M / NUM_WARPS;  // 8
constexpr int D_PER_LANE = HEAD_DIM / 32;           // 4

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

template <bool CAUSAL>
__global__ void attention_fwd_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    int seq_len, float scale, int num_heads, int kv_heads)
{
    const int m_tile = blockIdx.x;
    const int bh = blockIdx.y;  // fused batch*head index
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int d0 = lane * D_PER_LANE;  // first head index of this lane

    // GQA: each KV head serves num_heads/kv_heads consecutive query heads.
    const int b = bh / num_heads, h = bh % num_heads;
    const int kvh = h / (num_heads / kv_heads);
    const long long base = ((long long)b * num_heads + h) * seq_len * HEAD_DIM;
    const long long base_kv = ((long long)b * kv_heads + kvh) * seq_len * HEAD_DIM;

    __shared__ __align__(16) __nv_bfloat16 sK[BLOCK_N][HEAD_DIM];
    __shared__ __align__(16) __nv_bfloat16 sV[BLOCK_N][HEAD_DIM];

    float q_reg[ROWS_PER_WARP][D_PER_LANE];
    float acc[ROWS_PER_WARP][D_PER_LANE];
    float row_max[ROWS_PER_WARP];
    float row_sum[ROWS_PER_WARP];

#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        row_max[r] = -INFINITY;
        row_sum[r] = 0.f;
        const int grow = m_tile * BLOCK_M + warp * ROWS_PER_WARP + r;
        if (grow < seq_len) {
            // Vectorized 8-byte load: D_PER_LANE (4) consecutive bf16.
            const uint2 vq = *reinterpret_cast<const uint2*>(
                q + base + (long long)grow * HEAD_DIM + d0);
            const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&vq);
#pragma unroll
            for (int j = 0; j < D_PER_LANE; ++j) {
                acc[r][j] = 0.f;
                q_reg[r][j] = __bfloat162float(p[j]);
            }
        } else {
#pragma unroll
            for (int j = 0; j < D_PER_LANE; ++j) {
                acc[r][j] = 0.f;
                q_reg[r][j] = 0.f;
            }
        }
    }

    const int row_hi = min(m_tile * BLOCK_M + BLOCK_M - 1, seq_len - 1);
    int n_tiles = (seq_len + BLOCK_N - 1) / BLOCK_N;
    if (CAUSAL) n_tiles = min(n_tiles, row_hi / BLOCK_N + 1);

    for (int nt = 0; nt < n_tiles; ++nt) {
        // Cooperative K/V tile load into shared memory, 16-byte vectorized.
        for (int idx = threadIdx.x; idx < BLOCK_N * HEAD_DIM / 8; idx += NUM_THREADS) {
            const int row = idx / (HEAD_DIM / 8);
            const int col8 = (idx % (HEAD_DIM / 8)) * 8;  // first of 8 bf16
            const int grow_kv = nt * BLOCK_N + row;
            uint4* dst = reinterpret_cast<uint4*>(&sK[row][col8]);
            if (grow_kv < seq_len) {
                *dst = *reinterpret_cast<const uint4*>(
                    k + base_kv + (long long)grow_kv * HEAD_DIM + col8);
                *reinterpret_cast<uint4*>(&sV[row][col8]) =
                    *reinterpret_cast<const uint4*>(
                        v + base_kv + (long long)grow_kv * HEAD_DIM + col8);
            } else {
                const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
                *dst = zero;
                *reinterpret_cast<uint4*>(&sV[row][col8]) = zero;
            }
        }
        __syncthreads();

        // Restructured inner loop: each K/V column is read from shared once,
        // then all ROWS_PER_WARP query rows consume it (registers). This cuts
        // shared traffic ~8x (one column read per 8 rows instead of per row)
        // and halves FP32 dot-product work (K*Q and V*P share the column).
        for (int c = 0; c < BLOCK_N; ++c) {
            const int gcol = nt * BLOCK_N + c;
            const int grow_first = m_tile * BLOCK_M + warp * ROWS_PER_WARP;
            const int grow_last = grow_first + ROWS_PER_WARP - 1;
            if (gcol >= seq_len) break;                 // uniform across the warp
            if (CAUSAL && gcol > grow_last) break;      // no row needs this col

            // Vectorized 8-byte shared reads for the 4 head elements; convert
            // once to FP32 (shared across all ROWS_PER_WARP rows).
            const uint2 vk = *reinterpret_cast<const uint2*>(&sK[c][d0]);
            const __nv_bfloat16* pk = reinterpret_cast<const __nv_bfloat16*>(&vk);
            const uint2 vv = *reinterpret_cast<const uint2*>(&sV[c][d0]);
            const __nv_bfloat16* pv = reinterpret_cast<const __nv_bfloat16*>(&vv);
            float k_f[D_PER_LANE], v_f[D_PER_LANE];
#pragma unroll
            for (int j = 0; j < D_PER_LANE; ++j) {
                k_f[j] = __bfloat162float(pk[j]);
                v_f[j] = __bfloat162float(pv[j]);
            }

            float s[ROWS_PER_WARP];
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                float partial = 0.f;
#pragma unroll
                for (int j = 0; j < D_PER_LANE; ++j)
                    partial += q_reg[r][j] * k_f[j];
                s[r] = warp_sum(partial) * scale;
            }

            // Fused online softmax + output accumulation (tile size 1),
            // with FA2-style deferred rescaling: acc/sum are kept in the
            // domain of the current running max m; only when a new row max
            // arrives do we rescale history (rare for random scores, ~log S
            // times per row). Steady state costs 1 expf + 1 FADD + 4 FFMA
            // per (row, column) instead of 2 expf + corr rescale FMAs.
            // Causal: row grow may only attend columns gcol <= grow.
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                if (CAUSAL && (grow_first + r) < gcol) continue;  // row can't attend
                float p;
                if (s[r] > row_max[r]) {
                    // New running max: rescale history, p = exp(0) = 1.
                    const float rescale = __expf(row_max[r] - s[r]);
                    row_max[r] = s[r];
                    row_sum[r] *= rescale;
#pragma unroll
                    for (int j = 0; j < D_PER_LANE; ++j) acc[r][j] *= rescale;
                    p = 1.f;
                } else {
                    p = __expf(s[r] - row_max[r]);
                }
                row_sum[r] += p;
#pragma unroll
                for (int j = 0; j < D_PER_LANE; ++j)
                    acc[r][j] = fmaf(p, v_f[j], acc[r][j]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int grow = m_tile * BLOCK_M + warp * ROWS_PER_WARP + r;
        if (grow >= seq_len) continue;
        const float inv = row_sum[r] > 0.f ? 1.f / row_sum[r] : 0.f;
        // Vectorized 8-byte store of 4 consecutive bf16.
        uint2 vo;
        {
            uint16_t e[4];
#pragma unroll
            for (int j = 0; j < D_PER_LANE; ++j)
                e[j] = __bfloat16_as_ushort(__float2bfloat16(acc[r][j] * inv));
            vo.x = (uint32_t)e[0] | ((uint32_t)e[1] << 16);
            vo.y = (uint32_t)e[2] | ((uint32_t)e[3] << 16);
        }
        *reinterpret_cast<uint2*>(
            out + base + (long long)grow * HEAD_DIM + d0) = vo;
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
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "expected [B, H, S, D]");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(2) == k.size(2) &&
                q.size(3) == k.size(3) &&
                q.size(0) == v.size(0) && q.size(2) == v.size(2) &&
                q.size(3) == v.size(3), "q/k/v batch, seqlen, head_dim mismatch");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "heads must be a multiple of kv_heads");
    TORCH_CHECK(q.size(3) == HEAD_DIM, "head_dim must be ", HEAD_DIM);
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q/k/v must be contiguous");

    const at::cuda::CUDAGuard guard(q.device());
    const int B = q.size(0), H = q.size(1), KVH = k.size(1), S = q.size(2);
    TORCH_CHECK((long long)B * H <= 65535, "B*H exceeds grid.y limit");

    auto out = torch::empty_like(q);
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    dim3 grid((S + BLOCK_M - 1) / BLOCK_M, B * H);

    if (causal) {
        attention_fwd_kernel<true><<<grid, NUM_THREADS, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            S, scale, H, KVH);
    } else {
        attention_fwd_kernel<false><<<grid, NUM_THREADS, 0,
                                at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            S, scale, H, KVH);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
