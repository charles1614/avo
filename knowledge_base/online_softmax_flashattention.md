# Online softmax and the FlashAttention algorithm

## Problem
Attention: O = softmax(Q K^T / sqrt(d)) V with Q,K,V in R^{S x d}. Materializing
the S x S score matrix is memory-bound; Flash-style kernels tile K/V and keep a
running softmax so scores never leave registers/shared memory.

## Online softmax recurrence (per query row)
Maintain running max m, running denominator l, and un-normalized output acc.
For each new block of scores s_1..s_n (or a single score, tile size 1):

    m_new = max(m, max_i s_i)
    corr  = exp(m - m_new)              # rescales history
    p_i   = exp(s_i - m_new)
    l     = l * corr + sum_i p_i
    acc   = acc * corr + sum_i p_i * V_i
    m     = m_new

Epilogue: O = acc / l. The recurrence is exact for any blocking granularity —
column-at-a-time, tile-at-a-time, or two-pass-per-tile all give identical math.

Numerical notes:
- Initialize m = -inf, l = 0; exp(-inf - x) = 0 makes the first step self-correcting.
- Compute p in FP32 even when inputs are BF16; accumulate acc in FP32.
- exp2f(x * log2e) is cheaper than expf(x); fold log2(e) into the scale so the
  QK product directly produces log2-domain scores (FA2 does this).
- Rescaling every block is wasteful: FA2 defers the corr multiply, only
  rescaling when m actually changes, and normalizes once at the end.

## FlashAttention-2 structure (what a good kernel looks like)
- Grid: (num_q_tiles, batch*heads); each CTA owns one Q tile of BLOCK_M rows,
  loops over K/V tiles of BLOCK_N.
- Q tile loaded once into registers/shared; K/V tiles staged through shared
  memory with async copies (cp.async on Ampere; TMA on Hopper).
- S = Q K^T via tensor-core MMA (m16n8k16 bf16 on Ampere; wgmma on Hopper),
  FP32 accumulators.
- Softmax runs on the FP32 accumulator in registers; P is cast to BF16 for the
  P V MMA.
- Work partitioning (FA2 improvement over FA1): split Q rows across warps, so
  each warp owns whole rows and the softmax needs no cross-warp reduction; K/V
  loop is sequential per CTA.
- Causal masking: skip tiles entirely past the diagonal; only the diagonal
  tiles need per-element masks. FA2 splits the loop into "full tiles" (no mask
  check) + "diagonal tiles" (masked) to keep the hot loop branch-free.
- Pipelining: double-buffer K/V tiles (load tile i+1 while computing tile i).

## Typical progression from a scalar seed kernel
1. Vectorized global loads (float4 / 128-bit) and shared-memory swizzle to
   kill bank conflicts.
2. Tensor-core MMA for QK^T and PV instead of scalar FMA (order-of-magnitude).
3. cp.async double-buffering of K/V tiles overlapping compute.
4. Deferred rescaling + exp2 softmax.
5. Larger tiles (e.g. 128 x 64) once register pressure is under control;
   balance with occupancy.
6. Split-K / warp specialization (producer-consumer) on Hopper.
