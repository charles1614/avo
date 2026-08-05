# Hopper GH100 / sm_90a tuning (H100)

## Chip facts
- H100 SXM: 132 SMs, 228 KB shared/L1 per SM (dynamic shared up to 227 KB per
  block), 50 MB L2, HBM3 ~3.35 TB/s.
- Peak dense BF16 tensor ~990 TFLOPS (SXM). FA3 reaches ~740 TFLOPS forward on
  head_dim 128; SDPA/cuDNN similar ballpark. FA2 (Ampere-style code) lands
  around 500-600 TFLOPS on H100.
- MUST compile with arch=compute_90a,code=sm_90a: wgmma, TMA, and setmaxnreg
  are gated behind the "a" (architecture-specific) target; plain sm_90 hides them.

## Hopper-only machinery
- wgmma (warpgroup MMA): one warpgroup (4 warps, 128 threads) issues
  wgmma.mma_async.sync.aligned.m64nNk16 with operands read directly from
  shared memory (B) and registers/shared (A); accumulators live in registers
  spread across the warpgroup. Asynchronous: overlap with softmax of the
  previous tile (the FA3 "pingpong"/intra-warpgroup overlap ideas).
- TMA (cp.async.bulk.tensor): hardware tensor-tile copies driven by one
  thread; replaces per-lane cp.async loops; needs CUtensorMap (or CUTLASS/CuTe
  helpers) and mbarrier synchronization.
- setmaxnreg.inc/.dec: rebalance registers between producer (load) and
  consumer (MMA) warpgroups — producer warpgroups shrink to ~24-40 regs,
  consumers grow to 160-232. This is what FA3/CUTLASS warp specialization uses.
- Threadblock clusters + distributed shared memory: adjacent CTAs can read
  each other's shared memory; rarely needed for attention forward.

## Practical notes for attention here
- The proven Hopper attention shape (FA3): warp-specialized producer/consumer
  warpgroups, TMA loads of K/V tiles into a multi-stage shared pipeline, wgmma
  m64n64k16/m64n128k16 for QK^T and PV, softmax overlapped with the next
  wgmma, BLOCK_M 128/192, BLOCK_N 64/128 at head_dim 128.
- An Ampere-style kernel (cp.async + mma.sync m16n8k16) is portable to sm_90a
  and a reasonable intermediate step, but leaves ~30% on the table vs wgmma.
- Register file is the constraint at big tiles: 128 x 128 FP32 accumulators
  per warpgroup = 64 regs/thread just for acc; plan register budgets before
  choosing tiles, verify with -Xptxas=-v.
