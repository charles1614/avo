# CUDA optimization essentials

## Memory
- Coalescing: a warp's 32 lanes should touch one or two 128-byte segments.
  For row-major [S, 128] bf16 tensors, lane l reading elements [4l, 4l+4) of a
  row is fully coalesced (8 bytes/lane = 256 B/warp = two segments).
- Vectorize: reinterpret as float4/uint4 for 128-bit loads; bf16 pairs as
  __nv_bfloat162. Requires 16-byte alignment (torch tensors are aligned).
- Shared memory banks: 32 banks x 4 bytes. A [64][128] bf16 tile has a 256-byte
  row stride => column accesses hit the same bank. Fix with padding
  ([64][128+8]) or XOR swizzling. Profile with l1tex__data_bank_conflicts.
- cp.async (sm_80+): `cp.async.cg.shared.global [smem], [gmem], 16;` copies
  bypass registers and overlap with compute; commit groups + cp.async.wait_group
  for double buffering.

## Execution
- Occupancy is limited by registers/thread, shared/block, and blocks/SM. More
  occupancy is not always better — attention kernels usually want few blocks
  with big tiles and high ILP. Check with --ptxas-options=-v (or -Xptxas=-v).
- Register spills: watch for "spill stores/loads" in ptxas output; spills go to
  local memory (L1/L2) and are catastrophic in hot loops. --maxrregcount or
  __launch_bounds__ trade occupancy vs spills.
- Warp divergence: branches uniform across a warp are free; per-lane branches
  serialize. Prefer predication/min/max over if/else in hot loops.
- __syncthreads() only when data actually crosses warps; __syncwarp() or
  shuffle-based exchange within a warp.
- Warp shuffles: __shfl_xor_sync for butterfly reductions (5 steps for 32 lanes);
  no shared memory needed.

## Math
- expf is ~20 cycles; __expf / exp2f use the SFU (~4x faster, <2 ulp worse) —
  fine under BF16 tolerances. --use_fast_math applies this globally (also
  affects division/sqrt; check correctness after enabling).
- FMA contraction is on by default; don't break it with intermediate rounding.
- fdividef / __frcp_rn for fast reciprocal in epilogues.

## Timing & profiling
- cudaEvent elapsed time around the kernel only; warm up first (JIT, clocks).
- Lock clocks for stable numbers if root: nvidia-smi -lgc <sm_clock> (do NOT do
  this inside the harness; it needs sudo and affects the whole box).
- ncu --set full <binary> for per-kernel metrics; key ones: sm__throughput,
  dram__throughput, l1tex__data_bank_conflicts, launch__registers_per_thread.
