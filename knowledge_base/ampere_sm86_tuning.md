# Ampere GA102 / sm_86 tuning (RTX 3090 Ti)

## Chip facts
- 84 SMs; per SM: 128 FP32 lanes (64 FP32 + 64 FP32/INT32), 4 tensor cores,
  256 KB register file (65536 x 32-bit), max 255 registers/thread.
- L1/shared: 128 KB combined; shared carveout up to 100 KB per block (static
  __shared__ limit is 48 KB; >48 KB requires dynamic shared memory +
  cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, ...)).
- 6 MB L2. GDDR6X ~1008 GB/s.
- Peak dense BF16/FP16 tensor: ~160 TFLOPS (with FP32 accumulate).
  Peak FP32 FFMA: ~40 TFLOPS. => scalar attention tops out low; tensor cores
  are mandatory to approach SDPA/FA2 numbers (~50-70 TFLOPS geomean typical).
- 1536 max threads/SM on sm_86 (not 2048): occupancy math differs from A100.

## Tensor cores on Ampere (bf16)
- mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 is the workhorse
  (see ptx_mma_bf16_notes.md for the exact fragment layout).
- Feed with ldmatrix.sync.aligned.m8n8.x4.shared.b16 from shared memory;
  swizzle the shared layout so ldmatrix reads are conflict-free.
- cp.async available (16-byte variant, .cg bypasses L1); no TMA, no wgmma,
  no setmaxnreg — those are Hopper sm_90a features.

## Practical notes for attention here
- Good FA2-on-Ampere tile sizes for head_dim 128: BLOCK_M 128, BLOCK_N 32/64,
  4-8 warps, K/V double-buffered in shared (2 x 64 x 128 x 2 B x 2 = 64 KB
  needs dynamic shared).
- GDDR6X thermals on a 3090 Ti drift clocks over long benches; the harness
  uses median-of-repeats which damps this, but expect a few % run-to-run.
- BF16 on sm_86 is full-rate on tensor cores; FP16 accumulate-in-FP16 is 2x
  tensor rate but unacceptable numerically for softmax outputs — keep FP32
  accumulation.
