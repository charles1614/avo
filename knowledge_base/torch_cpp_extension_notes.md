# torch.utils.cpp_extension notes (how this task's kernels get built)

## How the harness builds
- All workspace .cu/.cpp files are compiled together via cpp_extension.load()
  with -O3 plus the config's -gencode flags; the module name and build dir are
  keyed on a content hash (no stale caches; unchanged code = instant rebuild).
- Optional workspace file `build_flags.json`:
  {"extra_cuda_cflags": ["--use_fast_math", "-Xptxas=-v", "--maxrregcount=128"]}
  Allowed: -O0..3, --use_fast_math, -lineinfo, --maxrregcount=N, -Xptxas...,
  -D defines. Anything else fails the compile stage.

## Useful flags
- -Xptxas=-v  -> per-kernel register/shared/spill report in the build log
  (visible in the evaluate output on compile, and via gpu_shell).
- --use_fast_math -> __expf, fast div/sqrt everywhere; big softmax win but
  re-check correctness.
- -lineinfo -> source-line attribution in ncu profiles.

## Common build errors
- "identifier __nv_bfloat16 is undefined": include <cuda_bf16.h>.
- "wgmma..." / "setmaxnreg..." unknown instruction: you compiled sm_90 (or
  sm_86) instead of sm_90a — these need arch=compute_90a,code=sm_90a.
- Static __shared__ arrays over 48 KB: "uses too much shared data" — switch to
  dynamic shared memory (extern __shared__) + cudaFuncSetAttribute(
  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) before launch.
- Kernel launch failures surface via C10_CUDA_KERNEL_LAUNCH_CHECK() as
  "CUDA error: invalid argument" — usually a grid/block/shared-size problem.
- ptxas "register allocation" errors at high --maxrregcount: 255 is the hard
  per-thread cap; large per-thread arrays must be provably index-constant to
  stay in registers (unrolled loops, constexpr bounds), else they spill to
  local memory.

## Runtime API bits used by fancier kernels
- at::cuda::getCurrentCUDAStream() — always launch on the ambient stream.
- Dynamic shared launch: kernel<<<grid, block, smem_bytes, stream>>>(...).
- cudaFuncSetAttribute must be called once per kernel before the first launch
  with oversized dynamic shared memory.
