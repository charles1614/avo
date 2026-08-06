# AVO run report

Run: `attention-3090-20260805-194458`

| version | step | score | change |
|---|---|---|---|
| v0000 | 0 | 2.3804 | seed |
| v0001 | 4 | 2.9561 | Enable --use_fast_math build flag: scalar kernel is dominated by per-column expf |
| v0002 | 8 | 3.8245 | Scalar kernel vectorization: BLOCK_M 32→64 with 8 warps, 8/16-byte vectorized gl |
| v0003 | 9 | 4.7903 | Scalar kernel inner-loop restructure: iterate K/V columns outer, query rows inne |
| v0004 | 10 | 4.8034 | Restructure scalar kernel inner loop: iterate K/V columns outer, all ROWS_PER_WA |
| v0005 | 16 | 6.5835 | Fuse online-softmax correction and output accumulation into one loop; hoist shar |
| v0006 | 18 | 6.8094 | FA2-style deferred-rescaling online softmax: keep acc/sum in the domain of the r |
| v0007 | 24 | 6.9271 | Compile-time causal specialization: template<bool CAUSAL> kernel with runtime br |

Seed -> best: 2.3804 -> 6.9271 (+191.0%)
- baseline sdpa_flash: 70.3950
- baseline sdpa_cudnn: 62.9700
- baseline sdpa_efficient: 47.6470
- baseline sdpa_math: 5.8690