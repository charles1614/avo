# BF16 tensor-core MMA via inline PTX (sm_80+)

## The m16n8k16 bf16 MMA
Computes D(16x8, f32) += A(16x16, bf16) x B(16x8, bf16) per warp.

```cuda
// a: 4x uint32 (8 bf16), b: 2x uint32 (4 bf16), c/d: 4x float, per lane
__device__ __forceinline__ void mma_m16n8k16_bf16(
    float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
```

Fragment layouts (row.col): lane l owns A elements at rows {l/4, l/4+8} and
2-column pairs {2*(l%4), 2*(l%4)+1} within each 8-col half of K; accumulator D
lane mapping: d[0..1] -> row l/4, cols {2*(l%4), 2*(l%4)+1}; d[2..3] -> row
l/4+8, same cols. Full tables: PTX ISA "Warp Level Matrix Multiply-Accumulate
Instructions", section on mma.m16n8k16.

## ldmatrix: shared memory -> fragments
```cuda
__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], const void* smem_ptr)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
```
Each of the 4 8x8 b16 matrices is addressed by one lane-group (lanes 0-7 give
row pointers of the first, 8-15 the second, ...). `.trans` variant transposes
on load — the usual way to get K^T fragments for row.col MMA.

## cp.async 16-byte copy (Ampere+)
```cuda
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem)
{
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(s), "l"(gmem));
}
// after issuing a tile:  asm volatile("cp.async.commit_group;\n");
// wait for all but N in flight: asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
```

## Gotchas
- All pointers into shared must go through __cvta_generic_to_shared.
- bf16 pairs pack little-endian into uint32 (element 0 = low half).
- mma.sync needs the full warp active — no divergent lanes.
- Swizzle shared layout so ldmatrix reads hit distinct banks; the standard
  128-byte swizzle is XOR of bits [4:6] of the byte offset with bits [7:9].
