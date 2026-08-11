"""Attention-task ban list (torch-free so it stays unit-testable offline).

The task is to hand-write the kernel: calling a pre-built fused attention op
measures the vendor library, not the agent (observed: ~608 TFLOPS by
delegating to SDPA/cuDNN, which made a cross-model comparison meaningless).
GEMM primitives (cuBLAS/CUTLASS) stay allowed — composing attention from GEMM
building blocks is legitimate kernel work.

Scanning itself lives in avo_harness.scan_banned_apis; a run can override this
list via `task_params.banned_apis`.
"""
BANNED_API_PATTERNS = [
    r"scaled_dot_product_attention",
    r"cudnnMultiHeadAttn", r"cudnn_attention", r"cudnnAttn", r"cudnnFusedAttn",
    r"_flash_attention", r"flash_attn_", r"mem_efficient_attention",
    r"at::native::[A-Za-z_]*attention", r"_scaled_dot_product", r"xformers",
]
