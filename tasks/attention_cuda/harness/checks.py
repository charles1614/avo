"""Torch-free source checks (importable offline).

Reusable by ANY hand-written-kernel task: the default pattern list bans fused
attention entry points, and a task can override or extend it via
`task_params.banned_apis` (a list of regexes) — e.g. a GEMM task would ban
`cublas.*Gemm|at::matmul`, a conv task `cudnnConvolutionForward`. Without such
a ban, an agent can delegate to a vendor library and the score measures the
library rather than the agent (observed: 608 TFLOPS via SDPA).
"""
from __future__ import annotations

import re
from pathlib import Path

# Pre-built fused attention entry points the candidate may NOT call — the task
# is to hand-write the kernel. cuBLAS/CUTLASS GEMM primitives stay allowed
# (composing attention from GEMM building blocks is legitimate kernel work).
BANNED_API_PATTERNS = [
    r"scaled_dot_product_attention",
    r"cudnnMultiHeadAttn", r"cudnn_attention", r"cudnnAttn",
    r"cudnnFusedAttn", r"_flash_attention", r"flash_attn_",
    r"mem_efficient_attention", r"at::native::[A-Za-z_]*attention",
    r"_scaled_dot_product", r"xformers",
]
_SRC_EXT = (".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp", ".py")


def scan_banned_apis(workspace, patterns: list[str] | None = None) -> str | None:
    """First banned-API symbol in workspace source, or None. Comments are
    stripped so a mention in a comment doesn't false-trip. `patterns`
    overrides the attention defaults (see task_params.banned_apis)."""
    pat = re.compile("|".join(patterns or BANNED_API_PATTERNS))
    for p in sorted(Path(workspace).rglob("*")):
        if p.suffix.lower() not in _SRC_EXT or not p.is_file():
            continue
        text = p.read_text(errors="replace")
        text = re.sub(r"//[^\n]*", "", text)               # C++ line comments
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)  # block comments
        text = re.sub(r"(?m)#.*$", "", text)               # py comments
        m = pat.search(text)
        if m:
            return f"{m.group(0)} in {p.name}"
    return None
