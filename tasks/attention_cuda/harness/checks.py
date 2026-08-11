"""Torch-free source checks for the attention harness (importable offline)."""
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


def scan_banned_apis(workspace) -> str | None:
    """First banned attention-API symbol in workspace source, or None.
    Comments are stripped so a mention in a comment doesn't false-trip."""
    pat = re.compile("|".join(BANNED_API_PATTERNS))
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
