"""Build the workspace kernel via torch.utils.cpp_extension with content-hash
module names and per-hash build directories: unchanged code never recompiles,
changed code never reuses a stale ninja cache, and long-lived processes never
hit already-imported-module collisions."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

ALLOWED_FLAG = re.compile(
    r"^(-O[0-3]"
    r"|--use_fast_math|-use_fast_math"
    r"|-lineinfo|--generate-line-info"
    r"|--maxrregcount=\d{1,3}"
    r"|-Xptxas[=,]?[-=,a-zA-Z0-9]*"
    r"|-D[A-Za-z_][A-Za-z0-9_]*(=[A-Za-z0-9_.]+)?"
    r")$")

SOURCE_EXTS = (".cu", ".cpp", ".cc")
HEADER_EXTS = (".cuh", ".h", ".hpp")


def workspace_sources(workspace: Path) -> list[Path]:
    return sorted(p for p in Path(workspace).rglob("*")
                  if p.suffix in SOURCE_EXTS and p.is_file())


def extra_flags(workspace: Path) -> list[str]:
    flags_file = Path(workspace) / "build_flags.json"
    if not flags_file.exists():
        return []
    data = json.loads(flags_file.read_text())
    flags = data.get("extra_cuda_cflags", [])
    for f in flags:
        if not ALLOWED_FLAG.match(f):
            raise ValueError(f"disallowed build flag: {f!r}")
    return list(flags)


def content_key(workspace: Path, cuda_cflags: list[str]) -> str:
    import torch
    h = hashlib.sha256()
    for p in sorted(Path(workspace).rglob("*")):
        if p.is_file() and p.suffix in SOURCE_EXTS + HEADER_EXTS:
            h.update(str(p.relative_to(workspace)).encode())
            h.update(p.read_bytes())
    h.update(" ".join(cuda_cflags).encode())
    h.update(torch.__version__.encode())
    return h.hexdigest()


def build(workspace: Path, arch_flags: list[str]):
    from torch.utils import cpp_extension

    workspace = Path(workspace)
    sources = workspace_sources(workspace)
    if not sources:
        raise FileNotFoundError("no .cu/.cpp sources in workspace")
    cuda_cflags = ["-O3"] + list(arch_flags) + extra_flags(workspace)
    key = content_key(workspace, cuda_cflags)
    build_root = Path(os.environ.get("AVO_BUILD_CACHE",
                                     "~/avo_scratch/build_cache")).expanduser()
    build_dir = build_root / key[:16]
    build_dir.mkdir(parents=True, exist_ok=True)
    # Concurrent routes with identical source share this content-hash dir;
    # two ninja invocations in one build directory race and can corrupt the
    # .so. Serialize per hash (flock auto-releases if a builder dies).
    import fcntl
    lock_path = build_root / f"{key[:16]}.lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return cpp_extension.load(
            name=f"avo_attn_{key[:12]}",
            sources=[str(s) for s in sources],
            extra_cuda_cflags=cuda_cflags,
            build_directory=str(build_dir),
            verbose=True,
        )
