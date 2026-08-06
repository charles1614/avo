#!/usr/bin/env python3
"""Fetch external knowledge-base material into knowledge_base/external/
(gitignored): FlashAttention-2/3 sources and the CUTLASS/CuTe headers they
build on. Records what was fetched in MANIFEST.json so
scripts/check_kb_freshness.py can detect upstream updates.

Network access; no LLM, no GPU.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPOS = {
    "flash-attention": {
        "repo": "https://github.com/Dao-AILab/flash-attention.git",
        "keep": ["csrc/flash_attn/src", "flash_attn/flash_attn_interface.py",
                 "hopper"],
    },
    "cutlass": {
        "repo": "https://github.com/NVIDIA/cutlass.git",
        # cute core + the arch-level mma/copy PTX wrappers: the layers an
        # attention-kernel author actually reads
        "keep": ["include/cute", "include/cutlass/arch"],
    },
    "kernelbench": {
        "repo": "https://github.com/ScalingIntelligence/KernelBench.git",
        # problem definitions only (reference Model + get_inputs per file);
        # consumed by scripts/run_kernelbench.py, evaluated by OUR harness
        "keep": ["KernelBench/level1", "KernelBench/level2",
                 "KernelBench/level3", "LICENSE.md"],
    },
}


def load_manifest(dest: Path) -> dict:
    p = dest / "MANIFEST.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_manifest(dest: Path, manifest: dict) -> None:
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))


def fetch_repo(name: str, spec: dict, dest: Path, ref: str | None) -> dict:
    clone_dir = dest / "_clone_tmp"
    final_dir = dest / name
    if final_dir.exists():
        print(f"{final_dir} already exists; delete it to re-fetch")
        return {}
    shutil.rmtree(clone_dir, ignore_errors=True)
    cmd = ["git", "clone", "--depth", "1", spec["repo"], str(clone_dir)]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if ref:
        subprocess.run(["git", "-C", str(clone_dir), "fetch", "--depth", "1",
                        "origin", ref], check=True)
        subprocess.run(["git", "-C", str(clone_dir), "checkout", "FETCH_HEAD"],
                       check=True)
    final_dir.mkdir()
    for rel in spec["keep"]:
        src = clone_dir / rel
        if not src.exists():
            print(f"  (skipping missing {rel})")
            continue
        dst = final_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)
        print(f"  kept {rel}")
    sha = subprocess.run(["git", "-C", str(clone_dir), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    (final_dir / "SOURCE.txt").write_text(f"{spec['repo']} @ {sha}\n")
    shutil.rmtree(clone_dir)
    print(f"done: {final_dir} (@{sha[:12]})")
    return {"type": "git", "repo": spec["repo"], "commit": sha,
            "kept": spec["keep"],
            "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None, help="tag/commit to fetch (default: HEAD)")
    ap.add_argument("--dest", default="knowledge_base/external")
    ap.add_argument("--only", choices=list(REPOS), default=None)
    ap.add_argument("--from-manifest", action="store_true",
                    help="restore the exact commits recorded in MANIFEST.json "
                         "(reproducible KB on a fresh clone)")
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(dest)
    for name, spec in REPOS.items():
        if args.only and name != args.only:
            continue
        ref = args.ref
        if args.from_manifest:
            pinned = manifest.get(name, {}).get("commit")
            if not pinned:
                print(f"{name}: no pinned commit in MANIFEST.json; skipping")
                continue
            ref = pinned
        entry = fetch_repo(name, spec, dest, ref)
        if entry:
            manifest[name] = entry
    save_manifest(dest, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
