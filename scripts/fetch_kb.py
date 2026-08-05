#!/usr/bin/env python3
"""Fetch external knowledge-base material (FlashAttention-2 sources) into
knowledge_base/external/ (gitignored). Network access, no LLM, no GPU."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/Dao-AILab/flash-attention.git"
KEEP = ["csrc/flash_attn/src", "flash_attn/flash_attn_interface.py", "hopper"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None, help="tag/commit to fetch (default: HEAD)")
    ap.add_argument("--dest", default="knowledge_base/external")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    clone_dir = dest / "_clone_tmp"
    final_dir = dest / "flash-attention"
    if final_dir.exists():
        print(f"{final_dir} already exists; delete it to re-fetch")
        return 0
    shutil.rmtree(clone_dir, ignore_errors=True)

    cmd = ["git", "clone", "--depth", "1", REPO, str(clone_dir)]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if args.ref:
        subprocess.run(["git", "-C", str(clone_dir), "fetch", "--depth", "1",
                        "origin", args.ref], check=True)
        subprocess.run(["git", "-C", str(clone_dir), "checkout", "FETCH_HEAD"],
                       check=True)

    final_dir.mkdir()
    for rel in KEEP:
        src = clone_dir / rel
        if not src.exists():
            print(f"  (skipping missing {rel})")
            continue
        dst = final_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"  kept {rel}")
    sha = subprocess.run(["git", "-C", str(clone_dir), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    (final_dir / "SOURCE.txt").write_text(f"{REPO} @ {sha}\n")
    shutil.rmtree(clone_dir)
    print(f"done: {final_dir} (@{sha[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
