#!/usr/bin/env python3
"""Report whether knowledge_base/external/ content is behind upstream.

- git sources (flash-attention, cutlass): compares the pinned commit in
  MANIFEST.json against `git ls-remote` HEAD.
- NVIDIA docs: compares the recorded doc version against the version string
  on the live docs page title (e.g. "... 13.3 documentation").

Read-only network checks; changes nothing. Exit code 1 if anything is stale.
Usage: python scripts/check_kb_freshness.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

MANIFEST = Path("knowledge_base/external/MANIFEST.json")
UA = "Mozilla/5.0 (Macintosh) avo-kb-fetch/0.1"


def remote_head(repo: str) -> str | None:
    try:
        out = subprocess.run(["git", "ls-remote", repo, "HEAD"],
                             capture_output=True, text=True, timeout=60).stdout
        return out.split()[0] if out.split() else None
    except (subprocess.TimeoutExpired, IndexError):
        return None


def live_doc_version(url: str) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            head = r.read(200_000).decode("utf-8", errors="replace")
    except Exception:
        return None
    m = re.search(r"(\d+\.\d+(?:\.\d+)?)\s+documentation", head)
    return m.group(1) if m else None


def main() -> int:
    if not MANIFEST.exists():
        print("no MANIFEST.json — run scripts/fetch_kb.py and "
              "scripts/fetch_nvidia_docs.py first")
        return 1
    manifest = json.loads(MANIFEST.read_text())
    stale = 0
    for name, entry in sorted(manifest.items()):
        if entry.get("type") == "git":
            head = remote_head(entry["repo"])
            if head is None:
                status = "UNKNOWN (ls-remote failed)"
            elif head == entry["commit"]:
                status = "up to date"
            else:
                status, stale = f"STALE (upstream {head[:12]}, local {entry['commit'][:12]})", stale + 1
        elif entry.get("type") == "doc":
            live = live_doc_version(entry["url"])
            if live is None:
                status = "UNKNOWN (no version string found)"
            elif live == entry.get("version"):
                status = "up to date"
            else:
                status, stale = f"STALE (upstream v{live}, local v{entry.get('version')})", stale + 1
        else:
            status = "UNKNOWN (unrecognized entry type)"
        print(f"{name:28} {status}   (fetched {entry.get('fetched_at', '?')[:10]})")
    if stale:
        print(f"\n{stale} source(s) stale. Re-fetch between runs with "
              "scripts/fetch_kb.py (delete the stale dir first) and/or "
              "scripts/fetch_nvidia_docs.py.")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
