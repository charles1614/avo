#!/usr/bin/env python3
"""Fetch official NVIDIA documentation (CUDA programming guide, best
practices, PTX ISA, arch tuning guides) into knowledge_base/external/ as
plain text, chunked so the KB indexer picks every part up.

Matches the paper's knowledge base K: 'CUDA programming guides, PTX ISA
documentation, architecture specifications'. Stdlib only — no new deps.
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

DOCS = {
    "cuda_c_programming_guide": "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html",
    "cuda_best_practices_guide": "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html",
    "ptx_isa": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html",
    "ampere_tuning_guide": "https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html",
    "hopper_tuning_guide": "https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html",
}
CHUNK_BYTES = 1_000_000  # keep well under the KB's per-file cap
UA = "Mozilla/5.0 (Macintosh) avo-kb-fetch/0.1"
SKIP_TAGS = {"script", "style", "nav", "footer", "header"}


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self._skip_depth += 1
        elif tag in ("p", "div", "li", "tr", "br", "h1", "h2", "h3", "h4",
                     "pre", "section", "table"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    ex = TextExtractor()
    ex.feed(html)
    text = "".join(ex.parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


TOC_LINE = re.compile(r"\.{6,}\s*\d+\s*$")
PAGE_NUM = re.compile(r"^\s*\d{1,4}\s*$")
CODE_CHARS = set("{};=()<>[]")


def clean_text(text: str) -> str:
    """Deterministic cleanup for converted docs: drop TOC dot-lines, bare page
    numbers, and per-page boilerplate (exact lines repeating >=10x that look
    like prose headers, not code); collapse blank runs. No LLM — the docs stay
    verbatim ground truth, just less noisy for grep-style retrieval."""
    from collections import Counter
    lines = text.splitlines()
    counts = Counter(l.strip() for l in lines if l.strip())
    boiler = {l for l, n in counts.items()
              if n >= 10 and 20 <= len(l) < 100
              and not (set(l) & CODE_CHARS)
              and any(c.isalpha() for c in l)}
    out: list[str] = []
    blank = 0
    for l in lines:
        s = l.strip()
        if s and (TOC_LINE.search(s) or PAGE_NUM.match(s) or s in boiler):
            continue
        blank = blank + 1 if not s else 0
        if blank > 1:
            continue
        out.append(l)
    return "\n".join(out)


def chunk_lines(text: str, chunk_bytes: int) -> list[str]:
    chunks, cur, size = [], [], 0
    for line in text.splitlines(keepends=True):
        b = len(line.encode())
        if size + b > chunk_bytes and cur:
            chunks.append("".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += b
    if cur:
        chunks.append("".join(cur))
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="knowledge_base/external/nvidia_docs")
    ap.add_argument("--clean-existing", action="store_true",
                    help="re-run the deterministic cleanup on already-fetched "
                         "files instead of downloading")
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    if args.clean_existing:
        for f in sorted(dest.glob("*.txt")):
            before = f.stat().st_size
            f.write_text(clean_text(f.read_text(errors="replace")))
            print(f"{f.name}: {before / 1e3:.0f}K -> {f.stat().st_size / 1e3:.0f}K")
        return 0

    import datetime
    import json
    manifest_path = dest.parent / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, url in DOCS.items():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                html = r.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"FAILED {name}: {type(e).__name__}: {e}")
            continue
        text = f"# {name} — source: {url}\n\n" + clean_text(html_to_text(html))
        chunks = chunk_lines(text, CHUNK_BYTES)
        for i, chunk in enumerate(chunks, 1):
            suffix = f".part{i:02d}" if len(chunks) > 1 else ""
            (dest / f"{name}{suffix}.txt").write_text(chunk)
        version = re.search(r"(\d+\.\d+(?:\.\d+)?)\s+documentation", html[:200_000])
        manifest[f"nvidia_docs/{name}"] = {
            "type": "doc", "url": url,
            "version": version.group(1) if version else None,
            "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        print(f"{name}: {len(html) / 1e6:.1f} MB html -> "
              f"{len(text) / 1e6:.1f} MB text in {len(chunks)} file(s)")
    manifest_path.write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
