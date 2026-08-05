"""Knowledge base K: curated docs + external kernel sources, with grep-style retrieval."""
from __future__ import annotations

import re
from pathlib import Path

from avo.types import truncate_middle

TEXT_EXTS = {".md", ".txt", ".rst", ".cu", ".cuh", ".h", ".hpp", ".c", ".cc",
             ".cpp", ".py", ".ptx", ".cmake", ".yaml", ".yml"}
MAX_FILE_BYTES = 2_000_000
MAX_RESULT_BYTES = 8_000
CONTEXT_LINES = 1  # lines of context around each hit -> 3-line snippets


class KnowledgeBase:
    def __init__(self, dirs: list[Path]):
        self.dirs = [Path(d) for d in dirs if Path(d).is_dir()]

    def _iter_files(self):
        for root in self.dirs:
            for p in sorted(root.rglob("*")):
                if (p.is_file() and p.suffix.lower() in TEXT_EXTS
                        and p.stat().st_size <= MAX_FILE_BYTES
                        and ".git" not in p.parts):
                    yield root, p

    def search(self, query: str, max_results: int = 30) -> str:
        """Regex search (falls back to literal) across all indexed files.
        Returns `path:line:` hits with surrounding context."""
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            pattern = re.compile(re.escape(query), re.IGNORECASE)
        chunks: list[str] = []
        hits = 0
        for root, p in self._iter_files():
            try:
                lines = p.read_text(errors="replace").splitlines()
            except OSError:
                continue
            rel = f"{root.name}/{p.relative_to(root)}"
            for i, line in enumerate(lines):
                if pattern.search(line):
                    lo, hi = max(0, i - CONTEXT_LINES), min(len(lines), i + CONTEXT_LINES + 1)
                    snippet = "\n".join(f"{rel}:{j + 1}: {lines[j][:240]}"
                                        for j in range(lo, hi))
                    chunks.append(snippet)
                    hits += 1
                    if hits >= max_results:
                        break
            if hits >= max_results:
                break
        if not chunks:
            return f"no matches for {query!r} in knowledge base"
        return truncate_middle("\n---\n".join(chunks), MAX_RESULT_BYTES)

    def read(self, rel_path: str, start_line: int | None = None,
             end_line: int | None = None) -> str:
        """Read a knowledge-base file by the `<root_name>/<relative>` path
        shown in search results (bare relative paths also work)."""
        for root in self.dirs:
            candidates = [root / rel_path]
            if rel_path.startswith(root.name + "/"):
                candidates.insert(0, root / rel_path[len(root.name) + 1:])
            for cand in candidates:
                try:
                    resolved = cand.resolve()
                    resolved.relative_to(root.resolve())  # confinement check
                except (ValueError, OSError):
                    continue
                if resolved.is_file():
                    lines = resolved.read_text(errors="replace").splitlines()
                    lo = (start_line - 1) if start_line else 0
                    hi = end_line if end_line else len(lines)
                    body = "\n".join(f"{i + 1}\t{l}" for i, l in
                                     enumerate(lines[lo:hi], start=lo))
                    return truncate_middle(body, 20_000)
        return f"file not found in knowledge base: {rel_path!r}"
