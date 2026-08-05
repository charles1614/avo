"""Eval cache keyed by content: identical (workspace, harness, params, target) => same result."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from avo.types import ScoreResult

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "build"}


def tree_hash(root: Path) -> str:
    """Deterministic hash of a directory's file contents (skips VCS/build dirs)."""
    h = hashlib.sha256()
    root = Path(root)
    for p in sorted(root.rglob("*")):
        if not p.is_file() or any(part in SKIP_DIRS for part in p.parts):
            continue
        h.update(str(p.relative_to(root)).encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\1")
    return h.hexdigest()


def eval_key(workspace: Path, harness: Path, params: dict, runner_identity: str) -> str:
    h = hashlib.sha256()
    h.update(tree_hash(workspace).encode())
    h.update(tree_hash(harness).encode())
    h.update(json.dumps(params, sort_keys=True).encode())
    h.update(runner_identity.encode())
    return h.hexdigest()


class EvalCache:
    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> ScoreResult | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            result = ScoreResult.from_dict(json.loads(p.read_text()))
        except (json.JSONDecodeError, KeyError):
            return None
        result.cached = True
        result.eval_hash = key
        return result

    def put(self, key: str, result: ScoreResult) -> None:
        result.eval_hash = key
        self._path(key).write_text(result.to_json())
