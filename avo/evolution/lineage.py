"""Git-backed single-lineage persistence.

The evolution workspace at runs/<id>/workspace/ is its own git repo, seeded
from tasks/<task>/seed/. Every gate-passing submit becomes a commit tagged
vNNNN with machine-readable trailers; lineage.jsonl mirrors the metadata (the
query path — git stays the durable artifact).
"""
from __future__ import annotations

import datetime
import shutil
import subprocess
from pathlib import Path

from avo.types import LineageEntry


def _git(workspace: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(["git", "-C", str(workspace), *args],
                          capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class Lineage:
    def __init__(self, workspace: Path, jsonl_path: Path):
        self.workspace = Path(workspace)
        self.jsonl_path = Path(jsonl_path)

    # -- setup ----------------------------------------------------------------

    @classmethod
    def init_run(cls, run_dir: Path, seed_dir: Path) -> "Lineage":
        workspace = run_dir / "workspace"
        shutil.copytree(seed_dir, workspace,
                        ignore=shutil.ignore_patterns(".git", "__pycache__"))
        _git(workspace, "init", "-b", "main")
        _git(workspace, "config", "user.name", "avo")
        _git(workspace, "config", "user.email", "avo@localhost")
        _git(workspace, "add", "-A")
        _git(workspace, "commit", "-m", "seed")
        lineage = cls(workspace, run_dir / "lineage.jsonl")
        return lineage

    @classmethod
    def load(cls, run_dir: Path) -> "Lineage":
        lineage = cls(run_dir / "workspace", run_dir / "lineage.jsonl")
        lineage.verify()
        return lineage

    # -- queries ----------------------------------------------------------------

    def entries(self) -> list[LineageEntry]:
        if not self.jsonl_path.exists():
            return []
        out = []
        import json
        for line in self.jsonl_path.read_text().splitlines():
            if line.strip():
                out.append(LineageEntry.from_dict(json.loads(line)))
        return out

    def best(self) -> LineageEntry | None:
        entries = self.entries()
        return max(entries, key=lambda e: e.score) if entries else None

    def tags(self) -> list[str]:
        return [t for t in _git(self.workspace, "tag", "--list", "v*").split()
                if t.strip()]

    def verify(self) -> None:
        """Cross-check lineage.jsonl against git tags (resume safety)."""
        tags = set(self.tags())
        versions = {e.version for e in self.entries()}
        if tags != versions:
            raise RuntimeError(
                f"lineage.jsonl and git tags disagree: jsonl={sorted(versions)} "
                f"tags={sorted(tags)}")

    # -- mutations ---------------------------------------------------------------

    def record_seed(self, score: float, eval_hash: str) -> LineageEntry:
        sha = _git(self.workspace, "rev-parse", "HEAD").strip()
        _git(self.workspace, "tag", "v0000")
        entry = LineageEntry(version="v0000", step=0, commit=sha, score=score,
                             message="seed", eval_hash=eval_hash, parent=None,
                             timestamp=_now())
        self._append(entry)
        return entry

    def commit_version(self, step: int, score: float, message: str,
                       eval_hash: str) -> LineageEntry:
        entries = self.entries()
        version = f"v{len(entries):04d}"
        parent = entries[-1].version if entries else None
        full_message = (f"{message}\n\nAVO-Score: {score:.6f}\nAVO-Step: {step}\n"
                        f"AVO-Eval-Hash: {eval_hash}")
        _git(self.workspace, "add", "-A")
        _git(self.workspace, "commit", "-m", full_message)
        sha = _git(self.workspace, "rev-parse", "HEAD").strip()
        _git(self.workspace, "tag", version)
        entry = LineageEntry(version=version, step=step, commit=sha, score=score,
                             message=message, eval_hash=eval_hash, parent=parent,
                             timestamp=_now())
        self._append(entry)
        return entry

    def capture_uncommitted_patch(self) -> str:
        """Diff of the working tree (incl. new files) vs HEAD, without committing."""
        _git(self.workspace, "add", "-A")
        patch = _git(self.workspace, "diff", "--cached")
        _git(self.workspace, "reset", "HEAD", check=False)
        return patch

    def reset_workspace(self) -> None:
        """Discard everything uncommitted (failed step cleanup)."""
        _git(self.workspace, "reset", "--hard", "HEAD")
        _git(self.workspace, "clean", "-fd")

    def last_commit_diff(self, max_chars: int = 12_000) -> str:
        entries = self.entries()
        if len(entries) < 2:
            return ""
        diff = _git(self.workspace, "show", "--format=%s", entries[-1].version)
        return diff[:max_chars] + ("\n... [diff truncated]" if len(diff) > max_chars else "")

    def _append(self, entry: LineageEntry) -> None:
        with open(self.jsonl_path, "a") as f:
            f.write(entry.to_json_line() + "\n")
