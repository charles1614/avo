"""Execution backends. A Runner stages {workspace, harness} and runs the
scoring harness (and ad-hoc shell commands) either locally or on a remote GPU
host. The harness copy always comes from the pristine task directory."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from avo.config import RunnerConfig
from avo.eval.scoring import encode_params
from avo.types import ScoreResult, ShellResult, truncate_middle

COPY_IGNORE = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache")


class Runner(ABC):
    def __init__(self, cfg: RunnerConfig):
        self.cfg = cfg

    @abstractmethod
    def score(self, workspace: Path, harness: Path, score_entry: str,
              params: dict) -> ScoreResult:
        ...

    @abstractmethod
    def run_shell(self, workspace: Path, command: str,
                  timeout_s: int | None = None) -> ShellResult:
        """Run a command with the (synced) workspace as cwd, on the eval target."""

    def identity(self) -> str:
        return self.cfg.identity()


def _run(cmd: list[str], cwd: Path | None, timeout_s: int) -> ShellResult:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout_s)
        return ShellResult(proc.returncode, proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired as e:
        return ShellResult(-1, _dec(e.stdout), _dec(e.stderr), timed_out=True)


def _dec(x) -> str:
    if x is None:
        return ""
    return x.decode(errors="replace") if isinstance(x, bytes) else str(x)


def _score_cmd(python: str, score_entry: str, params: dict) -> list[str]:
    return [python, f"harness/{score_entry}", "--workspace", "workspace",
            "--params-b64", encode_params(params), "--out", "result.json"]


def parse_result_file(path: Path, run_log: str) -> ScoreResult:
    if not path.exists():
        return ScoreResult.failure("harness", "harness produced no result.json",
                                   truncate_middle(run_log, 20_000))
    try:
        return ScoreResult.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return ScoreResult.failure("harness", f"invalid result.json: {e}",
                                   truncate_middle(run_log, 20_000))


class LocalRunner(Runner):
    def score(self, workspace: Path, harness: Path, score_entry: str,
              params: dict) -> ScoreResult:
        with tempfile.TemporaryDirectory(prefix="avo_eval_") as td:
            staged = Path(td)
            shutil.copytree(workspace, staged / "workspace", ignore=COPY_IGNORE)
            shutil.copytree(harness, staged / "harness", ignore=COPY_IGNORE)
            res = _run(_score_cmd(self.cfg.python, score_entry, params),
                       cwd=staged, timeout_s=self.cfg.eval_timeout_s)
            if res.timed_out:
                return ScoreResult.failure(
                    "harness", f"eval timed out after {self.cfg.eval_timeout_s}s",
                    res.render())
            return parse_result_file(staged / "result.json",
                                     res.stdout + "\n" + res.stderr)

    def run_shell(self, workspace: Path, command: str,
                  timeout_s: int | None = None) -> ShellResult:
        return _run(["/bin/bash", "-c", command], cwd=workspace,
                    timeout_s=timeout_s or self.cfg.shell_timeout_s)
