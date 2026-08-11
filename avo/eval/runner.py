"""Execution backends. A Runner stages {workspace, harness} and runs the
scoring harness (and ad-hoc shell commands) either locally or on a remote GPU
host. The harness copy always comes from the pristine task directory."""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
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


def _run(cmd: list[str], cwd: Path | None, timeout_s: int,
         env_extra: dict | None = None) -> ShellResult:
    env = {**os.environ, **env_extra} if env_extra else None
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout_s, env=env)
        return ShellResult(proc.returncode, proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired as e:
        return ShellResult(-1, _dec(e.stdout), _dec(e.stderr), timed_out=True)


def _dec(x) -> str:
    if x is None:
        return ""
    return x.decode(errors="replace") if isinstance(x, bytes) else str(x)


@contextmanager
def gpu_lock(path: str, wait_timeout_s: float):
    """Exclusive cross-process lock serializing GPU evals from concurrent AVO
    runs. Acquired BEFORE the eval subprocess starts, so waiting never eats
    into the eval's own timeout. Auto-released on process death (flock)."""
    if not path:
        yield
        return
    f = open(os.path.expanduser(path), "w")
    deadline = time.monotonic() + wait_timeout_s
    try:
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"GPU lock {path} still held after "
                        f"{wait_timeout_s:.0f}s") from None
                time.sleep(1)
        f.write(str(os.getpid()))
        f.flush()
        yield
    finally:
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
        except OSError:
            pass
        f.close()


def resolve_python(python: str) -> str:
    """Path-like interpreter settings (e.g. `.venv/bin/python`) resolve
    against the framework's launch directory — the harness subprocess runs
    with a staging temp dir as cwd, where a relative path would be wrong.
    Bare command names (`python3`) stay as PATH lookups."""
    if "/" in python:
        # absolute() not resolve(): a venv's python is a symlink, and
        # resolving it would bypass the venv (pyvenv.cfg lives beside the link)
        return str(Path(python).absolute())
    return python


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
            try:
                with gpu_lock(self.cfg.lock_path(),
                              wait_timeout_s=self.cfg.eval_timeout_s * 2 + 300):
                    res = _run(_score_cmd(resolve_python(self.cfg.python),
                                          score_entry, params),
                               cwd=staged, timeout_s=self.cfg.eval_timeout_s,
                               env_extra=self.cfg.eval_env())
            except TimeoutError as e:
                return ScoreResult.failure("harness", str(e))
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
