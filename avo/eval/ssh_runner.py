"""SSH runner: framework runs on the Mac, build/bench execute on the GPU host.

Plain subprocess ssh/rsync over the user's ssh config (no paramiko).
ControlMaster/ControlPersist keeps one connection alive across the many evals
of a long run. Remote layout:

    <scratch>/<run_id>/eval/{workspace/, harness/, result.json}   # scoring
    <scratch>/<run_id>/work/workspace/                            # gpu_shell
    <scratch>/build_cache/<hash>/                                 # nvcc builds (harness-managed)
"""
from __future__ import annotations

import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from avo.config import RunnerConfig
from avo.eval.runner import (COPY_IGNORE, Runner, _dec, _score_cmd,
                             parse_result_file)
from avo.types import ScoreResult, ShellResult

SSH_OPTS = ["-o", "ControlMaster=auto", "-o", "ControlPath=~/.ssh/avo-%r@%h-%p",
            "-o", "ControlPersist=600", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=15"]
TRANSPORT_EXIT = 255  # ssh's own failure code, distinct from remote command failure
RETRIES = 2


class SSHRunner(Runner):
    def __init__(self, cfg: RunnerConfig, run_id: str):
        super().__init__(cfg)
        if not cfg.host:
            raise ValueError("runner.kind=ssh requires runner.host")
        self.host = cfg.host
        self.run_id = run_id
        self._home: str | None = None

    # -- low-level helpers ---------------------------------------------------

    def _abs_scratch(self) -> str:
        """Resolve '~' in the scratch path to the remote $HOME once: quoted
        remote commands must never rely on shell tilde expansion."""
        s = self.cfg.scratch.rstrip("/")
        if s == "~" or s.startswith("~/"):
            if self._home is None:
                res = self._ssh("echo $HOME", 30)
                home = res.stdout.strip()
                if res.exit_code != 0 or not home.startswith("/"):
                    raise RuntimeError(f"cannot resolve remote $HOME: {res.render()}")
                self._home = home
            s = self._home + s[1:]
        return s

    def _remote(self, sub: str) -> str:
        return f"{self._abs_scratch()}/{self.run_id}/{sub}"

    def _ssh(self, remote_cmd: str, timeout_s: int) -> ShellResult:
        cmd = ["ssh", *SSH_OPTS, self.host, remote_cmd]
        last = ShellResult(TRANSPORT_EXIT, "", "ssh not attempted")
        for attempt in range(RETRIES + 1):
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=timeout_s)
                last = ShellResult(proc.returncode, proc.stdout, proc.stderr)
            except subprocess.TimeoutExpired as e:
                return ShellResult(-1, _dec(e.stdout), _dec(e.stderr), timed_out=True)
            if last.exit_code != TRANSPORT_EXIT:
                return last
            time.sleep(2 ** attempt)
        return last

    def _rsync(self, local_dir: Path, remote_dir: str) -> ShellResult:
        cmd = ["rsync", "-az", "--delete",
               "--exclude", ".git", "--exclude", "__pycache__",
               "-e", "ssh " + " ".join(SSH_OPTS),
               f"{local_dir}/", f"{self.host}:{remote_dir}/"]
        for attempt in range(RETRIES + 1):
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode == 0:
                return ShellResult(0, proc.stdout, proc.stderr)
            time.sleep(2 ** attempt)
        return ShellResult(proc.returncode, proc.stdout, proc.stderr)

    def _wrap_env(self, remote_cmd: str) -> str:
        if self.cfg.env_activate:
            return f"{self.cfg.env_activate} && {remote_cmd}"
        return remote_cmd

    # -- Runner interface ----------------------------------------------------

    def score(self, workspace: Path, harness: Path, score_entry: str,
              params: dict) -> ScoreResult:
        with tempfile.TemporaryDirectory(prefix="avo_stage_") as td:
            staged = Path(td)
            shutil.copytree(workspace, staged / "workspace", ignore=COPY_IGNORE)
            shutil.copytree(harness, staged / "harness", ignore=COPY_IGNORE)
            (staged / "result.json").unlink(missing_ok=True)

            remote_dir = self._remote("eval")
            mk = self._ssh(f"mkdir -p {shlex.quote(remote_dir)}", 60)
            if mk.exit_code != 0:
                return ScoreResult.failure("harness", "ssh mkdir failed", mk.render())
            up = self._rsync(staged, remote_dir)
            if up.exit_code != 0:
                return ScoreResult.failure("harness", "rsync to remote failed", up.render())

        score_cmd = " ".join(shlex.quote(a) for a in
                             _score_cmd(self.cfg.python, score_entry, params))
        remote_cmd = self._wrap_env(
            f"cd {shlex.quote(remote_dir)} && rm -f result.json && "
            f"timeout {self.cfg.eval_timeout_s} {score_cmd}")
        run = self._ssh(remote_cmd, self.cfg.eval_timeout_s + 120)
        if run.timed_out or run.exit_code == 124:  # 124 = remote `timeout`
            return ScoreResult.failure(
                "harness", f"eval timed out after {self.cfg.eval_timeout_s}s",
                run.render())

        cat = self._ssh(f"cat {shlex.quote(remote_dir)}/result.json", 60)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            local_result = Path(f.name)
            if cat.exit_code == 0:
                f.write(cat.stdout)
        try:
            if cat.exit_code != 0:
                local_result.unlink(missing_ok=True)
                return parse_result_file(Path("/nonexistent"),
                                         run.render() + "\n" + cat.render())
            return parse_result_file(local_result, run.render())
        finally:
            local_result.unlink(missing_ok=True)

    def run_shell(self, workspace: Path, command: str,
                  timeout_s: int | None = None) -> ShellResult:
        """gpu_shell: sync the workspace to the remote work dir, run there."""
        t = timeout_s or self.cfg.shell_timeout_s
        remote_dir = self._remote("work/workspace")
        mk = self._ssh(f"mkdir -p {shlex.quote(remote_dir)}", 60)
        if mk.exit_code != 0:
            return mk
        up = self._rsync(workspace, remote_dir)
        if up.exit_code != 0:
            return up
        return self._ssh(self._wrap_env(
            f"cd {shlex.quote(remote_dir)} && timeout {t} bash -c {shlex.quote(command)}"),
            t + 60)


def make_runner(cfg: RunnerConfig, run_id: str) -> Runner:
    from avo.eval.runner import LocalRunner
    if cfg.kind == "ssh":
        return SSHRunner(cfg, run_id)
    return LocalRunner(cfg)
