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
                             parse_result_file, stage_eval_tree)
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
        self._sandbox_resolved: str | None = None

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
            stage_eval_tree(staged, workspace, harness)
            (staged / "result.json").unlink(missing_ok=True)

            remote_dir = self._remote("eval")
            mk = self._ssh(f"mkdir -p {shlex.quote(remote_dir)}", 60)
            if mk.exit_code != 0:
                return ScoreResult.failure("harness", "ssh mkdir failed", mk.render())
            up = self._rsync(staged, remote_dir)
            if up.exit_code != 0:
                return ScoreResult.failure("harness", "rsync to remote failed", up.render())

        from avo.eval.runner import new_result_token, verify_token
        token = new_result_token()
        score_cmd = " ".join(shlex.quote(a) for a in
                             _score_cmd(self.cfg.python, score_entry, params,
                                        token))
        inner = f"timeout {self.cfg.eval_timeout_s} {score_cmd}"
        lock_wait = self.cfg.eval_timeout_s * 2 + 300
        env_prefix = "".join(f"{k}={shlex.quote(v)} "
                             for k, v in self.cfg.eval_env().items())
        if env_prefix:
            inner = env_prefix + inner
        if self.cfg.lock_path():
            # serialize with other AVO runs on the SAME remote GPU (per-device
            # lock path; util-linux flock)
            inner = (f"flock -w {lock_wait} {shlex.quote(self.cfg.lock_path())} "
                     f"-c {shlex.quote(inner)}")
        remote_cmd = self._wrap_env(
            f"cd {shlex.quote(remote_dir)} && rm -f result.json && {inner}")
        run = self._ssh(remote_cmd, self.cfg.eval_timeout_s + lock_wait + 120)
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
            return verify_token(parse_result_file(local_result, run.render()),
                                token)
        finally:
            local_result.unlink(missing_ok=True)

    def _sandbox_mode(self) -> str:
        """Resolve 'auto' by probing the remote host for a working bwrap once."""
        if self._sandbox_resolved is not None:
            return self._sandbox_resolved
        mode = self.cfg.sandbox
        if mode == "auto":
            probe = self._ssh(
                "bwrap --ro-bind / / --tmpfs /tmp true >/dev/null 2>&1 "
                "&& echo bwrap || echo none", 30)
            mode = "bwrap" if probe.stdout.strip() == "bwrap" else "none"
        self._sandbox_resolved = mode
        return mode

    def run_shell(self, workspace: Path, command: str,
                  timeout_s: int | None = None) -> ShellResult:
        """gpu_shell: sync the workspace to the remote work dir, run there —
        isolated from peer routes' remote scratch when bwrap is available."""
        from avo.agent.sandbox import build_remote_shell_cmd
        t = timeout_s or self.cfg.shell_timeout_s
        remote_dir = self._remote("work/workspace")
        mk = self._ssh(f"mkdir -p {shlex.quote(remote_dir)}", 60)
        if mk.exit_code != 0:
            return mk
        up = self._rsync(workspace, remote_dir)
        if up.exit_code != 0:
            return up
        inner = build_remote_shell_cmd(command, remote_dir,
                                       self._sandbox_mode(), self._abs_scratch())
        return self._ssh(self._wrap_env(f"timeout {t} {inner}"), t + 60)


def make_runner(cfg: RunnerConfig, run_id: str) -> Runner:
    from avo.eval.runner import LocalRunner
    if cfg.kind == "ssh":
        return SSHRunner(cfg, run_id)
    return LocalRunner(cfg)
