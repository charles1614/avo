"""Filesystem isolation for agent shell commands.

Multiple evolution routes on one host each own runs/<route>/workspace/. Without
isolation an agent can `cat`/`git show`/`cp` a peer route's workspace, lineage,
or git history (observed: direct solution copying), and can read anything the
experimenter left in a shared /tmp. A regex deny-list cannot stop this — shell
is a full language (`cat $(echo /runs/...)`, base64, etc.). Real confinement
needs a filesystem namespace.

Strategy (bubblewrap, the portable unprivileged option):
  bwrap --ro-bind / /              # whole system read-only (toolchain visible)
        --tmpfs <runs_dir>         # blank the ENTIRE runs tree — peers vanish
        --bind <this_workspace> …  # re-expose ONLY this route's workspace rw
        --tmpfs /tmp               # private /tmp (kills the shared-/tmp leak)
        --unshare-pid/ipc/uts, --die-with-parent, --chdir <workspace>
Later binds override earlier ones, so blanking runs/ then re-binding one
workspace leaves exactly one route visible and writable.

`none` (no sandbox available) confines cwd and applies the deny-list only —
documented as NOT providing cross-route integrity.
"""
from __future__ import annotations

import functools
import shlex
import subprocess
from pathlib import Path

BWRAP_UNSHARE = ["--unshare-pid", "--unshare-ipc", "--unshare-uts",
                 "--die-with-parent"]


@functools.lru_cache(maxsize=8)
def bwrap_works(bwrap: str = "bwrap") -> bool:
    """True iff bwrap can actually create a namespace here (setuid or
    unprivileged userns). Cached — the answer is constant per host."""
    try:
        r = subprocess.run(
            [bwrap, "--ro-bind", "/", "/", "--tmpfs", "/tmp", "true"],
            capture_output=True, timeout=15)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def build_shell_argv(command: str, workspace: Path, mode: str = "auto",
                     runs_dir: Path | None = None) -> tuple[list[str], str]:
    """Return (argv, effective_mode). mode: auto|bwrap|none.
    'auto' uses bwrap when it works, else falls back to none."""
    ws = Path(workspace).resolve()
    if mode == "auto":
        mode = "bwrap" if bwrap_works() else "none"
    if mode == "bwrap":
        # blank the runs tree (hide peers) then re-expose only this workspace
        runs = Path(runs_dir).resolve() if runs_dir else ws.parent.parent
        argv = ["bwrap", "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
                "--tmpfs", "/tmp", "--tmpfs", str(runs),
                "--bind", str(ws), str(ws), *BWRAP_UNSHARE,
                "--chdir", str(ws), "/bin/bash", "-c", command]
        return argv, "bwrap"
    return ["/bin/bash", "-c", command], "none"


def build_remote_shell_cmd(command: str, workspace_abs: str, mode: str,
                           runs_abs: str) -> str:
    """Same policy as a single remote shell string (SSH runner). mode here is
    explicit (bwrap|none) — remote capability isn't probed from the client."""
    if mode == "bwrap":
        inner = " ".join(shlex.quote(a) for a in [
            "bwrap", "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
            "--tmpfs", "/tmp", "--tmpfs", runs_abs,
            "--bind", workspace_abs, workspace_abs, *BWRAP_UNSHARE,
            "--chdir", workspace_abs, "/bin/bash", "-c", command])
        return inner
    return f"cd {shlex.quote(workspace_abs)} && /bin/bash -c {shlex.quote(command)}"
