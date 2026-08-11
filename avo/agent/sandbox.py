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
import os
import shlex
import shutil
import subprocess
from pathlib import Path

BWRAP_UNSHARE = ["--unshare-pid", "--unshare-ipc", "--unshare-uts",
                 "--die-with-parent"]


def route_uid(run_name: str, base: int = 60_000, span: int = 4_000) -> int:
    """Stable unprivileged uid for a route. No /etc/passwd entry is needed —
    setpriv accepts a bare numeric uid — so this creates no users and touches
    no container state."""
    import hashlib
    h = int(hashlib.sha256(run_name.encode()).hexdigest()[:8], 16)
    return base + (h % span)


@functools.lru_cache(maxsize=2)
def uid_isolation_available() -> bool:
    """POSIX-permission isolation: run agent shells as a per-route
    unprivileged uid so peers' 0700 run dirs are unreadable. Needs root (to
    drop privileges) but NO capabilities, NO user namespaces — it works in
    locked-down containers where bwrap/proot cannot."""
    return os.geteuid() == 0 and shutil.which("setpriv") is not None


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


class SandboxUnavailable(RuntimeError):
    """Raised when isolation was REQUIRED but no mechanism works here."""


def resolve_mode(mode: str) -> str:
    """auto/require -> the mechanism actually usable on this host.
    Order: bwrap (namespaces) > uid (POSIX permissions) > none.
    'require' raises instead of degrading — a multi-route experiment that
    silently loses isolation produces contaminated results (observed: a route
    bootstrapped from another run's /tmp residue)."""
    if mode in ("bwrap", "uid", "none"):
        return mode
    if bwrap_works():
        return "bwrap"
    if uid_isolation_available():
        return "uid"
    if mode == "require":
        raise SandboxUnavailable(
            "filesystem isolation REQUIRED but unavailable: bwrap cannot "
            "create a namespace (no CAP_SYS_ADMIN / unprivileged userns) and "
            "uid isolation needs root + setpriv. Options: (a) run one route "
            "per container/pod — the container is then the sandbox — "
            "(b) run the framework as root inside the container so per-route "
            "uid isolation can engage, or (c) set sandbox: none to accept "
            "NO isolation (single-route experiments only).")
    return "none"


def build_shell_argv(command: str, workspace: Path, mode: str = "auto",
                     runs_dir: Path | None = None,
                     uid: int | None = None,
                     tmpdir: Path | None = None) -> tuple[list[str], str]:
    """Return (argv, effective_mode). mode: auto|require|bwrap|uid|none."""
    ws = Path(workspace).resolve()
    mode = resolve_mode(mode)
    if mode == "uid":
        # drop to the route's uid: peers' run dirs are 0700 and owned by
        # different uids, so they are unreadable; TMPDIR is private
        env = ["env", f"TMPDIR={tmpdir or ws / '.tmp'}",
               f"TMP={tmpdir or ws / '.tmp'}", f"HOME={ws}"]
        argv = ["setpriv", f"--reuid={uid}", f"--regid={uid}",
                "--clear-groups", "--inh-caps=-all", *env,
                "/bin/bash", "-c", f"cd {shlex.quote(str(ws))} && {command}"]
        return argv, "uid"
    if mode == "bwrap":
        # blank the runs tree (hide peers) then re-expose only this workspace
        runs = Path(runs_dir).resolve() if runs_dir else ws.parent.parent
        argv = ["bwrap", "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
                "--tmpfs", "/tmp", "--tmpfs", str(runs),
                "--bind", str(ws), str(ws), *BWRAP_UNSHARE,
                "--chdir", str(ws), "/bin/bash", "-c", command]
        return argv, "bwrap"
    return ["/bin/bash", "-c", command], "none"


AVO_RESIDUE_GLOBS = ("avo_eval_*", "avo_stage_*", "avo_*", "*attention*.cu",
                     "*.ptx", "*kernel*.cu")


def readable_residue(tmp_dir: str = "/tmp", limit: int = 12) -> list[str]:
    """Readable leftovers in a shared /tmp that an agent could bootstrap from.

    This is exactly how R4 was contaminated: a route ran `ls /tmp`, found a
    previous run's kernel sources, and reached 181 TFLOPS in 3 steps instead
    of deriving anything. Without isolation the framework must at least refuse
    to start on a dirty /tmp.
    """
    hits: list[str] = []
    root = Path(tmp_dir)
    if not root.is_dir():
        return hits
    for pattern in AVO_RESIDUE_GLOBS:
        for p in root.glob(pattern):
            try:
                if os.access(p, os.R_OK):
                    hits.append(str(p))
            except OSError:
                continue
            if len(hits) >= limit:
                return sorted(set(hits))
    return sorted(set(hits))


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
