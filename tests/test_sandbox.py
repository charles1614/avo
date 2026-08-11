"""Filesystem-isolation command construction (offline; real bwrap isolation
must be validated on a Linux box — see the test at the bottom that self-skips)."""
import shutil

import pytest

from avo.agent.sandbox import (build_remote_shell_cmd, build_shell_argv,
                               bwrap_works)


def test_bwrap_argv_hides_runs_and_reexposes_workspace(tmp_path):
    runs = tmp_path / "runs"
    ws = runs / "routeA" / "workspace"
    ws.mkdir(parents=True)
    argv, mode = build_shell_argv("cat x", ws, mode="bwrap", runs_dir=runs)
    assert mode == "bwrap" and argv[0] == "bwrap"
    s = " ".join(argv)
    # whole FS read-only, then the runs tree blanked, then THIS ws re-bound rw
    assert "--ro-bind / /" in s
    assert f"--tmpfs {runs.resolve()}" in s          # peers vanish
    assert f"--bind {ws.resolve()} {ws.resolve()}" in s
    assert "--tmpfs /tmp" in s                        # private /tmp
    assert argv[-3:] == ["/bin/bash", "-c", "cat x"]
    # ordering: runs blanked BEFORE this workspace is re-exposed
    assert s.index(f"--tmpfs {runs.resolve()}") < s.index(f"--bind {ws.resolve()}")


def test_none_mode_is_plain_bash(tmp_path):
    argv, mode = build_shell_argv("ls", tmp_path, mode="none")
    assert mode == "none" and argv == ["/bin/bash", "-c", "ls"]


def test_auto_falls_back_to_none_without_bwrap(tmp_path, monkeypatch):
    monkeypatch.setattr("avo.agent.sandbox.bwrap_works", lambda *a, **k: False)
    argv, mode = build_shell_argv("ls", tmp_path, mode="auto")
    assert mode == "none"


def test_remote_bwrap_cmd_blanks_scratch(tmp_path):
    cmd = build_remote_shell_cmd("nvcc x.cu", "/home/u/avo_scratch/r/work/workspace",
                                 "bwrap", "/home/u/avo_scratch")
    assert "bwrap" in cmd and "--tmpfs /home/u/avo_scratch" in cmd
    assert "/home/u/avo_scratch/r/work/workspace" in cmd
    none_cmd = build_remote_shell_cmd("ls", "/home/u/avo_scratch/r/work/workspace",
                                      "none", "/home/u/avo_scratch")
    assert none_cmd.startswith("cd ") and "bwrap" not in none_cmd


def test_require_mode_refuses_instead_of_degrading(monkeypatch):
    """The R4 lesson: degrading to no isolation silently invalidated a
    multi-route experiment. `require` must abort with actionable guidance."""
    from avo.agent.sandbox import SandboxUnavailable, resolve_mode
    monkeypatch.setattr("avo.agent.sandbox.bwrap_works", lambda *a, **k: False)
    monkeypatch.setattr("avo.agent.sandbox.uid_isolation_available",
                        lambda: False)
    assert resolve_mode("auto") == "none"          # degrades, with a warning
    with pytest.raises(SandboxUnavailable) as e:
        resolve_mode("require")
    msg = str(e.value)
    assert "one route per container" in msg and "root" in msg


def test_uid_mode_drops_privileges_and_pins_tmpdir(tmp_path):
    ws = tmp_path / "runs" / "routeA" / "workspace"
    ws.mkdir(parents=True)
    argv, mode = build_shell_argv("nvcc -V", ws, mode="uid", uid=60123,
                                  tmpdir=tmp_path / "private")
    assert mode == "uid" and argv[0] == "setpriv"
    s = " ".join(argv)
    assert "--reuid=60123" in s and "--regid=60123" in s
    assert "--clear-groups" in s and "--inh-caps=-all" in s
    assert f"TMPDIR={tmp_path / 'private'}" in s   # no shared /tmp
    assert str(ws) in s                             # cwd is the workspace


def test_uid_preferred_when_bwrap_blocked(monkeypatch):
    """Containers without CAP_SYS_ADMIN/userns: bwrap fails but uid isolation
    still works, so auto must pick it rather than giving up."""
    monkeypatch.setattr("avo.agent.sandbox.bwrap_works", lambda *a, **k: False)
    monkeypatch.setattr("avo.agent.sandbox.uid_isolation_available",
                        lambda: True)
    from avo.agent.sandbox import resolve_mode
    assert resolve_mode("auto") == "uid" and resolve_mode("require") == "uid"


def test_route_uids_are_stable_and_distinct():
    from avo.agent.sandbox import route_uid
    a, b = route_uid("attention-h100-r1"), route_uid("attention-h100-r2")
    assert a != b and a == route_uid("attention-h100-r1")
    assert 60_000 <= a < 64_000  # unprivileged, no passwd entry needed


def test_residue_detector_finds_bootstrappable_leftovers(tmp_path):
    """The exact R4 vector: a previous run's kernel sources readable in /tmp."""
    from avo.agent.sandbox import readable_residue
    (tmp_path / "avo_eval_abc123").mkdir()
    (tmp_path / "d2_attention_v3.cu").write_text("__global__ void k(){}")
    hits = readable_residue(str(tmp_path))
    assert any("avo_eval_abc123" in h for h in hits)
    assert any("attention" in h for h in hits)
    assert readable_residue(str(tmp_path / "nonexistent")) == []


@pytest.mark.skipif(not shutil.which("bwrap") or not bwrap_works(),
                    reason="bwrap unavailable on this host (macOS/CI)")
def test_bwrap_actually_hides_peer(tmp_path):
    import subprocess
    runs = tmp_path / "runs"
    (runs / "routeA" / "workspace").mkdir(parents=True)
    (runs / "routeB" / "workspace").mkdir(parents=True)
    (runs / "routeB" / "workspace" / "secret.cu").write_text("peer solution")
    argv, _ = build_shell_argv(
        "cat ../../routeB/workspace/secret.cu 2>&1 || echo BLOCKED",
        runs / "routeA" / "workspace", mode="bwrap", runs_dir=runs)
    out = subprocess.run(argv, capture_output=True, text=True, timeout=30).stdout
    assert "peer solution" not in out and "BLOCKED" in out
