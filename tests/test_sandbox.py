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
