"""Cross-process/thread eval serialization for shared-GPU multi-run setups."""
import json
import textwrap
import threading
import time

import pytest

from avo.config import RunnerConfig
from avo.eval.runner import LocalRunner, gpu_lock

# harness that records its own execution interval, so overlap is measurable
INTERVAL_HARNESS = textwrap.dedent("""\
    import argparse, json, time
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--params-b64", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.monotonic()
    time.sleep(0.4)
    t1 = time.monotonic()
    Path(a.out).write_text(json.dumps(
        {"correct": True, "score": 1.0, "error": None,
         "configs": [{"t0": t0, "t1": t1}], "meta": {}}))
""")


@pytest.fixture
def staged_task(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "x.txt").write_text("x")
    harness = tmp_path / "harness"
    harness.mkdir()
    (harness / "score.py").write_text(INTERVAL_HARNESS)
    return ws, harness


def test_concurrent_evals_serialize(tmp_path, staged_task):
    ws, harness = staged_task
    lock_path = str(tmp_path / "gpu.lock")
    runner = LocalRunner(RunnerConfig(kind="local", gpu_lock=lock_path,
                                      eval_timeout_s=30))
    results = []

    def worker():
        results.append(runner.score(ws, harness, "score.py", {}))

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    intervals = sorted((r.configs[0]["t0"], r.configs[0]["t1"])
                       for r in results if r.correct)
    assert len(intervals) == 3
    for (a0, a1), (b0, b1) in zip(intervals, intervals[1:]):
        assert a1 <= b0 + 1e-3, f"evals overlapped: {(a0, a1)} vs {(b0, b1)}"


def test_no_lock_when_disabled(staged_task, tmp_path):
    ws, harness = staged_task
    runner = LocalRunner(RunnerConfig(kind="local", gpu_lock="",
                                      eval_timeout_s=30))
    r = runner.score(ws, harness, "score.py", {})
    assert r.correct


def test_lock_wait_timeout_is_structured_failure(tmp_path):
    lock_path = str(tmp_path / "gpu.lock")
    hold = threading.Event()
    release = threading.Event()

    def holder():
        with gpu_lock(lock_path, wait_timeout_s=5):
            hold.set()
            release.wait(timeout=10)

    t = threading.Thread(target=holder)
    t.start()
    hold.wait(timeout=5)
    with pytest.raises(TimeoutError, match="still held"):
        with gpu_lock(lock_path, wait_timeout_s=1.5):
            pass
    release.set()
    t.join()


def test_ssh_runner_wraps_score_in_flock(tmp_path):
    from avo.eval.ssh_runner import SSHRunner
    from avo.types import ShellResult
    cfg = RunnerConfig(kind="ssh", host="x", gpu_lock="/tmp/avo_gpu.lock")
    r = SSHRunner(cfg, "run1")
    r._home = "/home/u"
    cmds: list[str] = []

    def fake_ssh(cmd, t):
        cmds.append(cmd)
        return ShellResult(0, "", "")

    r._ssh = fake_ssh
    r._rsync = lambda a, b: ShellResult(0, "", "")
    ws, h = tmp_path / "w", tmp_path / "h"
    ws.mkdir()
    h.mkdir()
    (h / "score.py").write_text("x")
    r.score(ws, h, "score.py", {})
    score_cmds = [c for c in cmds if "score.py" in c]
    assert score_cmds and "flock -w" in score_cmds[0]
    assert "/tmp/avo_gpu.lock" in score_cmds[0]