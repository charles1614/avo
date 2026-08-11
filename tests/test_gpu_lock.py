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
    ap.add_argument("--result-token", default="")
    a = ap.parse_args()
    t0 = time.monotonic()
    time.sleep(0.4)
    t1 = time.monotonic()
    Path(a.out).write_text(json.dumps(
        {"correct": True, "score": 1.0, "error": None,
         "configs": [{"t0": t0, "t1": t1}], "meta": {"result_token": a.result_token}}))
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


def test_per_device_lock_paths_and_env():
    a = RunnerConfig(kind="local", cuda_device="0")
    b = RunnerConfig(kind="local", cuda_device="3")
    # different GPUs on one node => independent locks (no false serialization)
    assert a.lock_path() != b.lock_path()
    assert a.lock_path() == "/tmp/avo_gpu0.lock"
    assert b.eval_env() == {"CUDA_VISIBLE_DEVICES": "3"}
    # unset device: single shared lock, no pinning
    plain = RunnerConfig(kind="local")
    assert plain.lock_path() == "/tmp/avo_gpu0.lock" and plain.eval_env() == {}
    # same GPU => same lock (routes sharing a card still serialize)
    assert RunnerConfig(kind="local", cuda_device="3").lock_path() == b.lock_path()
    # cache identity must NOT depend on which identical GPU ran it
    assert a.identity() == b.identity()


def test_local_runner_pins_device(tmp_path, staged_task):
    ws, harness = staged_task
    probe = harness / "score.py"
    probe.write_text(
        "import argparse, json, os\n"
        "from pathlib import Path\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--workspace'); ap.add_argument('--params-b64')\n"
        "ap.add_argument('--out')\n"
        "ap.add_argument('--result-token', default='')\n"
        "a = ap.parse_args()\n"
        "Path(a.out).write_text(json.dumps({'correct': True, 'score': 1.0,\n"
        "  'error': None, 'configs': [], 'meta': {\n"
        "  'result_token': a.result_token,\n"
        "  'seen': os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}}))\n")
    runner = LocalRunner(RunnerConfig(kind="local", cuda_device="5",
                                      gpu_lock="", eval_timeout_s=30))
    r = runner.score(ws, harness, "score.py", {})
    assert r.correct and r.meta["seen"] == "5"


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