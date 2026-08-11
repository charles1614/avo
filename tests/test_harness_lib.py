"""The shared integrity layer: primitives, and the enforced scoring sequence.

These test the FRAMEWORK guarantees — a task cannot omit the recheck, the ban
scan, or the result token, because they are not the task's code.
"""
import base64
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "avo_harness", REPO / "harness_lib" / "avo_harness" / "__init__.py")
ah = importlib.util.module_from_spec(spec)
sys.modules["avo_harness"] = ah
spec.loader.exec_module(ah)


def make_args(tmp_path, params=None, token="tok123") -> "ah.HarnessArgs":
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    return ah.HarnessArgs(ws, params or {}, str(tmp_path / "result.json"), token)


def read(args):
    return json.loads(Path(args.out).read_text())


def test_write_result_always_stamps_token(tmp_path):
    args = make_args(tmp_path)
    ah.write_result(args, correct=True, score=3.5, configs=[{"a": 1}])
    r = read(args)
    assert r["meta"]["result_token"] == "tok123" and r["score"] == 3.5


def test_fail_is_structured_and_tokened(tmp_path):
    args = make_args(tmp_path)
    with pytest.raises(SystemExit):
        ah.fail(args, "compile", "nvcc exploded", "log" * 10)
    r = read(args)
    assert r["correct"] is False and r["score"] == 0.0
    assert r["error"]["stage"] == "compile"
    assert r["meta"]["result_token"] == "tok123"


def test_parse_args_merges_defaults(tmp_path, monkeypatch):
    params = {"repeats": 7}
    b64 = base64.b64encode(json.dumps(params).encode()).decode()
    monkeypatch.setattr(sys, "argv", ["score.py", "--workspace", str(tmp_path),
                                      "--params-b64", b64, "--out", "o.json",
                                      "--result-token", "T"])
    a = ah.parse_args({"repeats": 30, "warmup": 5})
    assert a.params == {"repeats": 7, "warmup": 5} and a.token == "T"


def test_banned_scan_is_task_configurable(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "k.cu").write_text("// uses cublasLtMatmul in a comment\n"
                             "cublasLtMatmul(handle);")
    assert ah.scan_banned_apis(ws) is None            # no default bans
    assert ah.scan_banned_apis(ws, [r"cublas\w*Matmul"])
    assert ah.scan_banned_apis(ws, [r"scaled_dot_product"]) is None


def test_geomean_and_max_err_helpers():
    assert abs(ah.geomean([4.0, 16.0]) - 8.0) < 1e-9
    assert ah.geomean([1.0, 0.0]) == 0.0


# -- the enforced sequence --------------------------------------------------

class FakeCandidate:
    def __init__(self):
        self.checks = 0


def hooks(check_impl, measure_impl=None):
    return ah.ScoringHooks(
        load=lambda args: FakeCandidate(),
        configs=lambda c, args: [{"size": 1}, {"size": 2}],
        check=check_impl,
        measure=measure_impl or (lambda c, cfg, args: {"metric_value": 10.0}),
        correctness_trials=2)


def run(tmp_path, monkeypatch, hooks_obj, params=None):
    """run_scoring with torch/GPU guards stubbed out."""
    monkeypatch.setattr(ah, "gpu_busy_reason", lambda: None)
    monkeypatch.setattr(ah, "gpu_meta", lambda: {"gpu": "fake"})
    fake_torch = type(sys)("torch")
    fake_torch.cuda = type("c", (), {"is_available": staticmethod(lambda: True)})()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    args = make_args(tmp_path, params)
    try:
        ah.run_scoring(args, hooks_obj)
    except SystemExit:
        pass
    return read(args)


def test_happy_path_scores_and_tokens(tmp_path, monkeypatch):
    r = run(tmp_path, monkeypatch, hooks(lambda c, cfg, s, a: {"ok": True}))
    assert r["correct"] and abs(r["score"] - 10.0) < 1e-9  # geomean float noise
    assert r["meta"]["result_token"] == "tok123" and len(r["configs"]) == 2


def test_post_bench_recheck_catches_memoizing_candidate(tmp_path, monkeypatch):
    """A candidate correct during the pre-bench trials but stale afterwards
    (the classic 'cache the output and time ~0 ms' exploit) must be caught by
    the framework, with no cooperation from the task harness."""
    state = {"n": 0}

    def check(c, cfg, seed, args):
        state["n"] += 1
        # 4 pre-bench checks (2 configs x 2 trials) pass, later ones fail
        return {"ok": state["n"] <= 4, "detail": "stale cached output"}

    r = run(tmp_path, monkeypatch, hooks(check))
    assert r["correct"] is False and r["score"] == 0.0
    assert "post-benchmark recheck FAILED" in r["error"]["detail"]
    assert state["n"] > 4, "recheck did not run after benchmarking"


def test_ban_scan_runs_before_load(tmp_path, monkeypatch):
    loaded = {"yes": False}

    def load(args):
        loaded["yes"] = True
        return FakeCandidate()

    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    (ws / "cheat.cu").write_text("at::scaled_dot_product_attention(q,k,v);")
    h = hooks(lambda c, cfg, s, a: {"ok": True})
    h.load = load
    r = run(tmp_path, monkeypatch, h,
            params={"banned_apis": [r"scaled_dot_product_attention"]})
    assert r["correct"] is False and r["error"]["stage"] == "compile"
    assert "forbidden" in r["error"]["detail"]
    assert not loaded["yes"], "candidate was loaded despite a banned API"


def test_busy_gpu_is_non_cached_harness_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(ah, "gpu_busy_reason", lambda: "9000 MiB in use")
    fake_torch = type(sys)("torch")
    fake_torch.cuda = type("c", (), {"is_available": staticmethod(lambda: True)})()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    args = make_args(tmp_path)
    try:
        ah.run_scoring(args, hooks(lambda c, cfg, s, a: {"ok": True}))
    except SystemExit:
        pass
    r = read(args)
    assert r["error"]["stage"] == "harness" and "busy" in r["error"]["detail"]
