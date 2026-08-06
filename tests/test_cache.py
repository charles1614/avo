from avo.eval.cache import EvalCache, eval_key, tree_hash
from avo.types import ScoreResult


def make_tree(root, files: dict):
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def test_tree_hash_stable_and_content_sensitive(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    make_tree(a, {"x.py": "1", "sub/y.cu": "2"})
    make_tree(b, {"x.py": "1", "sub/y.cu": "2"})
    assert tree_hash(a) == tree_hash(b)
    (b / "x.py").write_text("changed")
    assert tree_hash(a) != tree_hash(b)


def test_tree_hash_ignores_git_and_pycache(tmp_path):
    a = tmp_path / "a"
    make_tree(a, {"x.py": "1"})
    before = tree_hash(a)
    make_tree(a, {".git/config": "gitstuff", "__pycache__/x.pyc": "bin"})
    assert tree_hash(a) == before


def test_eval_key_varies_with_params_and_runner(tmp_path):
    ws, h = tmp_path / "ws", tmp_path / "h"
    make_tree(ws, {"k.cu": "kernel"})
    make_tree(h, {"score.py": "s"})
    k1 = eval_key(ws, h, {"seqlens": [1024]}, "ssh:asus:")
    k2 = eval_key(ws, h, {"seqlens": [2048]}, "ssh:asus:")
    k3 = eval_key(ws, h, {"seqlens": [1024]}, "ssh:h100:")
    assert len({k1, k2, k3}) == 3


def test_infrastructure_failures_not_cacheable():
    from avo.eval.scoring import cacheable
    assert cacheable(ScoreResult(correct=True, score=1.0))
    assert cacheable(ScoreResult.failure("compile", "syntax error"))
    assert cacheable(ScoreResult.failure("correctness", "wrong"))
    assert not cacheable(ScoreResult.failure("harness", "rsync to remote failed"))
    assert not cacheable(ScoreResult.failure("harness", "eval timed out"))


def test_ssh_runner_resolves_tilde_scratch(monkeypatch):
    from avo.config import RunnerConfig
    from avo.eval.ssh_runner import SSHRunner
    from avo.types import ShellResult
    r = SSHRunner(RunnerConfig(kind="ssh", host="x", scratch="~/avo_scratch"),
                  "run1")
    monkeypatch.setattr(r, "_ssh",
                        lambda cmd, t: ShellResult(0, "/home/u\n", ""))
    assert r._remote("eval") == "/home/u/avo_scratch/run1/eval"
    # cached: second call must not ssh again
    monkeypatch.setattr(r, "_ssh", lambda cmd, t: (_ for _ in ()).throw(AssertionError))
    assert r._remote("work") == "/home/u/avo_scratch/run1/work"


def test_resolve_python_paths():
    from avo.eval.runner import resolve_python
    assert resolve_python("python3") == "python3"  # PATH lookup untouched
    resolved = resolve_python(".venv/bin/python")
    assert resolved.startswith("/") and resolved.endswith("/.venv/bin/python")


def test_cache_round_trip(tmp_path):
    cache = EvalCache(tmp_path / "cache")
    r = ScoreResult(correct=True, score=12.5, configs=[{"seqlen": 1024}])
    cache.put("k" * 64, r)
    hit = cache.get("k" * 64)
    assert hit is not None and hit.cached and hit.score == 12.5
    assert hit.eval_hash == "k" * 64
    assert cache.get("m" * 64) is None
