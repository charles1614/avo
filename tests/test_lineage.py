import pytest

from avo.evolution.lineage import Lineage


@pytest.fixture
def run_dir(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "solution.py").write_text("def f(): return 1\n")
    rd = tmp_path / "run"
    rd.mkdir()
    return rd, seed


def test_init_seed_and_commit_flow(run_dir):
    rd, seed = run_dir
    lin = Lineage.init_run(rd, seed)
    lin.record_seed(score=1.0, eval_hash="h0")
    assert [e.version for e in lin.entries()] == ["v0000"]

    (lin.workspace / "solution.py").write_text("def f(): return 2\n")
    e = lin.commit_version(step=1, score=2.0, message="improve f", eval_hash="h1")
    assert e.version == "v0001" and e.parent == "v0000"
    assert lin.best().score == 2.0
    assert set(lin.tags()) == {"v0000", "v0001"}
    lin.verify()  # jsonl and tags agree


def test_reset_discards_changes_and_new_files(run_dir):
    rd, seed = run_dir
    lin = Lineage.init_run(rd, seed)
    lin.record_seed(1.0, "h0")
    (lin.workspace / "solution.py").write_text("broken")
    (lin.workspace / "junk.txt").write_text("junk")
    lin.reset_workspace()
    assert (lin.workspace / "solution.py").read_text() == "def f(): return 1\n"
    assert not (lin.workspace / "junk.txt").exists()


def test_capture_patch_without_committing(run_dir):
    rd, seed = run_dir
    lin = Lineage.init_run(rd, seed)
    lin.record_seed(1.0, "h0")
    (lin.workspace / "new.py").write_text("x = 1\n")
    patch = lin.capture_uncommitted_patch()
    assert "new.py" in patch and "x = 1" in patch
    assert lin.tags() == ["v0000"]  # nothing committed


def test_load_verifies_consistency(run_dir):
    rd, seed = run_dir
    lin = Lineage.init_run(rd, seed)
    lin.record_seed(1.0, "h0")
    reloaded = Lineage.load(rd)
    assert [e.version for e in reloaded.entries()] == ["v0000"]

    # corrupt: extra tag without jsonl entry
    import subprocess
    subprocess.run(["git", "-C", str(lin.workspace), "tag", "v0009"], check=True)
    with pytest.raises(RuntimeError, match="disagree"):
        Lineage.load(rd)


def test_commit_message_trailers(run_dir):
    rd, seed = run_dir
    lin = Lineage.init_run(rd, seed)
    lin.record_seed(1.0, "h0")
    (lin.workspace / "solution.py").write_text("v2")
    lin.commit_version(step=3, score=4.5, message="msg", eval_hash="abc")
    import subprocess
    body = subprocess.run(["git", "-C", str(lin.workspace), "log", "-1",
                           "--format=%B"], capture_output=True, text=True).stdout
    assert "AVO-Score: 4.5" in body and "AVO-Step: 3" in body \
        and "AVO-Eval-Hash: abc" in body
