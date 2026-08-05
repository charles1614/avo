import datetime

from avo.config import SupervisorConfig
from avo.evolution.supervisor import Supervisor
from avo.types import LineageEntry


def entry(version: str, score: float) -> LineageEntry:
    return LineageEntry(version=version, step=0, commit="c", score=score,
                        message="m", eval_hash="h", parent=None,
                        timestamp=datetime.datetime.now().isoformat())


def make(cfg=None, tmp_path=None):
    return Supervisor(cfg or SupervisorConfig(), tmp_path / "sup.jsonl")


def test_stagnation_trigger_and_rearm(tmp_path):
    sup = make(SupervisorConfig(stagnation_steps=2), tmp_path)
    assert sup.should_reflect(0, []) is None
    assert sup.should_reflect(1, []) is None
    reason = sup.should_reflect(2, [])
    assert reason and "without a commit" in reason
    sup.note_triggered(2)
    # not re-triggered immediately after firing
    assert sup.should_reflect(3, []) is None
    # re-arms after another full stagnation window
    assert sup.should_reflect(4, []) is not None


def test_commit_resets_trigger_state(tmp_path):
    sup = make(SupervisorConfig(stagnation_steps=2), tmp_path)
    sup.note_triggered(2)
    sup.note_commit()
    assert sup.should_reflect(2, []) is not None


def test_low_improvement_window_triggers(tmp_path):
    sup = make(SupervisorConfig(stagnation_steps=2, min_rel_improvement=0.01,
                                window=3), tmp_path)
    entries = [entry("v0000", 10.0), entry("v0001", 10.001),
               entry("v0002", 10.002), entry("v0003", 10.003)]
    reason = sup.should_reflect(0, entries)
    assert reason and "improved only" in reason


def test_healthy_improvement_no_trigger(tmp_path):
    sup = make(SupervisorConfig(window=3), tmp_path)
    entries = [entry("v0000", 10.0), entry("v0001", 11.0),
               entry("v0002", 12.0), entry("v0003", 13.0)]
    assert sup.should_reflect(0, entries) is None


def test_reflect_logs_and_returns_guidance(tmp_path, fake_llm_factory):
    from avo.types import TextBlock
    sup = make(None, tmp_path)
    llm = fake_llm_factory([[TextBlock("try tensor cores; try cp.async")]])
    guidance = sup.reflect(llm, [entry("v0000", 1.0)], ["step 1: failed"],
                           "kernel source", "stalled")
    assert "tensor cores" in guidance
    assert (tmp_path / "sup.jsonl").exists()
