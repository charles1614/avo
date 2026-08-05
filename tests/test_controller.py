"""End-to-end controller test, fully offline: a FakeLLM drives the real
controller/lineage/runner stack on a trivial task whose score is read from a
file the agent edits."""
import json
import textwrap
from pathlib import Path

import pytest

from avo.config import RunConfig
from avo.evolution.controller import Controller
from avo.types import TextBlock
from tests.conftest import FakeLLM, tool_use

HARNESS = textwrap.dedent("""\
    import argparse, base64, json
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--params-b64", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        value = float((Path(a.workspace) / "value.txt").read_text().strip())
        ok = value >= 0
        payload = {"correct": ok, "score": value if ok else 0.0,
                   "error": None if ok else {"stage": "correctness",
                                             "detail": "negative", "log_tail": ""},
                   "configs": [{"size": 1, "throughput": value}], "meta": {}}
    except Exception as e:
        payload = {"correct": False, "score": 0.0,
                   "error": {"stage": "correctness", "detail": str(e),
                             "log_tail": ""}, "configs": [], "meta": {}}
    Path(a.out).write_text(json.dumps(payload))
""")


@pytest.fixture
def project(tmp_path):
    root = tmp_path
    task = root / "tasks" / "value"
    (task / "seed").mkdir(parents=True)
    (task / "harness").mkdir()
    (task / "seed" / "value.txt").write_text("1.0\n")
    (task / "harness" / "score.py").write_text(HARNESS)
    (task / "task.yaml").write_text(
        "name: value\nbrief: Maximize the number in value.txt.\n")
    (root / "knowledge_base").mkdir()
    return root


def make_config(**overrides) -> RunConfig:
    base = {"run_name": "t", "task": "tasks/value",
            "llm": {"provider": "anthropic", "model": "fake",
                    "price_input_per_mtok": 1.0, "price_output_per_mtok": 1.0},
            "budgets": {"max_versions": 2, "max_steps": 4,
                        "max_turns_per_step": 6, "max_evals_per_step": 4,
                        "max_usd": 10.0, "max_total_tokens": 100000}}
    base.update(overrides)
    return RunConfig.model_validate(base)


def set_value(v: str):
    return tool_use("write_file", f"w{v}", path="value.txt", content=v)


def test_full_run_two_commits(project):
    script = [
        # step 1: improve to 2.0 and submit
        [set_value("2.0")], [tool_use("submit", "s1", message="bump to 2")],
        # step 2: improve to 3.0 and submit
        [set_value("3.0")], [tool_use("submit", "s2", message="bump to 3")],
    ]
    llm = FakeLLM(script)
    ctrl = Controller(make_config(), llm, project_root=project)
    summary = ctrl.run(log=lambda *a: None)

    assert summary["versions"] == 2
    assert summary["seed_score"] == 1.0 and summary["best_score"] == 3.0
    entries = ctrl.lineage.entries()
    assert [e.version for e in entries] == ["v0000", "v0001", "v0002"]
    assert (ctrl.lineage.workspace / "value.txt").read_text() == "3.0"
    assert summary["usd"] > 0


def test_failed_step_resets_workspace_and_records_summary(project):
    script = [
        # step 1: regress to 0.5, submit rejected, then run out of turns
        [set_value("0.5")],
        [tool_use("submit", "s1", message="worse")],
        [TextBlock("hmm")],
        # failure-summary LLM call answers:
        [TextBlock("tried 0.5; rejected as regression; try larger values")],
        # step 2: do it right
        [set_value("5.0")], [tool_use("submit", "s2", message="bump to 5")],
        # step 3+: nothing useful; failure summary again
        [TextBlock("idle")], [TextBlock("idle")], [TextBlock("idle")],
        [TextBlock("no attempt made")],
    ]
    cfg = make_config(budgets={"max_versions": 1, "max_steps": 2,
                               "max_turns_per_step": 3, "max_evals_per_step": 4,
                               "max_usd": 10.0, "max_total_tokens": 100000})
    llm = FakeLLM(script)
    ctrl = Controller(cfg, llm, project_root=project)
    summary = ctrl.run(log=lambda *a: None)

    assert summary["versions"] == 1 and summary["best_score"] == 5.0
    # workspace was reset after the failed step before step 2 started
    assert (ctrl.lineage.workspace / "value.txt").read_text() == "5.0"
    # failure patch + summary recorded
    patches = list((ctrl.run_dir / "logs").glob("step_*_final.patch"))
    assert patches, "failed step should leave a patch"
    state = json.loads((ctrl.run_dir / "state.json").read_text())
    assert state["steps_done"] >= 1


def test_resume_continues_without_duplicates(project):
    script1 = [[set_value("2.0")], [tool_use("submit", "s1", message="bump to 2")]]
    cfg = make_config(budgets={"max_versions": 1, "max_steps": 3,
                               "max_turns_per_step": 5, "max_evals_per_step": 4,
                               "max_usd": 10.0, "max_total_tokens": 100000})
    ctrl1 = Controller(cfg, FakeLLM(script1), project_root=project)
    ctrl1.run(log=lambda *a: None)
    run_dir = ctrl1.run_dir

    # resume with a higher version budget; agent commits one more
    cfg2 = make_config(budgets={"max_versions": 2, "max_steps": 6,
                                "max_turns_per_step": 5, "max_evals_per_step": 4,
                                "max_usd": 10.0, "max_total_tokens": 100000})
    script2 = [[set_value("4.0")], [tool_use("submit", "s2", message="bump to 4")]]
    ctrl2 = Controller(cfg2, FakeLLM(script2), project_root=project,
                       run_dir=run_dir)
    summary = ctrl2.run(log=lambda *a: None)
    versions = [e.version for e in ctrl2.lineage.entries()]
    assert versions == ["v0000", "v0001", "v0002"]
    assert summary["best_score"] == 4.0
    # cumulative accounting survived the resume
    assert summary["tokens"] > FakeLLM([]).usage.total_tokens


def test_run_dir_lock_rejects_second_controller(project):
    script = [[set_value("2.0")], [tool_use("submit", "s1", message="b2")]]
    ctrl = Controller(make_config(), FakeLLM(script), project_root=project)
    with pytest.raises(RuntimeError, match="already running"):
        Controller(make_config(), FakeLLM([]), project_root=project,
                   run_dir=ctrl.run_dir)


def test_seed_eval_is_cached_for_identical_content(project):
    script = [[set_value("2.0")], [tool_use("submit", "s1", message="b2")]]
    ctrl = Controller(make_config(), FakeLLM(script), project_root=project)
    r1 = ctrl.evaluate_workspace()
    r2 = ctrl.evaluate_workspace()
    assert not r1.cached and r2.cached and r1.score == r2.score
