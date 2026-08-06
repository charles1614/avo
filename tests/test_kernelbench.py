"""Campaign-runner logic, fully offline (no torch, no GPU, no LLM)."""
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "run_kernelbench", REPO / "scripts" / "run_kernelbench.py")
rkb = importlib.util.module_from_spec(spec)
sys.modules["run_kernelbench"] = rkb
spec.loader.exec_module(rkb)

PROBLEM_SRC = """\
import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, x): return torch.relu(x)
def get_inputs(): return [torch.randn(16, 16)]
def get_init_inputs(): return []
"""


def fake_problems(tmp_path, monkeypatch):
    root = tmp_path / "kb" / "KernelBench"
    (root / "level1").mkdir(parents=True)
    for name in ("19_ReLU", "40_LayerNorm"):
        (root / "level1" / f"{name}.py").write_text(PROBLEM_SRC)
    monkeypatch.setattr(rkb, "PROBLEMS_ROOT", root)
    return root


def test_discover_and_slug(tmp_path, monkeypatch):
    fake_problems(tmp_path, monkeypatch)
    probs = rkb.discover("level1")
    assert [p.name for p in probs] == ["19_ReLU.py", "40_LayerNorm.py"]
    one = rkb.discover("level1/19_ReLU.py")
    assert len(one) == 1
    assert rkb.slug("level1/19_ReLU.py") == "level1-19-relu"


def test_materialize_seed_and_config(tmp_path, monkeypatch):
    root = fake_problems(tmp_path, monkeypatch)
    monkeypatch.setattr(rkb, "SEED_TEMPLATE",
                        REPO / "tasks" / "kernelbench" / "seed_template")
    seed = rkb.materialize_seed(root / "level1" / "19_ReLU.py", tmp_path / "seeds")
    assert (seed / "problem.py").read_text() == PROBLEM_SRC
    assert "ModelNew" in (seed / "model_new.py").read_text()

    base = {"run_name": "kb", "task": "tasks/kernelbench",
            "llm": {"provider": "openai_compat", "model": "m",
                    "base_url": "https://x", "api_key_env": "K"}}
    cfg = rkb.problem_config(base, root / "level1" / "19_ReLU.py", seed)
    assert cfg.run_name == "kb-level1-19-relu"
    assert cfg.task_params["problem_source"] == PROBLEM_SRC
    assert cfg.task_params["problem_name"] == "level1/19_ReLU.py"
    assert cfg.seed_dir == str(seed)


def test_finished_run_detection_and_fastp(tmp_path):
    runs = tmp_path / "runs"
    # finished run for ReLU (speedup 2.5), unfinished dir for LayerNorm
    done = runs / "kb-level1-19-relu-20260806-1"
    done.mkdir(parents=True)
    (done / "summary.json").write_text(json.dumps(
        {"best_score": 2.5, "versions": 2, "usd": 0.4}))
    (runs / "kb-level1-40-layernorm-20260806-1").mkdir()

    assert rkb.finished_run(runs, "kb-level1-19-relu") == done
    assert rkb.finished_run(runs, "kb-level1-40-layernorm") is None

    names = {"level1/19_ReLU.py": "kb-level1-19-relu",
             "level1/40_LayerNorm.py": "kb-level1-40-layernorm"}
    report = rkb.aggregate(runs, names, tmp_path / "report.md")
    assert "fast_1 (best speedup > 1x): 100.00% (1/1)" in report
    assert "fast_2 (best speedup > 2x): 100.00% (1/1)" in report
    assert "pending/failed" in report and "2.500x" in report
