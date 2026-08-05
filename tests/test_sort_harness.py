"""The sort_py harness as a subprocess — the same way runners invoke it."""
import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASK = ROOT / "tasks" / "sort_py"


def run_harness(tmp_path, solution_src: str, params: dict | None = None) -> dict:
    staged = tmp_path / "staged"
    (staged / "workspace").mkdir(parents=True)
    shutil.copytree(TASK / "harness", staged / "harness")
    (staged / "workspace" / "solution.py").write_text(solution_src)
    params_b64 = base64.b64encode(
        json.dumps(params or {"sizes": [200, 500], "rng_seed": 42}).encode()
    ).decode()
    proc = subprocess.run(
        [sys.executable, "harness/score.py", "--workspace", "workspace",
         "--params-b64", params_b64, "--out", "result.json"],
        cwd=staged, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return json.loads((staged / "result.json").read_text())


def test_seed_scores_correct(tmp_path):
    seed = (TASK / "seed" / "solution.py").read_text()
    result = run_harness(tmp_path, seed)
    assert result["correct"] is True and result["score"] > 0
    assert len(result["configs"]) == 2


def test_banned_sorted_rejected(tmp_path):
    result = run_harness(tmp_path, "def sort_list(a):\n    return sorted(a)\n")
    assert result["correct"] is False
    assert "sorted" in result["error"]["detail"]


def test_banned_method_sort_rejected(tmp_path):
    src = "def sort_list(a):\n    b = list(a)\n    b.sort()\n    return b\n"
    result = run_harness(tmp_path, src)
    assert result["correct"] is False


def test_wrong_result_rejected(tmp_path):
    result = run_harness(tmp_path, "def sort_list(a):\n    return list(a)\n")
    assert result["correct"] is False
    assert result["error"]["stage"] == "correctness"


def test_mutation_rejected(tmp_path):
    src = ("def sort_list(a):\n"
           "    a.append(0) if a else None\n"
           "    out = []\n"
           "    for x in a:\n"
           "        i = 0\n"
           "        while i < len(out) and out[i] < x: i += 1\n"
           "        out.insert(i, x)\n"
           "    return out\n")
    result = run_harness(tmp_path, src)
    assert result["correct"] is False


def test_better_algorithm_scores_higher(tmp_path):
    seed = (TASK / "seed" / "solution.py").read_text()
    merge = (
        "def sort_list(arr):\n"
        "    a = list(arr)\n"
        "    if len(a) <= 1: return a\n"
        "    mid = len(a) // 2\n"
        "    l, r = sort_list(a[:mid]), sort_list(a[mid:])\n"
        "    out, i, j = [], 0, 0\n"
        "    while i < len(l) and j < len(r):\n"
        "        if l[i] <= r[j]: out.append(l[i]); i += 1\n"
        "        else: out.append(r[j]); j += 1\n"
        "    out.extend(l[i:]); out.extend(r[j:])\n"
        "    return out\n")
    slow = run_harness(tmp_path, seed)
    tmp2 = tmp_path / "second"
    tmp2.mkdir()
    fast = run_harness(tmp2, merge)
    assert fast["correct"] and fast["score"] > slow["score"]


def test_syntax_error_reported(tmp_path):
    result = run_harness(tmp_path, "def sort_list(a:\n")
    assert result["correct"] is False
