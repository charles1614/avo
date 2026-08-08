"""Profile tool: ncu CSV parsing, conditional registration, plumbing."""
import importlib.util
import sys
from pathlib import Path

from avo.agent.tools import ToolContext, ToolRegistry
from avo.knowledge.kb import KnowledgeBase
from avo.types import ScoreResult

REPO = Path(__file__).resolve().parent.parent


def load_harness(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


NCU_CSV = '''some ncu banner line
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"
"0","1","python","local","attention_fwd_kernel","1","7","GPU Speed Of Light Throughput","Compute (SM) Throughput","%","11.52"
"0","1","python","local","attention_fwd_kernel","1","7","GPU Speed Of Light Throughput","Memory Throughput","%","47.80"
"0","1","python","local","attention_fwd_kernel","1","7","GPU Speed Of Light Throughput","Duration","usecond","14887.28"
"0","1","python","local","attention_fwd_kernel","1","7","Occupancy","Achieved Occupancy","%","33.10"
"0","1","python","local","attention_fwd_kernel","1","7","Launch Statistics","Registers Per Thread","register/thread","112"
'''


def test_attention_ncu_parse_and_summary():
    prof = load_harness("tasks/attention_cuda/harness/profile.py", "attn_prof")
    rows = prof.parse_ncu_csv(NCU_CSV)
    assert len(rows) == 5
    summary, metrics = prof.summarize(rows, {"seqlen": 1024, "causal": False})
    m = metrics["attention_fwd_kernel"]
    assert m["compute_sol_pct"].startswith("11.52")
    assert m["memory_sol_pct"].startswith("47.80")
    assert "achieved_occupancy_pct" in m and "registers_per_thread" in m
    assert "attention_fwd_kernel" in summary and "compute_sol_pct" in summary


def test_kernelbench_two_stage_rendering():
    prof = load_harness("tasks/kernelbench/harness/profile.py", "kb_prof")
    stage1 = {"ops": [{"name": "aten::linear", "device_us": 900.0, "calls": 24},
                      {"name": "aten::gelu", "device_us": 100.0, "calls": 24}],
              "kernels": [{"name": "sm90_gemm_kernel", "device_us": 850.0,
                           "calls": 24},
                          {"name": "gelu_kernel", "device_us": 95.0,
                           "calls": 24}]}
    s1 = prof.render_stage1(stage1)
    assert s1.index("aten::linear") < s1.index("aten::gelu")
    assert "sm90_gemm_kernel" in s1 and "89.9%" in s1  # 850/945

    header = ('"ID","Process ID","Process Name","Host Name","Kernel Name",'
              '"Context","Stream","Section Name","Metric Name","Metric Unit",'
              '"Metric Value"')
    csv_text = "\n".join([header,
        '"0","1","p","h","sm90_gemm_kernel","1","7","S","Compute (SM) Throughput","%","72.4"',
        '"0","1","p","h","sm90_gemm_kernel","1","7","S","Memory Throughput","%","31.0"'])
    s2 = prof.render_stage2(prof.parse_ncu_csv(csv_text))
    assert "sm90_gemm_kernel" in s2 and "72.4" in s2 and "Reading:" in s2


def test_profile_tool_registration_and_plumbing(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    calls = {}

    def profile_fn(config_index=0):
        calls["idx"] = config_index
        return ScoreResult(correct=True, score=0.0,
                           meta={"summary": "compute_sol 11% memory_sol 48%"})

    base = dict(workspace=ws, kb=KnowledgeBase([]),
                evaluate_fn=lambda quick=False: ScoreResult(correct=True, score=1.0),
                submit_fn=lambda m: (True, "ok"), max_evals=3)
    without = ToolRegistry(ToolContext(**base))
    assert "profile" not in [s.name for s in without.specs()]

    reg = ToolRegistry(ToolContext(**base, profile_fn=profile_fn))
    assert "profile" in [s.name for s in reg.specs()]
    out = reg.dispatch("profile", {"config_index": 2})
    assert not out.is_error and "compute_sol 11%" in out.content
    assert calls["idx"] == 2
    assert reg.ctx.evals_used == 1  # counts against the eval budget

    failing = ToolRegistry(ToolContext(
        **base, profile_fn=lambda config_index=0: ScoreResult.failure(
            "harness", "ERR_NVGPUCTRPERM: counters restricted")))
    out2 = failing.dispatch("profile", {})
    assert out2.is_error and "NVGPUCTRPERM" in out2.content
