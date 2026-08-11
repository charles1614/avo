"""Experiment-integrity gates: banned-API scan (attention) + KB preflight."""
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def test_banned_attention_apis_detected(tmp_path):
    common = load("tasks/attention_cuda/harness/checks.py", "attn_checks_scan")
    # a kernel that delegates to SDPA is rejected...
    ws = tmp_path / "cheat"
    ws.mkdir()
    (ws / "attention.cu").write_text(
        "torch::Tensor attention_forward(...) {\n"
        "  return at::scaled_dot_product_attention(q, k, v);\n}\n")
    hit = common.scan_banned_apis(ws)
    assert hit and "scaled_dot_product_attention" in hit

    # ...cuDNN fused attention too
    (ws / "attention.cu").write_text("x = cudnnMultiHeadAttnForward(...);")
    assert common.scan_banned_apis(ws)


def test_banned_scan_ignores_comments_and_allows_gemm(tmp_path):
    common = load("tasks/attention_cuda/harness/checks.py", "attn_checks_scan2")
    ws = tmp_path / "ok"
    ws.mkdir()
    (ws / "attention.cu").write_text(
        "// do NOT call scaled_dot_product_attention here\n"
        "/* mem_efficient_attention is banned */\n"
        "cublasLtMatmul(...);  // GEMM primitive is allowed\n"
        "wmma::mma_sync(acc, a, b, acc);\n")
    assert common.scan_banned_apis(ws) is None


def test_kb_preflight_warns_on_missing_sources(tmp_path):
    from avo.config import RunConfig
    from avo.evolution.controller import Controller
    # minimal project with a manifest whose sources are absent
    root = tmp_path
    task = root / "tasks" / "value"
    (task / "seed").mkdir(parents=True)
    (task / "harness").mkdir()
    (task / "seed" / "value.txt").write_text("1.0\n")
    (task / "harness" / "score.py").write_text(
        "import argparse,json;from pathlib import Path\n"
        "a=argparse.ArgumentParser();a.add_argument('--workspace');"
        "a.add_argument('--params-b64');a.add_argument('--out');x=a.parse_args();"
        "Path(x.out).write_text(json.dumps({'correct':True,'score':1.0,"
        "'error':None,'configs':[],'meta':{}}))\n")
    (task / "task.yaml").write_text("name: value\nbrief: x\n")
    ext = root / "knowledge_base" / "external"
    ext.mkdir(parents=True)
    (ext / "MANIFEST.json").write_text(json.dumps(
        {"flash-attention": {"type": "git", "commit": "abc"},
         "cutlass": {"type": "git", "commit": "def"}}))

    from tests.conftest import FakeLLM
    cfg = RunConfig.model_validate(
        {"run_name": "t", "task": "tasks/value",
         "llm": {"provider": "anthropic", "model": "fake",
                 "price_input_per_mtok": 1.0, "price_output_per_mtok": 1.0},
         "budgets": {"max_steps": 0, "max_versions": 0}})
    ctrl = Controller(cfg, FakeLLM([]), project_root=root)
    logs = []
    ctrl._preflight(logs.append)
    warned = [l for l in logs if "NOT on disk" in l]
    assert warned and "flash-attention" in warned[0] and "cutlass" in warned[0]
