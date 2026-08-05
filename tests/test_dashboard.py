import json

from avo.report.dashboard import build, collect


def make_run(tmp_path):
    run = tmp_path / "run-x"
    (run / "logs").mkdir(parents=True)
    (run / "evals").mkdir()
    entries = [
        {"version": "v0000", "step": 0, "commit": "a", "score": 2.4,
         "message": "seed", "eval_hash": "h0", "parent": None, "timestamp": "t"},
        {"version": "v0001", "step": 2, "commit": "b", "score": 3.1,
         "message": "fast math", "eval_hash": "h1", "parent": "v0000",
         "timestamp": "t"},
    ]
    (run / "lineage.jsonl").write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n")
    (run / "state.json").write_text(json.dumps(
        {"steps_done": 2, "stagnation": 0, "usd": 0.5,
         "input_tokens": 1000, "output_tokens": 200, "elapsed_s": 60}))
    (run / "evals" / "h1.json").write_text(json.dumps(
        {"correct": True, "score": 3.1,
         "configs": [{"batch": 8, "heads": 16, "seqlen": 1024, "head_dim": 128,
                      "causal": False, "tflops": 3.2, "median_ms": 1.0}]}))
    (run / "logs" / "step_0001.jsonl").write_text(json.dumps(
        {"ts": 1.0, "kind": "assistant", "payload": {"role": "assistant",
                                                     "blocks": []}}) + "\n")
    (run / "logs" / "step_0002.jsonl").write_text(json.dumps(
        {"ts": 2.0, "kind": "tool",
         "payload": {"name": "evaluate", "input": {}, "is_error": False,
                     "content": "ok"}}) + "\n")
    baselines_dir = tmp_path / "baselines"
    baselines_dir.mkdir()
    (baselines_dir / "b.json").write_text(json.dumps(
        {"geomeans": {"sdpa_flash": 70.4},
         "per_config": {"sdpa_flash": {"s1024_b8_full": 71.0}}}))
    return run


def test_collect_shapes(tmp_path):
    run = make_run(tmp_path)
    d = collect(run)
    assert d["run"] == "run-x" and d["running"] is True
    assert [e["version"] for e in d["lineage"]] == ["v0000", "v0001"]
    assert d["lineage"][1]["configs"][0]["tflops"] == 3.2
    outcomes = {s["step"]: s["outcome"] for s in d["steps"]}
    assert outcomes == {1: "failed", 2: "committed"}
    assert d["baselines"]["sdpa_flash"] == 70.4


def test_build_html(tmp_path):
    run = make_run(tmp_path)
    out = build(run)
    html_text = out.read_text()
    assert out.name == "dashboard.html"
    assert "v0001" in html_text and "sdpa_flash" in html_text
    assert "<script>" in html_text and "http-equiv" not in html_text
    out2 = build(run, refresh_s=30)
    assert 'http-equiv="refresh" content="30"' in out2.read_text()
    # no LLM anywhere: dashboard must not import the LLM package
    import inspect

    import avo.report.dashboard as mod
    src = inspect.getsource(mod)
    assert "avo.llm" not in src and "make_client" not in src
