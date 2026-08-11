"""Contamination auditing: detect what isolation could not prevent."""
import json
from pathlib import Path

from avo.evolution.integrity import audit_run, write_report


def make_run(tmp_path, tool_calls) -> Path:
    run = tmp_path / "kb-routeA-20260811-1"
    (run / "logs").mkdir(parents=True)
    lines = []
    for i, (tool, inp) in enumerate(tool_calls):
        lines.append(json.dumps({"ts": float(i), "kind": "tool",
                                 "payload": {"name": tool, "input": inp,
                                             "is_error": False,
                                             "content": "ok"}}))
    (run / "logs" / "step_0001.jsonl").write_text("\n".join(lines))
    return run


def test_clean_run_is_not_flagged(tmp_path):
    run = make_run(tmp_path, [
        ("shell", {"command": "ls -la"}),
        ("read_file", {"path": "attention.cu"}),
        ("shell", {"command": "nvcc --version && git diff HEAD"}),
    ])
    r = audit_run(run, isolation="uid")
    assert not r.contaminated and r.commands_scanned == 3


def test_peer_route_access_detected(tmp_path):
    """The R3 vector: reading/copying another route's workspace."""
    run = make_run(tmp_path, [
        ("shell", {"command": "cat runs/dsv4flash-high-2026/workspace/attention.cu"}),
        ("shell", {"command": "cp runs/pro-max-2026/workspace/attention.cu ./"}),
    ])
    r = audit_run(run, isolation="none")
    assert r.contaminated
    kinds = {f.kind for f in r.findings}
    assert "peer_route" in kinds and len(r.findings) >= 2


def test_foreign_tmp_bootstrap_detected(tmp_path):
    """The R4 vector: reading a previous run's residue the route never wrote."""
    run = make_run(tmp_path, [
        ("shell", {"command": "ls /tmp"}),
        ("shell", {"command": "cat /tmp/d2_attention_v3.cu"}),
    ])
    r = audit_run(run, isolation="none")
    assert r.contaminated
    kinds = {f.kind for f in r.findings}
    assert "foreign_tmp" in kinds and "recon" in kinds


def test_own_scratch_files_are_not_contamination(tmp_path):
    """Precision: an agent writing then re-reading its own /tmp scratch is
    normal work. Flagging it would make the auditor useless."""
    run = make_run(tmp_path, [
        ("shell", {"command": "cat > /tmp/probe2.cu <<EOF\n__global__ void k(){}\nEOF"}),
        ("shell", {"command": "nvcc /tmp/probe2.cu -o /tmp/probe2 && /tmp/probe2"}),
        ("shell", {"command": "cat /tmp/probe2.cu"}),
    ])
    r = audit_run(run, isolation="none")
    assert not r.contaminated, [f.kind + ':' + f.evidence for f in r.findings]


def test_recon_alone_is_reported_but_not_condemning(tmp_path):
    run = make_run(tmp_path, [("shell", {"command": "find / -name '*.cu'"})])
    r = audit_run(run, isolation="none")
    assert not r.contaminated
    assert {f.kind for f in r.findings} == {"recon"}


def test_own_run_dir_access_is_allowed(tmp_path):
    run = make_run(tmp_path, [
        ("shell", {"command": "cat runs/kb-routeA-20260811-1/workspace/x.cu"}),
    ])
    assert not audit_run(run, isolation="none").contaminated


def test_report_written_and_machine_readable(tmp_path):
    run = make_run(tmp_path, [("shell", {"command": "find /tmp -name '*.cu'"})])
    r = audit_run(run, isolation="none")
    out = write_report(run, r)
    data = json.loads(Path(out).read_text())
    assert data["isolation"] == "none"
    assert data["by_kind"]["recon"] >= 1
    assert data["findings"][0]["step"] == 1
