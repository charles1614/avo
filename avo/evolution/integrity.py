"""Post-hoc contamination audit of a run's agent transcripts.

When the host cannot enforce filesystem isolation (a single container without
root, no user namespaces), cross-route reads cannot be *prevented*. They can
still be *detected*: every shell command and file read is in the transcripts,
so a run can be labelled contaminated instead of silently producing a number
that looks valid.

Detects the two observed vectors:
  * peer-route access  — reading another run's workspace/lineage/.git
  * shared-/tmp bootstrap — listing or reading kernel/eval residue in /tmp
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

# command/path patterns that indicate looking outside this route's workspace
PEER_PATTERNS = [
    r"runs/(?!\Z)[\w.\-]+/(workspace|lineage\.jsonl|logs|evals|\.git)",
    r"\.\./\.\./[\w.\-]+/workspace",
    r"git\s+(?:-C\s+\S*runs/|--git-dir)",
]
# reconnaissance: enables discovery of peer residue, but isn't proof of use
RECON_PATTERNS = [r"\bls\s+(-\w+\s+)*/tmp\b", r"\bfind\s+/tmp\b",
                  r"\bfind\s+/\s", r"\bls\s+(-\w+\s+)*/\s"]
OUTSIDE_PATH = re.compile(r"(?<![\w./])(/tmp/[\w./\-]+)")
# a path the agent itself created is its own scratch, not contamination
WRITE_CONTEXT = re.compile(
    r"(>|>>|\btee\b|\bcp\b|\bmv\b|\btouch\b|\bmkdir\b|-o\s|\binstall\b|"
    r"write_text|open\([^)]*['\"]w)")
READ_TOOLS = ("shell", "gpu_shell", "read_file", "list_dir")


@dataclass
class Finding:
    step: int
    tool: str
    kind: str          # "peer_route" | "shared_tmp"
    evidence: str


@dataclass
class IntegrityReport:
    run: str
    isolation: str
    findings: list = field(default_factory=list)
    commands_scanned: int = 0

    @property
    def contaminated(self) -> bool:
        """Only genuine cross-contamination counts: reading a peer route, or
        reading a /tmp file this route never created. Recon (`ls /tmp`) and
        the agent's own scratch files are reported but do not condemn a run —
        an auditor that cries wolf gets ignored."""
        return any(f.kind in ("peer_route", "foreign_tmp") for f in self.findings)

    def summary(self) -> dict:
        kinds: dict[str, int] = {}
        for f in self.findings:
            kinds[f.kind] = kinds.get(f.kind, 0) + 1
        return {"contaminated": self.contaminated, "isolation": self.isolation,
                "commands_scanned": self.commands_scanned, "by_kind": kinds,
                "first_findings": [asdict(f) for f in self.findings[:5]]}


def _own_run_name(run_dir: Path) -> str:
    return run_dir.name


def audit_run(run_dir: Path, isolation: str = "unknown") -> IntegrityReport:
    run_dir = Path(run_dir)
    report = IntegrityReport(run=_own_run_name(run_dir), isolation=isolation)
    peer_rx = [re.compile(p) for p in PEER_PATTERNS]
    recon_rx = [re.compile(p) for p in RECON_PATTERNS]
    own = _own_run_name(run_dir)
    self_created: set[str] = set()   # /tmp paths this route wrote first

    for f in sorted((run_dir / "logs").glob("step_*.jsonl")):
        try:
            step = int(f.stem.split("_")[1])
        except (IndexError, ValueError):
            step = -1
        for line in f.read_text(errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") != "tool":
                continue
            payload = rec.get("payload", {})
            tool = payload.get("name", "")
            if tool not in READ_TOOLS:
                continue
            text = json.dumps(payload.get("input", {}))
            report.commands_scanned += 1
            for rx in peer_rx:
                m = rx.search(text)
                # a route reading its OWN run dir is fine
                if m and own not in m.group(0):
                    report.findings.append(
                        Finding(step, tool, "peer_route", m.group(0)[:200]))
                    break
            for rx in recon_rx:
                m = rx.search(text)
                if m:
                    report.findings.append(
                        Finding(step, tool, "recon", m.group(0)[:200]))
                    break
            writing = bool(WRITE_CONTEXT.search(text))
            for path in OUTSIDE_PATH.findall(text):
                if writing or path in self_created:
                    self_created.add(path)      # the agent's own scratch file
                    continue
                report.findings.append(
                    Finding(step, tool, "foreign_tmp", path[:200]))
    return report


def write_report(run_dir: Path, report: IntegrityReport) -> Path:
    out = Path(run_dir) / "logs" / "integrity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {**report.summary(), "findings": [asdict(f) for f in report.findings]},
        indent=1))
    return out
