"""Agent tool registry: the software-engineering toolkit the paper describes
(file editing, shell, navigation, doc retrieval) plus evaluate/submit.

Design notes:
- All file tools are confined to the workspace; `.git` is never writable.
- `shell` runs locally in the workspace; `gpu_shell` (registered only for SSH
  runners) syncs the workspace to the remote host and runs there.
- `evaluate` and `submit` are controller-provided closures: the framework, not
  the agent, owns scoring and the commit gate.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from avo.eval.runner import Runner
from avo.knowledge.kb import KnowledgeBase
from avo.types import ScoreResult, ShellResult, ToolSpec, truncate_middle

OUTPUT_CAP = 20_000

DENY_PATTERNS = [
    r"\bsudo\b", r"\bsu\s", r"\bshutdown\b", r"\breboot\b", r"\bpoweroff\b",
    r"\bmkfs\b", r"\bdd\s+if=", r"rm\s+(-[a-zA-Z]+\s+)*[\"']?/(\s|$|[\"'])",
    r"\bpip3?\s+install\b", r"\bconda\s+install\b", r"\bapt(-get)?\b",
    r"\byum\b", r"\bbrew\s+install\b",
    r"nvidia-smi\s+.*(-lgc|-rgc|-pl\b|--lock-gpu-clocks|--power-limit)",
    r":\s*\(\s*\)\s*\{",  # fork bomb
    r"\bcurl\b.*\|\s*(ba)?sh", r"\bwget\b.*\|\s*(ba)?sh",
]


def command_denied(command: str) -> str | None:
    for pat in DENY_PATTERNS:
        if re.search(pat, command):
            return pat
    return None


@dataclass
class ToolOutcome:
    content: str
    is_error: bool = False
    signal: str | None = None  # "committed" ends the step


@dataclass
class ToolContext:
    workspace: Path
    kb: KnowledgeBase
    evaluate_fn: Callable[[], ScoreResult]
    # returns (committed, verdict_text)
    submit_fn: Callable[[str], "tuple[bool, str]"]
    runner: Runner | None = None  # SSH runner => gpu_shell available
    profile_fn: Callable[..., ScoreResult] | None = None  # task ships profile.py
    revert_fn: Callable[[], str] | None = None  # discard uncommitted work
    sandbox: str = "auto"  # filesystem isolation mode for shell (see sandbox.py)
    runs_dir: Path | None = None  # runs tree to hide from shell (peer routes)
    route_uid: int | None = None  # uid-isolation mode: this route's uid
    private_tmp: Path | None = None  # per-route TMPDIR
    shell_timeout_s: int = 60
    gpu_shell_timeout_s: int = 120
    evals_used: int = 0
    max_evals: int = 8
    counters: dict = field(default_factory=dict)


def _spec(name: str, description: str, props: dict, required: list[str]) -> ToolSpec:
    return ToolSpec(name=name, description=description,
                    input_schema={"type": "object", "properties": props,
                                  "required": required})


class ToolRegistry:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx

    # -- specs ---------------------------------------------------------------

    def specs(self) -> list[ToolSpec]:
        remote_gpu = self.ctx.runner is not None
        shell_desc = (
            "Run a shell command with the local workspace as cwd, on the "
            "framework host. The GPU lives on the remote host — use gpu_shell "
            "for GPU probes." if remote_gpu else
            "Run a shell command with the workspace as cwd, on this machine "
            "(which has the GPU). Fine for probes (nvidia-smi, nvcc, ncu); do "
            "NOT build your own test/benchmark environment — only `evaluate` "
            "uses the scoring environment, and ad-hoc results will not match.")
        s = [
            _spec("read_file",
                  "Read a file. Workspace-relative paths, plus knowledge-base "
                  "paths as shown by kb_search (knowledge_base/...) are served "
                  "read-only. Returns numbered lines.",
                  {"path": {"type": "string", "description": "workspace-relative path"},
                   "offset": {"type": "integer", "description": "1-based first line"},
                   "limit": {"type": "integer", "description": "max lines"}},
                  ["path"]),
            _spec("write_file", "Create or overwrite a file in the workspace.",
                  {"path": {"type": "string"}, "content": {"type": "string"}},
                  ["path", "content"]),
            _spec("edit_file",
                  "Replace an exact, unique occurrence of old_string with new_string in a workspace file.",
                  {"path": {"type": "string"}, "old_string": {"type": "string"},
                   "new_string": {"type": "string"}},
                  ["path", "old_string", "new_string"]),
            _spec("list_dir", "List a workspace directory.",
                  {"path": {"type": "string", "description": "workspace-relative, '' for root"}},
                  []),
            _spec("shell", shell_desc,
                  {"command": {"type": "string"},
                   "timeout_s": {"type": "integer"}},
                  ["command"]),
            _spec("kb_search",
                  "Search the domain knowledge base (docs + kernel sources). Regex or keywords.",
                  {"query": {"type": "string"}}, ["query"]),
            _spec("kb_read", "Read a knowledge-base file (path as shown by kb_search).",
                  {"path": {"type": "string"},
                   "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
                  ["path"]),
            _spec("evaluate",
                  "Score the current workspace: correctness check then benchmark. "
                  "Returns the full scoring result. Counts against your eval budget. "
                  "quick=true scores a reduced grid with fewer repeats — much faster, "
                  "use it for inner-loop iteration; submit always re-scores the full grid.",
                  {"quick": {"type": "boolean"}}, []),
            _spec("submit",
                  "Authoritatively evaluate and commit the current workspace as the next "
                  "version if it is correct AND matches-or-improves the best committed score. "
                  "message = concise description of the change (used as commit message).",
                  {"message": {"type": "string"}}, ["message"]),
            _spec("revert",
                  "Discard ALL uncommitted changes and reset the workspace to the best "
                  "committed version. Use when the current line of work is a dead end.",
                  {}, []),
        ]
        if self.ctx.profile_fn is not None:
            s.insert(-1, _spec(
                "profile",
                "Profile the current workspace's kernel(s) with Nsight Compute "
                "(ncu) in the pristine scoring environment: memory vs compute "
                "throughput (%SOL), achieved occupancy, registers/thread, "
                "shared memory, launch shape. Use it to DIAGNOSE the bottleneck "
                "before choosing an optimization. Counts against the eval "
                "budget; slower than evaluate.",
                {"config_index": {"type": "integer",
                                  "description": "benchmark-grid config to profile (default 0)"}},
                []))
        if self.ctx.runner is not None:
            s.insert(5, _spec(
                "gpu_shell",
                "Sync the workspace to the GPU host and run a shell command there "
                "(cwd = synced workspace). Use for nvcc probes, nvidia-smi, quick tests.",
                {"command": {"type": "string"}, "timeout_s": {"type": "integer"}},
                ["command"]))
        return s

    # -- dispatch ------------------------------------------------------------

    def dispatch(self, name: str, args: dict) -> ToolOutcome:
        handler = getattr(self, f"_t_{name}", None)
        if handler is None:
            return ToolOutcome(f"unknown tool: {name}", is_error=True)
        try:
            return handler(**args)
        except TypeError as e:
            return ToolOutcome(f"bad arguments for {name}: {e}", is_error=True)
        except Exception as e:  # tool bugs must not kill the run
            return ToolOutcome(f"tool {name} failed: {type(e).__name__}: {e}",
                               is_error=True)

    # -- path safety ---------------------------------------------------------

    def _safe_path(self, rel: str, writing: bool = False) -> Path:
        p = (self.ctx.workspace / rel).resolve()
        p.relative_to(self.ctx.workspace.resolve())  # raises ValueError if outside
        if writing and ".git" in p.relative_to(self.ctx.workspace.resolve()).parts:
            raise ValueError("editing .git is not allowed")
        return p

    # -- file tools ----------------------------------------------------------

    def _kb_root_names(self) -> set[str]:
        return {d.name for d in self.ctx.kb.dirs}

    def _t_read_file(self, path: str, offset: int = 1, limit: int = 2000) -> ToolOutcome:
        # kb_search shows knowledge-base paths that look like ordinary files;
        # serve them read-only instead of failing and derailing the agent
        first = path.split("/", 1)[0]
        if first in self._kb_root_names():
            return ToolOutcome(self.ctx.kb.read(path, start_line=offset,
                                                end_line=offset + limit - 1))
        try:
            p = self._safe_path(path)
        except ValueError:
            return ToolOutcome(f"path escapes workspace: {path}", is_error=True)
        if not p.is_file():
            return ToolOutcome(f"not a file: {path}", is_error=True)
        lines = p.read_text(errors="replace").splitlines()
        lo = max(0, offset - 1)
        chunk = lines[lo:lo + limit]
        body = "\n".join(f"{i + 1}\t{l}" for i, l in enumerate(chunk, start=lo))
        return ToolOutcome(truncate_middle(body, OUTPUT_CAP))

    def _t_write_file(self, path: str, content: str) -> ToolOutcome:
        try:
            p = self._safe_path(path, writing=True)
        except ValueError as e:
            return ToolOutcome(f"invalid path {path}: {e}", is_error=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return ToolOutcome(f"wrote {len(content)} bytes to {path}")

    def _t_edit_file(self, path: str, old_string: str, new_string: str) -> ToolOutcome:
        try:
            p = self._safe_path(path, writing=True)
        except ValueError as e:
            return ToolOutcome(f"invalid path {path}: {e}", is_error=True)
        if not p.is_file():
            return ToolOutcome(f"not a file: {path}", is_error=True)
        text = p.read_text(errors="replace")
        n = text.count(old_string)
        if n == 0:
            return ToolOutcome("old_string not found; read the file and retry "
                               "with an exact match", is_error=True)
        if n > 1:
            return ToolOutcome(f"old_string matches {n} times; provide more "
                               "context to make it unique", is_error=True)
        p.write_text(text.replace(old_string, new_string, 1))
        return ToolOutcome(f"edited {path}")

    def _t_list_dir(self, path: str = "") -> ToolOutcome:
        try:
            p = self._safe_path(path or ".")
        except ValueError:
            return ToolOutcome(f"path escapes workspace: {path}", is_error=True)
        if not p.is_dir():
            return ToolOutcome(f"not a directory: {path}", is_error=True)
        entries = []
        for child in sorted(p.iterdir()):
            if child.name in (".git", "__pycache__"):
                continue
            suffix = "/" if child.is_dir() else f"  ({child.stat().st_size}B)"
            entries.append(child.name + suffix)
        return ToolOutcome("\n".join(entries) or "(empty)")

    # -- shell tools ---------------------------------------------------------

    def _t_shell(self, command: str, timeout_s: int | None = None) -> ToolOutcome:
        denied = command_denied(command)
        if denied:
            return ToolOutcome(f"command denied by policy (matched {denied!r})",
                               is_error=True)
        t = min(timeout_s or self.ctx.shell_timeout_s, 600)
        from avo.agent.sandbox import build_shell_argv
        argv, _ = build_shell_argv(command, self.ctx.workspace,
                                   mode=self.ctx.sandbox,
                                   runs_dir=self.ctx.runs_dir,
                                   uid=self.ctx.route_uid,
                                   tmpdir=self.ctx.private_tmp)
        env = None
        if self.ctx.private_tmp:  # keep this route's temp files out of /tmp
            import os as _os
            env = {**_os.environ, "TMPDIR": str(self.ctx.private_tmp),
                   "TMP": str(self.ctx.private_tmp)}
        try:
            # cwd handled by the wrapper (--chdir under bwrap); pass for none-mode
            proc = subprocess.run(argv, cwd=self.ctx.workspace, env=env,
                                  capture_output=True, text=True, timeout=t)
            res = ShellResult(proc.returncode, proc.stdout, proc.stderr)
        except subprocess.TimeoutExpired:
            res = ShellResult(-1, "", "", timed_out=True)
        return ToolOutcome(res.render(OUTPUT_CAP), is_error=res.exit_code != 0)

    def _t_gpu_shell(self, command: str, timeout_s: int | None = None) -> ToolOutcome:
        if self.ctx.runner is None:
            return ToolOutcome("gpu_shell unavailable (local runner)", is_error=True)
        denied = command_denied(command)
        if denied:
            return ToolOutcome(f"command denied by policy (matched {denied!r})",
                               is_error=True)
        t = min(timeout_s or self.ctx.gpu_shell_timeout_s, 900)
        res = self.ctx.runner.run_shell(self.ctx.workspace, command, timeout_s=t)
        return ToolOutcome(res.render(OUTPUT_CAP), is_error=res.exit_code != 0)

    # -- knowledge base --------------------------------------------------------

    def _t_kb_search(self, query: str) -> ToolOutcome:
        return ToolOutcome(self.ctx.kb.search(query))

    def _t_kb_read(self, path: str, start_line: int | None = None,
                   end_line: int | None = None, offset: int | None = None,
                   limit: int | None = None) -> ToolOutcome:
        if offset is not None:  # accept read_file-style pagination too
            start_line = offset
            end_line = offset + (limit or 2000) - 1
        return ToolOutcome(self.ctx.kb.read(path, start_line, end_line))

    # -- evaluate / submit -----------------------------------------------------

    def _budget_left(self) -> int:
        return self.ctx.max_evals - self.ctx.evals_used

    def _t_evaluate(self, quick: bool = False) -> ToolOutcome:
        if self._budget_left() <= 0:
            return ToolOutcome("eval budget exhausted for this step", is_error=True)
        self.ctx.evals_used += 1
        result = self.ctx.evaluate_fn(quick=quick)
        note = f"\n[evals remaining this step: {self._budget_left()}]"
        if quick:
            note += " [QUICK eval — reduced grid; submit re-scores the full grid]"
        return ToolOutcome(result.to_json() + note, is_error=not result.correct)

    def _t_profile(self, config_index: int = 0) -> ToolOutcome:
        if self.ctx.profile_fn is None:
            return ToolOutcome("profiling not available for this task", is_error=True)
        if self._budget_left() <= 0:
            return ToolOutcome("eval budget exhausted for this step", is_error=True)
        self.ctx.evals_used += 1
        result = self.ctx.profile_fn(config_index=config_index)
        if not result.correct:
            err = result.error or {}
            return ToolOutcome(f"profiling failed ({err.get('stage')}): "
                               f"{err.get('detail', '')[:800]}", is_error=True)
        summary = result.meta.get("summary", "(no profile summary)")
        return ToolOutcome(truncate_middle(summary, OUTPUT_CAP)
                           + f"\n[evals remaining this step: {self._budget_left()}]")

    def _t_revert(self) -> ToolOutcome:
        if self.ctx.revert_fn is None:
            return ToolOutcome("revert not available", is_error=True)
        return ToolOutcome(self.ctx.revert_fn())

    def _t_submit(self, message: str) -> ToolOutcome:
        if self._budget_left() <= 0:
            return ToolOutcome("eval budget exhausted for this step; submit "
                               "would need an authoritative eval", is_error=True)
        self.ctx.evals_used += 1
        committed, verdict = self.ctx.submit_fn(message)
        if committed:
            return ToolOutcome(verdict, signal="committed")
        return ToolOutcome(verdict + "\nYou may continue improving and submit again.",
                           is_error=True)
