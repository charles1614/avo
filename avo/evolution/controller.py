"""The AVO main loop.

    while budgets allow and versions remain:
        maybe reflect (supervisor) -> guidance
        step_prompt = lineage + last diff + failure summaries + guidance
        agent runs one variation step (edit-evaluate-diagnose ... submit)
        commit passed the gate -> lineage grows; else -> patch + failure summary

The controller owns scoring and the commit gate; the agent only proposes.
State is persisted after every step so a killed run resumes cleanly.
"""
from __future__ import annotations

import datetime
import fcntl
import json
import os
import time
from pathlib import Path

from avo.agent.prompts import (FAILURE_SUMMARY_PROMPT, build_step_prompt,
                               build_system_prompt)
from avo.agent.tools import ToolContext, ToolRegistry
from avo.agent.transcript import Transcript
from avo.agent.variation import VariationAgent
from avo.config import RunConfig, load_task_spec
from avo.eval.cache import EvalCache, eval_key
from avo.eval.scoring import cacheable, gate
from avo.eval.ssh_runner import make_runner
from avo.evolution.lineage import Lineage
from avo.evolution.supervisor import Supervisor
from avo.knowledge.kb import KnowledgeBase
from avo.llm.base import LLMClient
from avo.types import ChatMessage, ScoreResult, TextBlock, Usage

SOURCE_SNAPSHOT_EXTS = {".cu", ".cuh", ".cpp", ".h", ".py", ".json"}


class BudgetTracker:
    def __init__(self, config: RunConfig, llm: LLMClient, prior_usage: Usage,
                 prior_elapsed_s: float):
        self.cfg = config.budgets
        self.llm = llm
        self.prior_usage = prior_usage
        self.prior_elapsed_s = prior_elapsed_s
        self.session_start = time.monotonic()

    @property
    def elapsed_s(self) -> float:
        return self.prior_elapsed_s + (time.monotonic() - self.session_start)

    @property
    def total_tokens(self) -> int:
        return self.prior_usage.total_tokens + self.llm.usage.total_tokens

    @property
    def usd(self) -> float:
        prior = (self.prior_usage.input_tokens * self.llm.cfg.price_input_per_mtok
                 + self.prior_usage.output_tokens * self.llm.cfg.price_output_per_mtok) / 1e6
        return prior + self.llm.cost_usd

    def exhausted(self) -> str | None:
        if self.usd >= self.cfg.max_usd:
            return f"max_usd ({self.usd:.2f} >= {self.cfg.max_usd})"
        if self.total_tokens >= self.cfg.max_total_tokens:
            return f"max_total_tokens ({self.total_tokens})"
        if self.elapsed_s >= self.cfg.max_wall_clock_s:
            return f"max_wall_clock_s ({self.elapsed_s:.0f}s)"
        return None


class Controller:
    def __init__(self, config: RunConfig, llm: LLMClient, project_root: Path,
                 run_dir: Path | None = None):
        self.config = config
        self.llm = llm
        self.root = Path(project_root)
        self.task_dir = self.root / config.task
        self.task = load_task_spec(self.task_dir)
        self.resuming = run_dir is not None

        if run_dir is None:
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            run_dir = self.root / config.runs_dir / f"{config.run_name}-{stamp}"
            run_dir.mkdir(parents=True)
        self.run_dir = Path(run_dir)
        self.state_path = self.run_dir / "state.json"
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        # Exactly one controller per run dir: two agents racing in the same
        # workspace corrupt each other's commits. Lock held for process life.
        self._lock_file = open(self.run_dir / ".lock", "w")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError(
                f"another controller is already running on {self.run_dir} "
                "(runs/<id>/.lock is held); stop it before resuming") from None
        self._lock_file.write(str(os.getpid()))
        self._lock_file.flush()

        self.runner = make_runner(config.runner, self.run_dir.name)
        self.cache = EvalCache(self.run_dir / "evals")
        self.kb = KnowledgeBase([self.root / d for d in config.kb_dirs])
        self.supervisor = Supervisor(config.supervisor,
                                     self.logs_dir / "supervisor.jsonl")
        self.state = self._load_state()
        self.budget = BudgetTracker(
            config, llm,
            prior_usage=Usage(self.state.get("input_tokens", 0),
                              self.state.get("output_tokens", 0)),
            prior_elapsed_s=self.state.get("elapsed_s", 0.0))

        if self.resuming:
            self.lineage = Lineage.load(self.run_dir)
            # an interrupted step's uncommitted work is starting material for
            # the next attempt — capture it before the reset discards it
            patch = self.lineage.capture_uncommitted_patch()
            if patch.strip():
                interrupted = self.state["steps_done"] + 1
                (self.logs_dir / f"step_{interrupted:04d}_final.patch"
                 ).write_text(patch)
                self.state["failure_summaries"].append(
                    f"step {interrupted}: interrupted by a restart before it "
                    "could finish; its uncommitted diff is provided below.")
            self.lineage.reset_workspace()
        else:
            (self.run_dir / "config.yaml").write_text(
                json.dumps(json.loads(config.model_dump_json()), indent=1))
            seed = (self.root / config.seed_dir if config.seed_dir
                    else self.task_dir / self.task.seed_dir)
            if not seed.is_dir():
                raise FileNotFoundError(f"seed directory not found: {seed}")
            self.lineage = Lineage.init_run(self.run_dir, seed)

    # -- state ------------------------------------------------------------------

    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"steps_done": 0, "stagnation": 0, "failure_summaries": [],
                "elapsed_s": 0.0, "input_tokens": 0, "output_tokens": 0}

    def _save_state(self) -> None:
        self.state["elapsed_s"] = self.budget.elapsed_s
        self.state["input_tokens"] = (self.budget.prior_usage.input_tokens
                                      + self.llm.usage.input_tokens)
        self.state["output_tokens"] = (self.budget.prior_usage.output_tokens
                                       + self.llm.usage.output_tokens)
        self.state["usd"] = round(self.budget.usd, 4)
        self.state_path.write_text(json.dumps(self.state, indent=1))

    # -- evaluation ---------------------------------------------------------------

    def evaluate_workspace(self, fresh: bool = False,
                           quick: bool = False) -> ScoreResult:
        harness = self.task_dir / self.task.harness_dir
        task_params = dict(self.config.task_params)
        overrides = task_params.pop("quick_overrides", None)
        if quick and overrides:
            task_params.update(overrides)
        key = eval_key(self.lineage.workspace, harness, task_params,
                       self.runner.identity())
        if not fresh:
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        # Correctness data seed derived from the content hash: any code change
        # shifts the test data unpredictably (no hardcoding outputs), while
        # identical code keeps a stable seed (cache stays meaningful).
        params = {**task_params, "rng_seed": int(key[:12], 16) % (2**31)}
        result = self.runner.score(self.lineage.workspace, harness,
                                   self.task.score_entry, params)
        if cacheable(result):
            self.cache.put(key, result)
        else:
            result.eval_hash = key
        return result

    def profile_workspace(self, config_index: int = 0) -> ScoreResult:
        """Run the task's profile.py harness entry (ncu diagnostics) through
        the same staging/lock/cache pipeline as scoring."""
        harness = self.task_dir / self.task.harness_dir
        base = {**self.config.task_params, "_entry": "profile",
                "profile_config_index": int(config_index)}
        base.pop("quick_overrides", None)
        key = eval_key(self.lineage.workspace, harness, base,
                       self.runner.identity())
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        result = self.runner.score(self.lineage.workspace, harness,
                                   "profile.py", base)
        if cacheable(result):
            self.cache.put(key, result)
        return result

    # -- main loop -----------------------------------------------------------------

    def run(self, log=print) -> dict:
        if not self.lineage.entries():
            log("[avo] evaluating seed ...")
            seed = self.evaluate_workspace()
            if not seed.correct:
                raise RuntimeError(f"seed fails its own scoring: {seed.brief()}")
            self.lineage.record_seed(seed.score, seed.eval_hash or "")
            log(f"[avo] seed committed as v0000, score={seed.score:.4f}")

        while True:
            stop = self._stop_reason()
            if stop:
                log(f"[avo] stopping: {stop}")
                break
            step = self.state["steps_done"] + 1
            guidance = self._maybe_reflect(log)
            result = self._run_step(step, guidance, log)
            self.state["steps_done"] = step
            if result.committed:
                self.state["stagnation"] = 0
                self.state["failure_summaries"] = []
                self.supervisor.note_commit()
            else:
                self.state["stagnation"] += 1
            self._save_state()

        summary = self._finalize(log)
        return summary

    def _stop_reason(self) -> str | None:
        versions = len(self.lineage.entries()) - 1  # exclude seed
        if versions >= self.config.budgets.max_versions:
            return f"max_versions reached ({versions})"
        if self.state["steps_done"] >= self.config.budgets.max_steps:
            return f"max_steps reached ({self.state['steps_done']})"
        return self.budget.exhausted()

    def _maybe_reflect(self, log) -> str:
        reason = self.supervisor.should_reflect(self.state["stagnation"],
                                                self.lineage.entries())
        if not reason:
            return ""
        log(f"[avo] supervisor reflecting: {reason}")
        self.supervisor.note_triggered(self.state["stagnation"])
        guidance = self.supervisor.reflect(
            self.llm, self.lineage.entries(),
            self.state["failure_summaries"], self._source_snapshot(), reason)
        return guidance

    def _run_step(self, step: int, guidance: str, log):
        best = self.lineage.best()
        best_score = best.score if best else 0.0
        transcript = Transcript(self.logs_dir / f"step_{step:04d}.jsonl")

        has_profiler = (self.task_dir / self.task.harness_dir / "profile.py").exists()
        ctx = ToolContext(
            workspace=self.lineage.workspace,
            kb=self.kb,
            evaluate_fn=self.evaluate_workspace,
            submit_fn=lambda msg: self._submit(step, msg, log),
            runner=self.runner if self.config.runner.kind == "ssh" else None,
            profile_fn=self.profile_workspace if has_profiler else None,
            max_evals=self.config.budgets.max_evals_per_step,
        )
        registry = ToolRegistry(ctx)
        agent = VariationAgent(
            self.llm, registry, transcript,
            max_turns=self.config.budgets.max_turns_per_step,
            budget_abort_fn=self.budget.exhausted)

        system = build_system_prompt(
            self.task.brief, best_score,
            self.config.budgets.max_turns_per_step,
            self.config.budgets.max_evals_per_step,
            self.config.gpu_sheet)
        prev = (self.state["failure_summaries"][-1]
                if self.state["failure_summaries"] else "")
        prev_patch = ""
        if prev:  # a failed or interrupted attempt left a diff to build on:
            # this step's own patch exists when it was interrupted mid-attempt,
            # otherwise fall back to the previous step's failed attempt
            for candidate in (f"step_{step:04d}_final.patch",
                              f"step_{step - 1:04d}_final.patch"):
                patch_file = self.logs_dir / candidate
                if patch_file.exists():
                    prev_patch = patch_file.read_text()[:8000]
                    break
        step_prompt = build_step_prompt(self.lineage.entries(),
                                        self.lineage.last_commit_diff(),
                                        prev, guidance, prev_patch=prev_patch)

        log(f"[avo] step {step}: starting variation (best={best_score:.4f})")
        try:
            result = agent.run_step(system, step_prompt)
        except Exception as e:  # LLM/transport failure must not kill the run
            log(f"[avo] step {step}: aborted by error: {type(e).__name__}: {e}")
            from avo.agent.variation import StepResult
            result = StepResult(committed=False, submit_message=None,
                                turns_used=0, evals_used=ctx.evals_used,
                                last_eval_brief=f"step crashed: {type(e).__name__}: {e}",
                                stop_cause="llm_error")
        log(f"[avo] step {step}: {result.stop_cause} after {result.turns_used} turns, "
            f"{result.evals_used} evals, ${self.budget.usd:.2f} spent")

        if not result.committed:
            patch = self.lineage.capture_uncommitted_patch()
            if patch.strip():
                (self.logs_dir / f"step_{step:04d}_final.patch").write_text(patch)
            summary = self._failure_summary(patch, result.last_eval_brief)
            self.state["failure_summaries"].append(f"step {step}: {summary}")
            self.lineage.reset_workspace()
        return result

    def _submit(self, step: int, message: str, log) -> tuple[bool, str]:
        result = self.evaluate_workspace()
        best = self.lineage.best()
        best_score = best.score if best else 0.0
        passed, verdict = gate(result, best_score)
        if passed:
            entry = self.lineage.commit_version(step, result.score, message,
                                                result.eval_hash or "")
            log(f"[avo] step {step}: committed {entry.version} "
                f"score={result.score:.4f} ({message.splitlines()[0][:80]})")
            return True, f"{verdict}\nCommitted as {entry.version}."
        return False, verdict

    def _failure_summary(self, patch: str, last_eval: str) -> str:
        prompt = FAILURE_SUMMARY_PROMPT.format(
            diff=patch[:8000] or "(no changes were made)",
            last_eval=last_eval or "(no evaluation was run)")
        try:
            turn = self.llm.chat(
                system="You summarize failed optimization attempts factually.",
                messages=[ChatMessage("user", [TextBlock(prompt)])])
            return turn.message.text().strip()[:1200]
        except Exception as e:
            return f"(failure summary unavailable: {e}) last eval: {last_eval[:300]}"

    def _source_snapshot(self, max_chars: int = 24_000) -> str:
        parts = []
        for p in sorted(self.lineage.workspace.rglob("*")):
            if p.is_file() and p.suffix in SOURCE_SNAPSHOT_EXTS \
                    and ".git" not in p.parts:
                parts.append(f"===== {p.relative_to(self.lineage.workspace)} =====\n"
                             + p.read_text(errors="replace"))
        return "\n".join(parts)[:max_chars]

    def _finalize(self, log) -> dict:
        entries = self.lineage.entries()
        best = self.lineage.best()
        summary = {
            "run_dir": str(self.run_dir),
            "versions": len(entries) - 1 if entries else 0,
            "steps": self.state["steps_done"],
            "seed_score": entries[0].score if entries else None,
            "best_score": best.score if best else None,
            "best_version": best.version if best else None,
            "tokens": self.budget.total_tokens,
            "usd": round(self.budget.usd, 4),
            "elapsed_s": round(self.budget.elapsed_s),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        self._save_state()
        fcntl.flock(self._lock_file, fcntl.LOCK_UN)
        self._lock_file.close()
        log(f"[avo] done: {json.dumps(summary)}")
        return summary
