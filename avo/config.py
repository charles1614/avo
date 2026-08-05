"""Configuration models. All config comes from YAML; API keys come from env vars only."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

DEFAULT_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai_compat": "OPENAI_API_KEY"}


class LLMConfig(BaseModel):
    provider: Literal["anthropic", "openai_compat"]
    model: str
    base_url: str | None = None  # openai_compat only (Ollama/vLLM/DeepSeek/...)
    api_key_env: str | None = None
    max_tokens: int = 8192
    temperature: float | None = None  # None = provider default
    # Provider-specific request fields passed through verbatim, e.g. DeepSeek
    # thinking: {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}
    # or Anthropic {"thinking": {"type": "enabled", "budget_tokens": 8000}}.
    extra_body: dict = Field(default_factory=dict)
    # USD per million tokens; required (non-zero) for max_usd to mean anything.
    price_input_per_mtok: float = 0.0
    price_output_per_mtok: float = 0.0

    def key_env_name(self) -> str:
        return self.api_key_env or DEFAULT_KEY_ENV[self.provider]

    def resolve_api_key(self) -> str | None:
        return os.environ.get(self.key_env_name())


class RunnerConfig(BaseModel):
    kind: Literal["local", "ssh"] = "local"
    host: str | None = None  # ssh only; may be an ssh_config alias
    scratch: str = "~/avo_scratch"
    env_activate: str = ""  # e.g. "source ~/venvs/avo/bin/activate"
    python: str = "python3"
    arch_flags: list[str] = Field(default_factory=list)
    eval_timeout_s: int = 1800
    shell_timeout_s: int = 120

    def identity(self) -> str:
        """Part of the eval-cache key: same code on a different target != same eval."""
        return f"{self.kind}:{self.host or 'local'}:{' '.join(self.arch_flags)}"


class BudgetConfig(BaseModel):
    max_versions: int = 10
    max_steps: int = 30
    max_wall_clock_s: int = 43200
    max_turns_per_step: int = 40
    max_evals_per_step: int = 8
    max_usd: float = 20.0
    # Backstop that works even when token prices are left at 0.
    max_total_tokens: int = 2_000_000


class SupervisorConfig(BaseModel):
    stagnation_steps: int = 2
    min_rel_improvement: float = 0.01
    window: int = 3


class RunConfig(BaseModel):
    run_name: str
    task: str  # path to task dir, relative to project root
    llm: LLMConfig
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    task_params: dict = Field(default_factory=dict)
    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
    supervisor: SupervisorConfig = Field(default_factory=SupervisorConfig)
    kb_dirs: list[str] = Field(default_factory=lambda: ["knowledge_base"])
    gpu_sheet: str = ""  # short hardware description appended to the system prompt
    runs_dir: str = "runs"


class TaskSpec(BaseModel):
    """Loaded from tasks/<task>/task.yaml."""

    name: str
    brief: str  # task description injected into the agent's system prompt
    seed_dir: str = "seed"
    harness_dir: str = "harness"
    score_entry: str = "score.py"


def load_run_config(path: str | Path) -> RunConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return RunConfig.model_validate(data)


def load_task_spec(task_dir: str | Path) -> TaskSpec:
    with open(Path(task_dir) / "task.yaml") as f:
        data = yaml.safe_load(f)
    return TaskSpec.model_validate(data)
