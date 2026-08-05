# AVO — Agentic Variation Operators (reproduction)

From-scratch reproduction of **"AVO: Agentic Variation Operators for Autonomous
Evolutionary Search"** ([arXiv:2603.24517](https://arxiv.org/abs/2603.24517)):
an LLM coding agent replaces mutation/crossover in a single-lineage
evolutionary search — `Vary(P_t) = Agent(P_t, K, f)` — applied here to CUDA
attention kernels.

The agent gets software-engineering tools (file editing, shell, knowledge-base
retrieval, `evaluate`, `submit`); the framework owns scoring and the commit
gate: a candidate is committed (git commit + tag) only if it passes numerical
correctness **and** matches-or-improves the best committed score. Stagnation
triggers a supervisor reflection that reviews the trajectory and injects fresh
directions into the next step.

## Layout

- `avo/` — framework: LLM adapters (Anthropic + OpenAI-compatible), agent loop
  + tools, git lineage, controller, supervisor, local/SSH runners, eval cache,
  reporting. `avo/cli.py` is the entry point.
- `tasks/sort_py/` — CPU-only toy task (evolve a pure-Python sort); validates
  the whole loop locally.
- `tasks/attention_cuda/` — the real task: seed CUDA attention kernel
  (BF16 io / FP32 accum, deliberately scalar) + scoring harness (correctness
  vs FP32 reference with a BF16 error-floor tolerance, then TFLOPS via CUDA
  events over a config grid; geomean score).
- `knowledge_base/` — curated docs (online softmax/FA algorithm, CUDA, sm_86,
  sm_90a, PTX mma, build notes); `scripts/fetch_kb.py` adds FA2 sources.
- `configs/` — YAML run configs (task, LLM provider/model/prices, runner,
  grid, budgets, supervisor).

## Cost safety (non-negotiable)

Only `avo run`, `avo resume`, and `scripts/llm_smoke.py` can call an LLM, and
all three **require `--confirm-spend`** (otherwise they print the model,
prices, and budget caps and exit). `eval-once`, `baselines`, `report`,
`rebench`, and `pytest` make zero LLM calls by construction. API keys come
from env vars (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`); fill the
`price_*_per_mtok` fields so the `max_usd` cap works — `max_total_tokens` is
the backstop either way.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,report]"
pytest                      # offline; no keys, no network, no GPU needed
```

Remote GPU host (evals run there over SSH; framework stays local):

```bash
bash scripts/setup_remote.sh asus "source ~/venvs/avo/bin/activate"
```

## Usage

```bash
# score the seed kernel once on the GPU host (no LLM)
avo eval-once --config configs/attention_3090.yaml

# baselines: SDPA flash/cudnn/efficient/math + flash-attn 2/3 (no LLM)
avo baselines --config configs/attention_3090.yaml

# toy evolution run, local + cheap (LLM: requires key + --confirm-spend)
avo run --config configs/sort_py.yaml --confirm-spend

# the real thing (LLM + GPU)
avo run --config configs/attention_3090.yaml --confirm-spend
avo resume --run runs/<id> --confirm-spend       # after Ctrl-C / crash
avo report --run runs/<id>                        # table + plot vs baselines
avo rebench --run runs/<id> --config configs/attention_3090.yaml --rounds 10
```

Run artifacts live in `runs/<id>/`: `workspace/` (git repo; committed versions
tagged `v0000..`), `lineage.jsonl`, `logs/` (per-step transcripts, failure
patches, supervisor log), `evals/` (score cache), `state.json`, `summary.json`.

## Divergences from the paper (deliberate)

1. **Hardware**: RTX 3090 Ti (sm_86) / H100 (sm_90a) instead of B200;
   baselines are PyTorch SDPA + FlashAttention-2/3, not cuDNN 9.19/FA4.
2. **Fresh agent conversation per variation step** with the lineage table,
   last-commit diff, and failure summaries injected — instead of one
   multi-day conversation (bounded context, resumable runs).
3. **Supervisor algorithm and seed kernel are our own designs** — the paper
   specifies neither.
4. **Scaled budgets**: ~10 committed versions over hours, vs 40 versions over
   7 days of continuous evolution.
