# AVO — Agentic Variation Operators

**An open reproduction of [*AVO: Agentic Variation Operators for Autonomous Evolutionary Search*](https://arxiv.org/abs/2603.24517) — an LLM coding agent replaces mutation and crossover in an evolutionary search, and autonomously evolves CUDA attention kernels.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Tests](https://img.shields.io/badge/tests-68%20passing%20offline-brightgreen) ![License](https://img.shields.io/badge/license-MIT-green) ![LLM](https://img.shields.io/badge/LLM-Anthropic%20%7C%20OpenAI--compatible-8A2BE2)

The paper's core idea: instead of `Vary(P) = Generate(Sample(P))` with fixed heuristics, let an autonomous agent drive variation directly — **`Vary(P_t) = Agent(P_t, K, f)`** — with full access to the solution lineage `P_t`, a domain knowledge base `K`, and the scoring function `f`. This repo implements the complete architecture from scratch and applies it to BF16 attention kernels on consumer/datacenter NVIDIA GPUs (RTX 3090 Ti, H100), with a CPU-only toy task for zero-cost end-to-end validation.

---

## How it works

```mermaid
flowchart LR
    subgraph inputs["Inputs"]
        P["Lineage P_t<br/>(x_i, f(x_i)) as git tags"]
        K["Knowledge base K<br/>CUDA/PTX docs · FA2/FA3 · CUTLASS"]
        F["Scoring f<br/>correctness gate → TFLOPS geomean"]
    end
    subgraph loop["Agentic variation step"]
        A["Plan → Edit → Evaluate → Diagnose<br/>tools: files · shell · gpu_shell · kb_search · evaluate · submit"]
    end
    S["Supervisor<br/>stagnation → reflect → steer"]
    G{"Commit gate<br/>correct AND ≥ best?"}
    inputs --> loop
    S -. "conditional guidance" .-> loop
    loop --> G
    G -- "yes: git commit + tag vNNNN" --> P
    G -- "no: patch + failure summary" --> S
```

- **The framework never trusts the agent.** Scoring runs framework-side on a pristine harness copy; correctness inputs are seeded from the code's content hash (hardcoded outputs can't survive an edit); a candidate is committed only if it is numerically correct **and** matches-or-improves the best committed score.
- **Single-lineage evolution, git-native.** Every accepted version is a commit tagged `vNNNN` with score trailers; failed attempts leave a patch + an LLM failure summary that feeds the next step.
- **Self-supervision.** A stagnation monitor reviews the whole trajectory and injects fresh optimization directions when progress stalls.
- **Runs anywhere.** The framework runs on your laptop; kernels compile and benchmark on a remote GPU over plain ssh/rsync. Evals are content-hash cached; killed runs resume exactly where they stopped.

## Results

**Toy task (pure-Python sort, deepseek-v4-flash, $0.30):** the agent took a bubble-sort seed to a hybrid insertion/counting/radix design — **14.9 → 4,320 kElem/s (~290×)** in 3 committed versions, including a gate-rejected regression and a committed revert. Full artifact: [`results/sort-py-20260805-184703/`](results/sort-py-20260805-184703/).

**Attention forward (BF16, head_dim 128, RTX 3090 Ti):** evolution in progress. Reference points on the identical benchmark grid:

| implementation | geomean TFLOPS |
|---|---:|
| PyTorch SDPA (flash) | 70.4 |
| PyTorch SDPA (cuDNN) | 63.0 |
| PyTorch SDPA (efficient) | 47.6 |
| seed kernel (deliberately scalar) | 2.37 |

## Quick start

```bash
git clone <this-repo> && cd avo
uv sync --extra dev --extra report       # exact env from uv.lock
uv run pytest                            # 70 tests — no API key, network, or GPU
# without uv: python3 -m venv .venv && source .venv/bin/activate
#             pip install -r requirements-lock.txt && pip install -e . --no-deps

# fetch the knowledge base (FA2/FA3 + CUTLASS at pinned commits, NVIDIA docs)
uv run python scripts/fetch_kb.py --from-manifest
uv run python scripts/fetch_nvidia_docs.py
```

**Toy evolution (local CPU, ~$0.3 of LLM tokens):**

```bash
export DEEPSEEK_API_KEY=...              # or ANTHROPIC_API_KEY / OPENAI_API_KEY
python scripts/llm_smoke.py --config configs/sort_py.yaml --confirm-spend
avo run --config configs/sort_py.yaml --confirm-spend
avo dashboard --watch 30 --open          # live wandb-style dashboard
```

**Kernel evolution — run directly on the GPU machine** (the default: all
attention configs use `runner.kind: local`). Clone the repo on the box, then:

```bash
bash scripts/setup_host.sh        # CUDA-matched pinned torch + ninja into .venv (idempotent)
export PATH=/usr/local/cuda/bin:$PATH        # nvcc for the harness compiles
avo eval-once  --config configs/attention_3090.yaml            # seed compiles + scores (no LLM)
avo baselines  --config configs/attention_3090.yaml            # SDPA / flash-attn lines
avo run        --config configs/attention_3090.yaml --confirm-spend
avo dashboard --watch 30 --open
```

Same flow on an H100 with `configs/attention_h100.yaml` (paper grid, sm_90a).

<details><summary>Alternative topology: framework on a laptop, GPU over ssh</summary>

Set `runner: {kind: ssh, host: <alias>, scratch: "~/avo_scratch", env_activate:
"export PATH=/usr/local/cuda/bin:$PATH && source ~/avo_scratch/venv/bin/activate"}`
in the config, then provision the host once and preflight:

```bash
bash scripts/provision_remote.sh <ssh-host>
bash scripts/setup_remote.sh <ssh-host> \
  'export PATH=/usr/local/cuda/bin:$PATH && source ~/avo_scratch/venv/bin/activate'
```

Evals then rsync the workspace to the host and run there; the agent gets a
`gpu_shell` tool for remote probes (locally, plain `shell` reaches the GPU).
</details>

## Commands

| command | what it does | LLM cost |
|---|---|---|
| `avo run` / `avo resume` | start / continue an evolution run | **yes** — requires `--confirm-spend` |
| `avo eval-once` | score a workspace once | none |
| `avo baselines` | benchmark SDPA/flash-attn on the task grid | none |
| `avo dashboard` | self-contained live HTML dashboard (`--watch`, `--open`) | none |
| `avo report` / `avo rebench` | lineage plot · re-score all versions, mean±std | none |
| `avo export` | package a run into committable `results/` | none |

**Cost safety is structural**: only `run`, `resume`, and `llm_smoke.py` can reach an LLM API, all three require `--confirm-spend`, and every run is capped by `max_usd` (from API-reported usage × configured prices) plus a `max_total_tokens` backstop, per-step turn/eval budgets, and wall-clock limits. Keys come from environment variables only.

## Project structure

```
avo/                  framework: agent loop · tools · LLM adapters · lineage ·
                      controller · supervisor · local/SSH runners · eval cache ·
                      dashboard/report
tasks/sort_py/        CPU toy task (validates the loop for pennies)
tasks/attention_cuda/ seed CUDA kernel + scoring harness (correctness vs FP32
                      reference with BF16 error-floor tolerance; CUDA-event TFLOPS)
knowledge_base/       curated notes + official NVIDIA docs + FA2/FA3 + CUTLASS
                      (external sources pinned in MANIFEST.json)
configs/              per-run YAML: task · LLM/provider · runner/host · grid · budgets
scripts/              remote preflight · KB fetchers · freshness check · LLM smoke test
results/              exported, committable run artifacts
```

**Adding a new kernel task** requires zero framework changes: a task is a directory with `task.yaml`, a `seed/`, and a `harness/score.py` following the result-JSON contract (`{"correct": bool, "score": float, "configs": [...]}`). The CUDA build/timing utilities in `tasks/attention_cuda/harness/` are kernel-agnostic and copy-pastable.

## Reproducibility

Everything a result depends on is pinned and committed:

- **Code, configs, tests** — this repo; `requirements-lock.txt` pins the env.
- **Knowledge base** — `knowledge_base/external/MANIFEST.json` records exact repo commits and doc versions; `fetch_kb.py --from-manifest` restores them byte-identically; `check_kb_freshness.py` audits drift against upstream.
- **Run artifacts** — `avo export` packages lineage, full per-config score records, baselines, the dashboard, and the evolved solution's **complete git history** as `workspace.bundle` (`git clone workspace.bundle` to walk every version).

The LLM-driven search is not bit-reproducible (sampling); every *claim* is: each committed kernel re-scores to its recorded TFLOPS on matching hardware (`avo eval-once --workspace results/<id>/final_solution --fresh`, or `avo rebench` for mean±std over 10 independent rounds), and the method re-runs end-to-end from the seed.

## Faithfulness to the paper — and deliberate divergences

Implemented as described: single-lineage evolution with git persistence, correct-AND-improves commit gate, failed attempts kept out of the lineage, agentic edit-evaluate-diagnose variation with software-engineering tools and documentation retrieval, correctness-gated TFLOPS-geomean scoring, FA-style benchmark protocol (fixed token count, causal/non-causal, warmup + median of repeats), and conditional supervisor intervention.

Deliberately different:

1. **Hardware**: RTX 3090 Ti (sm_86) / H100 (sm_90a) instead of B200; baselines are PyTorch SDPA + FlashAttention-2/3 instead of cuDNN 9.19/FA4.
2. **Fresh agent conversation per variation step** (lineage table, last-commit diff, and failure summaries injected) instead of one multi-day conversation — bounded context, resumable runs.
3. **The supervisor algorithm and seed kernel are our own designs** — the paper specifies neither.
4. **Scaled budgets**: ~10 versions over hours rather than 40 versions over 7 days.

## Citation

```bibtex
@article{avo2026,
  title   = {AVO: Agentic Variation Operators for Autonomous Evolutionary Search},
  author  = {Chen, Terry and Ye, Zhifan and Xu, Bing and others},
  journal = {arXiv preprint arXiv:2603.24517},
  year    = {2026}
}
```

## License

[MIT](LICENSE) for the code in this repository. Knowledge-base content retains its upstream licenses: FlashAttention and CUTLASS (BSD-3-Clause) are never committed — they are restored at pinned commits by `scripts/fetch_kb.py --from-manifest`. The converted NVIDIA documentation text under `knowledge_base/external/nvidia_docs/` is © NVIDIA and is currently committed **for private-repo reproducibility only** — run `git rm -r --cached knowledge_base/external/nvidia_docs` before making this repository public (the fetch scripts restore it locally).
